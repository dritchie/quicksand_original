local m = terralib.require("mem")
local Vector = terralib.require("vector")
local Grid2D = terralib.require("grid").Grid2D
local ad = terralib.require("ad")
local linsolve = terralib.require("linsolve")


-- Generates functions that do Newton's method to try and solve non-linear systems
--    of equations
-- f: Function describing system to be solved. Takes a vector of input nums and fills in a
--    vector of output nums
-- linsolver: Function that solves the linear system Ax = b, taking a grid A and a vector b and
--    fills in a vector x. Can be a fully-determined solve or a least-squares/min-norm solve.
-- Returns true if iteration converged within allowed computation budget, otherwise returns false
function newton(f, linsolver, convergeThresh, maxIters)
	convergeThresh = convergeThresh or 1e-8
	maxIters = maxIters or 100
	return terra(x: &Vector(double))
		var xdual = [Vector(ad.num)].stackAlloc()
		var y = [Vector(ad.num)].stackAlloc()
		var J = [Grid2D(double)].stackAlloc(2, 2)
		var delta = [Vector(double)].stackAlloc()
		var b = [Vector(double)].stackAlloc()
		var converged = false
		for iter=0,maxIters do
			xdual:resize(x.size)
			for i=0,x.size do xdual(i) = ad.num(x(i)) end
			f(&xdual, &y)
			J:resize(y.size, x.size)
			ad.jacobian(&y, &xdual, &J)
			-- Update rule is J(x) * (x' - x) = -f(x)
			b:resize(y.size)
			for i=0,y.size do b(i) = -ad.val(y(i)) end
			linsolver(&J, &b, &delta)
			-- Update x in place
			var deltaNorm = 0.0
			for i=0,x.size do
				deltaNorm = deltaNorm + delta(i)*delta(i)
				x(i) = x(i) + delta(i)
			end
			-- Check if we've converged and are OK to terminate
			-- (by checking the norm of delta)
			if deltaNorm/x.size < convergeThresh then
				converged = true
				break
			end
		end
		m.destruct(b)
		m.destruct(delta)
		m.destruct(J)
		m.destruct(y)
		m.destruct(xdual)
		return converged
	end
end


function newtonLeastSquares(f, convergeThresh, maxIters)
	return newton(f, linsolve.leastSquares, convergeThresh, maxIters)
end

function newtonFullRank(f, convergeThresh, maxIters)
	return newton(f, linsolve.fullRankGeneral, convergeThresh, maxIters)
end

local _module = 
{
	newton = newton,
	newtonLeastSquares = newtonLeastSquares,
	newtonFullRank = newtonFullRank
}


--------- TESTS ----------

local util = terralib.require("util")
local C = terralib.includecstring [[
#include <stdio.h>
#include <math.h>
]]

local errThresh = 1e-6
local checkVal = macro(function(actual, target)
	return quote
		var err = C.fabs(actual-target)
		util.assert(err < errThresh, "Value was %g, should've been %g\n", actual, target)
	end
end)
local assertAllZero = macro(function(fn, vec)
	return quote
		var dualvec = [Vector(ad.num)].stackAlloc()
		for i=0,vec.size do dualvec(i) = ad.num(vec(i)) end
		var outvec = [Vector(ad.num)].stackAlloc()
		fn(&dualvec, &outvec)
		for i=0,outvec.size do
			checkVal(ad.val(outvec(i)), 0.0)
		end
		m.destruct(outvec)
		m.destruct(dualvec)
	end
end)

-- (x-4)*(y-5) = 0
local terra underdetermined(input: &Vector(ad.num), output: &Vector(ad.num))
	output:resize(1)
	output(0) = (input(0) - 4.0) * (input(1) - 5.0)
end

-- (x-4)*(y-5) = 0
-- x*y - 12 = 0
-- (Roots are (4,3) and (2.4,5))
local terra fullydetermined(input: &Vector(ad.num), output: &Vector(ad.num))
	output:resize(2)
	output(0) = (input(0) - 4.0) * (input(1) - 5.0)
	output(1) = input(0)*input(1) - 12.0
end

local terra tests()
	
	-- Underdetermined, least-squares
	var x = [Vector(double)].stackAlloc()
	x:push(0.0); x:push(0.0)
	[newtonLeastSquares(underdetermined)](&x)
	assertAllZero(underdetermined, x)

	-- Fully-determined, least-squares
	x:clear()
	x:push(3.0); x:push(2.0)
	[newtonLeastSquares(fullydetermined)](&x)
	-- C.printf("%g, %g\n", x(0), x(1))
	assertAllZero(fullydetermined, x)

	-- Fully-determined, full rank solver
	x:clear()
	x:push(3.0); x:push(2.0)
	[newtonFullRank(fullydetermined)](&x)
	-- C.printf("%g, %g\n", x(0), x(1))
	assertAllZero(fullydetermined, x)

end

tests()

-------------------------

return _module






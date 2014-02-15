terralib.require("prob")

local m = terralib.require("mem")
local Vector = terralib.require("vector")
local trace = terralib.require("prob.trace")
local inf = terralib.require("prob.inference")
local newton = terralib.require("newton")
local inheritance = terralib.require("inheritance")
local util = terralib.require("util")
local ad = terralib.require("ad")

local C = terralib.includecstring [[
#include <stdio.h>
]]

-- Turn traceUpdate into a function that can be passed into
--    a newton solver
local function makeNewtonFn(adTrace)
	return newton.wrapDualFn(macro(function(x, y)
		return quote
			-- Copy inputs into the nonstructural vars
			adTrace:setRawNonStructuralReals(x)
			-- Update the trace
			[trace.traceUpdate({structureChange=false})](adTrace)
			-- Copy the new manifold values out of the trace
			y:resize(adTrace.manifolds.size)
			for i=0,adTrace.manifolds.size do
				y(i) = adTrace.manifolds(i)
			end
			-- C.printf("%u x %u\n", x.size, y.size)
		end
	end))
end

-- Run Newton's method to project a trace's continuous variables onto
--    the manifold.
-- Modifies currTrace in place.
local terra newtonManifoldProjection(currTrace: &trace.BaseTrace(double))
	-- We need to touch the .manifolds member, so this only works for GlobalTrace
	util.assert([inheritance.isInstanceOf(trace.GlobalTrace(double))](currTrace))
	var globTrace = [&trace.GlobalTrace(double)](currTrace)
	var x = [Vector(double)].stackAlloc()
	globTrace:getRawNonStructuralReals(&x)
	var adTrace = [&trace.GlobalTrace(ad.num)]([trace.BaseTrace(double).deepcopy(ad.num)](globTrace))
	var retcode = [newton.newtonLeastSquares(makeNewtonFn(adTrace))](&x)
	m.delete(adTrace)
	globTrace:setRawNonStructuralReals(&x)
	[trace.traceUpdate({structureChange=false})](globTrace)
	return retcode
end

-- Alternate newton with hmc until we hit the manifold
-- May modify and/or delete currTrace. Returns the updated trace object; use that instead.
local function newtonPlusHMCManifoldProjection(computation, hmcKernelParams, mcmcParams, maxTotalSamps)
	-- Tell HMC to use relaxed manifolds
	hmcKernelParams.relaxManifolds = true
	local maxIters = maxTotalSamps / mcmcParams.numsamps
	return terra(currTrace: &trace.BaseTrace(double), samples: &inf.SampleVectorType(computation))
		var kernel = [HMC(hmcKernelParams)()]
		var retcode = newton.ReturnCodes.DidNotConverge
		[util.optionally(mcmcParams.verbose, function() return quote
			C.printf("Newton-HMC manifold projection\n")
		end end)]
		for iter=0,maxIters do
			[util.optionally(mcmcParams.verbose, function() return quote
				C.printf("iteration %d\n", iter)
			end end)]
			var newTrace = currTrace:deepcopy()
			-- Attempt a Newton projection
			var retcode = newtonManifoldProjection(newTrace)
			-- If that worked, return
			if retcode == newton.ReturnCodes.ConvergedToSolution then
				[util.optionally(mcmcParams.verbose, function() return quote
					C.printf("DONE (Newton projection succeeded)\n")
				end end)]
				m.delete(currTrace)
				currTrace = newTrace
				break
			-- If that didn't work, run HMC for a while before trying again
			else
				[util.optionally(mcmcParams.verbose, function() return quote
					C.printf("Newton projection failed; doing HMC...\n")
				end end)]
				currTrace = [inf.mcmcSample(computation, mcmcParams)](currTrace, kernel, samples)
			end
		end
		m.delete(kernel)
		return currTrace
	end
end


return
{
	newtonManifoldProjection = newtonManifoldProjection,
	newtonPlusHMCManifoldProjection = newtonPlusHMCManifoldProjection
}









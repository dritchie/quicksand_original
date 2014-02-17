local m = terralib.require("mem")
local templatize = terralib.require("templatize")
local inf = terralib.require("prob.inference")
local MCMCKernel = inf.MCMCKernel
local inheritance = terralib.require("inheritance")
local ad = terralib.require("ad")
local trace = terralib.require("prob.trace")
local BaseTraceD = trace.BaseTrace(double)
local BaseTraceAD = trace.BaseTrace(ad.num)
local erp = terralib.require("prob.erph")
local RandVarAD = erp.RandVar(ad.num)
local rand = terralib.require("prob.random")
local Vector = terralib.require("vector")
local util = terralib.require("util")
local larj = terralib.require("prob.larj")
local DualAverage = terralib.require("prob.dualAverage")
local Grid2D = terralib.require("grid").Grid2D
local linsolve = terralib.require("linsolve")
local newton = terralib.require("newton")

local C = terralib.includecstring [[
#include <stdio.h>
]]

-- Kernel for doing Hamiltonian Monte Carlo proposals
-- Only makes proposals to non-structural variables
local HMCKernel = templatize(function(stepSize, numSteps, usePrimalLP,
									  stepSizeAdapt, adaptationRate,
									  temperAcceptHamiltonian, temperGuideHamiltonian,
									  temperInitialMomentum,
									  tempTrajectoryMult,
									  constrainToManifold,
									  relaxManifolds,
									  pmrAlpha, verbosity)

	local doTemperedTrajectories = tempTrajectoryMult ~= 1.0
	local sqrtTemperingMult = math.sqrt(tempTrajectoryMult)
	local doingPMR = pmrAlpha > 0.0
	local targetAcceptRate = (numSteps == 1) and 0.57 or 0.65

	local struct HMCKernelT
	{
		stepSize: double,
		adapter: &DualAverage,
		proposalsMade: uint,
		proposalsAccepted: uint,

		-- Intermediates that we keep around for efficiency
		lastTrace: &BaseTraceD,
		positions: Vector(double),
		gradient: Vector(double),
		momenta: Vector(double),
		invMasses: Vector(double),
		origInvMasses: Vector(double),
		adTrace: &BaseTraceAD,
		dTrace: &BaseTraceD,
		adNonstructuralVars: Vector(&RandVarAD),
		adStructuralVars: Vector(&RandVarAD),
		indepVarNums: Vector(ad.num),

		-- Stuff specifically for dealing with LARJ annealing
		doingLARJ: bool,
		realCompsPerVariable: Vector(int),
		LARJOldComponents: Vector(int),
		LARJNewComponents: Vector(int)
	}
	if constrainToManifold then
		HMCKernelT.entries:insert({field="constraintJacobian", type=Grid2D(double)})
	end
	inheritance.dynamicExtend(MCMCKernel, HMCKernelT)

	terra HMCKernelT:__construct()
		self.stepSize = stepSize
		self.adapter = nil
		self.proposalsMade = 0
		self.proposalsAccepted = 0
		self.lastTrace = nil
		m.init(self.positions)
		m.init(self.gradient)
		m.init(self.momenta)
		m.init(self.invMasses)
		m.init(self.origInvMasses)
		self.adTrace = nil
		self.dTrace = nil
		m.init(self.adNonstructuralVars)
		m.init(self.adStructuralVars)
		m.init(self.indepVarNums)
		self.doingLARJ = false
		m.init(self.realCompsPerVariable)
		m.init(self.LARJOldComponents)
		m.init(self.LARJNewComponents)
		[util.optionally(constrainToManifold, function() return quote
			m.init(self.constraintJacobian)
		end end)]
	end

	terra HMCKernelT:__destruct() : {}
		m.destruct(self.positions)
		m.destruct(self.momenta)
		m.destruct(self.gradient)
		m.destruct(self.invMasses)
		m.destruct(self.origInvMasses)
		if self.adTrace ~= nil then m.delete(self.adTrace) end
		if self.dTrace ~= nil then m.delete(self.dTrace) end
		m.destruct(self.adNonstructuralVars)
		m.destruct(self.adStructuralVars)
		m.destruct(self.indepVarNums)
		if self.adapter ~= nil then m.delete(self.adapter) end
		m.destruct(self.realCompsPerVariable)
		m.destruct(self.LARJOldComponents)
		m.destruct(self.LARJNewComponents)
		[util.optionally(constrainToManifold, function() return quote
			m.destruct(self.constraintJacobian)
		end end)]
	end
	inheritance.virtual(HMCKernelT, "__destruct")

	-- 'update' is the main function we use to update the trace and derived quantites
	--    every time we make a tweak to the variable values
	-- If we're not doing CHMC, then we expect jac to be nil
	terra HMCKernelT:update(pos: &Vector(double), grad: &Vector(double), jac: &Grid2D(double))
		self.indepVarNums:resize(pos.size)
		for i=0,pos.size do
			self.indepVarNums:set(i, ad.num(pos:get(i)))
		end
		var index = 0U
		for i=0,self.adNonstructuralVars.size do
			self.adNonstructuralVars:get(i):setRawRealComponents(&self.indepVarNums, &index)
		end
		[trace.traceUpdate({structureChange=false, relaxManifolds=relaxManifolds})](self.adTrace)
		var lp = self.adTrace.logprob:val()
		var duallp = self.adTrace.logprob
		[util.optionally(temperGuideHamiltonian, function() return quote 
			duallp = duallp / self.adTrace.temperature
		end end)]
		-- If we're doing regular HMC, we just need the gradient
		--    of lp w.r.t to pos
		[util.optionally(not constrainToManifold, function() return quote
			duallp:grad(&self.indepVarNums, grad)	
		end end)]
		-- If we're doing CHMC, then we also need the constraint jacobian
		[util.optionally(constrainToManifold, function() return quote
			duallp:grad(&self.indepVarNums, grad, false)
			ad.jacobian(&([&trace.GlobalTrace(ad.num)](self.adTrace).manifolds),
						&self.indepVarNums, jac)
		end end)]
		return lp
	end

	-- Integrators
	if constrainToManifold then
		-- RATTLE
		local pHalf = macro(function(x, nvars, i) return `x(i) end)
		local q1 = macro(function(x, nvars, i) return `x(nvars + i) end)
		local lambda = macro(function(x, nvars, i) return `x(2*nvars + i) end)
		local function makeNewtonFunction(self, p0, q0, grad0, J0)
			return newton.wrapDualFn(macro(function(x, y)
				return quote
					y:resize(x.size)
					var nvars = q0.size
					var nconstrs = J0.rows
					-- Extract lambda, compute J0^T * lambda
					var lam = [Vector(ad.num)].stackAlloc(nconstrs, 0.0)
					for i=0,lam.size do lam(i) = lambda(x, nvars, i) end
					var j0transposeLambda = [Vector(ad.num)].stackAlloc(J0.cols, 0.0)
					for i=0,J0.cols do
						var sum = ad.num(0.0)
						for j=0,J0.rows do
							sum = sum + J0(j,i)*lam(j)
						end
						j0transposeLambda(i) = sum
					end
					-- Compute equation 1
					for i=0,p0.size do
						pHalf(y, nvars, i) = p0(i) - 0.5*self.stepSize * (grad0(i) + j0transposeLambda(i)) - pHalf(x, nvars, i)
					end
					-- Compute equation 2
					for i=0,q0.size do
						q1(y, nvars, i) = q0(i) + self.stepSize * pHalf(x, nvars, i) * self.invMasses(i) - q1(x, nvars, i)
					end
					-- Update the trace so we can get c(q1)
					var qOne = [Vector(ad.num)].stackAlloc(nvars, 0.0)
					for i=0,q0.size do
						qOne(i) = q1(x, nvars, i)
					end
					self.adTrace:setRawNonStructuralReals(&qOne)
					[trace.traceUpdate({structureChange=false})](self.adTrace)
					-- Compute equation 3
					var globTrace = [&trace.GlobalTrace(ad.num)](self.adTrace)
					for i=0,globTrace.manifolds.size do
						lambda(y, nvars, i) = globTrace.manifolds(i)
					end
					m.destruct(lam)
					m.destruct(j0transposeLambda)
					m.destruct(qOne)
				end
			end))
		end
		local p1 = macro(function(x, nvars, i) return `x(i) end)
		local mu = macro(function(x, nvars, i) return `x(nvars + i) end)
		terra HMCKernelT:integratorStep(pos: &Vector(double), mom: &Vector(double), grad: &Vector(double), jac: &Grid2D(double))
			-- First, solve nonlinear system:
			--   (1) 0 = p0 - step/2 * ( grad0 + J0^T*lambda ) - p1/2
			--   (2) 0 = q0 + step * (p1/2 / m) - q1
			--   (3) 0 = c(q1)

			-- Pack the solution vector as: p1/2, q1, lambda
			var nvars = pos.size 
			var nconstrs = jac.rows
			var x = [Vector(double)].stackAlloc(nvars + nvars + nconstrs, 0.0)
			-- Use p0, q0, 0 as initial guess
			for i=0,nvars do pHalf(x, nvars, i) = mom(i) end
			for i=0,nvars do q1(x, nvars, i) = pos(i) end
			var retcode = [newton.newtonLeastSquares(makeNewtonFunction(self, mom, pos, grad, jac))](&x)
			-- C.printf("%d               \n", retcode)

			-- Then, solve linear system:
			--   (1) p1 = p1/2 - step/2 * ( grad1 + J1^T*mu )
			--   (2) 0  = J1 * (p1 / m)
			-- As a matrix equation b = Ax, this is expressed as:
			--   | -p1/2 + step2 * grad1 | = | -I      -step/2*J1^T | * | p1 |
			--   | 0                     | = |  J1/m   0            | * | mu |

			-- Extract q1 into pos
			for i=0,nvars do
				pos(i) = q1(x, nvars, i)
			end
			-- Compute grad1 and J1
			var lp = self:update(pos, grad, jac)
			-- Set up b vector
			var b = [Vector(double)].stackAlloc(nvars + nconstrs, 0.0)
			for i=0,nvars do p1(b, nvars, i) = -pHalf(x, nvars, i) + self.stepSize/2.0 * grad(i) end
			-- Set up A matrix
			var A = [Grid2D(double)].stackAlloc(b.size, b.size, 0.0)
			for i=0,nvars do
				-- -I block
				A(i,i) = -1.0
			end
			for i=0,jac.rows do
				for j=0,jac.cols do
					-- -step/2*J1^T block
					A(j, nvars + i) = -0.5*self.stepSize*jac(i,j)
					-- J1/m block
					A(nvars + i, j) = jac(i,j)*self.invMasses(j)
				end
			end
			-- Solve
			linsolve.leastSquares(&A, &b, &x)
			-- Extract p1 into mom
			for i=0,nvars do mom(i) = p1(x, nvars, i) end

			m.destruct(x)
			m.destruct(b)
			m.destruct(A)

			return lp
		end
	else
		-- Leapfrog
		terra HMCKernelT:integratorStep(pos: &Vector(double), mom: &Vector(double), grad: &Vector(double), jac: &Grid2D(double))
			-- Momentum update (first half)
			for i=0,mom.size do
				mom(i) = mom(i) + 0.5*self.stepSize*grad(i)
			end
			-- Position update
			for i=0,pos.size do
				pos(i) = pos(i) + self.stepSize*mom(i)*self.invMasses(i)
			end
			-- Momentum update (second half)
			var lp = self:update(pos, grad, jac)
			for i=0,mom.size do
				mom(i) = mom(i) + 0.5*self.stepSize*grad(i)
			end
			return lp
		end
	end

	terra HMCKernelT:sampleNewMomentaRaw(mom: &Vector(double))
		mom:resize(self.positions.size)
		for i=0,mom.size do
			mom(i) = [rand.gaussian_sample(double)](0.0, 1.0) * self.invMasses:get(i)
		end
	end

	if constrainToManifold then
		terra HMCKernelT:sampleNewMomenta(mom: &Vector(double))
			-- Sample new momenta, project onto manifold cotangent bundle
			self:sampleNewMomentaRaw(mom)
			linsolve.nullSpaceProjection(&self.constraintJacobian, mom, mom)
		end
	else
		HMCKernelT.methods.sampleNewMomenta = HMCKernelT.methods.sampleNewMomentaRaw
	end

	if doingPMR then
		terra HMCKernelT:sampleMomenta(mom: &Vector(double))
			-- Can't do a partial update if we don't yet have an existing
			--    set of momenta at this dimensionality
			if mom.size ~= self.positions.size then
				self:sampleNewMomenta(mom)
			-- Otherwise, do a partial update
			else
				var coeff = ad.math.sqrt(1.0 - pmrAlpha*pmrAlpha)
				for i=0,mom.size do
					var m = [rand.gaussian_sample(double)](0.0, 1.0) * self.invMasses:get(i)
					mom(i) = pmrAlpha*mom(i) + coeff*m
				end
			end
		end
	else
		HMCKernelT.methods.sampleMomenta = HMCKernelT.methods.sampleNewMomenta
	end

	terra HMCKernelT:kineticEnergy(mom: &Vector(double))
		var K = 0.0
		for i=0,mom.size do
			var m = mom(i)
			var invmass = self.invMasses:get(i)
			K = K + m*m*invmass
		end
		return -0.5*K
	end

	-- Search for a decent step size
	-- (Code adapted from Stan)
	terra HMCKernelT:searchForStepSize()
		[util.optionally(verbosity > 2, function() return quote
			C.printf("Searching for HMC step size...\n")
		end end)]
		self.stepSize = 1.0
		var pos = m.copy(self.positions)
		var grad = m.copy(self.gradient)
		var mom = [Vector(double)].stackAlloc()
		var jac = [Grid2D(double)].stackAlloc()
		var jacptr : &Grid2D(double) = nil
		[util.optionally(constrainToManifold, function() return quote
			m.destruct(jac)
			jac = m.copy(self.constraintJacobian)
			jacptr = &jac
		end end)]
		self:sampleNewMomenta(&mom)
		var lastlp = self.adTrace.logprob:val()
		var lp = self:integratorStep(&pos, &mom, &grad, jacptr)
		var H = lp - lastlp
		var direction = -1
		if H > ad.math.log(0.5) then direction = 1 end
		while true do
			[util.optionally(verbosity > 2, function() return quote
				C.printf("-------\n")
				C.printf("stepSize: %g\n", self.stepSize)
				C.printf("lp: %g\n", lp)
			end end)]
			m.destruct(pos)
			pos = m.copy(self.positions)
			m.destruct(grad)
			grad = m.copy(self.gradient)
			[util.optionally(constrainToManifold, function() return quote
				m.destruct(jac)
				jac = m.copy(self.constraintJacobian)
				jacptr = &jac
			end end)]
			self:sampleNewMomenta(&mom)
			lp = self:integratorStep(&pos, &mom, &grad, jacptr)
			H = lp - lastlp
			-- If our initial step improved the posterior by more than 0.5, then
			--    keep doubling step size until the initial step improves by as
			--    close as possible to 0.5
			-- If our initial step improved the posterior by less than 0.5, then
			--    keep halving the step size until the initial step improves by
			--    as close as possible to 0.5
			if (direction == 1) and (H <= ad.math.log(0.5)) then
				break
			elseif (direction == -1) and (H >= ad.math.log(0.5)) then
				break
			elseif direction == 1 then
				[util.optionally(verbosity > 3, function() return quote
					C.printf("doubling...\n")
				end end)]
				self.stepSize = self.stepSize * 2.0
			else
				[util.optionally(verbosity > 3, function() return quote
					C.printf("halving...\n")
				end end)]
				self.stepSize = self.stepSize * 0.5
			end
			-- Check for divergence to infinity or collapse to zero.
			if self.stepSize > 1e300 then
				util.fatalError("Bad posterior - HMC step size search diverged to infinity.\n")
			end
			if self.stepSize == 0 then
				util.fatalError("Bad (discontinuous?) posterior - HMC step size search collapsed to zero.\n")
			end
		end
		[util.optionally(verbosity > 2, function() return quote
			C.printf("Done searching for HMC step size\n")
		end end)]
		m.destruct(pos)
		m.destruct(grad)
		m.destruct(mom)
		m.destruct(jac)
	end

	-- Code adapted from Stan
	terra HMCKernelT:updateAdaptiveStepSize(dH: double)
		var EdH = ad.math.exp(dH)
		if EdH > 1.0 then EdH = 1.0 end
		-- Supress NaNs
		if not (EdH == EdH) then EdH = 0.0 end
		var adaptGrad = targetAcceptRate - EdH
		-- Dual averaging
		self.stepSize = ad.math.exp(self.adapter:update(adaptGrad))
	end

	-- Different position variables can have different masses, which
	--    we use during LARJ annealing to make sure that variables being
	--    annealed in/out don't start going all over the place just because
	--    they no longer have any effect on the posterior.
	--    If we don't do this, then we'll end up rejecting almost every
	--    LARJ jump because the reverse log proposal probability will go to
	--    negative infinity.
	terra HMCKernelT:updateInverseMasses(currTrace: &BaseTraceD)
		if self.doingLARJ then
			var alpha = ([&larj.InterpolationTrace(double)](currTrace)).alpha
			-- Variables that are annealing in gradually get less mass (m -> 1)
			-- Variables that are annealing out gradually get more mass (m -> inf)
			var oldScale = (1.0-alpha)
			var newScale = alpha
			for i=0,self.LARJOldComponents.size do
				var j = self.LARJOldComponents:get(i)
				self.invMasses(j) = oldScale*self.origInvMasses(j)
			end
			for i=0,self.LARJNewComponents.size do
				var j = self.LARJNewComponents:get(i)
				self.invMasses(j) = newScale*self.origInvMasses(j)
			end
		end
	end
	terra HMCKernelT:initInverseMasses(currTrace: &BaseTraceD)
		self.origInvMasses:resize(self.positions.size)
		var index = 0
		for i=0,self.adNonstructuralVars.size do
			var numComps = self.realCompsPerVariable(i)
			for j=0,numComps do
				self.origInvMasses(index + j) = self.adNonstructuralVars(i).invMass
			end
			index = index + numComps
		end
		m.destruct(self.invMasses)
		self.invMasses = m.copy(self.origInvMasses)

		if [inheritance.isInstanceOf(larj.InterpolationTrace(double))](currTrace) then
			self.doingLARJ = true
			self.LARJOldComponents:clear()
			self.LARJNewComponents:clear()
			-- The trace can tell us which nonstructural variables are old (annealing out)
			--    or new (annealing in).
			-- Since each variable can have > 0 real components, we need to do some simple
			--    conversions to calculate which real components (i.e. which variables in
			--    the HMC phase space) are old vs. new
			var itrace = [&larj.InterpolationTrace(double)](currTrace)
			var oldnonstructs = itrace:oldNonStructuralVarBits()
			var newnonstructs = itrace:newNonStructuralVarBits()
			var currindex = 0
			for i=0,oldnonstructs.size do
				var numreals = self.realCompsPerVariable:get(i)
				if oldnonstructs:get(i) then
					for j=0,numreals do
						self.LARJOldComponents:push(currindex+j)
					end
				elseif newnonstructs:get(i) then
					for j=0,numreals do
						self.LARJNewComponents:push(currindex+j)
					end
				end
				currindex = currindex + numreals
			end
		else
			self.doingLARJ = false
		end
		self:updateInverseMasses(currTrace)
	end

	terra HMCKernelT:initWithNewTrace(currTrace: &BaseTraceD)
		-- Remember that this is the last trace we saw, so we can
		--    avoid doing all this work if this kernel is repeatedly
		--    called (and it will be).
		self.lastTrace = currTrace

		-- Make sure currTrace starts out reflecting the correct
		--    relaxManifolds status
		if [inheritance.isInstanceOf(trace.GlobalTrace(double))](currTrace) then
			[trace.traceUpdate({structureChange=false, relaxManifolds=relaxManifolds})](currTrace)
		end

		-- Get the real components of currTrace variables
		self.positions:clear()
		self.realCompsPerVariable:clear()
		var currVars = currTrace:freeVars(false, true)
		for i=0,currVars.size do
			var prevsize = self.positions.size
			currVars:get(i):getRawRealComponents(&self.positions)
			self.realCompsPerVariable:push(self.positions.size - prevsize)
		end
		m.destruct(currVars)
		-- We must have some real-valued, non-structural variables
		if self.positions.size == 0 then
			util.fatalError("Cannot use HMC on a program with zero real-valued nonstructurals\n")
		end

		-- Create an AD trace that we can use for calculations
		-- Also remember the nonstructural variables
		if self.adTrace ~= nil then m.delete(self.adTrace) end
		self.adTrace = [BaseTraceD.deepcopy(ad.num)](currTrace)
		m.destruct(self.adNonstructuralVars)
		self.adNonstructuralVars = self.adTrace:freeVars(false, true)
		m.destruct(self.adStructuralVars)
		self.adStructuralVars = self.adTrace:freeVars(true, false)
		-- If we're getting final logprobs from the primal program, then
		--    we also need to make a stratch double trace
		if usePrimalLP then
			if self.dTrace ~= nil then m.delete(self.dTrace) end
			self.dTrace = currTrace:deepcopy()
		end

		-- Initialize the inverse masses for the HMC phase space
		self:initInverseMasses(currTrace)

		-- Initialize the lp gradient (and constraint jacobian)
		[util.optionally(constrainToManifold, function() return quote
			self:update(&self.positions, &self.gradient, &self.constraintJacobian)
		end end)]
		[util.optionally(not constrainToManifold, function() return quote
			self:update(&self.positions, &self.gradient, nil)
		end end)]

		-- Clear out the momenta
		self.momenta:clear()

		-- If the stepSize wasn't specified, try to find a decent one.
		if self.stepSize <= 0.0 then
			self:searchForStepSize()
		end

		-- Initialize the stepsize adapter, if we're doing adaptation
		if stepSizeAdapt and self.adapter == nil then
			self.adapter = DualAverage.heapAlloc(self.stepSize, adaptationRate)
		end

		-- If doing CHMC, then verify:
		--    * That this is an instance of GlobalTrace, and not some other
		--      subclass of BaseTrace (i.e. manifold constraints don't make
		--      sense for LARJ interpolation traces)
		--    * That this trace is actually on the manifold
		--      (i.e. there are manifold constraints and they are satisfied)
		[util.optionally(constrainToManifold, function() return quote
			util.assert([inheritance.isInstanceOf(trace.GlobalTrace(double))](currTrace),
				"CHMC only defined for single execution traces (not interpolation traces)\n")
			var globTrace = [&trace.GlobalTrace(double)](currTrace)
			util.assert(globTrace.manifolds.size > 0,
				"CHMC only defined when manifold constraints are used\n")
			var thresh = 1e-8
			var mnorm = 0.0
			for i=0,globTrace.manifolds.size do
				var mval = globTrace.manifolds(i)
				mnorm = mnorm + mval*mval				
			end
			util.assert(mnorm < thresh,
				"CHMC only defined when manifold constraints are satisfied; manifold norm was %g (greater than threshold %g)\n", mnorm, thresh)
		end end)]
	end

	-- Useful utility
	local terra copyNonstructRealsIntoTrace(reals: &Vector(double), trace: &BaseTraceD)
		var index = 0U
		var currVars = trace:freeVars(false, true)
		for i=0,currVars.size do
			currVars:get(i):setRawRealComponents(reals, &index)
		end
		m.destruct(currVars)
	end

	-- Simulate Hamiltonian dynamics for some number of steps and return the
	-- final logprob
	terra HMCKernelT:simulateTrajectory(pos: &Vector(double), mom: &Vector(double), grad: &Vector(double), jac: &Grid2D(double))
		var newlp : double
		for i=1,numSteps+1 do
			-- Tempering pre-step momentum adjustment
			[util.optionally(doTemperedTrajectories, function() return quote
				if 2*(i-1) < numSteps then
					for j=0,mom.size do mom(j) = mom(j) * sqrtTemperingMult end
				else
					for j=0,mom.size do mom(j) = mom(j) / sqrtTemperingMult end
				end
			end end)]
			-- Simualte one time step
			newlp = self:integratorStep(pos, mom, grad, jac)
			-- Tempering post-step momentum adjustment
			[util.optionally(doTemperedTrajectories, function() return quote
				if 2*i > numSteps then
					for j=0,mom.size do mom(j) = mom(j) / sqrtTemperingMult end
				else
					for j=0,mom.size do mom(j) = mom(j) * sqrtTemperingMult end
				end
			end end)]
			-- Diagnostic output
			[util.optionally(verbosity > 1, function() return quote
				C.printf("lp: %g\n", newlp)
			end end)]
		end
		return newlp
	end

	terra HMCKernelT:next(currTrace: &BaseTraceD) : &BaseTraceD
		self.proposalsMade = self.proposalsMade + 1
		if currTrace ~= self.lastTrace then
			self:initWithNewTrace(currTrace)
		end
		-- In case temperature has been changed since we first copied the trace
		self.adTrace.temperature = currTrace.temperature

		-- Update inverse masses of position variables, if needed (LARJ)
		self:updateInverseMasses(currTrace)

		-- Sample momentum variables
		var mom = m.copy(self.momenta)
		self:sampleMomenta(&mom)
		[util.optionally(temperInitialMomentum, function() return quote
			for i=0,mom.size do
				mom(i) = mom(i) * currTrace.temperature
			end
		end end)]

		-- Initial Hamiltonian
		var H = (self:kineticEnergy(&mom) + currTrace.logprob)
		[util.optionally(temperAcceptHamiltonian, function() return quote
			H = H / currTrace.temperature
		end end)] 

		-- Simulate Hamiltonian dynamics trajectory
		[util.optionally(verbosity > 0, function() return quote
			C.printf("--- TRAJECTORY START ---\n")
			C.printf("dimension: %u\n", self.positions.size)
			C.printf("stepSize: %g\n", self.stepSize)
			C.printf("initialLP: %g\n", currTrace.logprob)
			C.printf("H: %g\n", H)
		end end)]
		var pos = m.copy(self.positions)
		var grad = m.copy(self.gradient)
		var jac = [Grid2D(double)].stackAlloc()
		var jacptr : &Grid2D(double) = nil
		[util.optionally(constrainToManifold, function() return quote
			m.destruct(jac)
			jac = m.copy(self.constraintJacobian)
			jacptr = &jac
		end end)]
		var newlp = self:simulateTrajectory(&pos, &mom, &grad, jacptr)
		[util.optionally(verbosity > 0, function() return quote
			C.printf("--- TRAJECTORY END ---\n")
			C.printf("finalLP: %g              \n", newlp)
		end end)]

		-- If we're doing PMR, we need to negate momentum
		[util.optionally(doingPMR, function() return quote
			for i=0,mom.size do mom(i) = -mom(i) end
		end end)]

		-- If we're using the primal program to calculate final logprobs,
		--    then we run it now
		[util.optionally(usePrimalLP, function() return quote
			copyNonstructRealsIntoTrace(&pos, self.dTrace)
			[trace.traceUpdate({structureChange=false, relaxManifolds=relaxManifolds})](self.dTrace)
			newlp = self.dTrace.logprob
		end end)]

		-- Final Hamiltonian
		var H_new = (self:kineticEnergy(&mom) + newlp)
		[util.optionally(temperAcceptHamiltonian, function() return quote
			H_new = H_new / currTrace.temperature
		end end)] 

		var dH = H_new - H
		[util.optionally(verbosity > 0, function() return quote
			C.printf("H: %g, H_new: %g, dH: %g, exp(dH): %g\n", H, H_new, dH, ad.math.exp(dH))
		end end)]

		-- Update step size, if we're doing adaptive step sizes
		if stepSizeAdapt and self.adapter.adapting then
			self:updateAdaptiveStepSize(dH)
		end

		-- Accept/reject test
		var accept = ad.math.log(rand.random()) < dH
		-- var accept = true
		if accept then
			self.proposalsAccepted = self.proposalsAccepted + 1
			m.destruct(self.positions)
			self.positions = pos
			m.destruct(self.gradient)
			self.gradient = grad
			m.destruct(self.momenta)
			self.momenta = mom
			[util.optionally(constrainToManifold, function() return quote
				m.destruct(self.constraintJacobian)
				self.constraintJacobian = jac
			end end)]
			[util.optionally(not constrainToManifold, function() return quote
				m.destruct(jac)
			end end)]
			-- Copy final reals back into currTrace, flush trace to reconstruct
			-- return value
			copyNonstructRealsIntoTrace(&self.positions, currTrace)
			[trace.traceUpdate({structureChange=false, factorEval=false})](currTrace)
			-- Set logprob, since we didn't eval factors on the flush run.
			[util.optionally(usePrimalLP, function() return quote
				[BaseTraceD.setLogprobFrom(double)](currTrace, self.dTrace)
			end end)]
			[util.optionally(not usePrimalLP, function() return quote
				[BaseTraceD.setLogprobFrom(ad.num)](currTrace, self.adTrace)
			end end)]
			[util.optionally(verbosity > 0, function() return quote
				C.printf("ACCEPT\n")
			end end )]
		else
			m.destruct(pos)
			m.destruct(grad)
			m.destruct(mom)
			m.destruct(jac)
			[util.optionally(verbosity > 0, function() return quote
				C.printf("REJECT\n")
			end end )]
		end

		-- If we're doing PMR, we negate momentum again (so we get momentum
		--    reversals on rejection, rather than on acceptance)
		[util.optionally(doingPMR, function() return quote
			for i=0,self.momenta.size do self.momenta(i) = -self.momenta(i) end
		end end)]

		return currTrace
	end
	inheritance.virtual(HMCKernelT, "next")

	terra HMCKernelT:name() : rawstring return [HMCKernelT.name] end
	inheritance.virtual(HMCKernelT, "name")

	terra HMCKernelT:stats() : {}
		C.printf("Acceptance ratio: %g (%u/%u)\n",
			[double](self.proposalsAccepted)/self.proposalsMade,
			self.proposalsAccepted,
			self.proposalsMade)
	end
	inheritance.virtual(HMCKernelT, "stats")

	m.addConstructors(HMCKernelT)
	return HMCKernelT
end)


-- Convenience method for generating new HMCKernels
local HMC = util.fnWithDefaultArgs(function(...)
	local HMCType = HMCKernel(...)
	return function() return `HMCType.heapAlloc() end
end,
{{"stepSize", -1.0}, {"numSteps", 1}, {"usePrimalLP", false},
 {"stepSizeAdapt", true}, {"adaptationRate", 0.05},
 {"temperAcceptHamiltonian", false}, {"temperGuideHamiltonian", false},
 {"temperInitialMomentum", false},
 {"tempTrajectoryMult", 1.0},
 {"constrainToManifold", false},
 {"relaxManifolds", false},
 {"pmrAlpha", 0.0}, {"verbosity", 0}})




return
{
	globals = { HMC = HMC }
}





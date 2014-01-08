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

local C = terralib.includecstring [[
#include <stdio.h>
]]



-- Kernel for doing Hamiltonian Monte Carlo proposals
-- Only makes proposals to non-structural variables
local HMCKernel = templatize(function(stepSize, numSteps, usePrimalLP,
									  stepSizeAdapt, adaptationRate,
									  pmrAlpha)
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
	inheritance.dynamicExtend(MCMCKernel, HMCKernelT)

	terra HMCKernelT:__construct()
		self.stepSize = stepSize
		self.adapter = nil
		self.proposalsMade = 0
		self.proposalsAccepted = 0
		self.lastTrace = nil
		self.positions = [Vector(double)].stackAlloc()
		self.gradient = [Vector(double)].stackAlloc()
		self.momenta = [Vector(double)].stackAlloc()
		self.invMasses = [Vector(double)].stackAlloc()
		self.origInvMasses = [Vector(double)].stackAlloc() 
		self.adTrace = nil
		self.dTrace = nil
		m.init(self.adNonstructuralVars)
		m.init(self.adStructuralVars)
		m.init(self.indepVarNums)
		self.doingLARJ = false
		self.realCompsPerVariable = [Vector(int)].stackAlloc()
		self.LARJOldComponents = [Vector(int)].stackAlloc()
		self.LARJNewComponents = [Vector(int)].stackAlloc()
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
	end
	inheritance.virtual(HMCKernelT, "__destruct")

	terra HMCKernelT:logProbAndGrad(pos: &Vector(double), grad: &Vector(double))
		self.indepVarNums:resize(pos.size)
		for i=0,pos.size do
			self.indepVarNums:set(i, ad.num(pos:get(i)))
		end
		var index = 0U
		for i=0,self.adNonstructuralVars.size do
			self.adNonstructuralVars:get(i):setRealComponents(&self.indepVarNums, &index)
		end
		-- Also need to force logprob updates for all the structural vars, since their
		--    logprob values will be invalidated be previous calls to grad()
		for i=0,self.adStructuralVars.size do
			self.adStructuralVars(i):rescore()
		end
		[trace.traceUpdate({structureChange=false})](self.adTrace)
		var lp = self.adTrace.logprob:val()
		self.adTrace.logprob:grad(&self.indepVarNums, grad)
		return lp
	end

	-- TODO: The updates in this function are a good candidate for vectorized multiply/add.
	terra HMCKernelT:leapfrog(pos: &Vector(double), grad: &Vector(double))
		-- C.printf("Initial pos:\n")
		-- for i=0,pos.size do C.printf("%g\n", pos(i)) end
		-- C.printf("Initial grad:\n")
		-- for i=0,grad.size do C.printf("%g\n", grad(i)) end
		-- Momentum update (first half)
		for i=0,self.momenta.size do
			self.momenta:set(i, self.momenta:get(i) + 0.5*self.stepSize*grad:get(i))
		end
		-- Position update
		for i=0,pos.size do
			pos:set(i, pos:get(i) + self.stepSize*self.momenta:get(i)*self.invMasses:get(i))
		end
		-- Momentum update (second half)
		var lp = self:logProbAndGrad(pos, grad)
		-- C.printf("Midpoint pos:\n")
		-- for i=0,pos.size do C.printf("%g\n", pos(i)) end
		-- C.printf("Midpoint grad:\n")
		-- for i=0,grad.size do C.printf("%g\n", grad(i)) end
		for i=0,self.momenta.size do
			self.momenta:set(i, self.momenta:get(i) + 0.5*self.stepSize*grad:get(i))
		end

		return lp
	end

	terra HMCKernelT:sampleMomenta()
		self.momenta:resize(self.positions.size)
		for i=0,self.momenta.size do
			var m = [rand.gaussian_sample(double)](0.0, 1.0) * self.invMasses:get(i)
			self.momenta:set(i, m)
		end
	end

	if doingPMR then
		-- Partial momentum refreshment version of momentum sampling
		terra HMCKernelT:sampleMomentaPMR()
			-- Can't do a partial update if we don't yet have an existing
			--    set of momenta at this dimensionality
			if self.momenta.size ~= self.positions.size then
				self:sampleMomenta()
			-- Otherwise, do a partial update
			else
				var coeff = ad.math.sqrt(1.0 - pmrAlpha*pmrAlpha)
				for i=0,self.momenta.size do
					var m = [rand.gaussian_sample(double)](0.0, 1.0) * self.invMasses:get(i)
					self.momenta:set(i, pmrAlpha*self.momenta(i) + coeff*m)
				end
			end
		end
	end

	terra HMCKernelT:kineticEnergy()
		var K = 0.0
		for i=0,self.momenta.size do
			var m = self.momenta:get(i)
			var invmass = self.invMasses:get(i)
			K = K + m*m*invmass
		end
		return -0.5*K
	end

	-- Search for a decent step size
	-- (Code adapted from Stan)
	terra HMCKernelT:searchForStepSize()
		self.stepSize = 1.0
		var pos = m.copy(self.positions)
		var grad = m.copy(self.gradient)
		self:sampleMomenta()
		var lastlp = self.adTrace.logprob:val()
		var lp = self:leapfrog(&pos, &grad)
		var H = lp - lastlp
		var direction = -1
		if H > ad.math.log(0.5) then direction = 1 end
		while true do
			m.destruct(pos)
			m.destruct(grad)
			pos = m.copy(self.positions)
			grad = m.copy(self.gradient)
			self:sampleMomenta()
			lp = self:leapfrog(&pos, &grad)
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
				self.stepSize = self.stepSize * 2.0
			else
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
		m.destruct(pos)
		m.destruct(grad)
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
		-- Get the real components of currTrace variables
		self.positions:clear()
		self.realCompsPerVariable:clear()
		var currVars = currTrace:freeVars(false, true)
		for i=0,currVars.size do
			var prevsize = self.positions.size
			currVars:get(i):getRealComponents(&self.positions)
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

		-- Initialize the gradient
		self:logProbAndGrad(&self.positions, &self.gradient) 

		-- If the stepSize wasn't specified, try to find a decent one.
		if self.stepSize <= 0.0 then
			self:searchForStepSize()
		end

		-- Initialize the stepsize adapter, if we're doing adaptation
		if stepSizeAdapt and self.adapter == nil then
			self.adapter = DualAverage.heapAlloc(self.stepSize, adaptationRate)
		end

		-- Remember that this is the last trace we saw, so we can
		--    avoid doing all this work if this kernel is repeatedly
		--    called (and it will be).
		self.lastTrace = currTrace
	end

	-- Useful utility
	local terra copyNonstructRealsIntoTrace(reals: &Vector(double), trace: &BaseTraceD)
		var index = 0U
		var currVars = trace:freeVars(false, true)
		for i=0,currVars.size do
			currVars:get(i):setRealComponents(reals, &index)
		end
		m.destruct(currVars)
	end

	terra HMCKernelT:next(currTrace: &BaseTraceD) : &BaseTraceD
		self.proposalsMade = self.proposalsMade + 1
		if currTrace ~= self.lastTrace then
			self:initWithNewTrace(currTrace)
		end

		-- Update inverse masses of position variables, if needed (LARJ)
		self:updateInverseMasses(currTrace)

		-- Sample momentum variables
		[util.optionally(doingPMR, function() return `self:sampleMomentaPMR() end)]
		[util.optionally(not doingPMR, function() return `self:sampleMomenta() end)]

		-- -- NEW VERSION OF TEMPERING: Momentum scaling
		-- for i=0,self.momenta.size do
		-- 	self.momenta(i) = currTrace.temperature*self.momenta(i)
		-- end

		-- Initial Hamiltonian
		var H = (self:kineticEnergy() + currTrace.logprob)/currTrace.temperature 

		-- Do leapfrog steps
		-- C.printf("--- TRAJECTORY START ---\n")
		var pos = m.copy(self.positions)
		var grad = m.copy(self.gradient)
		var newlp : double
		for i=0,numSteps do
			newlp = self:leapfrog(&pos, &grad)
		end
		-- C.printf("--- TRAJECTORY END ---\n")

		-- C.printf("%g, %g            \n", grad(grad.size-2), grad(grad.size-1))
		-- C.printf("%g                    \n", self.stepSize)
		-- C.printf("%g              \n", newlp)

		-- If we're doing PMR, we need to negate momentum
		[util.optionally(doingPMR, function() return quote
			for i=0,self.momenta.size do self.momenta(i) = -self.momenta(i) end
		end end)]

		-- If we're using the primal program to calculate final logprobs,
		--    then we run it now
		[util.optionally(usePrimalLP, function() return quote
			copyNonstructRealsIntoTrace(&pos, self.dTrace)
			[trace.traceUpdate({structureChange=false})](self.dTrace)
			newlp = self.dTrace.logprob
		end end)]

		-- Final Hamiltonian
		var H_new = (self:kineticEnergy() + newlp)/currTrace.temperature 

		var dH = H_new - H

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
			m.destruct(self.gradient)
			self.positions = pos
			self.gradient = grad
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
			-- C.printf("ACCEPT                   \n")
		else
			-- C.printf("REJECT                   \n")
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
 {"pmrAlpha", 0.0}})




return
{
	globals = { HMC = HMC }
}





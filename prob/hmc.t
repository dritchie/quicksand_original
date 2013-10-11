local m = terralib.require("mem")
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

local C = terralib.includecstring [[
#include <stdio.h>
]]


-- Univariate dual-averaging optimization (for HMC step size adaptation)
-- Adapted from Stan
local struct DualAverage
{
	gbar: double,
	xbar: double,
	x0: double,
	lastx: double,
	k: int,
	gamma: double,
	adapting: bool,
	minChange: double
}

terra DualAverage:__construct(x0: double, gamma: double, minChange: double) : {}
	self.k = 0
	self.x0 = x0
	self.lastx = x0
	self.gbar = 0.0
	self.xbar = 0.0
	self.gamma = gamma
	self.adapting = true
	self.minChange = minChange
end

terra DualAverage:__construct(x0: double, gamma: double) : {}
	DualAverage.__construct(self, x0, gamma, 0.0001)
end

terra DualAverage:update(g: double)
	if self.adapting then
		self.k = self.k + 1
		var avgeta = 1.0 / (self.k + 10)
		var xbar_avgeta = ad.math.pow(self.k, -0.75)
		var muk = 0.5 * ad.math.sqrt(self.k) / self.gamma
		self.gbar = avgeta*g + (1-avgeta)*self.gbar
		self.lastx = self.x0 - muk*self.gbar
		var oldxbar = self.xbar
		self.xbar = xbar_avgeta*self.lastx + (1-xbar_avgeta)*self.xbar
		if ad.math.fabs(oldxbar - self.xbar) < self.minChange then
			self.adapting = false
		end
	end
	return self.xbar
end

m.addConstructors(DualAverage)


-- Kernel for doing Hamiltonian Monte Carlo proposals
-- Only makes proposals to non-structural variables
-- TODO: automatically adapt step size? (dual averaging scheme from Stan)
-- TODO: variable mass adjustment based on LARJ annealing schedule
local struct HMCKernel
{
	stepSize: double,
	numSteps: uint,
	stepSizeAdapt: bool,
	targetAcceptRate: double,
	adaptationRate: double, 
	adapter: &DualAverage,
	proposalsMade: uint,
	proposalsAccepted: uint,

	-- Intermediates that we keep around for efficiency
	lastTrace: &BaseTraceD,
	positions: Vector(double),
	gradient: Vector(double),
	momenta: Vector(double),
	adTrace: &BaseTraceAD,
	adVars: Vector(&RandVarAD),
	indepVarNums: Vector(ad.num)
}
inheritance.dynamicExtend(MCMCKernel, HMCKernel)

terra HMCKernel:__construct(stepSize: double, numSteps: uint, stepSizeAdapt: bool,
							targetAcceptRate: double, adaptationRate: double)
	self.stepSize = stepSize
	self.numSteps = numSteps
	self.stepSizeAdapt = stepSizeAdapt
	self.targetAcceptRate = targetAcceptRate
	self.adaptationRate = adaptationRate
	self.adapter = nil
	self.proposalsMade = 0
	self.proposalsAccepted = 0
	self.lastTrace = nil
	self.positions = [Vector(double)].stackAlloc()
	self.gradient = [Vector(double)].stackAlloc()
	self.momenta = [Vector(double)].stackAlloc()
	self.adTrace = nil
	m.init(self.adVars)
	m.init(self.indepVarNums)
end

terra HMCKernel:__destruct() : {}
	m.destruct(self.positions)
	m.destruct(self.momenta)
	m.destruct(self.gradient)
	if self.adTrace ~= nil then m.delete(self.adTrace) end
	m.destruct(self.adVars)
	m.destruct(self.indepVarNums)
	if self.adapter ~= nil then m.delete(self.adapter) end
end
inheritance.virtual(HMCKernel, "__destruct")

terra HMCKernel:logProbAndGrad(pos: &Vector(double), grad: &Vector(double))
	self.indepVarNums:resize(pos.size)
	for i=0,pos.size do
		self.indepVarNums:set(i, ad.num(pos:get(i)))
	end
	var index = 0U
	for i=0,self.adVars.size do
		self.adVars:get(i):setRealComponents(&self.indepVarNums, &index)
	end
	[trace.traceUpdate({structureChange=false})](self.adTrace)
	var lp = self.adTrace.logprob:val()
	self.adTrace.logprob:grad(&self.indepVarNums, grad)
	return lp
end

-- TODO: The updates in this function are a good candidate for vectorized multiply/add.
terra HMCKernel:leapfrog(pos: &Vector(double), grad: &Vector(double))
	-- Momentum update (first half)
	for i=0,self.momenta.size do
		self.momenta:set(i, self.momenta:get(i) + 0.5*self.stepSize*grad:get(i))
	end
	-- Position update
	for i=0,pos.size do
		pos:set(i, pos:get(i) + self.stepSize*self.momenta:get(i))
	end
	-- Momentum update (second half)
	var lp = self:logProbAndGrad(pos, grad)
	for i=0,self.momenta.size do
		self.momenta:set(i, self.momenta:get(i) + 0.5*self.stepSize*grad:get(i))
	end
	return lp
end

terra HMCKernel:sampleMomenta()
	self.momenta:resize(self.positions.size)
	for i=0,self.momenta.size do self.momenta:set(i, [rand.gaussian_sample(double)](0.0, 1.0)) end
end

-- Search for a decent step size
-- (Code adapted from Stan)
terra HMCKernel:searchForStepSize()
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
		if (direction == 1) and not (H > ad.math.log(0.5)) then
			break
		elseif (direction == -1) and not (H < ad.math.log(0.5)) then
			break
		elseif direction == 1 then
			self.stepSize = self.stepSize * 2.0
		else
			self.stepSize = self.stepSize * 0.5
		end
		-- Check for divergence to infinity or collapse to zero.
		if self.stepSize > 1e300 then
			util.fatalError("Bad posterior - HMC step size search diverged to infinity.")
		end
		if self.stepSize == 0 then
			util.fatalError("Bad (discontinuous?) posterior - HMC step size search collapsed to zero.")
		end
	end
	m.destruct(pos)
	m.destruct(grad)
end

-- Code adapted from Stan
terra HMCKernel:updateAdaptiveStepSize(dH: double)
	var EdH = ad.math.exp(dH)
	if EdH > 1.0 then EdH = 1.0 end
	-- Supress NaNs
	if EdH ~= EdH then EdH = 0.0 end
	var adaptGrad = self.targetAcceptRate - EdH
	-- Dual averaging
	self.stepSize = self.adapter:update(adaptGrad)
end

terra HMCKernel:initWithNewTrace(currTrace: &BaseTraceD)
	-- Get the real components of currTrace variables
	self.positions:clear()
	var currVars = currTrace:freeVars(false, true)
	for i=0,currVars.size do
		currVars:get(i):getRealComponents(&self.positions)
	end
	m.destruct(currVars)

	-- Create an AD trace that we can use for calculations
	-- Also remember the nonstructural variables
	if self.adTrace ~= nil then m.delete(self.adTrace) end
	self.adTrace = [BaseTraceD.deepcopy(ad.num)](currTrace)
	m.destruct(self.adVars)
	self.adVars = self.adTrace:freeVars(false, true)

	-- Initialize the gradient
	self:logProbAndGrad(&self.positions, &self.gradient)

	-- If the stepSize wasn't specified, try to find a decent one.
	if self.stepSize <= 0.0 then
		self:searchForStepSize()
	end

	-- Initialize the stepsize adapter, if we're doing adaptation
	if self.stepSizeAdapt then
		self.adapter = DualAverage.heapAlloc(self.stepSize, self.adaptationRate)
	end

	-- Remember that this is the last trace we saw, so we can
	--    avoid doing all this work if this kernel is repeatedly
	--    called (and it will be).
	self.lastTrace = currTrace
end

terra HMCKernel:next(currTrace: &BaseTraceD) : &BaseTraceD
	self.proposalsMade = self.proposalsMade + 1
	if currTrace ~= self.lastTrace then
		self:initWithNewTrace(currTrace)
	end

	-- Sample momentum variables
	self:sampleMomenta()

	-- Initial Hamiltonian
	var H = 0.0
	for i=0,self.momenta.size do H = H + self.momenta:get(i)*self.momenta:get(i) end
	H = -0.5*H + currTrace.logprob 

	-- Do leapfrog steps
	var pos = m.copy(self.positions)
	var grad = m.copy(self.gradient)
	var newlp : double
	for i=0,self.numSteps do
		newlp = self:leapfrog(&pos, &grad)
	end

	-- Final Hamiltonian
	var H_new = 0.0
	for i=0,self.momenta.size do H_new = H_new + self.momenta:get(i)*self.momenta:get(i) end
	H_new = -0.5*H_new + newlp

	var dH = H_new - H

	-- Update step size, if we're doing adaptive step sizes
	if self.stepSizeAdapt and self.adapter.adapting then
		self:updateAdaptiveStepSize(dH)
	end

	-- Accept/reject test
	if ad.math.log(rand.random()) < dH then
		self.proposalsAccepted = self.proposalsAccepted + 1
		m.destruct(self.positions)
		m.destruct(self.gradient)
		self.positions = pos
		self.gradient = grad
		-- Copy final reals back into currTrace, flush trace to reconstruct
		-- return value
		var index = 0U
		var currVars = currTrace:freeVars(false, true)
		for i=0,currVars.size do
			currVars:get(i):setRealComponents(&self.positions, &index)
		end
		m.destruct(currVars)
		[trace.traceUpdate({structureChange=false, factorEval=false})](currTrace)
		[BaseTraceD.setLogprobFrom(ad.num)](currTrace, self.adTrace)
	end

	return currTrace
end
inheritance.virtual(HMCKernel, "next")

terra HMCKernel:name() : rawstring return [HMCKernel.name] end
inheritance.virtual(HMCKernel, "name")

terra HMCKernel:stats() : {}
	C.printf("Acceptance ratio: %g (%u/%u)\n",
		[double](self.proposalsAccepted)/self.proposalsMade,
		self.proposalsAccepted,
		self.proposalsMade)
end
inheritance.virtual(HMCKernel, "stats")

m.addConstructors(HMCKernel)


-- Convenience method for generating new HMCKernels
local HMC = inf.makeKernelGenerator(
	terra(stepSize: double, numSteps: uint, stepSizeAdapt: bool,
		  targetAcceptRate: double, adaptationRate: double)
		return HMCKernel.heapAlloc(stepSize, numSteps, stepSizeAdapt,
								   targetAcceptRate, adaptationRate)
	end,
	{stepSize = -1.0, numSteps = 1, stepSizeAdapt = true,
	 targetAcceptRate = 0.65, adaptationRate = 0.05})





return
{
	globals = { HMC = HMC }
}





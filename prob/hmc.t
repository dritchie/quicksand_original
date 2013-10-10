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

local C = terralib.includecstring [[
#include <stdio.h>
]]


-- Kernel for doing Hamiltonian Monte Carlo proposals
-- Only makes proposals to non-structural variables
-- TODO: automatically adapt step size? (dual averaging scheme from Stan)
-- TODO: variable mass adjustment based on LARJ annealing schedule
local struct HMCKernel
{
	stepSize: double,
	numSteps: uint,
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

terra HMCKernel:__construct(stepSize: double, numSteps: uint)
	self.stepSize = stepSize
	self.numSteps = numSteps
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
	var lp : double
	for s=0,self.numSteps do
		-- Momentum update (first half)
		lp = self:logProbAndGrad(pos, grad)
		for i=0,self.momenta.size do
			self.momenta:set(i, self.momenta:get(i) + 0.5*self.stepSize*grad:get(i))
		end
		-- Position update
		for i=0,pos.size do
			pos:set(i, pos:get(i) + self.stepSize*self.momenta:get(i))
		end
		-- Momentum update (second half)
		lp = self:logProbAndGrad(pos, grad)
		for i=0,self.momenta.size do
			self.momenta:set(i, self.momenta:get(i) + 0.5*self.stepSize*grad:get(i))
		end
	end
	return lp
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
	self.momenta:resize(self.positions.size)
	for i=0,self.momenta.size do self.momenta:set(i, [rand.gaussian_sample(double)](0.0, 1.0)) end

	-- Initial Hamiltonian
	var H = 0.0
	for i=0,self.momenta.size do H = H + self.momenta:get(i)*self.momenta:get(i) end
	H = -0.5*H + currTrace.logprob 

	-- Do an HMC step
	var pos = m.copy(self.positions)
	var grad = m.copy(self.gradient)
	var newlp = self:leapfrog(&pos, &grad)

	-- Final Hamiltonian
	var H_new = 0.0
	for i=0,self.momenta.size do H_new = H_new + self.momenta:get(i)*self.momenta:get(i) end
	H_new = -0.5*H_new + newlp

	-- Accept/reject test
	if ad.math.log(rand.random()) < H_new - H then
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
		currTrace.logprob = newlp
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
	terra(stepSize: double, numSteps: uint)
		return HMCKernel.heapAlloc(stepSize, numSteps)
	end,
	-- No way to provide a meaningful default step size...
	{numSteps = 1})





return
{
	globals = { HMC = HMC }
}





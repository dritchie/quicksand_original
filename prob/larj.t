local inheritance = terralib.require("inheritance")
local templatize = terralib.require("templatize")
local trace = terralib.require("prob.trace")
local BaseTrace = trace.BaseTrace
local GlobalTrace = trace.GlobalTrace
local m = terralib.require("mem")
local erp = terralib.require("prob.erph")
local RandVar = erp.RandVar
local util = terralib.require("util")
local specialize = terralib.require("prob.specialize")
local inf = terralib.require("prob.inference")
local MCMCKernel = inf.MCMCKernel
local Vector = terralib.require("vector")
local rand = terralib.require("prob.random")

local C = terralib.includecstring [[
#include <math.h>
#include <stdio.h>
]]


-- Random variable which wraps 2 random variables from different
--    traces but which have the same semantic meaning.
-- These are used as abstractions for simultaneously referring to both
--    instances of the same variable in two different traces through
--    a program.
local struct InterpolationRandVar
{
	rv1: &RandVar,
	rv2: &RandVar
}
inheritance.dynamicExtend(RandVar, InterpolationRandVar)

terra InterpolationRandVar:__construct(rv1: &RandVar, rv2: &RandVar)
	RandVar.__construct(self, rv1.isStructural, rv2.isDirectlyConditioned)
	self.logprob = rv1.logprob
	self.rv1 = rv1
	self.rv2 = rv2
end

terra InterpolationRandVar:__copy(other: &InterpolationRandVar)
	RandVar.__copy(self, other)
	self.rv1 = other.rv1
	self.rv2 = other.rv2
end

terra InterpolationRandVar:__destruct() : {}
	-- Does nothing
	-- I thought about putting this stub in RandVar and just not having a
	--    destructor here, but that's a dangerous pattern: if you forget to
	--    mark a destructor as virtual and there's a stub in the base class,
	--    then the program call the stub instead of the derived class
	--    destructor, and you'll never know this is happening.
end
inheritance.virtual(InterpolationRandVar, "__destruct")

terra InterpolationRandVar:valueTypeID() : uint64
	return self.rv1:valueTypeID()
end
inheritance.virtual(InterpolationRandVar, "valueTypeID")

terra InterpolationRandVar:pointerToValue() : &opaque
	return self.rv1:pointerToValue()
end
inheritance.virtual(InterpolationRandVar, "pointerToValue")

terra InterpolationRandVar:proposeNewValue() : {double, double}
	var fwdPropLP, rvsPropLP = self.rv1:proposeNewValue()
	self.rv2:setValue(self.rv1:pointerToValue())
	self.logprob = self.rv1.logprob
end
inheritance.virtual(InterpolationRandVar, "proposeNewValue")

m.addConstructors(InterpolationRandVar)





-- Trace for the linear interpolation of two programs
-- Only valid for programs with the same set of random variables.
-- This fine, since we only use this for LARJ proposals
local struct InterpolationTrace
{
	trace1: &GlobalTrace,
	trace2: &GlobalTrace,
	alpha: double,
	varlist: Vector(&RandVar),
	interpvars: Vector(&InterpolationRandVar)
}
inheritance.dynamicExtend(BaseTrace, InterpolationTrace)

-- Trace update stuff
local TraceUpdateFnPtr = {&BaseTrace}->{}
InterpolationTrace.traceUpdateVtable = global(Vector(TraceUpdateFnPtr))
Vector(TraceUpdateFnPtr).methods.__construct(InterpolationTrace.traceUpdateVtable:getpointer())
InterpolationTrace.traceUpdateVtableIsFilled = global(bool, false)

terra InterpolationTrace:clearVarList()
	for i=0,self.interpvars.size do
		m.delete(self.interpvars:get(i))
	end
	self.interpvars:clear()
	self.varlist:clear()
end

terra InterpolationTrace:buildVarList()
	self.varlist = [Vector(&RandVar)].stackAlloc()
	self.interpvars = [Vector(&InterpolationRandVar)].stackAlloc()
	var it = self.trace1.vars:iterator()
	util.foreach(it, [quote
		var k, v1 = it:keyvalPointer()
		var v2 = self.trace2.vars:getPointer(@k)
		var n1 = v1.size
		var n2 = 0
		if v2 ~= nil then n2 = v2.size end
		var minN = n1
		if n2 < n1 then minN = n2 end
		-- Variables shared by both traces
		for i=0,minN do
			var ivar = InterpolationRandVar.heapAlloc(v1:get(i), v2:get(i))
			self.varlist:push(ivar)
			self.interpvars:push(ivar)
		end
		-- Variables that only trace1 has
		for i=minN,n1 do
			self.varlist:push(v1:get(i))
		end
		-- Variables that only trace2 has
		for i=n1,n2 do
			self.varlist:push(v2:get(i))
		end
	end])
end

terra InterpolationTrace:updateLPCond()
	self.logprob = (1.0 - self.alpha)*self.trace1.logprob + self.alpha*self.trace2.logprob
	self.conditionsSatisfied = self.trace1.conditionsSatisfied and self.trace2.conditionsSatisfied
end

terra InterpolationTrace:__construct(t1: &GlobalTrace, t2: &GlobalTrace)
	BaseTrace.__construct(self)
	-- JIT traceUpdate functions, if we haven't compiled them yet
	if not [InterpolationTrace.traceUpdateVtableIsFilled] then
		trace.fillTraceUpdateVtable(self)	-- Calls back into Lua
	end
	-- Refer to the correct set of traceUpdate functions
	self.traceUpdateVtable = &[InterpolationTrace.traceUpdateVtable]
	-- Setup other stuff
	self.trace1 = t1
	self.trace2 = t2
	self.alpha = 0.0
	self:updateLPCond()
	self:buildVarList()
end

terra InterpolationTrace:__copy(trace: &InterpolationTrace)
	BaseTrace.__copy(self, trace)
	self.traceUpdateVtable = trace.traceUpdateVtable
	self.trace1 = [&GlobalTrace](trace.trace1:deepcopy())
	self.trace2 = [&GlobalTrace](trace.trace2:deepcopy())
	self.alpha = trace.alpha
	self:updateLPCond()
	self:buildVarList()
end

terra InterpolationTrace:__destruct() : {}
	self:clearVarList()
	m.delete(self.trace1)
	m.delete(self.trace2)
	m.destruct(self.varlist)
	m.destruct(self.interpvars)
end
inheritance.virtual(InterpolationTrace, "__destruct")

terra InterpolationTrace:deepcopy() : &BaseTrace
	var t = m.new(InterpolationTrace)
	t:__copy(self)
	return t
end
inheritance.virtual(InterpolationTrace, "deepcopy")

terra InterpolationTrace:varListPointer() : &Vector(&RandVar)
	return &self.varlist
end
inheritance.virtual(InterpolationTrace, "varListPointer")

terra InterpolationTrace:setAlpha(alpha: double)
	self.alpha = alpha
	self:updateLPCond()
end

-- Generate specialized 'traceUpdate code'
InterpolationTrace.traceUpdate = templatize(function(...)
	local paramtable = specialize.paramListToTable(...)
	return terra(self: &InterpolationTrace) : {}
		var t1 = self.trace1
		var t2 = self.trace2
		[trace.traceUpdate(t1, paramtable)]
		[trace.traceUpdate(t2, paramtable)]
		self:updateLPCond()
	end
end)

m.addConstructors(InterpolationTrace)





-- The actual LARJ algorithm, as an MCMC kernel
local struct LARJKernel
{
	diffusionKernel: &MCMCKernel,
	intervals: uint,
	stepsPerInterval: uint,
	jumpProposalsMade: uint,
	jumpProposalsAccepted: uint,
}
inheritance.dynamicExtend(MCMCKernel, LARJKernel)

-- NOTE: Assumes ownership of 'diffKernel'
terra LARJKernel:__construct(diffKernel: &MCMCKernel, intervals: uint, stepsPerInterval: uint)
	self.diffusionKernel = diffKernel
	self.intervals = intervals
	self.stepsPerInterval = stepsPerInterval
	self.jumpProposalsMade = 0
	self.jumpProposalsAccepted = 0
end

terra LARJKernel:__destruct() : {}
	m.delete(self.diffusionKernel)
end
inheritance.virtual(LARJKernel, "__destruct")

terra LARJKernel:next(currTrace: &BaseTrace)  : &BaseTrace
	self.jumpProposalsMade = self.jumpProposalsMade + 1
	var oldStructTrace = [&GlobalTrace](currTrace:deepcopy())
	var newStructTrace = [&GlobalTrace](currTrace:deepcopy())

	-- Randomly change a structural variable
	var freevars = newStructTrace:freeVars(true, false)
	var v = freevars:get(rand.uniformRandomInt(0, freevars.size))
	var fwdPropLP, rvsPropLP = v:proposeNewValue()
	[trace.traceUpdate(newStructTrace)]
	var oldNumStructVars = freevars.size
	var newNumStructVars = newStructTrace:numFreeVars(true, false)
	fwdPropLP = fwdPropLP + newStructTrace.newlogprob - C.log(oldNumStructVars)
	m.destruct(freevars)

	-- Do annealing, if more than zero annealing steps specified.
	var annealingLpRatio = 0.0
	if self.intervals > 0 and self.stepsPerInterval > 0 then
		var lerpTrace = InterpolationTrace.heapAlloc(oldStructTrace, newStructTrace)
		for ival=0,self.intervals do
			lerpTrace:setAlpha(ival/(self.intervals-1.0))
			for step=0,self.stepsPerInterval do
				annealingLpRatio = annealingLpRatio + lerpTrace.logprob
				lerpTrace = [&InterpolationTrace](self.diffusionKernel:next(lerpTrace))
				annealingLpRatio = annealingLpRatio - lerpTrace.logprob
			end
		end
		oldStructTrace = [&GlobalTrace](lerpTrace.trace1:deepcopy())
		newStructTrace = [&GlobalTrace](lerpTrace.trace2:deepcopy())
		m.delete(lerpTrace)
	end

	-- Finalize accept/reject decision
	rvsPropLP = oldStructTrace:lpDiff(newStructTrace) - C.log(newNumStructVars)
	var acceptanceProb = newStructTrace.logprob - currTrace.logprob + rvsPropLP - fwdPropLP + annealingLpRatio
	var accepted = newStructTrace.conditionsSatisfied and C.log(rand.random()) < acceptanceProb
	m.delete(oldStructTrace)
	if accepted then
		self.jumpProposalsAccepted = self.jumpProposalsAccepted + 1
		m.delete(currTrace)
		return newStructTrace
	else
		m.delete(newStructTrace)
		return currTrace
	end
end
inheritance.virtual(LARJKernel, "next")

terra LARJKernel:name() : rawstring return [LARJKernel.name] end
inheritance.virtual(LARJKernel, "name")

terra LARJKernel:stats() : {}
	C.printf("==JUMP==\nAcceptance ratio: %g (%u/%u)\n",
		[double](self.jumpProposalsAccepted)/self.jumpProposalsMade,
		self.jumpProposalsAccepted,
		self.jumpProposalsMade)
	C.printf("==ANNEALING==\n")
	self.diffusionKernel:stats()
end
inheritance.virtual(LARJKernel, "stats")

m.addConstructors(LARJKernel)



-- Convenience method for generating LARJ Multi-kernels (i.e. kernels that sometimes
--    do LARJ steps and other times do diffusion steps)
-- Can specify separate generators for diffusion and annealing kernels, but defaults to
--    using the same kernel for both if only one is specified.
local function LARJ(diffKernelGen, annealKernelGen)
	assert(diffKernelGen)
	annealKernelGen = annealKernelGen or diffKernelGen
	return inf.makeKernelGenerator(
		terra(jumpFreq: double, intervals: uint, stepsPerInterval: uint)
			var diffKernel = [diffKernelGen()]
			var jumpKernel = LARJKernel.heapAlloc([annealKernelGen()], intervals, stepsPerInterval)
			var kernels = [Vector(&MCMCKernel)].stackAlloc():fill(diffKernel, jumpKernel)
			var freqs = Vector.fromItems(1.0-jumpFreq, jumpFreq)
			return inf.MultiKernel.heapAlloc(kernels, freqs)
		end,
		{jumpFreq = 0.1, intervals = 0, stepsPerInterval = 1})
end


return
{
	globals = { LARJ = LARJ }
}









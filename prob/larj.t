local inheritance = terralib.require("inheritance")
local templatize = terralib.require("templatize")
local trace = terralib.require("trace")
local BaseTrace = trace.BaseTrace
local GlobalTrace = trace.GlobalTrace
local m = terralib.require("mem")
local erp = terralib.require("erph")
local RandVar = erp.RandVar
local util = terralib.require("util")
local specialize = terralib.require("specialize")
local inf = terralib.require("inference")
local MCMCKernel = inf.MCMCKernel
local Vector = terralib.require("vector")
local rand = terralib.require("random")

local C = terralib.includecstring [[
#include <math.h>
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
	BaseTrace.__construct(self, rv1.isStructural, rv2.isDirectlyConditioned)
	self.logprob = rv1.logprob
	self.rv1 = rv1
	self.rv2 = rv2
end

terra InterpolationRandVar:__copy(other: &InterpolationRandVar)
	BaseTrace.__copy(self, other)
	self.rv1 = other.rv1
	self.rv2 = other.rv2
end

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
	self:clearVarList()
	var n1 = self.trace1.varlist.size
	var n2 = self.trace2.varlist.size
	var minN = n1
	if n2 < n1 then minN = n2 end
	var it = self.trace1.vars:iterator()
	util.foreach(it, [quote
		var k, v1 = it:keyvalPointer()
		var v2 = self.trace2:getPointer(@k)
		-- Variables shared by both traces
		for i=0,minN do
			var ivar = InterpolationRandVar.heapAlloc(v1:get(i), v2:get(i))
			self.varlist:push(ivar)
			self.interpvars:push(ivar)
		end
		-- Variables that only trace1 has
		for i=minN,trace1.varlist.size do
			self.varlist:push(v1:get(i))
		end
		-- Variables that only trace2 has
		for i=trace1.varlist.size,trace2.varlist.size do
			self.varlist:push(v2:get(i))
		end
	end])
end

terra InterpolationTrace:updateLPCond()
	self.logprob = (1.0 - self.alpha)*self.trace1.logprob + alpha*self.trace2.logprob
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
	self.trace1 = trace.trace1:deepcopy()
	self.trace2 = trace.trace2:deepcopy()
	self.alpha = trace.alpha
	self:updateLPCond()
	self:buildVarList()
end

terra InterpolationTrace:__destruct()
	self:clearVarList()
	m.delete(self.trace1)
	m.delete(self.trace2)
	m.destruct(self.varlist)
	m.destruct(self.interpvars)
end

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

-- Generate specialized 'traceUpdate code'
InterpolationTrace.traceUpdate = templatize(function(...)
	local paramtable = specialize.paramListToTable(...)
	[trace.traceUpdate(self.trace1, paramtable)]
	[trace.traceUpdate(self.trace2, paramtable)]
	self:updateLPCond()
end)

m.addConstructors(InterpolationTrace)





-- The actual LARJ algorithm, as an MCMC kernel
local struct LARJKernel
{
	diffusionKernel: MCMCKernel,
	intervals: uint,
	stepsPerInterval: uint,
	jumpProposalsMade: uint,
	jumpProposalsAccepted: uint,
}

-- NOTE: Assumes ownership of 'diffKernel'
terra LARJKernel:__construct(diffKernel: MCMCKernel, intervals: uint, stepsPerInterval: uint)
	self.diffusionKernel = diffKernel
	self.intervals = intervals
	self.stepsPerInterval = stepsPerInterval
	self.jumpProposalsMade = 0
	self.jumpProposalsAccepted = 0
end

terra LARJKernel:__destruct()
	m.destruct(self.diffKernel)
end

terra LARJKernel:next(currTrace: &BaseTrace)
	self.jumpProposalsMade = self.jumpProposalsMade + 1
	var oldStructTrace = currTrace:deepcopy()
	var newStructTrace = currTrace:deepcopy()

	-- Randomly change a structural variable
	var freevars = newStructTrace:freeVars(true, false)
	var v = freevars:get(rand.uniformRandomInt(0, freevars.size))
	var fwdPropLP, rvsPropLP = v:proposeNewValue()
	[trace.traceUpdate(newStructTrace)]
	var oldNumStructVars = freevars.size
	var newNumStructVars = newStructTrace:numFreeVars(true, false)
	fwdPropLP = fwdPropLP + newStructTrace.logprob - C.log(oldNumStructVars)
	m.destruct(freevars)

	-- Do annealing, if more than zero annealing steps specified.
	var annealingLpRatio = 0.0
	if self.intervals > 0 and self.stepsPerInterval > 0 then
		var lerpTrace = InterpolationTrace.heapAlloc(oldStructTrace, newStructTrace)
		for ival=0,self.intervals do
			lerpTrace.alpha = ival/(self.intervals-1.0)
			for step=0,self.stepsPerInterval do
				annealingLpRatio = annealingLpRatio + lerpTrace.logprob
				lerpTrace = self.diffusionKernel:next(lerpTrace)
				annealingLpRatio = annealingLpRatio - lerpTrace.logprob
			end
		end
		oldStructTrace = lerpTrace.trace1:deepcopy()
		newStructTrace = lerpTrace.trace2:deepcopy()
		m.delete(lerpTrace)
	end

	-- Finalize accept/reject decision
	
end

terra LARJKernel:stats()
	C.printf("==JUMP==\nAcceptance ratio: %g (%u/%u)\n",
		[double](self.jumpProposalsAccepted)/self.jumpProposalsMade,
		self.jumpProposalsAccepted,
		self.jumpProposalsMade)
	if self.annealingProposalsMade > 0 then
		C.printf("==ANNEALING==\n")
		self.diffusionKernel:stats()
	end
end



-- Convenience method for generating LARJ Multi-kernels (i.e. kernels that sometimes
--  do LARJ steps and other times do diffusion steps)
local function LARJ(innerKernelName, innerKernelGen)
	return inf.makeKernelGenerator(
		terra(jumpFreq: double, intervals: uint, stepsPerInterval: uint)
			var diffKernel = innerKernelGen()
			var jumpKernel = LARJKernel.heapAlloc(innerKernelGen(), intervals, stepsPerInterval)
			var kernels = Vector.fromItems(diffKernel, jumpKernel)
			var names = Vector.fromItems(innerKernelName, "LARJ")
			var freqs = Vector.fromItems(1.0-jumpFreq, jumpFreq)
			return inf.MultiKernel.heapAlloc(kernels, names, freqs)
		end)
end


return
{
	LARJ = LARJ
}









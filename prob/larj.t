local inheritance = terralib.require("inheritance")
local templatize = terralib.require("templatize")
local virtualTemplate = terralib.require("vtemplate")
local trace = terralib.require("prob.trace")
local BaseTrace = trace.BaseTrace
local BaseTraceD = BaseTrace(double)
local GlobalTrace = trace.GlobalTrace
local m = terralib.require("mem")
local erp = terralib.require("prob.erph")
local RandVar = erp.RandVar
local util = terralib.require("util")
local spec = terralib.require("prob.specialize")
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
local InterpolationRandVar = templatize(function(ProbType)
	local RVar = RandVar(ProbType)
	local struct InterpolationRandVarT
	{
		rv1: &RVar,
		rv2: &RVar
	}
	inheritance.dynamicExtend(RVar, InterpolationRandVarT)

	terra InterpolationRandVarT:__construct(rv1: &RVar, rv2: &RVar)
		RVar.__construct(self, rv1.isStructural, rv2.isDirectlyConditioned)
		self.logprob = rv1.logprob
		self.rv1 = rv1
		self.rv2 = rv2
	end

	terra InterpolationRandVarT:__destruct() : {}
		-- Does nothing
		-- I thought about putting this stub in RandVar and just not having a
		--    destructor here, but that's a dangerous pattern: if you forget to
		--    mark a destructor as virtual and there's a stub in the base class,
		--    then the program call the stub instead of the derived class
		--    destructor, and you'll never know this is happening.
	end
	inheritance.virtual(InterpolationRandVarT, "__destruct")

	terra InterpolationRandVarT:valueTypeID() : uint64
		return self.rv1:valueTypeID()
	end
	inheritance.virtual(InterpolationRandVarT, "valueTypeID")

	terra InterpolationRandVarT:pointerToValue() : &opaque
		return self.rv1:pointerToValue()
	end
	inheritance.virtual(InterpolationRandVarT, "pointerToValue")

	terra InterpolationRandVarT:proposeNewValue() : {ProbType, ProbType}
		var fwdPropLP, rvsPropLP = self.rv1:proposeNewValue()
		self.rv2:setValue(self.rv1:pointerToValue())
		self.logprob = self.rv1.logprob
		return fwdPropLP, rvsPropLP
	end
	inheritance.virtual(InterpolationRandVarT, "proposeNewValue")

	terra InterpolationRandVarT:setValue(valptr: &opaque) : {}
		self.rv1:setValue(valptr)
		self.rv2:setValue(valptr)
		self.logprob = self.rv1.logprob
	end
	inheritance.virtual(InterpolationRandVarT, "setValue")

	terra InterpolationRandVarT:getRealComponents(comps: &Vector(ProbType)) : {}
		self.rv1:getRealComponents(comps)
	end
	inheritance.virtual(InterpolationRandVarT, "getRealComponents")

	terra InterpolationRandVarT:setRealComponents(comps: &Vector(ProbType), index: &uint) : {}
		var i = @index
		self.rv1:setRealComponents(comps, index)
		@index = i
		self.rv2:setRealComponents(comps, index)
		self.logprob = self.rv1.logprob
	end
	inheritance.virtual(InterpolationRandVarT, "setRealComponents")

	m.addConstructors(InterpolationRandVarT)
	return InterpolationRandVarT
end)





-- Trace for the linear interpolation of two programs
-- Only valid for programs with the same set of random variables.
-- This fine, since we only use this for LARJ proposals
local InterpolationTrace
InterpolationTrace = templatize(function(ProbType)
	local BaseTraceT = BaseTrace(ProbType)
	local GlobalTraceT = GlobalTrace(ProbType)
	local RandVarT = RandVar(ProbType)
	local InterpolationRandVarT = InterpolationRandVar(ProbType)
	local struct InterpolationTraceT
	{
		trace1: &GlobalTraceT,
		trace2: &GlobalTraceT,
		alpha: double,
		varlist: Vector(&RandVarT),
		interpvars: Vector(&InterpolationRandVarT)
	}
	inheritance.dynamicExtend(BaseTraceT, InterpolationTraceT)

	terra InterpolationTraceT:clearVarList()
		for i=0,self.interpvars.size do
			m.delete(self.interpvars:get(i))
		end
		self.interpvars:clear()
		self.varlist:clear()
	end

	terra InterpolationTraceT:buildVarList()
		self.varlist = [Vector(&RandVarT)].stackAlloc()
		self.interpvars = [Vector(&InterpolationRandVarT)].stackAlloc()
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
				var ivar = InterpolationRandVarT.heapAlloc(v1:get(i), v2:get(i))
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

	terra InterpolationTraceT:updateLPCond()
		self.logprob = (1.0 - self.alpha)*self.trace1.logprob + self.alpha*self.trace2.logprob
		self.conditionsSatisfied = self.trace1.conditionsSatisfied and self.trace2.conditionsSatisfied
	end

	-- NOTE: This assumes ownership of t1 and t2
	terra InterpolationTraceT:__construct(t1: &GlobalTraceT, t2: &GlobalTraceT)
		BaseTraceT.__construct(self)
		-- Setup other stuff
		self.trace1 = t1
		self.trace2 = t2
		self.alpha = 0.0
		self:updateLPCond()
		self:buildVarList()
	end

	InterpolationTraceT.__templatecopy = templatize(function(P)
		local BaseTraceP = BaseTrace(P)
		return terra(self: &InterpolationTraceT, other: &InterpolationTrace(P))
			[BaseTraceT.__templatecopy(P)](self, other)
			self.trace1 = [&GlobalTraceT]([BaseTraceP.deepcopy(ProbType)](other.trace1))
			self.trace2 = [&GlobalTraceT]([BaseTraceP.deepcopy(ProbType)](other.trace2))
			self.alpha = other.alpha
			self:updateLPCond()
			self:buildVarList()
		end
	end)

	terra InterpolationTraceT:__destruct() : {}
		self:clearVarList()
		if self.trace1 ~= nil then m.delete(self.trace1) end
		if self.trace2 ~= nil then m.delete(self.trace2) end
		m.destruct(self.varlist)
		m.destruct(self.interpvars)
	end
	inheritance.virtual(InterpolationTraceT, "__destruct")

	virtualTemplate(InterpolationTraceT, "deepcopy", function(P) return {}->{&BaseTrace(P)} end, function(P)
		local InterpolationTraceP = InterpolationTrace(P)
		return terra(self: &InterpolationTraceT)
			var t = m.new(InterpolationTraceP)
			[InterpolationTraceP.__templatecopy(ProbType)](t, self)
			return t
		end
	end)

	terra InterpolationTraceT:deepcopy() : &BaseTraceT
		return [BaseTraceT.deepcopy(ProbType)](self)
	end
	inheritance.virtual(InterpolationTraceT, "deepcopy")

	terra InterpolationTraceT:varListPointer() : &Vector(&RandVarT)
		return &self.varlist
	end
	inheritance.virtual(InterpolationTraceT, "varListPointer")

	terra InterpolationTraceT:setAlpha(alpha: double)
		self.alpha = alpha
		self:updateLPCond()
	end

	-- Generate specialized 'traceUpdate code'
	virtualTemplate(InterpolationTraceT, "traceUpdate", function(...) return {}->{} end, function(...)
		local paramtable = spec.paramListToTable(...)
		return terra(self: &InterpolationTraceT) : {}
			var t1 = self.trace1
			var t2 = self.trace2
			[trace.traceUpdate(paramtable)](t1)
			[trace.traceUpdate(paramtable)](t2)
			self:updateLPCond()
		end
	end)

	virtualTemplate(InterpolationTraceT, "setLogprobFrom", function(P) return {&BaseTrace(P)}->{} end,
	function(P)
		local val = macro(function(x)
			if P == ad.num and ProbType ~= P then
				return `x:val()
			else
				return x
			end
		end)
		return terra(self: &InterpolationTraceT, other: &BaseTrace(P))
			var otherDyn = [&InterpolationTrace(P)](other)
			self.logprob = val(other.logprob)
			self.rv1.logprob = val(other.rv1.logprob)
			self.rv2.logprob = val(other.rv2.logprob)
		end
	end)

	terra InterpolationTraceT:releaseSubtraces()
		var t1, t2 = self.trace1, self.trace2
		self.trace1 = nil
		self.trace2 = nil
		return t1, t2
	end

	m.addConstructors(InterpolationTraceT)
	return InterpolationTraceT
end)





-- The actual LARJ algorithm, as an MCMC kernel
local InterpolationTraceD = InterpolationTrace(double)
local GlobalTraceD = GlobalTrace(double)
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

terra LARJKernel:next(currTrace: &BaseTraceD)  : &BaseTraceD
	self.jumpProposalsMade = self.jumpProposalsMade + 1
	var oldStructTrace = [&GlobalTraceD](currTrace:deepcopy())
	var newStructTrace = [&GlobalTraceD](currTrace:deepcopy())

	-- Randomly change a structural variable
	var freevars = newStructTrace:freeVars(true, false)
	var v = freevars:get(rand.uniformRandomInt(0, freevars.size))
	var fwdPropLP, rvsPropLP = v:proposeNewValue()
	[trace.traceUpdate()](newStructTrace)
	var oldNumStructVars = freevars.size
	var newNumStructVars = newStructTrace:numFreeVars(true, false)
	fwdPropLP = fwdPropLP + newStructTrace.newlogprob - C.log(oldNumStructVars)
	m.destruct(freevars)

	-- Do annealing, if more than zero annealing steps specified.
	var annealingLpRatio = 0.0
	if self.intervals > 0 and self.stepsPerInterval > 0 then
		var lerpTrace = InterpolationTraceD.heapAlloc(oldStructTrace, newStructTrace)
		for ival=0,self.intervals do
			lerpTrace:setAlpha(ival/(self.intervals-1.0))
			for step=0,self.stepsPerInterval do
				annealingLpRatio = annealingLpRatio + lerpTrace.logprob
				lerpTrace = [&InterpolationTraceD](self.diffusionKernel:next(lerpTrace))
				annealingLpRatio = annealingLpRatio - lerpTrace.logprob
			end
		end
		oldStructTrace, newStructTrace = lerpTrace:releaseSubtraces()
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
--    using the same type kernel for both if only one is specified.
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









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
local ad = terralib.require("ad")
local DualAverage = terralib.require("prob.dualAverage")

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
		RVar.__construct(self, rv1.isStructural, rv1.isDirectlyConditioned, rv1.traceDepth, rv1.mass)
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

	terra InterpolationRandVarT:checkInvalidStructInterpOp()
		if self.isStructural then
			util.fatalError("Invalid operation on a structural InterpolationRandVar\n")
		end
	end
	util.inline(InterpolationRandVarT.methods.checkInvalidStructInterpOp)

	terra InterpolationRandVarT:pointerToValue() : &opaque
		self:checkInvalidStructInterpOp()
		return self.rv1:pointerToValue()
	end
	inheritance.virtual(InterpolationRandVarT, "pointerToValue")

	terra InterpolationRandVarT:proposeNewValue() : {ProbType, ProbType}
		self:checkInvalidStructInterpOp()
		var fwdPropLP, rvsPropLP = self.rv1:proposeNewValue()
		self.rv2:setRawValue(self.rv1:pointerToValue())
		self.logprob = self.rv1.logprob
		return fwdPropLP, rvsPropLP
	end
	inheritance.virtual(InterpolationRandVarT, "proposeNewValue")

	terra InterpolationRandVarT:setRawValue(valptr: &opaque) : {}
		self:checkInvalidStructInterpOp()
		self.rv1:setRawValue(valptr)
		self.rv2:setRawValue(valptr)
		self.logprob = self.rv1.logprob
	end
	inheritance.virtual(InterpolationRandVarT, "setRawValue")

	terra InterpolationRandVarT:getRealComponents(comps: &Vector(ProbType)) : {}
		self:checkInvalidStructInterpOp()
		self.rv1:getRealComponents(comps)
	end
	inheritance.virtual(InterpolationRandVarT, "getRealComponents")

	terra InterpolationRandVarT:setRealComponents(comps: &Vector(ProbType), index: &uint) : {}
		self:checkInvalidStructInterpOp()
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
-- This is used in LARJ, where trace1 stores the 'old structure'
--    and trace2 stores the 'new structure'
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
		interpvars: Vector(&InterpolationRandVarT),
		oldnonstructs: Vector(bool),
		newnonstructs: Vector(bool),
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
		self.oldnonstructs = [Vector(bool)].stackAlloc()
		self.newnonstructs = [Vector(bool)].stackAlloc()
		self.interpvars = [Vector(&InterpolationRandVarT)].stackAlloc()
		-- Look through all the addresses in trace1. This will give us
		--   all variables unique to trace1, as well as variables in both
		--   traces that share the same address
		var it = self.trace1.vars:iterator()
		[util.foreach(it, quote
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
				if not ivar.isStructural then
					self.oldnonstructs:push(false)
					self.newnonstructs:push(false)
				end
				self.interpvars:push(ivar)
			end
			-- Variables that only trace1 has
			for i=minN,n1 do
				self.varlist:push(v1:get(i))
				if not v1:get(i).isStructural then
					self.oldnonstructs:push(true)
					self.newnonstructs:push(false)
				end
			end
			-- Variables that only trace2 has
			for i=n1,n2 do
				self.varlist:push(v2:get(i))
				if not v2:get(i).isStructural then
					self.oldnonstructs:push(false)
					self.newnonstructs:push(true)
				end
			end
		end)]
		-- Now we have to look through all addresses in trace2 to find any
		--    other variables unique to trace2.
		it = self.trace2.vars:iterator()
		[util.foreach(it, quote
			var k, v2 = it:keyvalPointer()
			var v1 = self.trace1.vars:getPointer(@k)
			if v1 == nil then
				for i=0,v2.size do
					self.varlist:push(v2:get(i))
					if not v2:get(i).isStructural then
						self.oldnonstructs:push(false)
						self.newnonstructs:push(true)
					end
				end
			end
		end)]
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
		m.destruct(self.oldnonstructs)
		m.destruct(self.newnonstructs)
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

	terra InterpolationTraceT:oldNonStructuralVarBits()
		return &self.oldnonstructs
	end
	terra InterpolationTraceT:newNonStructuralVarBits()
		return &self.newnonstructs
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
			self.trace1.logprob = val(otherDyn.trace1.logprob)
			self.trace2.logprob = val(otherDyn.trace2.logprob)
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
local LARJKernel = templatize(function(intervals, stepsPerInterval, doDepthBiasedSelection,
	                                   branchFactor, branchFactorAdapt, targetAcceptRate, adaptRate)
	local struct LARJKernelT
	{
		diffusionKernel: &MCMCKernel,
		jumpProposalsMade: uint,
		jumpProposalsAccepted: uint,
		currBranchFactor: double,
		adapter: DualAverage
	}
	inheritance.dynamicExtend(MCMCKernel, LARJKernelT)

	-- NOTE: Assumes ownership of 'diffKernel'
	terra LARJKernelT:__construct(diffKernel: &MCMCKernel)
		self.diffusionKernel = diffKernel
		self.jumpProposalsMade = 0
		self.jumpProposalsAccepted = 0
		self.currBranchFactor = branchFactor
		self.adapter = DualAverage.stackAlloc(branchFactor, targetAcceptRate, adaptRate)
	end

	terra LARJKernelT:__destruct() : {}
		m.delete(self.diffusionKernel)
	end
	inheritance.virtual(LARJKernelT, "__destruct")

	terra LARJKernelT:updateAdaptiveBranchFactor(d: double)
		var e = ad.math.exp(d)
		if e > 1.0 then e = 1.0 end
		-- Suppres NaNs
		if not (e == e) then e = 0.0 end
		var agrad = targetAcceptRate - e
		-- Dual averaging
		self.currBranchFactor = self.adapter:update(agrad)
		-- Clamp, in case we're getting lots of negative feedback that's driving
		--    the branch factor down
		self.currBranchFactor = C.fmax(self.currBranchFactor, 1.0)
	end

	-- Returns the weights and the sum of weights
	terra LARJKernelT:depthBiasedSelectionWeights(vars: &Vector(&RandVar(double)))
		var weights = [Vector(double)].stackAlloc(vars.size, 0.0)
		var wsum = 0.0
		for i=0,vars.size do
			var fv = vars(i)
			weights(i) = C.pow(self.currBranchFactor, -1.0*fv.traceDepth)
			wsum = wsum + weights(i)
		end
		return weights, wsum
	end

	terra LARJKernelT:next(currTrace: &BaseTraceD)  : &BaseTraceD
		self.jumpProposalsMade = self.jumpProposalsMade + 1
		var oldStructTrace = [&GlobalTraceD](currTrace:deepcopy())
		var newStructTrace = [&GlobalTraceD](currTrace:deepcopy())

		-- Randomly change a structural variable
		var freevars = newStructTrace:freeVars(true, false)
		var v : &RandVar(double) = nil

		var fwdPropLP, rvsPropLP = 0.0, 0.0
		[util.optionally(doDepthBiasedSelection, function() return quote
			-- Skew variable selection based on trace depth
			var weights, wsum = self:depthBiasedSelectionWeights(&freevars)
			var which = [rand.multinomial_sample(double)](weights)
			v = freevars:get(which)
			-- C.printf("--------------                 \n")
			-- C.printf("currBranchFactor: %g\n", self.currBranchFactor)
			-- C.printf("which: %d\n", which)
			-- C.printf("weights(which): %g\n", weights(which))
			-- C.printf("wsum: %g\n", wsum)
			-- C.printf("weights(which)/wsum: %g\n", weights(which)/wsum)
			-- C.printf("C.log(weights(which)/wsum): %g\n", C.log(weights(which)/wsum))
			fwdPropLP = fwdPropLP + C.log(weights(which)/wsum)
			m.destruct(weights)
		end end)]
		[util.optionally(not doDepthBiasedSelection, function() return quote
			-- Select variable uniformly at random
			v = freevars:get(rand.uniformRandomInt(0, freevars.size))
			var oldNumStructVars = freevars.size
			fwdPropLP = fwdPropLP - C.log(oldNumStructVars)
		end end)]

		var fplp, rplp = v:proposeNewValue()
		fwdPropLP = fwdPropLP + fplp
		rvsPropLP = rvsPropLP + rplp
		[trace.traceUpdate()](newStructTrace)
		fwdPropLP = fwdPropLP + newStructTrace.newlogprob
		m.destruct(freevars)

		-- Do annealing, if more than zero annealing steps specified.
		var annealingLpRatio = 0.0
		[util.optionally(intervals > 0 and stepsPerInterval > 0,
		function() return quote
			var lerpTrace = InterpolationTraceD.heapAlloc(oldStructTrace, newStructTrace)
			for ival=0,intervals do
				lerpTrace:setAlpha(ival/(intervals-1.0))
				for step=0,stepsPerInterval do
					annealingLpRatio = annealingLpRatio + lerpTrace.logprob
					lerpTrace = [&InterpolationTraceD](self.diffusionKernel:next(lerpTrace))
					annealingLpRatio = annealingLpRatio - lerpTrace.logprob
				end
			end
			oldStructTrace, newStructTrace = lerpTrace:releaseSubtraces()
			m.delete(lerpTrace)
		end end)]

		rvsPropLP = rvsPropLP + oldStructTrace:lpDiff(newStructTrace)
		[util.optionally(doDepthBiasedSelection, function() return quote
			var newfreevars = newStructTrace:freeVars(true, false)
			var weights, wsum = self:depthBiasedSelectionWeights(&newfreevars)
			var which = [rand.multinomial_sample(double)](weights)
			rvsPropLP = rvsPropLP + C.log(weights(which)/wsum)
			m.destruct(weights)
			m.destruct(newfreevars)
		end end)]
		[util.optionally(not doDepthBiasedSelection, function() return quote
			var newNumStructVars = newStructTrace:numFreeVars(true, false)
			rvsPropLP = rvsPropLP - C.log(newNumStructVars)
		end end)]

		var acceptanceProb = (newStructTrace.logprob - currTrace.logprob)/currTrace.temperature  + rvsPropLP - fwdPropLP + annealingLpRatio

		-- C.printf("--------------                 \n")
		-- C.printf("newStructTrace.logprob: %g\n", newStructTrace.logprob)
		-- C.printf("currTrace.logprob: %g\n", currTrace.logprob)
		-- C.printf("rvsPropLP: %g\n", rvsPropLP)
		-- C.printf("fwdPropLP: %g\n", fwdPropLP)
		-- C.printf("acceptanceProb: %g\n", acceptanceProb)

		-- Adapt branchFactor, if requested
		[util.optionally(doDepthBiasedSelection and branchFactorAdapt, function()
			return `self:updateAdaptiveBranchFactor(acceptanceProb)
		end)]
		
		-- Finalize accept/reject decision
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
	inheritance.virtual(LARJKernelT, "next")

	terra LARJKernelT:name() : rawstring return [LARJKernelT.name] end
	inheritance.virtual(LARJKernelT, "name")

	terra LARJKernelT:stats() : {}
		C.printf("==JUMP==\nAcceptance ratio: %g (%u/%u)\n",
			[double](self.jumpProposalsAccepted)/self.jumpProposalsMade,
			self.jumpProposalsAccepted,
			self.jumpProposalsMade)
		C.printf("==ANNEALING==\n")
		self.diffusionKernel:stats()
	end
	inheritance.virtual(LARJKernelT, "stats")

	m.addConstructors(LARJKernelT)
	return LARJKernelT
end)



-- Convenience method for generating LARJ Multi-kernels (i.e. kernels that sometimes
--    do LARJ steps and other times do diffusion steps)
-- Can specify separate generators for diffusion and annealing kernels, but defaults to
--    using the same type kernel for both if only one is specified.
local function LARJ(diffKernelGen, annealKernelGen)
	assert(diffKernelGen)
	annealKernelGen = annealKernelGen or diffKernelGen
	return util.fnWithDefaultArgs(function(jumpFreq, ...)
		local haveManualJumpFreq = jumpFreq > 0.0
		local LARJType = LARJKernel(...)
		-- If a desired jump frequency is provided, then we use that
		-- Otherwise, we do jumps with freq. proportional to the % of struct. vars.
		local selectFn = terra(kernels: &Vector(&inf.MCMCKernel), currTrace: &BaseTraceD)
			var jumpThresh = 0.0
			[util.optionally(haveManualJumpFreq, function() return quote
				jumpThresh = jumpFreq
			end end)]
			[util.optionally(not haveManualJumpFreq, function() return quote
				var numstructs = currTrace:numFreeVars(true, false)
				var numnonstructs = currTrace:numFreeVars(false, true)
				jumpThresh = [double](numstructs)/(numstructs+numnonstructs)
			end end)]
			var r = rand.random()
			if r < jumpThresh then
				return kernels(1)
			else
				return kernels(0)
			end
		end
		local MultiKernelT = inf.MultiKernel(selectFn)
		return function()
			return quote
				var diffKernel = [diffKernelGen()]
				var annealKernel = [annealKernelGen()]
				var jumpKernel = LARJType.heapAlloc(annealKernel)
				var kernels = [Vector(&MCMCKernel)].stackAlloc():fill(diffKernel, jumpKernel)
			in
				MultiKernelT.heapAlloc(kernels)
			end
		end
	end,
	{{"jumpFreq", 0.0}, {"intervals", 0}, {"stepsPerInterval", 1},
	 {"doDepthBiasedSelection", false}, {"branchFactor", 1.0},
	 {"branchFactorAdapt", true}, {"targetAcceptRate", 0.25}, {"adaptRate", 0.05}})
end

return
{
	InterpolationTrace = InterpolationTrace,
	globals = { LARJ = LARJ }
}









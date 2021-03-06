local trace = require("prob.trace")
local spec = require("prob.specialize")
local BaseTrace = trace.BaseTrace
local BaseTraceD = BaseTrace(double)
local RandExecTrace = trace.RandExecTrace
local TraceWithRetVal = trace.TraceWithRetVal
local inheritance = require("inheritance")
local m = require("mem")
local rand = require("prob.random")
local templatize = require("templatize")
local Vector = require("vector")
local erp = require("prob.erph")
local ad = require("ad")
local util = require("util")

local C = terralib.includecstring [[
#include <stdio.h>
#include <math.h>
inline FILE* getstderr() { return stderr; }
inline void flushstderr() { fflush(stderr); }
]]


-- Base class for all MCMC kernels
local struct MCMCKernel {}
inheritance.purevirtual(MCMCKernel, "__destruct", {}->{})
inheritance.purevirtual(MCMCKernel, "next", {&BaseTraceD}->{&BaseTraceD})
inheritance.purevirtual(MCMCKernel, "stats", {}->{})
inheritance.purevirtual(MCMCKernel, "name", {}->{rawstring})




-- The basic random walk MH kernel
local RandomWalkKernel = templatize(function(structs, nonstructs)

	local struct RandomWalkKernelT
	{
		proposalsMade: uint,
		proposalsAccepted: uint
	}
	inheritance.dynamicExtend(MCMCKernel, RandomWalkKernelT)

	terra RandomWalkKernelT:__construct()
		self.proposalsMade = 0
		self.proposalsAccepted = 0
	end

	terra RandomWalkKernelT:__destruct() : {} end
	inheritance.virtual(RandomWalkKernelT, "__destruct")

	terra RandomWalkKernelT:next(currTrace: &BaseTraceD) : &BaseTraceD
		var nextTrace = currTrace
		var numvars = currTrace:numFreeVars(structs, nonstructs)
		-- If there are no free variables, then simply run the computation
		-- unmodified (nested query can make this happen)
		if numvars == 0 then
			[trace.traceUpdate()](currTrace)
		-- Otherwise, do an MH proposal
		else
			self.proposalsMade = self.proposalsMade + 1
			nextTrace = currTrace:deepcopy()
			var freevars = nextTrace:freeVars(structs, nonstructs)
			-- Select variable uniformly at random
			var v = freevars:get(rand.uniformRandomInt(0, freevars.size))
			var fwdPropLP, rvsPropLP = v:proposeNewValue()
			if v.isStructural then
				[trace.traceUpdate()](nextTrace)
			else
				[trace.traceUpdate({structureChange=false})](nextTrace)
			end
			if nextTrace.newlogprob ~= 0.0 or nextTrace.oldlogprob ~= 0.0 then
				var oldNumVars = numvars
				var newNumVars = nextTrace:numFreeVars(structs, nonstructs)
				fwdPropLP = fwdPropLP + nextTrace.newlogprob - C.log([double](oldNumVars))
				rvsPropLP = rvsPropLP + nextTrace.oldlogprob - C.log([double](newNumVars))
			end
			var acceptThresh = (nextTrace.logprob - currTrace.logprob)/currTrace.temperature + rvsPropLP - fwdPropLP
			if nextTrace.conditionsSatisfied and C.log(rand.random()) < acceptThresh then
				self.proposalsAccepted = self.proposalsAccepted + 1
				m.delete(currTrace)
			else
				m.delete(nextTrace)
				nextTrace = currTrace
			end
			m.destruct(freevars)
		end
		return nextTrace
	end
	inheritance.virtual(RandomWalkKernelT, "next")

	terra RandomWalkKernelT:name() : rawstring return [RandomWalkKernelT.name] end
	inheritance.virtual(RandomWalkKernelT, "name")

	terra RandomWalkKernelT:stats() : {}
		C.printf("Acceptance ratio: %g (%u/%u)\n",
			[double](self.proposalsAccepted)/self.proposalsMade,
			self.proposalsAccepted,
			self.proposalsMade)
	end
	inheritance.virtual(RandomWalkKernelT, "stats")

	m.addConstructors(RandomWalkKernelT)
	return RandomWalkKernelT
end)


-- Convenience method for making RandomWalkKernels
local RandomWalk = util.fnWithDefaultArgs(function(...)
	local RWType = RandomWalkKernel(...)
	return function() return `RWType.heapAlloc() end
end,
{{"structs", true}, {"nonstructs", true}})




-- Proposals to non-structural continuous random variables using Gaussian drift
local gdTargetAcceptRate = 0.234
local GaussianDriftKernel = templatize(function(bandwidth, bandwidthAdapt, adaptationRate)

	local struct GaussianDriftKernelT
	{
		proposalsMade: uint,
		proposalsAccepted: uint
	}
	inheritance.dynamicExtend(MCMCKernel, GaussianDriftKernelT)

	-- If adaptation was requested, then we need a bandwidth and some other data per variable.
	-- We also need to store the last trace we've seen, so we can know when we have to resize
	--   our per-variable data.
	if bandwidthAdapt then
		GaussianDriftKernelT.entries:insert({field = "bandwidths", type = Vector(double)})
		GaussianDriftKernelT.entries:insert({field = "propsPerVar", type = Vector(uint)})
		GaussianDriftKernelT.entries:insert({field = "acceptsPerVar", type = Vector(uint)})
		GaussianDriftKernelT.entries:insert({field = "lastTrace", type = &BaseTraceD})
	end

	terra GaussianDriftKernelT:__construct()
		self.proposalsMade = 0
		self.proposalsAccepted = 0
		[util.optionally(bandwidthAdapt, quote
			m.init(self.bandwidths)
			m.init(self.propsPerVar)
			m.init(self.acceptsPerVar)
			self.lastTrace = nil
		end)]
	end

	terra GaussianDriftKernelT:__destruct() : {}
		[util.optionally(bandwidthAdapt, quote
			m.destruct(self.bandwidths)
			m.destruct(self.propsPerVar)
			m.destruct(self.acceptsPerVar)
		end)]
	end
	inheritance.virtual(GaussianDriftKernelT, "__destruct")

	-- Retrieves correct bandwidth, whether we're adapting or not
	GaussianDriftKernelT.methods.getBandwidth = macro(function(self, i)
		if bandwidthAdapt then
			return `self.bandwidths(i)
		else
			return bandwidth
		end
	end)

	-- Check if we need to resize our bandwidth / adapter vectors (which happens)
	--    if we've never seen this trace before
	-- I elect not to ever shrink these vectors; if the set of variables grows again,
	--    the partially-adapted bandwidths stored there may be better than fresh ones.
	GaussianDriftKernelT.methods.preprocess = macro(function(self, currTrace, nvarsnew)
		if bandwidthAdapt then
			return quote
				if self.lastTrace ~= currTrace then
					var nvarsold = self.bandwidths.size
					if nvarsnew > nvarsold then
						self.bandwidths:resize(nvarsnew)
						self.propsPerVar:resize(nvarsnew)
						self.acceptsPerVar:resize(nvarsnew)
						for i=nvarsold,nvarsnew do
							self.bandwidths(i) = bandwidth
							self.propsPerVar(i) = 0
							self.acceptsPerVar(i) = 0
						end
					end
					self.lastTrace = currTrace
				end
			end
		else
			return quote end
		end
	end)

	-- Adapt the bandwidth of the changed variable depending upon how the change went (code adapted from Stan)
	-- If we accepted the change, then update self.lastTrace
	GaussianDriftKernelT.methods.postprocess = macro(function(self, nextTrace, changedVarIndex, didAccept)
		if bandwidthAdapt then
			return quote
				self.propsPerVar(changedVarIndex) = self.propsPerVar(changedVarIndex) + 1
				if didAccept then
					self.lastTrace = nextTrace
					self.acceptsPerVar(changedVarIndex) = self.acceptsPerVar(changedVarIndex) + 1
				end

				var acceptRatio = self.acceptsPerVar(changedVarIndex) / double(self.propsPerVar(changedVarIndex))
				if acceptRatio > gdTargetAcceptRate then
					self.bandwidths(changedVarIndex) = adaptationRate*self.bandwidths(changedVarIndex)
				else
					self.bandwidths(changedVarIndex) = (1/adaptationRate)*self.bandwidths(changedVarIndex)
				end
			end
		else
			return quote end
		end
	end)

	terra GaussianDriftKernelT:next(currTrace: &BaseTraceD) : &BaseTraceD
		self.proposalsMade = self.proposalsMade + 1
		var nextTrace = currTrace:deepcopy()
		-- Grab the real components of all nonstructural variables
		var freevars = nextTrace:freeVars(false, true)
		self:preprocess(currTrace, freevars.size)
		var realcomps = [Vector(double)].stackAlloc()
		for i=0,freevars.size do
			freevars(i):getRealComponents(&realcomps)
		end
		-- Choose a component at random, make a gaussian perturbation to it
		var i = rand.uniformRandomInt(0, realcomps.size)
		var changedVarIndex = i
		realcomps(i) = [rand.gaussian_sample(double)](realcomps(i), self:getBandwidth(i))
		-- Re-run trace with new value, make accept/reject decision
		var index = 0U
		for i=0,freevars.size do
			freevars(i):setRealComponents(&realcomps, &index)
		end
		[trace.traceUpdate({structureChange=false})](nextTrace)
		var acceptThresh = (nextTrace.logprob - currTrace.logprob)/currTrace.temperature
		var accepted = nextTrace.conditionsSatisfied and C.log(rand.random()) < acceptThresh
		if accepted then
			self.proposalsAccepted = self.proposalsAccepted + 1
			m.delete(currTrace)
		else
			m.delete(nextTrace)
			nextTrace = currTrace
		end
		m.destruct(realcomps)
		m.destruct(freevars)

		self:postprocess(nextTrace, changedVarIndex, accepted)

		return nextTrace
	end
	inheritance.virtual(GaussianDriftKernelT, "next")

	terra GaussianDriftKernelT:name() : rawstring return [GaussianDriftKernelT.name] end
	inheritance.virtual(GaussianDriftKernelT, "name")

	terra GaussianDriftKernelT:stats() : {}
		C.printf("Acceptance ratio: %g (%u/%u)\n",
			[double](self.proposalsAccepted)/self.proposalsMade,
			self.proposalsAccepted,
			self.proposalsMade)
	end
	inheritance.virtual(GaussianDriftKernelT, "stats")

	m.addConstructors(GaussianDriftKernelT)
	return GaussianDriftKernelT

end)

-- Convenience method for making GaussianDriftKernels
local GaussianDrift = util.fnWithDefaultArgs(function(...)
	local GDType = GaussianDriftKernel(...)
	return function() return `GDType.heapAlloc() end
end,
{{"bandwidth", 1.0}, {"bandwidthAdapt", false}, {"adaptationRate", 1.01}})





-- MCMC Kernel that probabilistically selects between multiple sub-kernels
local MultiKernel = templatize(function(selectFn)
	local struct MultiKernelT
	{
		kernels: Vector(&MCMCKernel)
	}
	inheritance.dynamicExtend(MCMCKernel, MultiKernelT)

	-- NOTE: Assumes ownership of arguments (read: does not copy)
	terra MultiKernelT:__construct(kernels: Vector(&MCMCKernel))
		self.kernels = kernels
	end

	terra MultiKernelT:__destruct() : {}
		for i=0,self.kernels.size do m.delete(self.kernels:get(i)) end
		m.destruct(self.kernels)
	end
	inheritance.virtual(MultiKernelT, "__destruct")

	terra MultiKernelT:next(currTrace: &BaseTraceD) : &BaseTraceD
		var kernel = selectFn(&self.kernels, currTrace)
		return kernel:next(currTrace)
	end
	inheritance.virtual(MultiKernelT, "next")

	terra MultiKernelT:name() : rawstring return [MultiKernelT.name] end
	inheritance.virtual(MultiKernelT, "name")

	terra MultiKernelT:stats() : {}
		for i=0,self.kernels.size do
			C.printf("------ Kernel %d (%s) ------\n", i+1, self.kernels:get(i):name())
			self.kernels:get(i):stats()
		end
	end
	inheritance.virtual(MultiKernelT, "stats")

	m.addConstructors(MultiKernelT)
	return MultiKernelT
end)





-- MCMC Kernel that does some computation based on how long
--    inference has been running.
local SchedulingKernel = templatize(function(innerKernelGen, schedule)
	
	local struct SchedulingKernelT
	{
		innerKernel: &MCMCKernel,
		currIter: uint
	}
	inheritance.dynamicExtend(MCMCKernel, SchedulingKernelT)

	-- Assumes ownership of innerKernel
	terra SchedulingKernelT:__construct()
		self.innerKernel = [innerKernelGen()]
		self.currIter = 0
	end

	terra SchedulingKernelT:__destruct() : {}
		m.delete(self.innerKernel)
	end
	inheritance.virtual(SchedulingKernelT, "__destruct")

	terra SchedulingKernelT:next(currTrace: &BaseTraceD) : &BaseTraceD
		schedule(self.currIter, currTrace)
		var nextTrace = self.innerKernel:next(currTrace) 
		self.currIter = self.currIter + 1
		return nextTrace
	end
	inheritance.virtual(SchedulingKernelT, "next")

	terra SchedulingKernelT:name() : rawstring return [SchedulingKernelT.name] end
	inheritance.virtual(SchedulingKernelT, "name")

	terra SchedulingKernelT:stats() : {}
		self.innerKernel:stats()
	end
	inheritance.virtual(SchedulingKernelT, "stats")

	m.addConstructors(SchedulingKernelT)
	return SchedulingKernelT

end)

local function Schedule(innerKernelGen, schedule)
	return function()
		return `[SchedulingKernel(innerKernelGen, schedule)].heapAlloc()
	end
end




---- Methods for actually doing some kind of inference:


-- Samples are what we get out of MCMC
local Sample = templatize(function(ValType)
	local struct Samp
	{
		value: ValType,
		logprob: double
	}
	Samp.ValueType = ValType

	-- Assumes ownership of (i.e. does not copy) val
	terra Samp:__construct(val: ValType, lp: double)
		self.value = val
		self.logprob = lp
	end

	terra Samp:__construct()
		m.init(self.value)
		self.logprob = 0.0
	end

	terra Samp:__copy(s: &Samp)
		self.value = m.copy(s.value)
		self.logprob = s.logprob
	end

	terra Samp:__destruct()
		m.destruct(self.value)
	end

	m.addConstructors(Samp)
	return Samp
end)


-- Convenience methods for managing types related to samples from computations.
local ReturnType = function(computation)
	return computation():getdefinitions()[1]:gettype().returntype
end
local SampleType = function(computation)
	return Sample(ReturnType(computation))
end
local SampleVectorType = function(computation)
	return Vector(SampleType(computation))
end

local function extractReturnValue(computation)
	local DerivedTraceType = RandExecTrace(double, computation)
	return macro(function(currTrace)
		return `[&DerivedTraceType](currTrace).returnValue
	end)
end

-- Draw unconditioned prior samples from computation by just running it
--    (i.e. not even creating any traces)
local function forwardSample(computation, numsamps)
	computation = spec.ensureProbComp(computation)
	local comp = computation()
	return terra()
		var samps = [SampleVectorType(computation)].stackAlloc()
		for i=0,numsamps do
			var retval = comp()
			var samp = [SampleType(computation)].stackAlloc(retval, 0.0)
			samps:push(samp)
			m.destruct(samp)
		end
		return samps
	end
end


-- Runs an mcmc sampling loop on trace using the transition kernel specified by
--    kernel, storing the results in samples
-- samples is optional; if nil, the samples will be discarded
-- params are: verbose, numsamps, lag, burnin
local function mcmcSample(computation, params)
	computation = spec.ensureProbComp(computation)
	local lag = params.lag or 1
	local iters = params.numsamps * lag
	local verbose = params.verbose or false
	local burnin = params.burnin or 0
	return terra(currTrace: &BaseTraceD, kernel: &MCMCKernel, samples: &SampleVectorType(computation))
		var t0 = 0.0
		for i=0,iters do
			if verbose then
				C.fprintf(C.getstderr(), " iteration: %d\r", i+1)
				C.flushstderr()
				if i == 1 then t0 = util.currentTimeInSeconds() end
			end
			currTrace = kernel:next(currTrace)
			if samples ~= nil and i % lag == 0 and i >= burnin then
				var derivedTrace = [&RandExecTrace(double, computation)](currTrace)
				var si = samples.size
				samples:resize(si+1)
				samples(si).value = m.copy(derivedTrace.returnValue)
				samples(si).logprob = derivedTrace.logprob
			end
		end
		if verbose then
			C.fprintf(C.getstderr(), "\n")
			kernel:stats()
			var t1 = util.currentTimeInSeconds()
			C.printf("Time: %g\n", t1 - t0)
		end
		return currTrace
	end
end

-- Generates code to initialize a trace from a computation and run an mcmc sampling loop on it, creating
--    and returning the list of resulting samples
local function mcmc(computation, kernelgen, params)
	computation = spec.ensureProbComp(computation)
	return terra()
		var currTrace : &BaseTraceD = [trace.newTrace(computation)]
		var kernel = [kernelgen()]
		var samps = [SampleVectorType(computation)].stackAlloc()
		currTrace = [mcmcSample(computation, params)](currTrace, kernel, &samps)
		m.delete(kernel)
		m.delete(currTrace)
		return samps
	end
end

-- Compute the mean of a vector of values
-- Value type must define + and scalar division
local expectation = macro(function(values)
	return quote
		var vals = values
		var me = m.copy(vals(0))
		for i=1,vals.size do
			var oldme = me
			me = me + vals(i)
			m.destruct(oldme)
		end
	in
		me / double(vals.size)
	end
end)

-- If ValType is a struct, look for a :dot method
-- Default to using the * operator
local function innerProd(ValType)
	local iprod = function(x,y) return `x*y end
	if ValType:isstruct() and ValType.methods.dot then
		iprod = function(x,y) return `x:dot(y) end
	end
	return iprod
end

-- Compute the variance of a vector of values
-- Value type must define + and scalar division
local variance = macro(function(values, mean)
	local iprod = innerProd(values:gettype().ValueType)
	return quote
		var vals = values
		var me = mean
		var v = 0.0
		for i=0,vals.size do
			var diff = vals(i) - me
			v = v + [iprod(diff, diff)]
			m.destruct(diff)
		end
	in
		v / vals.size
	end
end)

-- Compute the autocorrelation of a set of values
-- If optmean and optvar are not provided, use the
--    mean and variance of vals
local autocorrelation = macro(function(values, optmean, optvar)
	local ValType = values:gettype().ValueType
	local iprod = innerProd(ValType)
	local computeMeanVar = optmean == nil or optvar == nil
	return quote
		var vals = values
		-- Set up desired mean and variance
		var me : ValType
		var v = 0.0
		[util.optionally(computeMeanVar, function() return quote
			me = expectation(vals)
			v = variance(vals, me)
		end end)]
		[util.optionally(not computeMeanVar, function() return quote
			me = m.copy(optmean)
			v = optvar
		end end)]
		-- Compute autocorrelation
		var ac = [Vector(double)].stackAlloc(vals.size, 0.0)
		for t=0,vals.size do
			var n = vals.size - t
			for i=0,n do
				var tmp1 = vals(i) - me
				var tmp2 = vals(i+t) - me
				ac(t) = ac(t) + [iprod(tmp1, tmp2)]
				m.destruct(tmp1)
				m.destruct(tmp2)
			end
			if n > 0 then
				ac(t) = ac(t) / (n * v)
			end
		end
		m.destruct(me)
	in
		ac
	end
end)

-- Write autocorrelations to a CSV for later analysis
local terra saveAutocorrelation(ac: &Vector(double), filename: rawstring)
	var f = C.fopen(filename, "w")
	C.fprintf(f, "lag,autocorrelation\n")
	for i=0,ac.size do
		C.fprintf(f, "%d,%g\n", i, ac(i))
	end
	C.fclose(f)
end

-- Extract just the values from a set of samples
local sampleValues = macro(function(samps)
	local SampVectorType = samps:gettype()
	if SampVectorType:ispointer() then SampVectorType = SampVectorType.type end
	local SampType = SampVectorType.ValueType
	local ValType = SampType.ValueType
	local VectorType = Vector(ValType)
	return quote
		var vals = VectorType.stackAlloc(samps.size)
		for i=0,samps.size do vals(i) = samps(i).value end
	in
		vals
	end
end)


-- Compute expectation over a vector of samples
-- The value type must define + and scalar division
local sampleExpectation = macro(function(samps)
	return quote
		var vals = sampleValues(samps)
		var me = expectation(vals)
		m.destruct(vals)
	in
		me
	end
end)

-- Find the highest scoring sample
local MAP = macro(function(samps)
	return quote
		var bestindex = 0
		for i=1,samps.size do
			if samps(i).logprob > samps(bestindex).logprob then
				bestindex = i
			end
		end
	in
		samps(bestindex)
	end
end)

-- Sample from a computation by repeatedly burning in from a random initial
--    configuration
local function sampleByRepeatedBurnin(computation, kernelgen, mcmcparams, numsamps)
	computation = spec.ensureProbComp(computation)
	local terra doSampling()
		C.printf("Sampling by repeated burn-in...\n")
		var samps = [SampleVectorType(computation)].stackAlloc()
		var tmpsamps = [SampleVectorType(computation)].stackAlloc()
		for i=0,numsamps do
			C.printf("Iteration %d / %d\n", i+1, numsamps)
			tmpsamps:clear()
			var kernel = [kernelgen()]
			var currTrace : &BaseTraceD = [trace.newTrace(computation)]
			currTrace = [mcmcSample(computation, mcmcparams)](currTrace, kernel, &tmpsamps)
			m.delete(currTrace)
			m.delete(kernel)
			samps:push(MAP(tmpsamps))
		end
		m.destruct(tmpsamps)
		C.printf("DONE\n")
		return samps
	end
	return `doSampling()
end

-- Draw a sample from a computation via rejection
local function rejectionSample(computation)
	computation = spec.ensureProbComp(computation)
	return quote
		var tr = [trace.newTrace(computation)]
		var retval = m.copy(tr.returnValue)
		m.delete(tr)
	in
		retval
	end
end


-- Generally-useful utility for assembling factors
local softeq = macro(function(actual, target, softness)
	return `[rand.gaussian_logprob(actual:gettype())](actual, target, softness)
end)


return
{
	MCMCKernel = MCMCKernel,
	MultiKernel = MultiKernel,
	ReturnType = ReturnType,
	SampleType = SampleType,
	SampleVectorType = SampleVectorType,
	extractReturnValue = extractReturnValue,
	mcmcSample = mcmcSample,
	globals = {
		RandomWalk = RandomWalk,
		GaussianDrift = GaussianDrift,
		Schedule = Schedule,
		forwardSample = forwardSample,
		ReturnType = ReturnType,
		SampleType = SampleType,
		SampleVectorType = SampleVectorType,
		mcmc = mcmc,
		rejectionSample = rejectionSample,
		sampleByRepeatedBurnin = sampleByRepeatedBurnin,
		expectation = expectation,
		variance = variance,
		autocorrelation = autocorrelation,
		saveAutocorrelation = saveAutocorrelation,
		sampleValues = sampleValues,
		sampleExpectation = sampleExpectation,
		MAP = MAP,
		softeq = softeq
	}
}






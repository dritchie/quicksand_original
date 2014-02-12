local trace = terralib.require("prob.trace")
local spec = terralib.require("prob.specialize")
local BaseTrace = trace.BaseTrace
local BaseTraceD = BaseTrace(double)
local RandExecTrace = trace.RandExecTrace
local TraceWithRetVal = trace.TraceWithRetVal
local inheritance = terralib.require("inheritance")
local m = terralib.require("mem")
local rand = terralib.require("prob.random")
local templatize = terralib.require("templatize")
local Vector = terralib.require("vector")
local erp = terralib.require("prob.erph")
local ad = terralib.require("ad")
local util = terralib.require("util")

local C = terralib.includecstring [[
#include <stdio.h>
#include <math.h>
inline void flush() { fflush(stdout); }
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
		self.proposalsMade = self.proposalsMade + 1
		var nextTrace = currTrace
		var numvars = currTrace:numFreeVars(structs, nonstructs)
		-- If there are no free variables, then simply run the computation
		-- unmodified (nested query can make this happen)
		if numvars == 0 then
			[trace.traceUpdate()](currTrace)
		-- Otherwise, do an MH proposal
		else
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
local GaussianDriftKernel = templatize(function(bandwidth)

	local struct GaussianDriftKernelT
	{
		proposalsMade: uint,
		proposalsAccepted: uint
	}
	inheritance.dynamicExtend(MCMCKernel, GaussianDriftKernelT)

	-- 'bandwidth' can either be a constant number or a function that maps temperature
	--    to bandwidth
	local getBandwidth = macro(function(currTrace)
		if type(bandwidth) == "function" then
			return bandwidth(`currTrace.temperature)
		else
			return bandwidth
		end
	end)

	terra GaussianDriftKernelT:__construct()
		self.proposalsMade = 0
		self.proposalsAccepted = 0
	end

	terra GaussianDriftKernelT:__destruct() : {} end
	inheritance.virtual(GaussianDriftKernelT, "__destruct")

	terra GaussianDriftKernelT:next(currTrace: &BaseTraceD) : &BaseTraceD
		self.proposalsMade = self.proposalsMade + 1
		var nextTrace = currTrace:deepcopy()
		-- Grab the real components of all nonstructural variables
		var freevars = nextTrace:freeVars(false, true)
		var realcomps = [Vector(double)].stackAlloc()
		for i=0,freevars.size do
			freevars(i):getRealComponents(&realcomps)
		end
		-- Choose a component at random, make a gaussian perturbation to it
		var i = rand.uniformRandomInt(0, realcomps.size)
		realcomps(i) = [rand.gaussian_sample(double)](realcomps(i), getBandwidth(currTrace))
		-- Re-run trace with new value, make accept/reject decision
		var index = 0U
		for i=0,freevars.size do
			freevars(i):setRealComponents(&realcomps, &index)
		end
		[trace.traceUpdate({structureChange=false})](nextTrace)
		var acceptThresh = (nextTrace.logprob - currTrace.logprob)/currTrace.temperature
		if nextTrace.conditionsSatisfied and C.log(rand.random()) < acceptThresh then
			self.proposalsAccepted = self.proposalsAccepted + 1
			m.delete(currTrace)
		else
			m.delete(nextTrace)
			nextTrace = currTrace
		end
		m.destruct(freevars)
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
{{"bandwidth", 1.0}})





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

	terra Samp:__construct(val: ValType, lp: double)
		self.value = m.copy(val)
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

local function forwardSample(computation, numsamps)
	computation = spec.probcomp(computation)
	local comp = computation()
	local CompType = comp:getdefinitions()[1]:gettype()
	local RetValType = CompType.returns[1]
	local terra fn()
		var samps = [Vector(Sample(RetValType))].stackAlloc()
		for i=0,numsamps do
			var retval = comp()
			samps:push([Sample(RetValType)].stackAlloc(retval, 0.0))
			m.destruct(retval)
		end
		return samps
	end
	return `fn()
end

-- Takes a computation, initializes a trace for it, then runs any user-provided function
--    on that trace.
-- fn is a lua function that gets the computation as an argument and is expected to return
--    a Terra function that takes the trace as an argument (&BaseTrace(double)). This Terra
--    function assumes ownership of the trace and must delete it.
-- Returns a no-arg terra function that does the processing
local function processTrace(computation, fn)
	computation = spec.ensureProbComp(computation)
	return terra()
		var currTrace : &BaseTraceD = [trace.newTrace(computation)]
		var retval = [fn(computation)](currTrace)
		return retval
	end
end

-- Convenience methods for managing types related to samples from computations.
local ReturnType = templatize(function(computation)
	return computation():getdefinitions()[1]:gettype().returntype
end)
local SampleType = templatize(function(computation)
	return Sample(ReturnType(computation))
end)
local SampleVectorType = templatize(function(computation)
	return Vector(SampleType(computation))
end)

-- Runs an mcmc sampling loop on trace using the transition kernel specified by
--    kernel, storing the results in samples.
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
				C.printf(" iteration: %d\r", i+1)
				C.flush()
				if i == 1 then t0 = util.currentTimeInSeconds() end
			end
			currTrace = kernel:next(currTrace)
			if i % lag == 0 and i >= burnin then
				var derivedTrace = [&RandExecTrace(double, computation)](currTrace)
				samples:push([SampleType(computation)].stackAlloc(derivedTrace.returnValue, derivedTrace.logprob))
			end
		end
		if verbose then
			C.printf("\n")
			kernel:stats()
			var t1 = util.currentTimeInSeconds()
			C.printf("Time: %g\n", t1 - t0)
		end
		return currTrace
	end
end

-- Generates code to initialize a trace from a computation and run an mcmc sampling loop on it, creating
--    and returning the list of resulting samples
-- (exists for backwards-compatibility with old code)
local function mcmc(computation, kernelgen, params)
	computation = spec.ensureProbComp(computation)
	local function domcmc(comp)
		return terra(currTrace: &BaseTraceD)
			var kernel = [kernelgen()]
			var samps = [SampleVectorType(comp)].stackAlloc()
			currTrace = [mcmcSample(comp, params)](currTrace, kernel, &samps)
			m.delete(kernel)
			m.delete(currTrace)
			return samps
		end
	end
	return `[processTrace(computation, domcmc)]()
end

-- Compute the mean of a vector of values
-- Value type must define + and scalar division
local mean = macro(function(vals)
	return quote
		var m = m.copy(vals:get(0))
		for i=1,vals.size do
			m = m + vals:get(i)
		end
		m = m / [double](vals.size)
	in
		m
	end
end)

-- Compute expectation over a vector of samples
-- The value type must define + and scalar division
local expectation = macro(function(samps)
	return quote
		var m = m.copy(samps:get(0).value)
		for i=1,samps.size do
			m = m + samps:get(i).value
		end
		m = m / [double](samps.size)
	in
		m
	end
end)

-- Find the highest scoring sample
local MAP = macro(function(samps)
	return quote
		var best = m.copy(samps(0))
		for i=1,samps.size do
			var s = samps:getPointer(i)
			if s.logprob > best.logprob then
				best.logprob = s.logprob
				m.destruct(best.value)
				best.value = m.copy(s.value)
			end
		end
	in
		best.value
	end
end)

-- Draw a sample from a computation via rejection
local function rejectionSample(computation)
	computation = spec.probcomp(computation)
	return quote
		var tr = [trace.newTrace(computation)]
		var retval = m.copy(tr.returnValue)
		m.delete(tr)
	in
		retval
	end
end


return
{
	MCMCKernel = MCMCKernel,
	MultiKernel = MultiKernel,
	globals = {
		RandomWalk = RandomWalk,
		GaussianDrift = GaussianDrift,
		Schedule = Schedule,
		forwardSample = forwardSample,
		processTrace = processTrace,
		types = 
		{
			ReturnType = ReturnType,
			SampleType = SampleType,
			SampleVectorType = SampleVectorType
		},
		mcmcSample = mcmcSample,
		mcmc = mcmc,
		rejectionSample = rejectionSample,
		mean = mean,
		expectation = expectation,
		MAP = MAP
	}
}






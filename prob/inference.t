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
#include <sys/time.h>
inline double currentTimeInSeconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}
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





-- -- Random walk kernel that runs an automatically differentiated version
-- --    of the program.
-- -- Only works on fixed-structure programs whose ERPs all have type double
-- -- This is not useful in practice, but it's useful for testing that AD dual
-- --     nums are working properly.
-- local RandVarAD = erp.RandVar(ad.num)
-- local BaseTraceAD = BaseTrace(ad.num)
-- local struct ADRandomWalkKernelT
-- {
-- 	proposalsMade: uint,
-- 	proposalsAccepted: uint,
-- 	adTrace: &BaseTraceAD,
-- 	lastTrace: &BaseTraceD,
-- 	adVars: Vector(&RandVarAD),
-- 	values: Vector(double)
-- }
-- inheritance.dynamicExtend(MCMCKernel, ADRandomWalkKernelT)

-- terra ADRandomWalkKernelT:__construct()
-- 	self.proposalsMade = 0
-- 	self.proposalsAccepted = 0
-- 	self.adTrace = nil
-- 	self.lastTrace = nil
-- 	m.init(self.adVars)
-- 	m.init(self.values)
-- end

-- terra ADRandomWalkKernelT:__destruct() : {}
-- 	m.destruct(self.adVars)
-- 	m.destruct(self.values)
-- 	if self.adTrace ~= nil then
-- 		m.delete(self.adTrace)
-- 	end
-- end
-- inheritance.virtual(ADRandomWalkKernelT, "__destruct")

-- terra ADRandomWalkKernelT:initWithNewTrace(currTrace: &BaseTraceD)
-- 	self.lastTrace = currTrace
-- 	if self.adTrace ~= nil then m.delete(self.adTrace) end
-- 	self.adTrace = [BaseTraceD.deepcopy(ad.num)](currTrace)
-- 	m.destruct(self.adVars)
-- 	self.adVars = self.adTrace:freeVars(false, true)
-- 	self.values:resize(self.adVars.size)
-- 	for i=0,self.values.size do
-- 		var val = [erp.valueAs(ad.num)](self.adVars:get(i))
-- 		self.values:set(i, val:val())
-- 	end
-- end

-- terra ADRandomWalkKernelT:next(currTrace: &BaseTraceD) : &BaseTraceD
-- 	self.proposalsMade = self.proposalsMade + 1
-- 	if self.lastTrace ~= currTrace then
-- 		self:initWithNewTrace(currTrace)
-- 	end
-- 	var nextTrace = self.adTrace
-- 	var freevars = &self.adVars
-- 	-- Set the variable values for our working AD trace
-- 	for i=0,self.values.size do
-- 		var adv = self.adVars:get(i)
-- 		@[&ad.num](adv:pointerToValue()) = self.values:get(i)
-- 	end
-- 	-- Have to traceUpdate once to flush any changes to parameters before
-- 	--    we can safely call proposeNewValue()
-- 	[trace.traceUpdate({structureChange=false, factorEval=false})](nextTrace)
-- 	-- Propose change to randomly-selected variable
-- 	var whichVar = rand.uniformRandomInt(0, freevars.size)
-- 	var v = freevars:get(whichVar)
-- 	var fwdPropLP, rvsPropLP = v:proposeNewValue()
-- 	-- Update trace
-- 	[trace.traceUpdate({structureChange=false})](nextTrace)
-- 	var acceptThresh = (nextTrace.logprob - currTrace.logprob)/currTrace.temperature  + rvsPropLP - fwdPropLP
-- 	if nextTrace.conditionsSatisfied and C.log(rand.random()) < acceptThresh then
-- 		self.proposalsAccepted = self.proposalsAccepted + 1
-- 		-- Copy values back into currTrace, and into self.values
-- 		var oldfreevars = currTrace:freeVars(false, true)
-- 		for i=0,oldfreevars.size do
-- 			var newval = [erp.valueAs(ad.num)](freevars:get(i)):val()
-- 			oldfreevars:get(i):setValue(&newval)
-- 			self.values:set(i, newval)
-- 		end
-- 		m.destruct(oldfreevars)
-- 		-- Reconstruct return value by running a traceUpdate
-- 		-- (No need to evaluate any factors)
-- 		[trace.traceUpdate({structureChange=false, factorEval=false})](currTrace)
-- 		-- Set the final logprob (since we didn't evaluate factors)
-- 		[BaseTraceD.setLogprobFrom(ad.num)](currTrace, nextTrace)
-- 	end
-- 	-- Recover tape memory so we don't run out
-- 	ad.recoverMemory()
-- 	return currTrace
-- end
-- inheritance.virtual(ADRandomWalkKernelT, "next")

-- terra ADRandomWalkKernelT:name() : rawstring return [ADRandomWalkKernelT.name] end
-- inheritance.virtual(ADRandomWalkKernelT, "name")

-- terra ADRandomWalkKernelT:stats() : {}
-- 	C.printf("Acceptance ratio: %g (%u/%u)\n",
-- 		[double](self.proposalsAccepted)/self.proposalsMade,
-- 		self.proposalsAccepted,
-- 		self.proposalsMade)
-- end
-- inheritance.virtual(ADRandomWalkKernelT, "stats")

-- m.addConstructors(ADRandomWalkKernelT)


-- -- Convenience method for making ADRandomWalkKernelTs
-- local ADRandomWalk = makeKernelGenerator(
-- 	terra()
-- 		return ADRandomWalkKernelT.heapAlloc()
-- 	end,
-- 	{})






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
	return macro(function()
		return `[SchedulingKernel(innerKernelGen, schedule)].heapAlloc()
	end)
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

-- Single-chain MCMC
-- params are: verbose, numsamps, lag, burnin
-- returns a Vector of samples, where a sample is a
--    struct with a 'value' field and a 'logprob' field
local function mcmc(computation, kernelgen, params)
	computation = spec.probcomp(computation)
	local lag = params.lag or 1
	local iters = params.numsamps * lag
	local verbose = params.verbose or false
	local burnin = params.burnin or 0
	local comp = computation()
	local CompType = comp:getdefinitions()[1]:gettype()
	local RetValType = CompType.returns[1]
	local terra chain()
		var kernel = [kernelgen()]
		var samps = [Vector(Sample(RetValType))].stackAlloc()
		var currTrace : &BaseTraceD = [trace.newTrace(computation)]
		var t0 = 0.0
		for i=0,iters do
			if verbose then
				C.printf(" iteration: %d\r", i+1)
				C.flush()
				if i == 1 then t0 = C.currentTimeInSeconds() end
			end
			currTrace = kernel:next(currTrace)
			if i % lag == 0 and i > burnin then
				var derivedTrace = [&RandExecTrace(double, computation)](currTrace)
				samps:push([Sample(RetValType)].stackAlloc(derivedTrace.returnValue, derivedTrace.logprob))
			end
		end
		if verbose then
			C.printf("\n")
			kernel:stats()
			var t1 = C.currentTimeInSeconds()
			C.printf("Time: %g\n", t1 - t0)
		end
		m.delete(kernel)
		return samps
	end
	return `chain()
end

-- Compute the mean of a set of values
-- Value type must define + and scalar division
local mean = templatize(function(V)
	return terra(vals: &Vector(V))
		var m = m.copy(vals:get(0))
		for i=1,vals.size do
			m = m + vals:get(i)
		end
		return m / [double](vals.size)
	end
end)

-- Compute expectation over a set of samples
-- The return value type must define + and scalar division
local expectation = templatize(function(RetValType)
	return terra(samps: &Vector(Sample(RetValType)))
		var m = m.copy(samps:get(0).value)
		for i=1,samps.size do
			m = m + samps:get(i).value
		end
		return m / [double](samps.size)
	end
end)

-- Find the highest scoring sample
local MAP = templatize(function(RetValType)
	return terra(samps: &Vector(Sample(RetValType)))
		var best = [Sample(RetValType)].stackAlloc()
		best.logprob = [-math.huge]
		for i=0,samps.size do
			var s = samps:getPointer()
			if s.logprob > best.logprob then
				best.logprob = s.logprob
				m.destruct(best.value)
				best.value = m.copy(s.value)
			end
		end
		return best.value
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
		-- ADRandomWalk = ADRandomWalk,
		Schedule = Schedule,
		mcmc = mcmc,
		rejectionSample = rejectionSample,
		mean = mean,
		expectation = expectation,
		MAP = MAP
	}
}






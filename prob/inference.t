local trace = terralib.require("prob.trace")
local spec = terralib.require("prob.specialize")
local BaseTrace = trace.BaseTrace
local RandExecTrace = trace.RandExecTrace
local TraceWithRetVal = trace.TraceWithRetVal
local inheritance = terralib.require("inheritance")
local m = terralib.require("mem")
local rand = terralib.require("prob.random")
local templatize = terralib.require("templatize")
local Vector = terralib.require("vector")

local C = terralib.includecstring [[
#include <stdio.h>
#include <math.h>
inline void flush() { fflush(stdout); }
]]


-- Base class for all MCMC kernels
local struct MCMCKernel {}
inheritance.purevirtual(MCMCKernel, "__destruct", {}->{})
inheritance.purevirtual(MCMCKernel, "next", {&BaseTrace}->{&BaseTrace})
inheritance.purevirtual(MCMCKernel, "stats", {}->{})
inheritance.purevirtual(MCMCKernel, "name", {}->{rawstring})



-- Convenience method for generating new top-level MCMC kernels
-- Arguments:
--    * kernelfn: A (Terra) function which takes some arguments
--         and returns a kernel.
--    * defaults: A table of default argument values.
-- This returns a (Lua) function which will take a table of
--    named parameters and return a no-arg function which generates
--    the desired kernel.
-- The names used in the parameter table must match the names
--    used in the definition of kernelfn.
local function makeKernelGenerator(kernelfn, defaults)
	assert(terralib.isfunction(kernelfn))
	-- No overloaded kernel functions (what are you, crazy?!?)
	assert(#kernelfn:getdefinitions() == 1)
	return function(paramtable)
		paramtable = paramtable or {}
		-- Extract all needed arguments, either from paramtable
		--    or from defaults
		-- Throw an error if something is missing.
		local args = {}
		for _,param in ipairs(kernelfn:getdefinitions()[1].untypedtree.parameters) do
			local n = param.name
			local arg = paramtable[n]
			if arg == nil then arg = defaults[n] end
			if arg == nil then
				error(string.format("No provided or default value for kernel parameter '%s'", n)) 
			end
			table.insert(args, arg)
		end
		return function() return `kernelfn([args]) end
	end
end


-- The basic random walk MH kernel
local struct RandomWalkKernel
{
	structs: bool,
	nonstructs: bool,
	proposalsMade: uint,
	proposalsAccepted: uint
}
inheritance.dynamicExtend(MCMCKernel, RandomWalkKernel)

terra RandomWalkKernel:__construct(structs: bool, nonstructs: bool)
	self.structs = structs
	self.nonstructs = nonstructs
	self.proposalsMade = 0
	self.proposalsAccepted = 0
end

terra RandomWalkKernel:__destruct() : {} end
inheritance.virtual(RandomWalkKernel, "__destruct")

terra RandomWalkKernel:next(currTrace: &BaseTrace) : &BaseTrace
	self.proposalsMade = self.proposalsMade + 1
	var nextTrace = currTrace
	var numvars = currTrace:numFreeVars(self.structs, self.nonstructs)
	-- If there are no free variables, then simply run the computation
	-- unmodified (nested query can make this happen)
	if numvars == 0 then
		[trace.traceUpdate(currTrace)]
	-- Otherwise, do an MH proposal
	else
		nextTrace = currTrace:deepcopy()
		var freevars = nextTrace:freeVars(self.structs, self.nonstructs)
		var v = freevars:get(rand.uniformRandomInt(0, freevars.size))
		var fwdPropLP, rvsPropLP = v:proposeNewValue()
		if v.isStructural then
			[trace.traceUpdate(nextTrace)]
		else
			[trace.traceUpdate(nextTrace, {structureChange=false})]
		end
		if nextTrace.newlogprob ~= 0.0 or nextTrace.oldlogprob ~= 0.0 then
			var oldNumVars = numvars
			var newNumVars = nextTrace:numFreeVars(self.structs, self.nonstructs)
			fwdPropLP = fwdPropLP + nextTrace.newlogprob - C.log([double](oldNumVars))
			rvsPropLP = rvsPropLP + nextTrace.oldlogprob - C.log([double](newNumVars))
		end
		var acceptThresh = nextTrace.logprob - currTrace.logprob + rvsPropLP - fwdPropLP
		-- C.printf("--------------------------\n")
		-- C.printf("currTrace.logprob:    %g\n", currTrace.logprob)
		-- C.printf("nextTrace.logprob:    %g\n", nextTrace.logprob)
		-- C.printf("nextTrace.newlogprob:    %g\n", nextTrace.newlogprob)
		-- C.printf("nextTrace.oldlogprob:    %g\n", nextTrace.oldlogprob)
		-- C.printf("fwdPropLP:    %g\n", fwdPropLP)
		-- C.printf("rvsPropLP:    %g\n", rvsPropLP)
		if nextTrace.conditionsSatisfied and C.log(rand.random()) < acceptThresh then
			-- C.printf("ACCEPTED\n")
			self.proposalsAccepted = self.proposalsAccepted + 1
			m.delete(currTrace)
		else
			-- C.printf("REJECTED\n")
			m.delete(nextTrace)
			nextTrace = currTrace
		end
		m.destruct(freevars)
	end
	return nextTrace
end
inheritance.virtual(RandomWalkKernel, "next")

terra RandomWalkKernel:name() : rawstring return [RandomWalkKernel.name] end
inheritance.virtual(RandomWalkKernel, "name")

terra RandomWalkKernel:stats() : {}
	C.printf("Acceptance ratio: %g (%u/%u)\n",
		[double](self.proposalsAccepted)/self.proposalsMade,
		self.proposalsAccepted,
		self.proposalsMade)
end
inheritance.virtual(RandomWalkKernel, "stats")

m.addConstructors(RandomWalkKernel)



-- Convenience method for making RandomWalkKernels
local RandomWalk = makeKernelGenerator(
	terra(structural: bool, nonstructural: bool)
		return RandomWalkKernel.heapAlloc(structural, nonstructural)
	end,
	{structural=true, nonstructural=true})



-- MCMC Kernel that probabilistically selects between multiple sub-kernels
local struct MultiKernel
{
	kernels: Vector(&MCMCKernel),
	freqs: Vector(double)
}
inheritance.dynamicExtend(MCMCKernel, MultiKernel)

-- NOTE: Assumes ownership of arguments (read: does not copy)
terra MultiKernel:__construct(kernels: Vector(&MCMCKernel), freqs: Vector(double))
	self.kernels = kernels
	self.freqs = freqs
end

terra MultiKernel:__destruct() : {}
	for i=0,self.kernels.size do m.delete(self.kernels:get(i)) end
	m.destruct(self.kernels)
	m.destruct(self.freqs)
end
inheritance.virtual(MultiKernel, "__destruct")

terra MultiKernel:next(currTrace: &BaseTrace) : &BaseTrace
	var whichKernel = [rand.multinomial_sample(double)](self.freqs)
	return self.kernels:get(whichKernel):next(currTrace)
end
inheritance.virtual(MultiKernel, "next")

terra MultiKernel:name() : rawstring return [MultiKernel.name] end
inheritance.virtual(MultiKernel, "name")

terra MultiKernel:stats() : {}
	for i=0,self.kernels.size do
		C.printf("------ Kernel %d (%s) ------\n", i+1, self.kernels:get(i):name())
		self.kernels:get(i):stats()
	end
end
inheritance.virtual(MultiKernel, "stats")

m.addConstructors(MultiKernel)




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
	computation = spec.specializablethunk(computation)
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
		var currTrace : &BaseTrace = [trace.newTrace(computation)]
		for i=0,iters do
			if verbose then
				C.printf(" iteration: %d\r", i+1)
				C.flush()
			end
			currTrace = kernel:next(currTrace)
			if i % lag == 0 and i > burnin then
				var derivedTrace = [&RandExecTrace(computation)](currTrace)
				samps:push([Sample(RetValType)].stackAlloc(derivedTrace.returnValue, derivedTrace.logprob))
			end
		end
		if verbose then
			C.printf("\n")
			kernel:stats()
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
	computation = spec.specializablethunk(computation)
	return quote
		var tr = [trace.newTrace(computation)]
		var retval = tr.returnValue
		m.delete(tr)
	in
		retval
	end
end


return
{
	MCMCKernel = MCMCKernel,
	makeKernelGenerator = makeKernelGenerator,
	MultiKernel = MultiKernel,
	globals = {
		RandomWalk = RandomWalk,
		mcmc = mcmc,
		rejectionSample = rejectionSample,
		mean = mean,
		expectation = expectation,
		MAP = MAP
	}
}






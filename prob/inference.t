local trace = terralib.require("prob.trace")
local BaseTrace = trace.BaseTrace
local RandExecTrace = trace.RandExecTrace
local TraceWithRetVal = trace.TraceWithRetVal
local iface = terralib.require("interface")
local m = terralib.require("mem")
local rand = terralib.require("prob.random")
local templatize = terralib.require("templatize")
local Vector = terralib.require("vector")

local C = terralib.includecstring [[
#include <stdio.h>
#include <math.h>
inline void flush() { fflush(stdout); }
]]


-- Interface for all MCMC kernels
local MCMCKernel = iface.create {
	next = {&BaseTrace} -> {&BaseTrace};
	stats = {} -> {}
}


-- Convenience method for generating new top-level MCMC kernels
-- Arguments:
--    * kernelfn: A (Terra) function which takes some arguments
--         and returns a kernel.
--    * defaults: A table of default argument values.
-- This returns a (Lua) function which will take a table of
--    named parameters and return a no-arg macro which generates
--    the desired kernel.
-- The names used in the parameter table must match the names
--    used in the definition of kernelfn.
local function makeKernelGenerator(kernelfn, defaults)
	assert(terralib.isfunction(kernelfn))
	-- No overloaded kernel functions (what are you, crazy?!?)
	assert(#kernelfn:getdefinitions() == 1)
	return function(paramtable)
		paramtable = paramtable or nil
		-- Extract all needed arguments, either from paramtable
		--    or from defaults
		-- Throw an error if something is missing.
		local args = {}
		for _,param in ipairs(kernelfn:getdefinitions()[1].untypedtree.parameters) do
			local n = param.name
			local arg = paramtable[n] or defaults[n]
			if not arg then
				error(string.format("No provided or default value for kernel parameter '%s'", n)) 
			end
			table.insert(args, arg)
		end
		return macro(function() return `kernelfn([args]) end)
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

terra RandomWalkKernel:__construct(structs: bool, nonstructs: bool)
	self.structs = structs
	self.nonstructs = nonstructs
	self.proposalsMade = 0
	self.proposalsAccepted = 0
end

terra RandomWalkKernel:next(currTrace: &BaseTrace)
	self.proposalsMade = self.proposalsMade + 1
	var nextTrace = currTrace
	var numvars = currTrace:numFreeVars(self.structs, self.nonstructs)
	-- If there are no free variables, then simply run the computation
	-- unmodified (nested query can make this happen)
	if numvars == 0 then
		currTrace:traceUpdate()
	-- Otherwise, do an MH proposal
	else
		nextTrace = currTrace:deepcopy()
		var freevars = nextTrace:freeVars(self.structs, self.nonstructs)
		var v = freevars:get(rand.uniformRandomInt(0, freevars.size))
		var fwdPropLP, rvsPropLP = v:proposeNewValue()
		nextTrace:traceUpdate()
		if nextTrace.newlogprob ~= 0.0 or nextTrace.oldlogprob ~= 0.0 then
			var oldNumVars = numvars
			var newNumVars = freevars.size
			fwdPropLP = fwdPropLP + nextTrace.newlogprob - C.log(oldNumVars)
			rvsPropLP = rvsPropLP + nextTrace.oldlogprob - C.log(newNumVars)
		end
		var acceptThresh = nextTrace.logprob - currTrace.logprob + rvsPropLP - fwdPropLP
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

terra RandomWalkKernel:stats()
	C.printf("Acceptance ratio: %g (%u/%u)\n",
		[double](self.proposalsAccepted)/self.proposalsMade,
		self.proposalsAccepted,
		self.proposalsMade)
end

m.addConstructors(RandomWalkKernel)



-- Convenience method for making RandomWalkKernels
local RandomWalk = makeKernelGenerator(
	terra()
		return RandomWalkKernel.stackAlloc(true, true)
	end,
	{})




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
	local lag = params.lag or 1
	local iters = params.numsamps * lag
	local verbose = params.verbose or false
	local burnin = params.burnin or 0
	local CompType = computation:getdefinitions()[1]:gettype()
	local RetValType = CompType.returns[1]
	local terra chain()
		var kernel = kernelgen()
		var samps = [Vector(Sample(RetValType))].stackAlloc()
		var comp = computation
		var currTrace : &BaseTrace = trace.newTrace(comp)
		for i=0,iters do
			if verbose then
				C.printf(" iteration: %d\r", i+1)
				C.flush()
			end
			currTrace = kernel:next(currTrace)
			if i % lag == 0 and i > burnin then
				var derivedTrace = [&RandExecTrace(CompType)](currTrace)
				samps:push([Sample(RetValType)].stackAlloc(derivedTrace.returnValue, derivedTrace.logprob))
			end
		end
		if verbose then
			C.printf("\n")
			kernel:stats()
		end
		m.destruct(kernel)
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
local rejectionSample = macro(function(computation)
	return quote
		var tr = trace.newTrace(computation)
	in
		tr.returnValue
	end
end)


return
{
	makeKernelGenerator = makeKernelGenerator,
	RandomWalk = RandomWalk,
	mcmc = mcmc,
	rejectionSample = rejectionSample,
	mean = mean,
	expectation = expectation,
	MAP = MAP
}






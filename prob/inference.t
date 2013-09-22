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
--    named parameters and return the desired kernel
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
		local kernel = kernelfn(unpack(args))
		m.gc(kernel)
		return kernel
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
	var freevars = currTrace:freeVars(self.structs, self.nonstructs)
	-- If there are no free variables, then simply run the computation
	-- unmodified (nested query can make this happen)
	if freevars.size == 0 then
		currTrace:traceUpdate()
	-- Otherwise, do an MH proposal
	else
		nextTrace = currTrace:deepcopy()
		var v = freevars:get(rand.uniformRandomInt(0, freevars.size))
		var fwdPropLP, rvsPropLP = v:proposeNewValue()
		nextTrace:traceUpdate()
		if nextTrace.newlogprob ~= 0.0 or nextTrace.oldlogprob ~= 0.0 then
			var oldNumVars = freevars.size
			var newNumVars = nextTrace:numFreeVars(self.structs, self.nonstructs)
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
	end
	m.destruct(freevars)
	return nextTrace
end

terra RandomWalkKernel:stats()
	C.printf("Acceptance ratio: %g (%u/%u\n",
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
local function mcmc(computation, kernel, params)
	local lag = params.lag or 1
	local iters = params.numsamps * lag
	local verbose = params.verbose or false
	local burnin = params.burnin or 0
	local CompType = computation:getdefinitions()[1]:gettype()
	local RetValType = CompType.returns[1]
	local terra chain()
		var samps = [Vector(Sample(RetValType))].stackAlloc()
		var comp = computation
		var currTrace : &BaseTrace = trace.newTrace(comp)
		for i=0,iters do
			if verbose then
				C.printf(" iteration: %d\r", i+1)
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
		return samps
	end
	local samps = chain()
	m.gc(samps)
	return samps
end


return
{
	makeKernelGenerator = makeKernelGenerator,
	RandomWalk = RandomWalk,
	mcmc = mcmc
}






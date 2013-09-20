local trace = terralib.require("prob.trace")
local BaseTrace = trace.BaseTrace
local iface = terralib.require("interface")

local C = terralib.includecstring [[
#include <stdio.h>
]]


-- Interface for all MCMC kernels
local MCMCKernel = iface.create {
	next = {&BaseTrace} -> {&BaseTrace};
	stats = {} -> {}
}


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

terra RandomWalkKernel:next(currtrace: &BaseTrace)
	--
end

terra RandomWalkKernel:stats()
	C.printf("Acceptance ratio: %g (%u/%u\n",
		[double](self.proposalsAccepted)/self.proposalsMade,
		self.proposalsAccepted,
		self.proposalsMade)
end
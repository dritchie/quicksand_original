local spec = require("prob.specialize")
local rand = require("prob.random")

----------------------------------
-- Specialization parameters and their defaults

-- Compile-time params
-- IMPORTANT: Don't add any more. Otherwise, we start getting lots of incompatible types, and that makes
--    programming really gnarly.
spec.addParam({
	name = "computation",
	stage = "Compile",
	default = spec.probcomp(function() error("Specialization parameter 'computation' is not defined") end)
})
spec.addParam({
	name = "scalarType",
	stage = "Compile",
	default = double
})

-- Runtime params
spec.addParam({
	name = "doingInference",
	stage = "Runtime",
	type = bool,
	default = false
})
spec.addParam({
	name = "factorEval",
	stage = "Runtime",
	type = bool,
	default = true
})
spec.addParam({
	name = "structureChange",
	stage = "Runtime",
	type = bool,
	default = true
})
spec.addParam({
	name = "relaxManifolds",
	stage = "Runtime",
	type = bool,
	default = false
})

local terra setRuntimeDefaults()
	[spec.genSetRuntimeVars()]
end
setRuntimeDefaults()
-----------------------------------

-- Add all the 'global' exports from the module named 'name'
--    to the global environment.
local function processModule(name)
	local mod = require(string.format("prob.%s", name))
	local globals = mod.globals
	for k,v in pairs(globals) do
		local val = v
		-- If this export is actually a specializable function, then
		-- we register it globally and return the default specialization
		-- (invoking with no arguments)
		if spec.isSpecializable(v) then
			spec.registerGlobalSpecializable(k, v)
			val = v()
		end
		rawset(_G, k, val)
	end
end
local function processModules(...)
	for i=1,select("#",...) do processModule((select(i,...))) end
end

processModules("erph",
			   "erp",
			   "trace",
			   "inference",
			   "memoize",
			   "larj",
			   "hmc",
			   "specialize")

-- Seed the random number generator
rand.initrand()





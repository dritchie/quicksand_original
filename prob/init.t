local spec = terralib.require("prob.specialize")

-- Add all the 'global' exports from the module named 'name'
--    to the global environment.
local function processModule(name)
	local mod = terralib.require(string.format("prob.%s", name))
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

processModules("erp", "trace", "inference", "memoize", "larj")
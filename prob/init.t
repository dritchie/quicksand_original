local modules = {}
table.insert(modules, terralib.require("prob.erp"))
table.insert(modules, terralib.require("prob.trace"))
table.insert(modules, terralib.require("prob.inference"))
table.insert(modules, terralib.require("prob.memoize"))
table.insert(modules, terralib.require("prob.larj"))

-- Forward exports
local exports = {}
for _,m in ipairs(modules) do
	for k,v in pairs(m) do
		exports[k] = v
	end
end

return exports
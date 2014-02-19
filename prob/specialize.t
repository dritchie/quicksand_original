local templatize = terralib.require("templatize")
local util = terralib.require("util")

local currindex = 1
local defaults = {}
local indices = {}
local names = {}

local function addParam(name, defaultValue)
	defaults[name] = defaultValue
	indices[name] = currindex
	currindex = currindex + 1
	table.insert(names, name)
end

-- Extract parameter by name from ordered list (all present)
local function paramFromList(name, ...)
	local ret = (select(indices[name], ...))
	return ret
end

-- Extract parameter by name from table (some may be missing)
local function paramFromTable(name, tbl)
	if tbl[name] ~= nil then return tbl[name] else return defaults[name] end
end

-- Convert parameter table into ordered list
local function paramTableToList(tbl)
	local lst = {}
	for name,value in pairs(defaults) do
		lst[indices[name]] = paramFromTable(name, tbl)
	end
	return lst
end

local function paramListToTable(...)
	local tbl = {}
	for i=1,select("#",...) do
		tbl[names[i]] = (select(i,...))
	end
	return tbl
end


-- Specializable computations
local specializableMT = {
	__call = function(self, paramTable)
		paramTable = paramTable or {}
		local plist = paramTableToList(paramTable)
		return self.fn(unpack(plist))
	end
}
local function specializable(fn)
	local newobj = { fn = templatize(fn) }
	setmetatable(newobj, specializableMT)
	return newobj
end
local function isSpecializable(obj)
	return getmetatable(obj) == specializableMT
end

-- Register different events to run before doing a global specialization
local preGlobalEvents = {}
local function registerPreGlobalSpecializationEvent(event)
	table.insert(preGlobalEvents, event)
end
local function runPreGlobalSpecializationEvents()
	for _,e in ipairs(preGlobalEvents) do
		e()
	end
end

-- The registry of global specializables, and methods for modifying
-- the global environment according to specialization parameters
local globalSpecs = {}
local function registerGlobalSpecializable(name, obj)
	globalSpecs[name] = obj
end
local function executeUnderGlobalSpecialization(thunk, paramTable)
	runPreGlobalSpecializationEvents()
	local globalEnv = util.copytable(_G)
	-- Set up global specialization environment
	for name,obj in pairs(globalSpecs) do
		rawset(_G, name, obj(paramTable))
	end
	-- Execute under new specialization environment
	local ret = thunk()
	ret:compile()
	-- Restore previous versions
	for name,_ in pairs(globalSpecs) do
		rawset(_G, name, globalEnv[name])
	end
	return ret
end

-- Specialization wrapper for overall probabilistic computation thunks
local specthunkmt =
{
	__call = function(self, paramTable)
		paramTable = paramTable or {}
		paramTable.computation = self
		return self.templatethunk(unpack(paramTableToList(paramTable)))
	end
}
local function probcomp(thunk)
	local newobj = 
	{
		templatethunk = templatize(function(...)
			local paramTable = paramListToTable(...)
			return executeUnderGlobalSpecialization(thunk, paramTable)
		end)
	}
	setmetatable(newobj, specthunkmt)
	return newobj
end
local function isProbComp(comp)
	return getmetatable(comp) == specthunkmt
end
local function ensureProbComp(comp)
	if isProbComp(comp) then
		return comp
	else
		return probcomp(comp)
	end
end


return
{
	addParam = addParam,
	paramFromList = paramFromList,
	paramFromTable = paramFromTable,
	paramTableToList = paramTableToList,
	paramListToTable = paramListToTable,
	specializable = specializable,
	isSpecializable = isSpecializable,
	probcomp = probcomp,
	isProbComp = isProbComp,
	ensureProbComp = ensureProbComp,
	registerPreGlobalSpecializationEvent = registerPreGlobalSpecializationEvent,
	registerGlobalSpecializable = registerGlobalSpecializable,
	globals =
	{
		probcomp = probcomp
	}
}







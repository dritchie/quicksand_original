local templatize = terralib.require("templatize")
local util = terralib.require("util")


local currindex = 1
local values = {}
local defaults = {}
local indices = {}
local names = {}

local function addParam(name, defaultValue)
	values[name] = defaultValue
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


-- Associating unique IDs with param tables
local psetid = 1
local ParamSetID = templatize(function(...)
	psetid = psetid + 1
	return psetid-1
end)
local function paramTableID(paramTable)
	return ParamSetID(unpack(paramTableToList(paramTable)))
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

-- The registry of global specializables, and methods for modifying
-- the global environment according to specialization parameters
local globalSpecs = {}
local function registerGlobalSpecializable(name, obj)
	globalSpecs[name] = obj
end
local function executeUnderGlobalSpecialization(thunk, paramTable)
	local globalEnv = util.copytable(_G)
	-- Set up global specialization environment
	for name,obj in pairs(globalSpecs) do
		rawset(_G, name, obj(paramTable))
	end
	-- Execute under new specialization environment
	local ret = thunk()
	-- Restore previous versions
	for name,_ in pairs(globalSpecs) do
		rawset(_G, name, globalEnv[name])
	end
	return ret
end

-- Specialization wrapper for overall probabilistic computation thunks
local function specializablethunk(thunk)
	local templatefn = templatize(function(...)
		local paramTable = paramListToTable(...)
		return executeUnderGlobalSpecialization(thunk, paramTable)
	end)
	return function(paramTable)
		paramTable = paramTable or {}
		return templatefn(unpack(paramTableToList(paramTable)))
	end
end

----------------------------------
-- The actual parameters
addParam("structureChange", true)
addParam("factorEval", true)

----------------------------------

-- -- TEST
-- assert(globalParam("structureChange") == true)
-- assert(globalParam("factorEval") == true)
-- local t = {structureChange=false}
-- local l = paramTableToList(t)
-- assert(paramFromTable("structureChange", t) == false)
-- assert(paramFromTable("factorEval", t) == true)
-- assert(paramFromList("structureChange", unpack(l)) == false)
-- assert(paramFromList("factorEval", unpack(l)) == true)
-- t = paramListToTable(unpack(l))
-- assert(paramFromTable("structureChange", t) == false)
-- assert(paramFromTable("factorEval", t) == true)

return
{
	paramFromList = paramFromList,
	paramFromTable = paramFromTable,
	paramTableToList = paramTableToList,
	paramListToTable = paramListToTable,
	paramTableID = paramTableID,
	specializable = specializable,
	specializablethunk = specializablethunk,
	isSpecializable = isSpecializable,
	registerGlobalSpecializable = registerGlobalSpecializable
}







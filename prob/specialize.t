local templatize = terralib.require("templatize")

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

local function resetParamsToDefaults()
	for k,v in pairs(defaults) do
		values[k] = v
	end
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

-- Get current value of one of the globals
local function globalParam(name)
	return paramFromTable(name, values)
end

local specializeWithParams = templatize(function(computation, ...)
	local ptbl = paramListToTable(...)
	for name,value in pairs(ptbl) do
		values[name] = value
	end
	local comp = computation()
	resetParamsToDefaults()
	return comp
end)

local function specializeWithGlobals(computation)
	return specializeWithParams(computation, unpack(paramTableToList(values)))
end

local function default(computation)
	return specializeWithParams(computation, unpack(paramTableToList({})))
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
	globalParam = globalParam,
	paramTableID = paramTableID,
	specializeWithParams = specializeWithParams,
	specializeWithGlobals = specializeWithGlobals,
	default = default
}







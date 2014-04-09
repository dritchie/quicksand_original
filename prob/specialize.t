local templatize = terralib.require("templatize")
local util = terralib.require("util")


local currindex = 1
local defs = {}
local orderedNames = {}
local runtimeGlobals = {}


local function addParam(def)
	util.luaAssertWithTrace(def.name, "Specialization parameter must have a name.")
	util.luaAssertWithTrace(def.stage, "Specialization parameter must have a stage (Compile or Runtime).")
	if def.stage == "Runtime" then util.luaAssertWithTrace(def.type, "Runtime specialization parameter must have a type.") end
	util.luaAssertWithTrace(def.default ~= nil, "Specialization parameter must have a default value.")

	def.index = currindex
	currindex = currindex + 1
	defs[def.name] = def
	table.insert(orderedNames, def.name)

	if def.stage == "Runtime" then
		runtimeGlobals[def.name] = global(def.type)
	end
end

local function genSetRuntimeVars(varVals)
	varVals = varVals or {}
	local stmts = {}
	for name,def in pairs(defs) do
		if def.stage == "Runtime" then
			local val = def.default
			if varVals[name] ~= nil then val = varVals[name] end
			table.insert(stmts, quote [runtimeGlobals[name]] = [val] end)
		end
	end
	return stmts
end

local function getRuntimeVar(name)
	return runtimeGlobals[name]
end

-- Extract parameter by name from ordered list (all present)
local function paramFromList(name, ...)
	-- assert(defs[name].stage == "Compile",
	-- 	string.format("Use spec.getRuntimeVar to get the value of a Runtime specialization parameter.\n(Attempted to get '%s')", name))
	local ret = (select(defs[name].index, ...))
	return ret
end

-- Extract parameter by name from table (some may be missing)
local function paramFromTable(name, tbl)
	-- assert(defs[name].stage == "Compile",
	-- 	string.format("Use spec.getRuntimeVar to get the value of a Runtime specialization parameter.\n(Attempted to get '%s')", name))
	if tbl[name] ~= nil then return tbl[name] else return defs[name].default end
end

-- Convert parameter table into ordered list
local function paramTableToList(tbl)
	local lst = {}
	for name,def in pairs(defs) do
		local val = def.default
		if tbl[name] ~= nil then val = tbl[name] end
		lst[def.index] = val
	end
	return lst
end

-- Undo the above
local function paramListToTable(...)
	local tbl = {}
	for i=1,select("#",...) do
		tbl[orderedNames[i]] = (select(i,...))
	end
	return tbl
end

-- Extract just the compile time parameters into a list
-- (Facilitates templating something on those parameters)
local function sortByIndex(def1, def2)
	return def1.index < def2.index
end
local function compileTimeParamList(tbl)
	local cmpldefs = {}
	for name,def in pairs(defs) do
		if def.stage == "Compile" then
			table.insert(cmpldefs, def)
		end
	end
	table.sort(cmpldefs, sortByIndex)
	local lst = {}
	for _,def in ipairs(cmpldefs) do
		local val = def.default
		if tbl[def.name] ~= nil then val = tbl[def.name] end
		table.insert(lst, val)
	end
	return lst
end


-- Specializable computations
local specializableMT = {
	__call = function(self, paramTable)
		paramTable = paramTable or {}
		return self.callimpl(paramTable)
	end
}
local function specializable(fn)
	local currSpecParamTable = nil
	local templated = templatize(function(...)
		return fn(unpack(paramTableToList(currSpecParamTable)))
	end)
	local newobj =
	{
		callimpl = function(paramTable)
			currSpecParamTable = paramTable
			-- Only need to templatize on the compile time params (reduces overcompilation)
			return templated(unpack(compileTimeParamList(paramTable)))
		end
	}
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
	-- Execute (and compile) under new specialization environment
	local ret = thunk()
	util.luaAssertWithTrace(terralib.isfunction(ret), "Return value of a probcomp must be a Terra function")
	util.luaAssertWithTrace(#ret:gettype().parameters == 0, "Return value of a probcomp must be a no-argument Terra function")
	ret:compile()
	-- Restore previous versions
	for name,_ in pairs(globalSpecs) do
		rawset(_G, name, globalEnv[name])
	end
	return ret
end

-- Specialization wrapper for overall probabilistic computation thunks
local probcompmt =
{
	__call = function(self, paramTable)
		paramTable = paramTable or {}
		paramTable.computation = self
		local fn = self.callimpl(paramTable)
		-- Wrap returned Terra function with another that will set and restore
		--    the runtime vars
		return terra()
			[genSetRuntimeVars(paramTable)]
			var x = fn()
			[genSetRuntimeVars()]
			return x
		end
	end
}
local currExecutingProbComp = nil
local function probcomp(thunk)
	local currSpecParamTable = nil
	local wrapped = function(paramTable)
		currExecutingProbComp = paramTable.computation
		local ret = executeUnderGlobalSpecialization(thunk, paramTable)
		currExecutingProbComp = nil
		return ret
	end
	local templated = templatize(function(...)
		-- print("======== templated")
		-- print((select(1, ...)))
		-- print((select(2, ...)))
		-- print("-----------")
		return wrapped(currSpecParamTable)
	end)
	local newobj = 
	{
		callimpl = function(paramTable)
			currSpecParamTable = paramTable
			return templated(unpack(compileTimeParamList(paramTable)))
		end
	}
	setmetatable(newobj, probcompmt)
	return newobj
end
local function isProbComp(comp)
	return getmetatable(comp) == probcompmt
end
local function ensureProbComp(comp)
	if isProbComp(comp) then
		return comp
	else
		return probcomp(comp)
	end
end

-- probmodules are re-usable chunks of code that probcomps can use.
-- They can be called with no arguments inside of a probcomp.
--    In this version, a return value will be generated if necessary, retrieved if possible
-- Otherwise, they must be passed a probcomp as argument.
--    In this version, we attempt to retrieve a memoized value. If this fails, we call
--    the probcomp, in the hopes that it will generate and memoize the value we are looking
--    for. If this fails, we throw an error.
-- The reason for all of this hoopla is to guarantee that staging code is alwaysexecuted (and that
--    types are generated) in the order defined by the probcomp. I'm sure this is an overly-
--    conservative requirement, but I'm working with it for now.
local probmodulemt = 
{
	-- (Uses the current global value of 'real')
	__call = function(self, pcomp)
		if currExecutingProbComp then
			return self:cacheLookupOrCreate(real, currExecutingProbComp)
		else
			util.luaAssertWithTrace(pcomp, "Argument to a probmodule can only be omitted if calling it from within a probcomp.")
			util.luaAssertWithTrace(isProbComp(pcomp), "Argument to a probmodule must be a probcomp.")
			local val = self:cacheLookup(real, pcomp)
			if not val then
				pcomp()	-- Hope that this generates and memoizes the value we're looking for.
				val = self:cacheLookup(real, pcomp)
				util.luaAssertWithTrace(val, "probmodule must be used in the probcomp passed to it as argument.")
				return val
			else
				return val
			end
		end
	end,

	cacheLookup = function(self, real, pcomp)
		local key = util.stringify(real, pcomp)
		return self.cache[key]
	end,

	cachePut = function(self, real, pcomp, val)
		local key = util.stringify(real, pcomp)
		self.cache[key] = val
	end,

	cacheLookupOrCreate = function(self, real, pcomp)
		local val = self:cacheLookup(real, pcomp)
		if not val then
			val = self.thunk()
			self:cachePut(real, pcomp, val)
		end
		return val
	end
}
probmodulemt.__index = probmodulemt
local function probmodule(thunk)
	local newobj = 
	{
		thunk = thunk,
		cache = {}
	}
	setmetatable(newobj, probmodulemt)

	return newobj
end
local function isProbModule(mod)
	return getmetatable(mod) == probmodulemt
end


return
{
	addParam = addParam,
	genSetRuntimeVars = genSetRuntimeVars,
	getRuntimeVar = getRuntimeVar,
	paramFromList = paramFromList,
	paramFromTable = paramFromTable,
	paramTableToList = paramTableToList,
	paramListToTable = paramListToTable,
	specializable = specializable,
	isSpecializable = isSpecializable,
	probcomp = probcomp,
	isProbComp = isProbComp,
	probmodule = probmodule,
	isProbModule = isProbModule,
	ensureProbComp = ensureProbComp,
	registerPreGlobalSpecializationEvent = registerPreGlobalSpecializationEvent,
	registerGlobalSpecializable = registerGlobalSpecializable,
	globals =
	{
		probcomp = probcomp,
		probmodule = probmodule
	}
}







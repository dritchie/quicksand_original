
local random = terralib.require("prob.random")
local templatize = terralib.require("templatize")
local inheritance = terralib.require("inheritance")
local m = terralib.require("mem")

local C = terralib.includecstring [[
#include <stdio.h>
#include <stdlib.h>
]]

local terra notImplementedError(methodname: &int8, classname: &int8)
	C.printf("Virtual function '%s' not implemented in class '%s'", methodname, classname)
	C.exit(1)
end


-- Base class for all random variables
local struct RandVar
{
	logprob: double,
	isStructural: bool,
	isDirectlyConditioned: bool,
	isActive: bool
}

terra RandVar:__construct(lp: double, isstruct: bool, iscond: bool)
	self.logprob = lp
	self.isStructural = isstruct
	self.isDirectlyConditioned = iscond
	self.isActive = true
end

terra RandVar:valueTypeID() : uint64
	notImplementedError("valueTypeID", "RandVar")
end
inheritance.virtual(RandVar.methods.valueTypeID)

terra RandVar:pointerToValue() : &opaque
	notImplementedError("pointerToValue", "RandVar")
end
inheritance.virtual(RandVar.methods.valueTypeID)

terra RandVar:updateLogprob() : {}
	notImplementedError("updateLogprob", "RandVar")
end
inheritance.virtual(RandVar.methods.updateLogprob)


-- Functions to inspect the value type of any random variable
local typeToIDMap = {}
local currID = 0
local function typeToID(terratype)
	local id = typeToIDMap[terratype]
	if not id then
		id = currID
		currID = currID + 1
		typeToIDMap[terratype] = id
	end
	return id
end
local valueIs = templatize(function(T)
	local Ttype = typeToID(T)
	return terra(randvar: &RandVar)
		return randvar:valueTypeID() == Ttype
	end
end)
local valueAs = templatize(function(T)
	return terra(randvar: &RandVar)
		return @([&T](randvar:pointerToValue()))
	end
end)


-- Every random variable has some value type; this intermediate
-- class manages that
local RandVarWithVal = templatize(function(ValType)
	local struct RandVarWithValT
	{
		value: ValType
	}

	terra RandVarWithValT:__construct(val: ValType, lp: double, isstruct: bool, iscond: bool, isact: bool)
		RandVar.__construct(self, lp, isstruct, iscond, isact)
		self.value = val
	end

	terra RandVarWithValT:__destruct()
		m.destruct(self.value)	
	end

	local ValTypeID = typeToID(ValType)
	terra RandVarWithValT:valueTypeID() : uint64
		return ValTypeID
	end
	inheritance.virtual(RandVarWithValT.methods.valueTypeID)

	terra RandVarWithValT:pointerToValue() : &opaque
		return [&opaque](&self.value)
	end
	inheritance.virtual(RandVarWithValT.methods.pointerToValue)

	terra RandVarWithValT:proposeNewValue() : {ValType, double, double}
		notImplementedError("proposeNewValue", [string.format("RandVarWithVal(%s)", tostring(ValType))])
	end
	inheritance.virtual(RandVarWithValT.methods.proposeNewValue)

	inheritance.dynamicExtend(RandVar, RandVarWithValT)
	return RandVarWithValT
end)


-- Finally, at the bottom of the hierarchy, we have random primitives defined by a set of functions:
--    * A sampling function
--    * A log probability function
--    * (Optional) A proposal function
--    * (Optional) A proposal log probability function
local function RandVarFromFunctions(sample, logprob, propose, logProposalProb)
	assert(#sample:getdefinitions()[1]:gettype().returns = 1)	-- Can't handle functions with multiple return values
	local ValType = sample:getdefinitions()[1]:gettype().returns[1]
	local ParamTypes = sample:getdefinitions()[1]:gettype().parameters

	-- If we don't have propose or logProposalProb functions, make macros for default
	-- versions of these things given the functions we do have
	if not propose then
		-- Default: sample a new value irrespective of the current value (argument 1)
		propose = macro(function(...)
			local numargs = #ParamTypes + 1
			local args = {}
			for i=2,numargs do
				table.insert(args, select(i,...))
			end
			return `sample([args])
		end)
	end
	if not logProposalProb then
		-- Default: Score the proposed value irrespective of the current value
		logProposalProb = macro(function(...)
			local numargs = #ParamTypes + 2
			local args = {}
			for i=2,numargs do
				table.insert(args, select(i,...))
			end
			return `sample([args])
		end)
	end

	-- Initialize the class we're building
	local struct RandVarFromFunctionsT {}

	-- Add one field for each parameter
	local paramFieldNames = {}
	for i,t in ipairs(ParamTypes) do
		local n = string.format("param%d", i-1),
		table.insert(paramFieldNames, n)
		RandVarFromFunctionsT.entries:insert({ field = n, type = t})
	end

	local function genDestructBlock(self)
		local statements = {}
		for i,n in paramFieldNames do
			table.insert(statements, `m.destruct(self.[n]))
		end
		return statements
	end
	terra RandVarFromFunctionsT:__destruct()
		[RandVarWithVal(ValType)].__destruct(self)
		[genDestructBlock(self)]
	end

	-- UpdateLogprob

	-- Propose

	inheritance.dynamicExtend(RandVarWithVal(ValType), RandVarFromFunctionsT)
	return RandVarFromFunctionsT
end








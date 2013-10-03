local inheritance = terralib.require("inheritance")
local templatize = terralib.require("templatize")

local C = terralib.includecstring [[
#include <stdio.h>
#include <stdlib.h>
]]

local terra notImplementedError(methodname: &int8, classname: &int8)
	C.printf("Virtual function '%s' not implemented in class '%s'\n", methodname, classname)
	terralib.traceback(nil)
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

terra RandVar:__construct(isstruct: bool, iscond: bool)
	self.isStructural = isstruct
	self.isDirectlyConditioned = iscond
	self.isActive = true
	self.logprob = 0.0
end

terra RandVar:__copy(othervar: &RandVar)
	self.isStructural = othervar.isStructural
	self.isDirectlyConditioned = othervar.isDirectlyConditioned
	self.isActive = othervar.isActive
	self.logprob = othervar.logprob
end

terra RandVar:__destruct() : {}
	notImplementedError("__destruct", "RandVar")
end
inheritance.virtual(RandVar, "__destruct")

terra RandVar:deepcopy() : &RandVar
	notImplementedError("deepcopy", "RandVar")
end
inheritance.virtual(RandVar, "deepcopy")

terra RandVar:valueTypeID() : uint64
	notImplementedError("valueTypeID", "RandVar")
end
inheritance.virtual(RandVar, "valueTypeID")

terra RandVar:pointerToValue() : &opaque
	notImplementedError("pointerToValue", "RandVar")
end
inheritance.virtual(RandVar, "pointerToValue")

terra RandVar:proposeNewValue() : {double, double}
	notImplementedError("proposeNewValue", "RandVar")
end
inheritance.virtual(RandVar, "proposeNewValue")

terra RandVar:setValue(valptr: &opaque) : {}
	notImplementedError("setValue", "RandVar")
end
inheritance.virtual(RandVar, "setValue")



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


return
{
	RandVar = RandVar,
	notImplementedError = notImplementedError,
	typeToID = typeToID,
	valueIs = valueIs,
	valueAs = valueAs
}
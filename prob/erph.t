local inheritance = terralib.require("inheritance")
local templatize = terralib.require("templatize")


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

inheritance.purevirtual(RandVar, "__destruct", {}->{})
inheritance.purevirtual(RandVar, "deepcopy", {}->{&RandVar})
inheritance.purevirtual(RandVar, "valueTypeID", {}->{uint64})
inheritance.purevirtual(RandVar, "pointerToValue", {}->{&opaque})
inheritance.purevirtual(RandVar, "proposeNewValue", {}->{double,double})
inheritance.purevirtual(RandVar, "setValue", {&opaque}->{})


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
	typeToID = typeToID,
	valueIs = valueIs,
	valueAs = valueAs
}
local inheritance = terralib.require("inheritance")
local templatize = terralib.require("templatize")
local virtualTemplate = terralib.require("vtemplate")
local ad = terralib.require("ad")
local spec = terralib.require("prob.specialize")

local C = terralib.includecstring [[
#include <stdio.h>
]]


-- Base class for all random variables
local RandVar
RandVar = templatize(function(ProbType)
	local struct RandVarT
	{
		logprob: ProbType,
		isStructural: bool,
		isDirectlyConditioned: bool,
		isActive: bool
	}

	terra RandVarT:__construct(isstruct: bool, iscond: bool)
		self.isStructural = isstruct
		self.isDirectlyConditioned = iscond
		self.isActive = true
		self.logprob = ProbType(0.0)
	end

	RandVarT.__templatecopy = templatize(function(P)
		return terra(self: &RandVarT, other: &RandVar(P))
			self.isStructural = other.isStructural
			self.isDirectlyConditioned = other.isDirectlyConditioned
			self.isActive = other.isActive
			self.logprob = other.logprob	-- a cast had better exist
		end
	end)

	inheritance.purevirtual(RandVarT, "__destruct", {}->{})
	inheritance.purevirtual(RandVarT, "valueTypeID", {}->{uint64})
	inheritance.purevirtual(RandVarT, "pointerToValue", {}->{&opaque})
	inheritance.purevirtual(RandVarT, "proposeNewValue", {}->{ProbType,ProbType})
	inheritance.purevirtual(RandVarT, "setValue", {&opaque}->{})

	RandVarT.deepcopy = virtualTemplate(RandVarT, "deepcopy", function(P) return {}->{&RandVar(P)} end)

	return RandVarT
end)


-- Functions to inspect the value type of any random variable
local currID = 1
local typeToID = templatize(function(terratype)
	currID = currID+1
	return currID-1
end)
local valueIs = templatize(function(T)
	local Ttype = typeToID(T)
	return macro(function(randvar)
		return `randvar:valueTypeID() == Ttype
	end)
end)
local valueAs = templatize(function(T)
	return macro(function(randvar)
		return `@([&T](randvar:pointerToValue()))
	end)
end)


-- A utility that can convert templated functions of a certain form into
--    overloaded function.
-- This takes a function templated on a single Value type and an arbitrary
--    number of Param types. It returns a function that is templated on 
--    the Value type but overloaded on possible Param types.
-- A Param type is a scalar type -- either double or ad.num.
-- A Value type is also a scalar type.
-- This is useful for creating samplers and scorers for ERPs
--    (macros are another option).
local function overloadOnParams(numparams, fntemplate)
	-- Templatize on the value type
	return templatize(function(V)
		-- Generate overloads for all combinations of
		-- parameter types.
		if numparams == 0 then
			return fntemplate(V)
		elseif V ~= ad.num then
			local types = {}
			for i=1,numparams do table.insert(types, double) end
			return fntemplate(V, unpack(types))
		else
			local overallfn = nil
			local numVariants = 2 ^ numparams
			local bitstring = 0
			for i=1,numVariants do
				local types = {}
				for j=0,numparams-1 do
					if bit.band(bit.tobit(2^j), bit.tobit(bitstring)) == 0 then
						table.insert(types, double)
					else
						table.insert(types, ad.num)
					end
				end
				local fn = fntemplate(V, unpack(types))
				if not overallfn then
					overallfn = fn
				else
					overallfn:adddefinition(fn:getdefinitions()[1])
				end
				bitstring = bitstring + 1
			end
			return overallfn
		end
	end)
end


-- We sometimes need to have a unique id for every ERP callsite in the program.
local erpid = 1
local function getCurrentERPID()
	erpid = erpid + 1
	return erpid - 1
end
local function resetERPID() erpid = 1 end
-- The id counter needs to reset at the beginning of compiling a computation,
--    so that corresponding ERPs from different specializations have the same id. 
spec.registerPreGlobalSpecializationEvent(resetERPID)


return
{
	RandVar = RandVar,
	typeToID = typeToID,
	valueIs = valueIs,
	valueAs = valueAs,
	overloadOnParams = overloadOnParams,
	getCurrentERPID = getCurrentERPID,
}





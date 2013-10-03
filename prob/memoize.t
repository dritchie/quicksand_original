local m = terralib.require("mem")
local templatize = terralib.require("templatize")
local hash = terralib.require("hash")
local HashMap = terralib.require("hashmap")
local util = terralib.require("util")

-- Generally useful helpers
local function fields(inst, num)
	local exps = {}
	for i=1,num do table.insert(exps, `inst.[string.format("_%d", i-1)]) end
	return exps
end
local function initwrap(exps)
	local wrappedexps = {}
	for _,e in ipairs(exps) do table.insert(wrappedexps, `m.init(e)) end
	return wrappedexps
end
local function copywrap(exps)
	local wrappedexps = {}
	for _,e in ipairs(exps) do table.insert(wrappedexps, `m.copy(e)) end
	return wrappedexps
end
local function destructwrap(exps)
	local wrappedexps = {}
	for _,e in ipairs(exps) do table.insert(wrappedexps, `m.destruct(e)) end
	return wrappedexps
end

-- Build a struct that has one member of every argument type.
-- We use these both as key and value entries in mem cache hash maps
-- The __hash method simply invokes __hash on all members, puts all the resulting
--    numbers into an array, and the hashes that array data.
-- The __eq method simply checks equality of all members
local MultiTypeMemberStruct = templatize(function(...)
	-- Setup struct and its (unnamed) entries
	local struct aggregate {}
	local numtypes = select("#",...)
	for i=1,numtypes do aggregate.entries:insert((select(i,...))) end
	-- Some helpers
	local function elems(array, len)
		local exps = {}
		for i=1,len do table.insert(exps, `array[i-1]) end
		return exps
	end
	local dohash = macro(function(val)
		local t = val:gettype()
		local hashfn = hash.gethashfn(t)
		return `hashfn(val)
	end)
	local function hashwrap(exps)
		local wrappedexps = {}
		for _,e in ipairs(exps) do table.insert(wrappedexps, `dohash(e)) end
		return wrappedexps
	end
	local function checkequality(exps1, exps2)
		local eqexp = `([exps1[1]] == [exps2[1]])
		for i=2,#exps1 do eqexp = `eqexp and ([exps1[i]] == [exps2[i]]) end
		return eqexp
	end
	-- Generate methods
	terra aggregate:__construct()
		[initwrap(fields(self, numtypes))]
	end
	local params = {}
	for i=1,numtypes do table.insert(params, symbol(select(i,...))) end
	terra aggregate:__construct([params])
		[fields(self, numtypes)] = [copywrap(params)]
	end
	terra aggregate:__copy(other: &aggregate)
		[fields(self, numtypes)] = [copywrap(fields(other, numtypes))]
	end
	terra aggregate:__destruct()
		[destructwrap(fields(self, numtypes))]
	end
	terra aggregate:__hash()
		var hashintermediates : uint[numtypes]
		[elems(hashintermediates, numtypes)] = [hashwrap(fields(self, numtypes))]
		return hash.rawhash([&int8](&hashintermediates[0]), numtypes*sizeof(uint))
	end
	-- Does one of these types need to be 'aggregate' and not '&aggregate'?
	aggregate.metamethods.__eq = terra(self: &aggregate, other: &aggregate)
		return [checkequality(fields(self, numtypes), fields(other, numtypes))]
	end

	m.addConstructors(aggregate)
	return aggregate
end)

local MemFn = templatize(function(fn)
	-- Some helpers
	local function aggregates(fndef)
		local t = fndef:gettype()
		return MultiTypeMemberStruct(unpack(t.parameters)), MultiTypeMemberStruct(unpack(t.returns))
	end

	local struct MemFn {}
	-- One hash map per definition
	-- Record which hash map member was created from which parameter types
	local numdefs = #fn:getdefinitions()
	local paramRecToEntryID = {}
	for i=1,numdefs do
		local fndef = fn:getdefinitions()[i]
		local ParamRec, ReturnRec = aggregates(fndef)
		paramRecToEntryID[ParamRec] = i
		local MapType = HashMap(ParamRec, ReturnRec)
		MemFn.entries:insert(MapType)
	end
	-- Methods
	terra MemFn:__construct()
		[initwrap(fields(self, numdefs))]
	end
	terra MemFn:__copy(other: &MemFn)
		[fields(self, numdefs)] = [copywrap(fields(other, numdefs))]
	end
	terra MemFn:__destruct()
		[destructwrap(fields(self, numdefs))]
	end
	-- Invocation
	MemFn.metamethods.__apply = macro(function(self, ...)
		local args = {}
		for i=1,select("#",...) do table.insert(args, (select(i,...))) end
		local argtypes = {}
		for _,a in ipairs(args) do table.insert(argtypes, a:gettype()) end
		local ParamRec = MultiTypeMemberStruct(unpack(argtypes))
		local whichDef = paramRecToEntryID[ParamRec]
		if not whichDef then error("No definition matching arguments to memoized function.") end
		local numRets = #fn:getdefinitions()[whichDef]:gettype().returns
		local numParams = #fn:getdefinitions()[whichDef]:gettype().parameters
		local hmap = `self.[string.format("_%d", whichDef-1)]
		return quote
			var paramrec : ParamRec
			-- Bypass copy b/c we won't be keeping this around
			[fields(paramrec, numParams)] = [args]
			var retp, found = hmap:getOrCreatePointer(paramrec)
			if not found then
				-- Bypass copy b/c fn return gives us ownership
				[fields(retp, numRets)] = fn([args])
			end
		in
			[fields(retp, numRets)]
		end
	end)

	m.addConstructors(MemFn)
	return MemFn
end)

-- Memoize a terra function (or macro that forwards function definitions)
-- NOTE: caller takes ownership of return object's memory.
local function mem(fn)
	local MemFnType = MemFn(fn)
	return `MemFnType.stackAlloc()
end



return
{
	globals = { mem = mem }
}



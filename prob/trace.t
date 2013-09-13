local erp = terralib.require("erp.h")
local RandVar = erp.RandVar

local Vector = terralib.require("Vector")


-- VARIABLE ADDRESSING

local callsiteStack = global(Vector(int))
local loopnumStack = global(Vector(int))
Vector(int).methods.__construct(callsiteStack:get())
Vector(int).methods.__construct(loopnumStack:get())

-- Wrap a Terra function in a macro that, when called, will assign a
--    unique id to that call site.
local nextid = 0
local function pfn(fn)
	-- Prerequisites: fn must be a terra function, and if it is overloaded,
	--   all overloads must have the same number of return types
	assert(terralib.isfunction(fn))
	local numrets = #fn:getdefinitions()[1]:gettype().returns
	for i=2,#fn:getdefinitions() do
		assert(#fn:getdefinitions()[i]:gettype().returns == numrets)
	end

	-- Now do code gen
	local myid = nextid
	nextid = nextid + 1
	return macro(function(...)
		local args = {}
		for i=1,select("#",...) table.insert(args, (select(i,...))) end
		if numrets == 0 then
			return quote
				callsiteStack:push(myid)
				fn([args])
				callsiteStack:pop()
			end
		else
			local results = {}
			for i=1,numrets do table.insert(results, symbol()) end
			return quote
				callsiteStack:push(myid)
				[results] = fn([args])
				callsiteStack:pop()
			in
				[results]
			end
		end
	end)
end


-- PUBLIC INTERFACE

local terra lookupVariable(isStructural: bool) : &RandVar
	-- TODO: Complete!
	return nil
end

local terra addNewVariable(newvar: &RandVar) : {}
	--
end


return
{
	pfn = pfn,
	lookupVariable = lookupVariable,
	addNewVariable = addNewVariable
}
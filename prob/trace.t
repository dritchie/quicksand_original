local erp = terralib.require("prob.erph")
local RandVar = erp.RandVar
local iface = terralib.require("interface")
local Vector = terralib.require("vector")
local HashMap = terralib.require("hashmap")
local templatize = terralib.require("templatize")


-- ADDRESS TRANSFORM

local callsiteStack = global(Vector(int))
Vector(int).methods.__construct(callsiteStack:getpointer())

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
		for i=1,select("#",...) do table.insert(args, (select(i,...))) end
		local argintermediates = {}
		for _,a in ipairs(args) do table.insert(argintermediates, symbol(a:gettype())) end
		if numrets == 0 then
			return quote
				[argintermediates] = [args]
				callsiteStack:push(myid)
				fn([argintermediates])
				callsiteStack:pop()
			end
		else
			local results = {}
			for i=1,numrets do table.insert(results, symbol()) end
			return quote
				[argintermediates] = [args]
				callsiteStack:push(myid)
				[results] = fn([argintermediates])
				callsiteStack:pop()
			in
				[results]
			end
		end
	end)
end


-- TRACE MANAGEMENT

local GlobalTraceInterface = iface.create {
	lookupVariable: {bool} -> &RandVar;
	addNewVariable: {&RandVar} -> {};
	factor: {double} -> {};
	condition: {bool} -> {};
}

local globalTrace = global(GlobalTraceInterface)

local terra setGlobalTrace(trace: GlobalTraceInterface)
	globalTrace = trace
end


local RandExecTrace = templatize(function(ComputationType)

	local struct Trace
	{
		computation: &ComputationType
	}

end)



-- PUBLIC INTERFACE

local terra lookupVariable(isStructural: bool) : &RandVar
	-- TODO: Complete!
	return nil
end

local terra addNewVariable(newvar: &RandVar) : {}
	-- TODO: Complete!
end


return
{
	pfn = pfn,
	lookupVariable = lookupVariable,
	addNewVariable = addNewVariable
}
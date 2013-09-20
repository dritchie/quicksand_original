local erp = terralib.require("prob.erph")
local RandVar = erp.RandVar
local util = terralib.require("util")
local iface = terralib.require("interface")
local Vector = terralib.require("vector")
local HashMap = terralib.require("hashmap")
local templatize = terralib.require("templatize")
local m = terralib.require("mem")


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
local haveGlobalTrace = global(bool, false)

local terra setGlobalTrace(trace: GlobalTraceInterface)
	globalTrace = trace
	haveGlobalTrace = true
end
util.inline(setGlobalTrace)

local terra unsetGlobalTrace()
	haveGlobalTrace = false
end
util.inline(unsetGlobalTrace)

local terra globalTraceIsSet()
	return haveGlobalTrace
end
util.inline(globalTraceIsSet)


local RandExecTrace = templatize(function(ComputationType)

	-- For the time being at least, we restrict inference to
	-- computations with a single return value
	if #ComputationType.returns ~= 1 then
		error("Can only do inference on computations with a single return value.")
	end

	local IdSeq = Vector(int)

	-- TODO: The flat var list optimization idea from previous implementations

	local struct Trace
	{
		computation: &ComputationType,
		vars: HashMap(IdSeq, Vector(&RandVar)),
		loopcounters: HashMap(IdSeq, int),
		lastVarList: &Vector(&RandVar),
		inactiveVarNames: Vector(&IdSeq),
		logprob: double,
		newlogprob: double,
		oldlogprob: double,
		conditionsSatisfied: bool,
		returnValue: ComputationType.returns[1]
		hasReturnValue: bool
	}

	terra Trace:__construct(comp: &ComputationType)
		self.computation = comp
		self.vars = [HashMap(IdSeq, Vector(&RandVar))].stackAlloc()
		self.loopcounters = [HashMap(IdSeq, int)].stackAlloc()
		self.lastVarList = nil
		self.inactiveVarNames = [Vector(&IdSeq)].stackAlloc()
		self.logprob = 0.0
		self.newlogprob = 0.0
		self.oldlogprob = 0.0
		self.conditionsSatisfied = false
		self.hasReturnValue = false
	end

	terra Trace:__copy(trace: &Trace)
		self:__construct(trace.comp)
		self.logprob = trace.logprob
		self.oldlogprob = trace.oldlogprob
		self.newlogprob = trace.newlogprob
		self.conditionsSatisfied = trace.conditionsSatisfied
		self.hasReturnValue = trace.hasReturnValue
		if self.hasReturnValue then
			self.returnValue = m.copy(trace.returnValue)
		end
		-- Copy vars
		var it = trace.vars:iterator()
		util.foreach(it, [quote
			var k, v = it:keyvalPointer()
			var vlistp = self.vars:getOrCreatePointer(@k)
			for i=0,v.size do
				vlistp:push(v:get(i):deepcopy())
			end
		end])
	end

	terra Trace:__destruct()
		m.destruct(self.vars)
		m.destruct(self.loopcounters)
		m.destruct(self.inactiveVarNames)
		if self.hasReturnValue then
			m.destruct(self.returnValue)
		end
	end

	terra Trace:traceUpdate()
		var prevtrace = globalTrace
		globalTrace = self

		self.logprob = 0.0
		self.newlogprob = 0.0
		self.loopcounters:clear()
		self.conditionsSatisfied = true

		-- Mark all variables as inactive; only those reached by the computation
		-- will become active
		var it = self.vars:iterator()
		util.foreach(it, [quote
			it:valPointer().isActive = false
		end])

		-- Run computation
		self.returnValue = self.computation()
		self.hasReturnValue = true

		-- Clean up
		self.loopcounters:clear()

		-- Clear out any random variables that are no longer reachable
		self.oldlogprob = 0.0
		self.inactiveVarNames:clear()
		it = self.vars:iterator()
		util.foreach(it, [quote
			var vp = it:valPointer()
			if not vp.active then
				self.oldlogprob = self.oldlogprob + vp.logprob
				self.inactiveVarNames:push(it:keyPointer())
			end
		end])
		for i=0,self.inactiveVarNames.size do
			self.vars:remove(@self.inactiveVarNames:get(i))
		end
		self.inactiveVarNames:clear()
	end

	terra Trace:lookupVariable(isstruct: bool)
		-- How many times have we hit this lexical position (lexpos) before?
		-- (Zero if never)
		var lnump, foundlnum = self.loopcounters:getOrCreatePointer(callsiteStack)
		if not foundlnum then @lnump = 0 end
		var lnum = @lnump
		-- We've now hit this lexpos one more time, so we increment
		@lnump = @lnump + 1
		-- Grab all variables corresponding to this lexpos
		-- (getOrCreate means we will get an empty vector instead of nil)
		var vlistp = self.vars:getOrCreatePointer(callsiteStack)
		self.lastVarList = vlistp
		-- Return nil if no variables from this lexpos match the current loop num
		if vlistp.size <= lnum then
			return nil
		end
		-- Aha! We have a variable for this lexpos and loop num!
		-- Now we just need to verify that the structural types match
		var v = vlistp:get(lnum)
		if v.isStructural == isstruct then
			return v
		else
			return nil
		end
	end

	terra Trace:addNewVariable(newvar: &RandVar)
		-- Due to the bookkeeping we did in lookupVariable, the only thing
		-- we have to do here is just push the new variables onto lastVarList
		self.lastVarList:push(newvar)
	end

	terra Trace:factor(num: double)
		self.logprob = self.logprob + num
	end

	terra Trace:condition(cond: bool)
		self.conditionsSatisfied = self.conditionsSatisfied and cond
	end

	m.addConstructors(Trace)
	return Trace

end)





local lookupVariable = macro(function(isstruct)
	return `globalTrace:lookupVariable(isstruct)
end)

local addNewVariable = macro(function(newvar)
	return `globalTrace:addNewVariable(newvar)
end)

local factor = macro(function(num)
	return quote
		if haveGlobalTrace then
			globalTrace:factor(num)
		end
	end
end)

local condition = macro(function(pred)
	return quote
		if haveGlobalTrace then
			globalTrace:condition(pred)
		end
	end
end)


return
{
	pfn = pfn,
	globalTraceIsSet = globalTraceIsSet,
	lookupVariable = lookupVariable,
	addNewVariable = addNewVariable,
	factor = factor,
	condition = condition
}
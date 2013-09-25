local erp = terralib.require("prob.erph")
local RandVar = erp.RandVar
local notImplemented = erp.notImplementedError
local util = terralib.require("util")
local iface = terralib.require("interface")
local Vector = terralib.require("vector")
local HashMap = terralib.require("hashmap")
local templatize = terralib.require("templatize")
local inheritance = terralib.require("inheritance")
local m = terralib.require("mem")
local rand = terralib.require("prob.random")

local C = terralib.includecstring [[
#include <stdio.h>
]]



-- ADDRESS TRANSFORM

local callsiteStack = global(Vector(int))
local terra initGlobals()
	m.init(callsiteStack)
end
initGlobals()

-- For debugging
local terra printCallStack()
	for i=0,callsiteStack.size do
		C.printf("%d,", callsiteStack:get(i))
	end
	C.printf("\n")
end

-- Wrap a Terra function in a macro that, when called, will assign a
--    unique id to that call site.
-- There's some indirection trickery going on here to deal with
--    recursive functions.
local function isValidProbFn(fn)
	-- Prerequisites: fn must be a terra function, and if it is overloaded,
	-- all overloads must have the same number of return types
	assert(terralib.isfunction(fn))
	local s, t = fn:getdefinitions()[1]:peektype()
	local numrets = #t.returns
	for i=2,#fn:getdefinitions() do
		s, t = fn:getdefinitions()[i]:peektype()
		assert(#t.returns == numrets)
	end
end
local nextid = 0
local function pfn(fn)
	local data = { definition = fn }
	local ret = macro(function(...)
		local myid = nextid
		nextid = nextid + 1
		local args = {}
		for i=1,select("#",...) do table.insert(args, (select(i,...))) end
		local argintermediates = {}
		for _,a in ipairs(args) do table.insert(argintermediates, symbol(a:gettype())) end
		-- We fire this function when we know the type of data.definition.
		-- This may require the function to be compiled.
		local function whenTypeKnown()
			data.isCompiling = false
			isValidProbFn(data.definition)
			local s, typ = data.definition:peektype()
			local numrets = #typ.returns
			if numrets == 0 then
				return quote
					var [argintermediates] = [args]
					callsiteStack:push(myid)
					[data.definition]([argintermediates])
					callsiteStack:pop()
				end
			else
				local results = {}
				for i=1,numrets do table.insert(results, symbol()) end
				return quote
					var [argintermediates] = [args]
					callsiteStack:push(myid)
					var [results] = [data.definition]([argintermediates])
					callsiteStack:pop()
				in
					[results]
				end
			end
		end
		-- At this point, we need to get the type of the function being wrapped.
		-- However, it may already be compiling (if it's a recursive function).
		-- In this case, we attempt to peektype. If this fails, then we report
		--    a useful error
		local success, typ = data.definition:peektype()
		if success then
			return whenTypeKnown()
		elseif not data.isCompiling then
			data.isCompiling = true
			data.definition:compile()
			return whenTypeKnown()
		else
			error("Recursive probabilistic functions must have explicitly annotated return types.")
		end
	end)
	ret.data = data
	ret.define = function(self, fn)
		self.data.definition = fn
	end
	return ret
end





-- TRACE MANAGEMENT


-- We have a 'base class' for all traces, which will be extended
--    by i.e. the single trace and the LARJ trace classes.
local struct BaseTrace
{
	logprob: double,
	newlogprob: double,
	oldlogprob: double,
	conditionsSatisfied: bool,
}

terra BaseTrace:__construct()
	self.logprob = 0.0
	self.newlogprob = 0.0
	self.oldlogprob = 0.0
	self.conditionsSatisfied = false
end

terra BaseTrace:__copy(trace: &BaseTrace)
	self.logprob = trace.logprob
	self.newlogprob = trace.newlogprob
	self.oldlogprob = trace.oldlogprob
	self.conditionsSatisfied = trace.conditionsSatisfied
end

terra BaseTrace:deepcopy() : &BaseTrace
	notImplemented("deepcopy", "BaseTrace")
end
inheritance.virtual(BaseTrace, "deepcopy")

terra BaseTrace:traceUpdate() : {}
	notImplemented("traceUpdate", "BaseTrace")
end
inheritance.virtual(BaseTrace, "traceUpdate")

terra BaseTrace:varListPointer() : &Vector(&RandVar)
	notImplemented("varListPointer", "BaseTrace")
end
inheritance.virtual(BaseTrace, "varListPointer")

local terra isSatisfyingFreeVar(v: &RandVar, structs: bool, nonstructs: bool)
	return not v.isDirectlyConditioned and 
		((structs and v.isStructural) or (nonstructs and not v.isStructural))
end
util.inline(isSatisfyingFreeVar)

terra BaseTrace:numFreeVars(structs: bool, nonstructs: bool)
	var vlist = self:varListPointer()
	var counter = 0
	for i=0,vlist.size do
		if isSatisfyingFreeVar(vlist:get(i), structs, nonstructs) then
			counter = counter + 1
		end
	end
	return counter
end

-- Caller assumes ownership of the returned vector
terra BaseTrace:freeVars(structs: bool, nonstructs: bool)
	var fvars = [Vector(&RandVar)].stackAlloc()
	var vlist = self:varListPointer()
	for i=0,vlist.size do
		var v = vlist:get(i)
		if isSatisfyingFreeVar(v, structs, nonstructs) then
			fvars:push(v)
		end
	end
	return fvars
end

m.addConstructors(BaseTrace)






local IdSeq = Vector(int)
-- Functionality shared by single execution traces regardless of their return types
-- This is the type of the trace that gets stored as the global trace during
--    program execution.
local struct GlobalTrace
{
	vars: HashMap(IdSeq, Vector(&RandVar)),
	varlist: Vector(&RandVar),
	loopcounters: HashMap(IdSeq, int),
	lastVarList: &Vector(&RandVar)		
}
inheritance.dynamicExtend(BaseTrace, GlobalTrace)

-- TODO: The flat var list optimization idea from previous implementations

terra GlobalTrace:__construct()
	BaseTrace.__construct(self)
	self.vars = [HashMap(IdSeq, Vector(&RandVar))].stackAlloc()
	self.varlist = [Vector(&RandVar)].stackAlloc()
	self.loopcounters = [HashMap(IdSeq, int)].stackAlloc()
	self.lastVarList = nil	
end

terra GlobalTrace:__copy(trace: &GlobalTrace)
	self:__construct()
	BaseTrace.__copy(self, trace)
	-- Copy vars
	var old2new = [HashMap(&RandVar, &RandVar)].stackAlloc()
	--    First copy the name --> var map
	var it = trace.vars:iterator()
	util.foreach(it, [quote
		var k, v = it:keyvalPointer()
		var vlistp = (self.vars:getOrCreatePointer(@k))
		for i=0,v.size do
			var oldvar = v:get(i)
			var newvar = oldvar:deepcopy()
			old2new:put(oldvar, newvar)
			vlistp:push(newvar)
		end
	end])
	--   Then copy the flat var list
	self.varlist:resize(trace.varlist.size)
	for i=0,trace.varlist.size do
		self.varlist:set(i, @(old2new:getPointer(trace.varlist:get(i))))
	end
	m.destruct(old2new)
end	

terra GlobalTrace:__destruct()
	m.destruct(self.vars)
	m.destruct(self.varlist)
	m.destruct(self.loopcounters)
end

terra GlobalTrace:lookupVariable(isstruct: bool)
	-- printCallStack()
	-- How many times have we hit this lexical position (lexpos) before?
	-- (Zero if never)
	var lnump, foundlnum = self.loopcounters:getOrCreatePointer(callsiteStack)
	if not foundlnum then @lnump = 0 end
	var lnum = @lnump
	-- We've now hit this lexpos one more time, so we increment
	@lnump = @lnump + 1
	-- Grab all variables corresponding to this lexpos
	-- (getOrCreate means we will get an empty vector instead of nil)
	var vlistp = (self.vars:getOrCreatePointer(callsiteStack))
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

terra GlobalTrace:varListPointer() : &Vector(&RandVar)
	return &self.varlist
end
inheritance.virtual(GlobalTrace, "varListPointer")

terra GlobalTrace:factor(num: double)
	self.logprob = self.logprob + num
end
util.inline(GlobalTrace.methods.factor)

terra GlobalTrace:condition(cond: bool)
	self.conditionsSatisfied = self.conditionsSatisfied and cond
end
util.inline(GlobalTrace.methods.condition)

m.addConstructors(GlobalTrace)






-- The singleton global trace 
local globalTrace = global(&GlobalTrace, nil)






-- This is the normal, single trace that most inference uses.
--    It has to specialize on the type of function that it's tracking.
local RandExecTrace = templatize(function(ComputationType)

	-- For the time being at least, we restrict inference to
	-- computations with a single return value
	if #ComputationType.returns ~= 1 then
		error("Can only do inference on computations with a single return value.")
	end

	-- TODO: The flat var list optimization idea from previous implementations

	local struct Trace
	{
		computation: &ComputationType,
		returnValue: ComputationType.returns[1]
	}
	inheritance.dynamicExtend(GlobalTrace, Trace)

	terra Trace:__construct(comp: &ComputationType)
		GlobalTrace.__construct(self)
		self.computation = comp
		-- Initialize the trace with rejection sampling
		while not self.conditionsSatisfied do
			-- Clear out the existing vars
			var it = self.vars:iterator()
			util.foreach(it, [quote
				var vlistp = it:valPointer()
				for i=0,vlistp.size do m.delete(vlistp:get(i)) end
			end])
			self.vars:clear()
			-- Run the program forward
			self:traceUpdate()
		end
	end

	terra Trace:__copy(trace: &Trace)
		GlobalTrace.__copy(self, trace)
		self.computation = trace.computation
		self.returnValue = m.copy(trace.returnValue)
	end

	terra Trace:deepcopy() : &BaseTrace
		var t = m.new(Trace)
		t:__copy(self)
		return t
	end
	inheritance.virtual(Trace, "deepcopy")

	terra Trace:__destruct()
		GlobalTrace.__destruct(self)
		m.destruct(self.returnValue)
	end

	terra Trace:traceUpdate() : {}
		-- C.printf("======================\n")
		-- Assume ownership of the global trace
		var prevtrace = globalTrace
		globalTrace = self

		self.logprob = 0.0
		self.newlogprob = 0.0
		self.loopcounters:clear()
		self.conditionsSatisfied = true

		-- Clear out the flat var list so we can properly refill it
		self.varlist:clear()

		-- Mark all variables as inactive; only those reached by the computation
		-- will become active
		var it = self.vars:iterator()
		util.foreach(it, [quote
			var vlistp = it:valPointer()
			for i=0,vlistp.size do
				vlistp:get(i).isActive = false
			end
		end])

		-- Run computation
		self.returnValue = self.computation()

		-- Clean up
		self.loopcounters:clear()
		self.lastVarList = nil

		-- Clear out any random variables that are no longer reachable
		self.oldlogprob = 0.0
		it = self.vars:iterator()
		util.foreach(it, [quote
			var vlistp = it:valPointer()
			-- For common use cases (e.g. variables created in loops),
			-- iterating from last var to first will make removal more
			-- efficient (it'll just be a pop() in most cases)
			for i=[int](vlistp.size-1),-1,-1 do
				var vp = vlistp:get(i)
				if not vp.isActive then
					self.oldlogprob = self.oldlogprob + vp.logprob
					vlistp:remove(i)
					m.delete(vp)
					i = i + 1
				end
			end
		end])

		-- Reset the global trace data
		globalTrace = prevtrace
	end
	inheritance.virtual(Trace, "traceUpdate")


	m.addConstructors(Trace)
	return Trace

end)





---- Functions exposed to external modules

-- Caller assumes ownership of the returned trace
local newTrace = macro(function(computation)
	-- Assume computation is a function pointer
	local comptype = computation:gettype().type
	local TraceType = RandExecTrace(comptype)
	return `TraceType.heapAlloc(computation)
end)

local function lookupVariableValue(RandVarType, isstruct, iscond, condVal, params)
	return quote
		var retval: RandVarType.ValType
		-- If there's no global trace, just return the value
		if globalTrace == nil then
			retval = [iscond and condVal or (`RandVarType.sample([params]))]
		else
			-- Otherwise, proceed with trace interactions.
			var rv = [&RandVarType](globalTrace:lookupVariable(isstruct))
			if rv ~= nil then
				-- Check for changes that necessitate a logprob update
				[iscond and (`rv:checkForUpdates(condVal, [params])) or (`rv:checkForUpdates([params]))]
			else
				-- Make new variable, add to master list of vars, add to newlogprob
				rv = [iscond and (`RandVarType.heapAlloc(condVal, isstruct, iscond, [params])) or
								 (`RandVarType.heapAlloc(isstruct, iscond, [params]))]
				globalTrace.newlogprob = globalTrace.newlogprob + rv.logprob
				globalTrace.lastVarList:push(rv)
			end
			-- Add to logprob, set active, add to flat list
			rv.isActive = true
			globalTrace.logprob = globalTrace.logprob + rv.logprob
			globalTrace.varlist:push(rv)
			retval = m.copy(rv.value)
		end
	in
		retval
	end
end

local factor = macro(function(num)
	return quote
		if globalTrace ~= nil then
			globalTrace:factor(num)
		end
	end
end)

local condition = macro(function(pred)
	return quote
		if globalTrace ~= nil then
			globalTrace:condition(pred)
		end
	end
end)


return
{
	pfn = pfn,
	newTrace = newTrace,
	BaseTrace = BaseTrace,
	RandExecTrace = RandExecTrace,
	lookupVariableValue = lookupVariableValue,
	factor = factor,
	condition = condition
}





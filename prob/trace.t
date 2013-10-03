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
local spec = terralib.require("prob.specialize")

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
local terra printCallsiteStack()
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
	local s, t = fn:getdefinitions()[1]:peektype()
	local numrets = #t.returns
	for i=2,#fn:getdefinitions() do
		s, t = fn:getdefinitions()[i]:peektype()
		assert(#t.returns == numrets)
	end
end
local nextid = 0
local pfn = spec.specializable(function(...)
	local structureChange = spec.paramFromList("structureChange", ...)
	local doingInference = spec.paramFromList("doingInference", ...)
	return function(fn)
		local data = { definition = fn }
		local ret = macro(function(...)
			local args = {}
			for i=1,select("#",...) do table.insert(args, (select(i,...))) end
			-- If we're compiling a specialization with no structure change, or if we're running
			--    the code outside of an inference engine, then don't do any address tracking
			if not doingInference or not structureChange then
				return `[data.definition]([args])
			end
			-- Every call gets a unique id
			local myid = nextid
			nextid = nextid + 1
			local argintermediates = {}
			for _,a in ipairs(args) do table.insert(argintermediates, symbol(a:gettype())) end
			-- Does the function have an explicitly annotated return type?
			local success, typ = data.definition:peektype()
			-- If not, attempt to compile to determine the type
			if not success then typ = data.definition:gettype(true) end
			-- If this fails (asynchronous gettype returns nil), then we know we must have
			--    a recursive dependency.
			if not typ then
				error("Recursive probabilistic functions must have explicitly annotated return types.")
			else
				-- Generate code!
				isValidProbFn(data.definition)
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
		end)
		-- Provide mechanisms for the function to be defined after it has been declared
		-- This essentially provides a 'fix' operator for defining recursive functions.
		ret.data = data
		ret.define = function(self, fn)
			self.data.definition = fn
		end
		-- Allow this macro to masquerade as a Terra function
		ret.getdefinitions = function(self) return self.data.definition:getdefinitions() end
		ret.gettype = function(self) return self.data.definition:gettype() end
		ret.peektype = function(self) return self.data.definition:peektype() end
		return ret
	end
end)





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
local TraceUpdateFnPtr = {&BaseTrace}->{}
BaseTrace.entries:insert({field="traceUpdateVtable", type=&Vector(TraceUpdateFnPtr)})

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

terra BaseTrace:varListPointer() : &Vector(&RandVar)
	notImplemented("varListPointer", "BaseTrace")
end
inheritance.virtual(BaseTrace, "varListPointer")

terra BaseTrace:__destruct() : {}
	notImplemented("__destruct", "BaseTrace")
end
inheritance.virtual(BaseTrace, "__destruct")

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
	lastVarList: &Vector(&RandVar),
	currVarIndex: uint
}
inheritance.dynamicExtend(BaseTrace, GlobalTrace)

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

terra GlobalTrace:__destruct() : {}
	m.destruct(self.vars)
	m.destruct(self.varlist)
	m.destruct(self.loopcounters)
end
inheritance.virtual(GlobalTrace, "__destruct")

-- Algorithm for looking up the current variable when structure change is
-- possible. Uses the global callsiteStack
terra GlobalTrace:lookupVariableStructural(isstruct: bool)
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

-- Algorithm for looking up the current variable when structure change
-- is not possible. Simply looks up variables in order of creation
terra GlobalTrace:lookupVariableNonStructural()
	var v = self.varlist:get(self.currVarIndex)
	self.currVarIndex = self.currVarIndex + 1
	return v
end

-- Total lp from variables this trace has that other does not
terra GlobalTrace:lpDiff(other: &GlobalTrace)
	var total = 0.0
	var it = self.vars:iterator()
	util.foreach(it, [quote
		var k, v1 = it:keyvalPointer()
		var v2 = other.vars:getPointer(@k)
		var n1 = v1.size
		var n2 = 0
		if v2 ~= nil then n2 = v2.size end
		for i=n2,n1 do
			total = total + v1:get(i).logprob
		end
	end])
	return total
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





-- Specializing traceUpdate
local paramTables = {}
local function traceUpdate(trace, paramTable)
	paramTable = paramTable or {}
	paramTable.doingInference = true
	local id = spec.paramTableID(paramTable)
	local vtableindex = id-1
	paramTables[id] = paramTable
	return quote
		var fnptr : TraceUpdateFnPtr = [trace].traceUpdateVtable:get(vtableindex)
	in
		fnptr([trace])
	end
end
local vmethods = {} -- Hold on to these to prevent GC
local function fillTraceUpdateVtable(trace)
	local Trace = terralib.typeof(trace).type
	for _,pt in ipairs(paramTables) do
		local specfn = Trace.traceUpdate(unpack(spec.paramTableToList(pt)))
		table.insert(vmethods, specfn)
		Vector(TraceUpdateFnPtr).methods.push(Trace.traceUpdateVtable:getpointer(), specfn:getpointer())
	end
	Trace.traceUpdateVtableIsFilled:set(true)
end



-- This is the normal, single trace that most inference uses.
--    It has to specialize on the function that it's tracking.
local RandExecTrace = templatize(function(computation)

	-- Get the type of this computation (requires us to generate the default,
	--  unspecialized version)
	local comp = computation()
	local success, CompType = comp:peektype()
	if not success then CompType = comp:gettype() end

	-- For the time being at least, we restrict inference to
	-- computations with a single return value
	if #CompType.returns ~= 1 then
		error("Can only do inference on computations with a single return value.")
	end

	local struct Trace
	{
		returnValue: CompType.returns[1]
	}
	inheritance.dynamicExtend(GlobalTrace, Trace)

	Trace.traceUpdateVtable = global(Vector(TraceUpdateFnPtr))
	Vector(TraceUpdateFnPtr).methods.__construct(Trace.traceUpdateVtable:getpointer())
	Trace.traceUpdateVtableIsFilled = global(bool, false)

	terra Trace:__construct()
		GlobalTrace.__construct(self)
		-- JIT the traceUpdate functions, if we haven't compiled them yet.
		if not [Trace.traceUpdateVtableIsFilled] then
			fillTraceUpdateVtable(self)	-- Call back into Lua
		end
		-- Allow this instance to refer to the correct set of traceUpdate functions.
		self.traceUpdateVtable = &[Trace.traceUpdateVtable]
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
			[traceUpdate(self)]
		end
	end

	terra Trace:__copy(trace: &Trace)
		GlobalTrace.__copy(self, trace)
		self.traceUpdateVtable = trace.traceUpdateVtable
		self.returnValue = m.copy(trace.returnValue)
	end

	terra Trace:deepcopy() : &BaseTrace
		var t = m.new(Trace)
		t:__copy(self)
		return t
	end
	inheritance.virtual(Trace, "deepcopy")

	terra Trace:__destruct() : {}
		GlobalTrace.__rawdestruct(self)
		m.destruct(self.returnValue)
	end
	inheritance.virtual(Trace, "__destruct")

	-- Generate specialized 'traceUpdate' code
	Trace.traceUpdate = templatize(function(...)
		local structureChange = spec.paramFromList("structureChange",...)
		local speccomp = computation(spec.paramListToTable(...))
		return terra(self: &Trace) : {}
			-- Assume ownership of the global trace
			var prevtrace = globalTrace
			globalTrace = self

			self.logprob = 0.0
			self.newlogprob = 0.0
			self.oldlogprob = 0.0
			self.loopcounters:clear()
			self.conditionsSatisfied = true
			self.currVarIndex = 0

			-- Clear out the flat var list so we can properly refill it
			[structureChange and (`self.varlist:clear()) or (quote end)]

			-- Mark all variables as inactive; only those reached by the computation
			-- will become active
			[structureChange and 
			quote
				var it = self.vars:iterator()
				util.foreach(it, [quote
					var vlistp = it:valPointer()
					for i=0,vlistp.size do
						vlistp:get(i).isActive = false
					end
				end])
			end
				or
			quote end]

			-- Run computation
			self.returnValue = speccomp()

			-- Clean up
			self.loopcounters:clear()
			self.lastVarList = nil

			-- Clear out any random variables that are no longer reachable
			[structureChange and
			quote
				var it = self.vars:iterator()
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
			end
				or
			quote end]

			-- Reset the global trace data
			globalTrace = prevtrace
		end
	end)


	m.addConstructors(Trace)
	return Trace

end)




---- Functions exposed to external modules

-- Caller assumes ownership of the returned trace
local function newTrace(computation)
	local TraceType = RandExecTrace(computation)
	return `TraceType.heapAlloc()
end

local function lookupVariableValueStructural(RandVarType, isstruct, iscond, condVal, params)
	return quote
		var retval: RandVarType.ValType
		-- If there's no global trace, just return the value
		if globalTrace == nil then
			retval = [iscond and condVal or (`RandVarType.sample([params]))]
		else
			-- Otherwise, proceed with trace interactions.
			var rv = [&RandVarType](globalTrace:lookupVariableStructural(isstruct))
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

local function lookupVariableValueNonStructural(RandVarType, isstruct, iscond, condVal, params)
	return quote
		var retval: RandVarType.ValType
		var rv = [&RandVarType](globalTrace:lookupVariableNonStructural())
		-- Check for changes that necessitate a logprob update
		[iscond and (`rv:checkForUpdates(condVal, [params])) or (`rv:checkForUpdates([params]))]
		-- Add to logprob, set active
		rv.isActive = true
		globalTrace.logprob = globalTrace.logprob + rv.logprob
		retval = m.copy(rv.value)
	in
		retval
	end
end

local function lookupVariableValue(RandVarType, isstruct, iscond, condVal, params, specParams)
	local structureChange = spec.paramFromTable("structureChange", specParams)
	if structureChange then
		return lookupVariableValueStructural(RandVarType, isstruct, iscond, condVal, params)
	else
		return lookupVariableValueNonStructural(RandVarType, isstruct, iscond, condVal, params)
	end
end

local factor = spec.specializable(function(...)
	local factorEval = spec.paramFromList("factorEval", ...)
	local doingInference = spec.paramFromList("doingInference", ...)
	-- TODO: Also check "are we in inference?"
	return macro(function(num)
		-- Do not generate any code if we're compiling a specialization
		--    without factor evaluation, or if we're running the code outside of
		--    an inference engine
		if not doingInference or not factorEval then
			return quote end
		end
		return quote
			if globalTrace ~= nil then
				globalTrace:factor(num)
			end
		end
	end)
end)

local condition = spec.specializable(function(...)
	local doingInference = spec.paramFromList("doingInference", ...)
	return macro(function(pred)
		if not doingInference then
			return quote end
		end
		return quote
			if globalTrace ~= nil then
				globalTrace:condition(pred)
			end
		end
	end)
end)


return
{
	pfn = pfn,
	traceUpdate = traceUpdate,
	fillTraceUpdateVtable = fillTraceUpdateVtable,
	newTrace = newTrace,
	BaseTrace = BaseTrace,
	GlobalTrace = GlobalTrace,
	RandExecTrace = RandExecTrace,
	lookupVariableValue = lookupVariableValue,
	globals = {
		pfn = pfn,
		factor = factor,
		condition = condition
	}
}





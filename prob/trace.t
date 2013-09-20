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

-- We have an interface for 'the global trace'
-- This is the interface that global functions such as
--   'factor' and 'condition' see.
local GlobalTraceInterface = iface.create {
	lookupVariable = {bool} -> &RandVar;
	addNewVariable = {&RandVar} -> {};
	factor = {double} -> {};
	condition = {bool} -> {};
}

local globalTrace = global(GlobalTraceInterface)
local haveGlobalTrace = global(bool, false)

local terra globalTraceIsSet()
	return haveGlobalTrace
end
util.inline(globalTraceIsSet)


-- We also have a 'base class' for all traces, which will be extended
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

terra BaseTrace:randomFreeVar(structs: bool, nonstructs: bool)
	var fvars = self:freeVars(structs, nonstructs)
	var randindex = [uint]([rand.uniform_sample(double)](0, fvars.size))
	var chosenvar = fvars:randindex()
	m.destruct(fvars)
	return chosenvar
end

m.addConstructors(BaseTrace)



-- This is the normal, single trace that most inference uses.
--    It has to specialize on the type of function that it's tracking.
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
		varlist: Vector(&RandVar),
		loopcounters: HashMap(IdSeq, int),
		lastVarList: &Vector(&RandVar),
		returnValue: ComputationType.returns[1]
	}
	inheritance.dynamicExtend(BaseTrace, Trace)

	terra Trace:__construct(comp: &ComputationType)
		BaseTrace.__construct(self)
		self.computation = comp
		self.vars = [HashMap(IdSeq, Vector(&RandVar))].stackAlloc()
		self.varlist = [Vector(&RandVar)].stackAlloc()
		self.loopcounters = [HashMap(IdSeq, int)].stackAlloc()
		self.lastVarList = nil
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
		self:__construct(trace.comp)
		BaseTrace.__copy(self, trace)
		self.hasReturnValue = trace.hasReturnValue
		self.returnValue = m.copy(trace.returnValue)
		-- Copy vars
		var old2new = [HashMap(&RandVar, &RandVar)].stackAlloc()
		--    First copy the name --> var map
		var it = trace.vars:iterator()
		util.foreach(it, [quote
			var k, v = it:keyvalPointer()
			var vlistp = self.vars:getOrCreatePointer(@k)
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

	terra Trace:deepcopy() : &BaseTrace
		var t = m.new(Trace)
		t:__copy(self)
		return t
	end
	inheritance.virtual(Trace, "deepcopy")

	terra Trace:varListPointer() : &Vector(&RandVar)
		return &self.varlist
	end
	inheritance.virtual(Trace, "varListPointer")

	terra Trace:__destruct()
		m.destruct(self.vars)
		m.destruct(self.varlist)
		m.destruct(self.loopcounters)
		m.destruct(self.inactiveVarNames)
		m.destruct(self.returnValue)
	end

	terra Trace:traceUpdate() : {}
		-- Assume ownership of the global trace
		var prevtrace = globalTrace
		var prevHasGlobalTrace = haveGlobalTrace
		globalTrace = self
		haveGlobalTrace = true

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

		-- Clear out any random variables that are no longer reachable
		self.oldlogprob = 0.0
		it = self.vars:iterator()
		util.foreach(it, [quote
			var vlistp = it:valPointer()
			-- For common use cases (e.g. variables created in loops),
			-- iterating from last var to first will make removal more
			-- efficient (it'll just be a pop() in most cases)
			for i=vlistp.size-1,-1,-1 do
				var vp = vlistp:get(i)
				if not vp.active then
					self.oldlogprob = sellf.oldlogprob + vp.logprob
					vlistp:remove(i)
					m.delete(vp)
					i = i + 1
				end
			end
		end])

		-- Reset the global trace datat
		globalTrace = prevtrace
		haveGlobalTrace = prevHasGlobalTrace
	end
	inheritance.virtual(Trace, "traceUpdate")

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
		-- (Also push to the flat list so we capture vars in order of creation)
		self.varlist:push(newvar)
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


-- Caller assumes ownership of the returned trace
local newTrace = macro(function(computation)
	-- Assume computation is a function pointer
	local comptype = computation:gettype().type
	local TraceType = RandExecTrace(comptype)
	return `TraceType.heapAlloc(computation)
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
	newTrace = newTrace,
	BaseTrace = BaseTrace,
	globalTraceIsSet = globalTraceIsSet,
	lookupVariable = lookupVariable,
	addNewVariable = addNewVariable,
	factor = factor,
	condition = condition
}





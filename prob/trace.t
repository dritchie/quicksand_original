local erp = terralib.require("prob.erph")
local RandVar = erp.RandVar
local util = terralib.require("util")
local iface = terralib.require("interface")
local Vector = terralib.require("vector")
local HashMap = terralib.require("hashmap")
local templatize = terralib.require("templatize")
local virtualTemplate = terralib.require("vtemplate")
local inheritance = terralib.require("inheritance")
local m = terralib.require("mem")
local rand = terralib.require("prob.random")
local spec = terralib.require("prob.specialize")
local ad = terralib.require("ad")

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
	-- all overloads must have the same return type
	local s, t = fn:getdefinitions()[1]:peektype()
	local rettype = t.returntype
	for i=2,#fn:getdefinitions() do
		s, t = fn:getdefinitions()[i]:peektype()
		assert(t.returntype == rettype)
	end
end
local nextid = 0
local function pfn(fn, opts)
	local structureChange = spec.getRuntimeVar("structureChange")
	local doingInference = spec.getRuntimeVar("doingInference")
	local ismethod = opts and opts.ismethod
	local data = { definition = fn }
	local ret = macro(function(...)
		local args = {...}
		local argtypes = {}
		for _,a in ipairs(args) do table.insert(argtypes, a:gettype()) end
		-- If the function being wrapped is a method, and the first argument is not already
		--    a pointer type, convert it to a pointer (this is the 'self' argument)
		if ismethod and (not argtypes[1]:ispointertostruct()) then
			args[1] = `&[args[1]]
			argtypes[1] = &argtypes[1]
		end
		local argintermediates = {}
		for _,t in ipairs(argtypes) do table.insert(argintermediates, symbol(t)) end
		-- Does the function have an explicitly annotated return type?
		local success, typ = data.definition:peektype()
		-- If not, attempt to compile to determine the type
		if not success then typ = data.definition:gettype(true) end
		-- If this fails (asynchronous gettype returns nil), then we know we must have
		--    a recursive dependency.
		if not typ then
			error("Recursive probabilistic functions must have explicitly annotated return types.")
		else
			-- Every call gets a unique id
			local myid = nextid
			nextid = nextid + 1
			-- Generate code!
			isValidProbFn(data.definition)
			return quote
				var result : typ.returntype
				-- If we're guaranteed no structure change, or if we're running the code outside 
				--    of an inference engine, then don't do any address tracking.
				if not doingInference or not structureChange then
					result = [data.definition]([args])
				-- Otherwise, do stack-based address tracking
				else
					var [argintermediates] = [args]
					callsiteStack:push(myid)
					result = [data.definition]([argintermediates])
					callsiteStack:pop()
				end
			in
				result
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
	ret.gettype = function(self, async) return self.data.definition:gettype(async) end
	ret.peektype = function(self) return self.data.definition:peektype() end
	return ret
end

local function pmethod(fn, opts)
	opts = opts or {}
	opts.ismethod = true
	return pfn(fn, opts)
end

-- pfor is like a for loop wrapped in its own function block.
-- Namely, it pushes/pops the address stack, so that the structure
--    of nested for loops is preserved.
-- Arguments are one of:
--   * indexVar, lower, upper, quoteBlock
--   * indexVar, lower, upper, step, quoteBlock
local function pfor(...)
	local indexVar, lower, upper, step, quoteBlock
	local numArgs = select("#", ...)
	if numArgs == 5 then
		indexVar = (select(1, ...))
		lower = (select(2, ...))
		upper = (select(3, ...))
		step = (select(4, ...))
		quoteBlock = (select(5, ...))
	elseif numArgs == 4 then
		indexVar = (select(1, ...))
		lower = (select(2, ...))
		upper = (select(3, ...))
		step = 1
		quoteBlock = (select(4, ...))
	else
		error(string.format("Unexpected number of arguments to pfor -- got %u, expected 4 or 5", numArgs))
	end
	local myid = nextid
	nextid = nextid	+ 1
	return quote
		callsiteStack:push(myid)
		for i=lower,upper,step do
			indexVar = i
			[quoteBlock]
		end
		callsiteStack:pop()
	end
end



-- TRACE MANAGEMENT


-- We have a 'base class' for all traces, which will be extended
--    by i.e. the single trace and the LARJ trace classes.
local BaseTrace
BaseTrace = templatize(function(ProbType)
	local RVar = RandVar(ProbType)
	local struct BaseTraceT
	{
		logprob: ProbType,
		newlogprob: ProbType,
		oldlogprob: ProbType,
		conditionsSatisfied: bool,
		temperature: double
	}

	terra BaseTraceT:__construct()
		self.logprob = 0.0
		self.newlogprob = 0.0
		self.oldlogprob = 0.0
		self.conditionsSatisfied = false
		self.temperature = 1.0
	end

	BaseTraceT.__templatecopy = templatize(function(P)
		return terra(self: &BaseTraceT, other: &BaseTrace(P))
			self.logprob = other.logprob
			self.newlogprob = other.newlogprob
			self.oldlogprob = other.oldlogprob
			self.conditionsSatisfied = other.conditionsSatisfied
			self.temperature = other.temperature
		end
	end)

	inheritance.purevirtual(BaseTraceT, "__destruct", {}->{})
	inheritance.purevirtual(BaseTraceT, "varListPointer", {}->{&Vector(&RVar)})
	inheritance.purevirtual(BaseTraceT, "deepcopy", {}->{&BaseTraceT})

	BaseTraceT.deepcopy = virtualTemplate(BaseTraceT, "deepcopy", function(P) return {}->{&BaseTrace(P)} end)

	local terra isSatisfyingFreeVar(v: &RVar, structs: bool, nonstructs: bool)
		return not v.isDirectlyConditioned and 
			((structs and v.isStructural) or (nonstructs and not v.isStructural))
	end
	util.inline(isSatisfyingFreeVar)

	terra BaseTraceT:numFreeVars(structs: bool, nonstructs: bool)
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
	terra BaseTraceT:freeVars(structs: bool, nonstructs: bool)
		var fvars = [Vector(&RVar)].stackAlloc()
		var vlist = self:varListPointer()
		for i=0,vlist.size do
			var v = vlist:get(i)
			if isSatisfyingFreeVar(v, structs, nonstructs) then
				fvars:push(v)
			end
		end
		return fvars
	end

	-- A set of utilities for getting/setting the non-structural, continuous variables
	--    in a trace (I got tired of typing out repeated boilerplate)
	terra BaseTraceT:getNonStructuralReals(v: &Vector(ProbType))
		var nonstructs = self:freeVars(false, true)
		for i=0,nonstructs.size do
			nonstructs(i):getRealComponents(v)
		end
		m.destruct(nonstructs)
	end
	terra BaseTraceT:getRawNonStructuralReals(v: &Vector(ProbType))
		var nonstructs = self:freeVars(false, true)
		for i=0,nonstructs.size do
			nonstructs(i):getRawRealComponents(v)
		end
		m.destruct(nonstructs)
	end
	terra BaseTraceT:setNonStructuralReals(v: &Vector(ProbType))
		var nonstructs = self:freeVars(false, true)
		var index = 0U
		for i=0,nonstructs.size do
			nonstructs(i):setRealComponents(v, &index)
		end
		m.destruct(nonstructs)
	end
	terra BaseTraceT:setRawNonStructuralReals(v: &Vector(ProbType))
		var nonstructs = self:freeVars(false, true)
		var index = 0U
		for i=0,nonstructs.size do
			nonstructs(i):setRawRealComponents(v, &index)
		end
		m.destruct(nonstructs)
	end

	BaseTraceT.traceUpdate = virtualTemplate(BaseTraceT, "traceUpdate", function(...) return {}->{} end)

	BaseTraceT.setLogprobFrom = virtualTemplate(BaseTraceT, "setLogprobFrom", function(P) return {&BaseTrace(P)}->{} end)

	m.addConstructors(BaseTraceT)
	return BaseTraceT
end)





local function traceUpdate(paramTable)
	paramTable = paramTable or {}
	paramTable.doingInference = true
	return macro(function(inst)
		local TraceType = inst:gettype().type
		-- ProbType is the first template parameter
		local ProbType = TraceType.__templateParams[1]
		paramTable.scalarType = ProbType
		return `[BaseTrace(ProbType).traceUpdate(unpack(spec.paramTableToList(paramTable)))](inst)
	end)
end





local IdSeq = Vector(int)
-- Functionality shared by single execution traces regardless of their return types
-- This is the type of the trace that gets stored as the global trace during
--    program execution.
local GlobalTrace
GlobalTrace = templatize(function(ProbType)
	local RVar = RandVar(ProbType)
	local ParentClass = BaseTrace(ProbType)
	local struct GlobalTraceT
	{
		vars: HashMap(IdSeq, Vector(&RVar)),
		varlist: Vector(&RVar),
		loopcounters: HashMap(IdSeq, int),
		lastVarList: &Vector(&RVar),
		currVarIndex: uint,
		manifolds: Vector(ProbType)
	}
	inheritance.dynamicExtend(ParentClass, GlobalTraceT)

	terra GlobalTraceT:__construct()
		ParentClass.__construct(self)
		self.vars = [HashMap(IdSeq, Vector(&RVar))].stackAlloc()
		self.varlist = [Vector(&RVar)].stackAlloc()
		self.loopcounters = [HashMap(IdSeq, int)].stackAlloc()
		self.lastVarList = nil
		m.init(self.manifolds)
	end

	GlobalTraceT.__templatecopy = templatize(function(P)
		local RVarP = RandVar(P)
		return terra(self: &GlobalTraceT, other: &GlobalTrace(P))
			self:__construct()
			[ParentClass.__templatecopy(P)](self, other)
			-- Copy vars
			var old2new = [HashMap(&RVarP, &RVar)].stackAlloc()
			--    First copy the name --> var map
			var it = other.vars:iterator()
			[util.foreach(it, quote
				var k, v = it:keyvalPointer()
				var vlistp, didGet = self.vars:getOrCreatePointer(@k)
				for i=0,v.size do
					var oldvar = v:get(i)
					var newvar = [RVarP.deepcopy(ProbType)](oldvar)
					old2new:put(oldvar, newvar)
					vlistp:push(newvar)
				end
			end)]
			--   Then copy the flat var list
			self.varlist:resize(other.varlist.size)
			for i=0,other.varlist.size do
				self.varlist:set(i, @(old2new:getPointer(other.varlist:get(i))))
			end
			m.destruct(old2new)
			-- Copy manifolds (if any)
			m.init(self.manifolds)
			self.manifolds:resize(other.manifolds.size)
			for i=0,other.manifolds.size do
				self.manifolds(i) = other.manifolds(i)
			end
		end
	end)

	terra GlobalTraceT:__destruct() : {}
		m.destruct(self.vars)
		m.destruct(self.varlist)
		m.destruct(self.loopcounters)
		m.destruct(self.manifolds)
	end
	inheritance.virtual(GlobalTraceT, "__destruct")

	-- Algorithm for looking up the current variable when structure change is
	-- possible. Uses the global callsiteStack
	terra GlobalTraceT:lookupVariableStructural(isstruct: bool)
		-- How many times have we hit this lexical position (lexpos) before?
		-- (Zero if never)
		var lnump, foundlnum = self.loopcounters:getOrCreatePointer(callsiteStack)
		if not foundlnum then @lnump = 0 end
		var lnum = @lnump
		-- We've now hit this lexpos one more time, so we increment
		@lnump = @lnump + 1
		-- Grab all variables corresponding to this lexpos
		-- (getOrCreate means we will get an empty vector instead of nil)
		var vlistp, didGet = (self.vars:getOrCreatePointer(callsiteStack))
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
	terra GlobalTraceT:lookupVariableNonStructural()
		var v = self.varlist:get(self.currVarIndex)
		self.currVarIndex = self.currVarIndex + 1
		return v
	end

	-- Total lp from variables this trace has that other does not
	terra GlobalTraceT:lpDiff(other: &GlobalTraceT)
		var total = 0.0
		var it = self.vars:iterator()
		[util.foreach(it, quote
			var k, v1 = it:keyvalPointer()
			var v2 = other.vars:getPointer(@k)
			var n1 = v1.size
			var n2 = 0
			if v2 ~= nil then n2 = v2.size end
			for i=n2,n1 do
				total = total + v1:get(i).logprob
			end
		end)]
		return total
	end

	terra GlobalTraceT:varListPointer() : &Vector(&RVar)
		return &self.varlist
	end
	inheritance.virtual(GlobalTraceT, "varListPointer")

	terra GlobalTraceT:factor(num: ProbType)
		self.logprob = self.logprob + num
		-- C.printf("add factor: %g, lp = %g\n", ad.val(num), ad.val(self.logprob))
	end
	util.inline(GlobalTraceT.methods.factor)

	terra GlobalTraceT:manifold(num: ProbType)
		self.manifolds:push(num)
	end
	util.inline(GlobalTraceT.methods.manifold)

	terra GlobalTraceT:condition(cond: bool)
		self.conditionsSatisfied = self.conditionsSatisfied and cond
	end
	util.inline(GlobalTraceT.methods.condition)

	m.addConstructors(GlobalTraceT)
	return GlobalTraceT
end)






-- The singleton global trace
-- (There's one for every ProbType used by the program)
local globalTrace = templatize(function(ProbType)
	return global(&GlobalTrace(ProbType), nil)
end)







-- This is the normal, single trace that most inference uses.
--    It has to specialize on the function that it's tracking.
local RandExecTrace
RandExecTrace = templatize(function(ProbType, computation)

	local ParentClass = GlobalTrace(ProbType)

	-- We need the return type of this computation, which requires
	-- us to generate a specialization of it
	local comp = computation({scalarType=ProbType})
	local success, CompType = comp:peektype()
	if not success then CompType = comp:gettype() end

	local struct Trace
	{
		returnValue: CompType.returntype,
		hasReturnValue: bool
	}
	inheritance.dynamicExtend(ParentClass, Trace)

	terra Trace:__construct()
		ParentClass.__construct(self)
		self.hasReturnValue = false
		-- Initialize the trace with rejection sampling
		while not self.conditionsSatisfied do
			-- Clear out the existing vars
			var it = self.vars:iterator()
			[util.foreach(it, quote
				var vlistp = it:valPointer()
				for i=0,vlistp.size do m.delete(vlistp:get(i)) end
			end)]
			self.vars:clear()
			-- Run the program forward
			[traceUpdate()](self)
		end
	end

	Trace.__templatecopy = templatize(function(P)
		return terra(self: &Trace, other: &RandExecTrace(P, computation))
			[ParentClass.__templatecopy(P)](self, other)
			self.hasReturnValue = false
		end
	end)

	virtualTemplate(Trace, "deepcopy", function(P) return {}->{&BaseTrace(P)} end, function(P)
		local TraceP = RandExecTrace(P, computation)
		return terra(self: &Trace)
			var t = m.new(TraceP)
			[TraceP.__templatecopy(ProbType)](t, self)
			return t
		end
	end)

	terra Trace:deepcopy() : &BaseTrace(ProbType)
		return [BaseTrace(ProbType).deepcopy(ProbType)](self)
	end
	inheritance.virtual(Trace, "deepcopy")

	terra Trace:__destruct() : {}
		ParentClass.__rawdestruct(self)
		if self.hasReturnValue then m.destruct(self.returnValue) end
	end
	inheritance.virtual(Trace, "__destruct")

	-- Generate specialized 'traceUpdate' code
	virtualTemplate(Trace, "traceUpdate", function(...) return {}->{} end, function(...)
		-- Normally, structureChange is a runtime variable. But it's ok to access it at compile time in this case,
		--   since we are compiling traceUpdate, not the computation itself.
		local structureChange = spec.paramFromList("structureChange", ...)
		local speccomp = computation(spec.paramListToTable(...))
		local globTrace = globalTrace(ProbType)
		assert(ProbType == spec.paramFromList("scalarType", ...)) --Just checking
		return terra(self: &Trace) : {}
			-- Assume ownership of the global trace
			var prevtrace = globTrace
			globTrace = self

			self.logprob = 0.0
			self.newlogprob = 0.0
			self.oldlogprob = 0.0
			self.loopcounters:clear()
			self.conditionsSatisfied = true
			self.currVarIndex = 0
			self.manifolds:clear()

			-- Clear out the flat var list so we can properly refill it
			if structureChange then self.varlist:clear() end

			-- Mark all variables as inactive; only those reached by the computation
			-- will become active
			if structureChange then
				var it = self.vars:iterator()
				[util.foreach(it, quote
					var vlistp = it:valPointer()
					for i=0,vlistp.size do
						vlistp:get(i).isActive = false
					end
				end)]
			end

			-- Run computation
			if self.hasReturnValue then m.destruct(self.returnValue) end
			self.returnValue = speccomp()
			self.hasReturnValue = true

			-- Clean up
			self.loopcounters:clear()
			self.lastVarList = nil

			-- Clear out any random variables that are no longer reachable
			if structureChange then
				var it = self.vars:iterator()
				[util.foreach(it, quote
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
				end)]
			end

			-- Reset the global trace data
			globTrace = prevtrace
		end
	end)

	virtualTemplate(Trace, "setLogprobFrom", function(P) return {&BaseTrace(P)}->{} end,
	function(P)
		local val = macro(function(x)
			if P == ad.num and ProbType ~= P then
				return `x:val()
			else
				return x
			end
		end)
		return terra(self: &Trace, other: &BaseTrace(P))
			self.logprob = val(other.logprob)
		end
	end)


	m.addConstructors(Trace)
	return Trace
end)






-- Caller assumes ownership of the returned trace
local function newTrace(computation, ProbType)
	local ProbType = ProbType or double
	local TraceType = RandExecTrace(ProbType, computation)
	return `TraceType.heapAlloc()
end

local function lookupVariableValueStructural(RandVarType, opstruct, OpstructType, params, specParams)
	local globTrace = globalTrace(spec.paramFromTable("scalarType", specParams))
	local isstruct = erp.opts.getIsStruct(opstruct, OpstructType)
	local hasPrior = erp.opts.getHasPrior(opstruct, OpstructType)
	return quote
		var rv = [&RandVarType](globTrace:lookupVariableStructural(isstruct))
		if rv ~= nil then
			-- Check for changes that necessitate a logprob update
			rv:checkForUpdates(opstruct, [params])
		else
			var depth = callsiteStack.size
			-- Make new variable, add to master list of vars, add to newlogprob
			rv = RandVarType.heapAlloc(depth, opstruct, [params])
			globTrace.newlogprob = globTrace.newlogprob + rv.logprob
			globTrace.lastVarList:push(rv)
		end
		-- Add to logprob, set active, add to flat list
		rv.isActive = true
		if hasPrior then
			globTrace.logprob = globTrace.logprob + rv.logprob
		end
		globTrace.varlist:push(rv)
		var retval = m.copy(rv:getValue())
	in
		retval
	end
end

local function lookupVariableValueNonStructural(RandVarType, opstruct, OpstructType, params, specParams)
	local globTrace = globalTrace(spec.paramFromTable("scalarType", specParams))
	local hasPrior = erp.opts.getHasPrior(opstruct, OpstructType)
	return quote
		var rv = [&RandVarType](globTrace:lookupVariableNonStructural())
		-- Check for changes that necessitate a logprob update
			rv:checkForUpdates(opstruct, [params])
		-- Add to logprob, set active
		rv.isActive = true
		if hasPrior then
			globTrace.logprob = globTrace.logprob + rv.logprob
			-- C.printf("add prior: %g, lp = %g\n", ad.val(rv.logprob), ad.val(globTrace.logprob))
		end
		var retval = m.copy(rv:getValue())
	in
		retval
	end
end

local function lookupVariableValue(RandVarType, opstruct, OpstructType, params, specParams)
	local doingInference = spec.getRuntimeVar("doingInference")
	local structureChange = spec.getRuntimeVar("structureChange")
	return quote
		var result : RandVarType.ValType
		-- If we're not running in an inference engine, then just return the value directly.
		if not doingInference then
			result = [(erp.opts.getCondVal(opstruct, OpstructType) or (`RandVarType.sample([params])))]
		else
			-- Otherwise, the algorithm for variable lookup depends on whether the program control
			--    structure is fixed or variable.
			if structureChange then
				result = [lookupVariableValueStructural(RandVarType, opstruct, OpstructType, params, specParams)]
			else
				result = [lookupVariableValueNonStructural(RandVarType, opstruct, OpstructType, params, specParams)]
			end
		end
	in
		result
	end
end

local factor = spec.specializable(function(...)
	local scalarType = spec.paramFromList("scalarType", ...)
	local factorEval = spec.getRuntimeVar("factorEval")
	local doingInference = spec.getRuntimeVar("doingInference")
	local globTrace = globalTrace(scalarType)
	return macro(function(num)
		return quote
			if doingInference and factorEval then
				globTrace:factor(num)
			end
		end
	end)
end)

local manifold = spec.specializable(function(...)
	local scalarType = spec.paramFromList("scalarType", ...)
	local factorEval = spec.getRuntimeVar("factorEval")
	local doingInference = spec.getRuntimeVar("doingInference")
	local relaxManifolds = spec.getRuntimeVar("relaxManifolds")
	local globTrace = globalTrace(scalarType)
	return macro(function(num, softness)
		return quote
			if doingInference and factorEval then
				if relaxManifolds then
					globTrace:factor([rand.gaussian_logprob(scalarType)](num, 0.0, softness))
				else
					globTrace:manifold(num)
				end
			end
		end
	end)
end)

local condition = spec.specializable(function(...)
	local doingInference = spec.getRuntimeVar("doingInference")
	local globTrace = globalTrace(spec.paramFromList("scalarType", ...))
	return macro(function(pred)
		return quote
			if doingInference then
				globTrace:condition(pred)
			end
		end
	end)
end)


return
{
	pfn = pfn,
	traceUpdate = traceUpdate,
	newTrace = newTrace,
	BaseTrace = BaseTrace,
	GlobalTrace = GlobalTrace,
	RandExecTrace = RandExecTrace,
	lookupVariableValue = lookupVariableValue,
	globals = {
		pfn = pfn,
		pmethod = pmethod,
		pfor = pfor,
		factor = factor,
		manifold = manifold,
		condition = condition
	}
}





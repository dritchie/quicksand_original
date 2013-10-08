
local random = terralib.require("prob.random")
local templatize = terralib.require("templatize")
local virtualTemplate = terralib.require("vtemplate")
local inheritance = terralib.require("inheritance")
local Vector = terralib.require("vector")
local m = terralib.require("mem")
local util = terralib.require("util")
local erph = terralib.require("prob.erph")
local RandVar = erph.RandVar
local typeToID = erph.typeToID
local trace = terralib.require("prob.trace")
local spec = terralib.require("prob.specialize")


-- Every random variable has some value type; this intermediate
-- class manages that
local RandVarWithVal
RandVarWithVal = templatize(function(ProbType, ValType)
	local struct RandVarWithValT
	{
		value: ValType
	}
	local RVar = RandVar(ProbType)
	inheritance.dynamicExtend(RVar, RandVarWithValT)

	terra RandVarWithValT:__construct(val: ValType, isstruct: bool, iscond: bool)
		RVar.__construct(self, isstruct, iscond)
		self.value = m.copy(val)
	end

	RandVarWithValT.__templatecopy = templatize(function(P, V)
		return terra(self: &RandVarWithValT, other: &RandVarWithVal(P, V))
			[RVar.__templatecopy(P)](self, other)
			-- template copy based on ProbType, in case ValType is a type templated
			--    on ProbType (e.g. ValType = Vector(ProbType)).
			-- This defaults to normal m.copy if there is no __templatecopy method.
			self.value = [m.templatecopy(ProbType)](other.value)
		end
	end)

	terra RandVarWithValT:__destruct() : {}
		m.destruct(self.value)
	end
	inheritance.virtual(RandVarWithValT, "__destruct")

	local valTypeID = typeToID(ValType)
	terra RandVarWithValT:valueTypeID() : uint64
		return valTypeID
	end
	inheritance.virtual(RandVarWithValT, "valueTypeID")

	terra RandVarWithValT:pointerToValue() : &opaque
		return [&opaque](&self.value)
	end
	inheritance.virtual(RandVarWithValT, "pointerToValue")

	return RandVarWithValT
end)


-- Finally, at the bottom of the hierarchy, we have random primitives defined by a set of functions
--    * The type of scalar values (doubles or ad.nums)
--    * A sampling function. It may be overloaded, but all overloads must have the same return type
--    * A log probability function
--    * A proposal function
--    * ... Variadic arguments are the types of the parameters to the ERP (essentially specifying which overload
--          of the provided functions we're using)
local RandVarFromFunctions
RandVarFromFunctions = templatize(function(scalarType, sampleTemplate, logprobTemplate, proposeTemplate, ...)
	local paramtypes = {}
	for i=1,select("#", ...) do table.insert(paramtypes, (select(i,...))) end

	local sample = sampleTemplate(scalarType)
	local logprobfn = logprobTemplate(scalarType)
	local propose = proposeTemplate(scalarType)

	-- Can't handle functions with multiple return values
	assert(#sample:getdefinitions()[1]:gettype().returns == 1)
	-- All overloads of the sampling function must have the same return type
	local ValType = sample:getdefinitions()[1]:gettype().returns[1]
	for i=2,#sample:getdefinitions() do assert(sample:getdefinitions()[i]:gettype().returns[1] == ValType) end

	local ProbType = scalarType

	-- Initialize the class we're building
	local struct RandVarFromFunctionsT {}
	RandVarFromFunctionsT.ValType = ValType
	local ParentClass = RandVarWithVal(ProbType, ValType)
	inheritance.dynamicExtend(ParentClass, RandVarFromFunctionsT)

	-- Add one field for each parameter
	local paramFieldNames = {}
	for i,t in ipairs(paramtypes) do
		local n = string.format("param%d", i-1)
		table.insert(paramFieldNames, n)
		RandVarFromFunctionsT.entries:insert({ field = n, type = t})
	end

	local function genParamFieldsExpList(self)
		local exps = {}
		for i,n in ipairs(paramFieldNames) do
			table.insert(exps, `self.[n])
		end
		return exps
	end
	local function wrapExpListWithCopies(explist)
		local ret = {}
		for _,exp in ipairs(explist) do
			table.insert(ret, `m.copy([exp]))
		end
		return ret
	end
	local function wrapExpListWithTemplateCopies(explist, ...)
		local ret = {}
		for _,exp in ipairs(explist) do
			table.insert(ret, `[m.templatecopy(...)]([exp]))
		end
		return ret
	end

	---- Constructors take extra parameters
	-- Ctor 1: Take in a value argument 
	local paramsyms = {}
	for i,t in ipairs(paramtypes) do table.insert(paramsyms, symbol(t)) end
	terra RandVarFromFunctionsT:__construct(val: ValType, isstruct: bool, iscond: bool, [paramsyms])
		ParentClass.__construct(self, val, isstruct, iscond)
		[genParamFieldsExpList(self)] = [wrapExpListWithCopies(paramsyms)]
		self:updateLogprob()
	end
	-- Ctor 2: No value argument, sample one instead
	paramsyms = {}
	for i,t in ipairs(paramtypes) do table.insert(paramsyms, symbol(t)) end
	terra RandVarFromFunctionsT:__construct(isstruct: bool, iscond: bool, [paramsyms])
		var val = sample([paramsyms])
		ParentClass.__construct(self, val, isstruct, iscond)
		[genParamFieldsExpList(self)] = [wrapExpListWithCopies(paramsyms)]
		self:updateLogprob()
	end

	-- Exposing the sample function
	paramsyms = {}
	for i,t in ipairs(paramtypes) do table.insert(paramsyms, symbol(t)) end
	RandVarFromFunctionsT.methods.sample = terra([paramsyms])
		return sample([paramsyms])
	end

	-- Copy constructor
	-- Variadic args are paramtypes
	RandVarFromFunctionsT.__templatecopy = templatize(function(P, ...)
		local RVFFP = RandVarFromFunctions(P, sampleTemplate, logprobTemplate, proposeTemplate, ...)
		local V = RVFFP.ValType
		return terra(self: &RandVarFromFunctionsT, other: &RVFFP)
			[ParentClass.__templatecopy(P, V)](self, other)
			[genParamFieldsExpList(self)] = [wrapExpListWithTemplateCopies(genParamFieldsExpList(other), ProbType)]
		end
	end)

	-- Destructor should clean up any parameters
	local function genDestructBlock(self)
		local statements = {}
		for i,n in ipairs(paramFieldNames) do
			table.insert(statements, `m.destruct(self.[n]))
		end
		return statements
	end
	terra RandVarFromFunctionsT:__destruct() : {}
		ParentClass.__rawdestruct(self)
		[genDestructBlock(self)]
	end
	inheritance.virtual(RandVarFromFunctionsT, "__destruct")

	-- Check if we need to update log probabilities do to changes in:
	--    1) Parameters
	--    2) Conditioned value
	local function checkParams(self, hasChanges)
		local checkexps = {}
		for i,p in ipairs(paramsyms) do
			local n = paramFieldNames[i]
			table.insert(checkexps,
				quote
					if not (self.[n] == p) then
						m.destruct(self.[n])
						self.[n] = m.copy(p)
						hasChanges = true
					end
				end)
		end
		return checkexps
	end
	paramsyms = {}
	for i,t in ipairs(paramtypes) do table.insert(paramsyms, symbol(t)) end
	terra RandVarFromFunctionsT:checkForUpdates([paramsyms])
		var hasChanges = false
		[checkParams(self, hasChanges)]
		if hasChanges then
			self:updateLogprob()
		end
	end
	paramsyms = {}
	for i,t in ipairs(paramtypes) do table.insert(paramsyms, symbol(t)) end
	terra RandVarFromFunctionsT:checkForUpdates(val: ValType, [paramsyms])
		var hasChanges = false
		[checkParams(self, hasChanges)]
		if not (self.value == val) then
			m.destruct(self.value)
			self.value = m.copy(val)
			hasChanges = true
		end
		if hasChanges then
			self:updateLogprob()
		end
	end

	-- Update log probability
	terra RandVarFromFunctionsT:updateLogprob() : {}
		self.logprob = logprobfn(self.value, [genParamFieldsExpList(self)])
	end

	-- Propose new value
	terra RandVarFromFunctionsT:proposeNewValue() : {ProbType, ProbType}
		var newval, fwdPropLP, rvsPropLP = propose(self.value, [genParamFieldsExpList(self)])
		m.destruct(self.value)
		self.value = newval
		self:updateLogprob()
		return fwdPropLP, rvsPropLP
	end
	inheritance.virtual(RandVarFromFunctionsT, "proposeNewValue")

	-- Set the value directly
	terra RandVarFromFunctionsT:setValue(valptr: &opaque) : {}
		m.destruct(self.value)
		self.value = m.copy(@([&ValType](valptr)))
		self:updateLogprob()
	end
	inheritance.virtual(RandVarFromFunctionsT, "setValue")

	m.addConstructors(RandVarFromFunctionsT)
	return RandVarFromFunctionsT
end)


-- OK, I lied: this isn't quite the bottom of the hierarchy. We actually need to have a unique
--    type for every ERP callsite, which requires a minor extension of the above class.
-- This does not use the normal templating mechanism, since creation and retrieval of classes
--    need to happen through different interfaces.
-- As above, variadic args are the parameter types for the ERP.
local randVarFromCallsiteCache = {}
local function getRandVarFromCallsite(scalarType, computation, callsiteID)
	local key = util.stringify(scalarType, computation, callsiteID)
	return randVarFromCallsiteCache[key]
end
-- This is the public interface to getting the ERP type
-- External code can treat this as if it were a normal template.
local function RandVarFromCallsite(scalarType, computation, callsiteID)
	-- Invoke the computation template to guarantee that the ERP specialization will exist.
	local paramTable = {scalarType=scalarType, doingInference=true}
	computation(paramTable)	-- Throw away the return value; we only care about the side effects.
	local class = getRandVarFromCallsite(scalarType, computation, callsiteID)
	-- It had better exist after we specialized computation!
	assert(class)
	return class
end
local function createRandVarFromCallsite(scalarType, sample, logprobfn, propose, computation, ...)
	local id = erph.getCurrentERPID()

	-- Check if we've already built this specialization
	local class = getRandVarFromCallsite(scalarType, computation, id)
	if class then return class end

	-- Otherwise, we need to build the class
	local struct RandVarFromCallsiteT {}
	RandVarFromCallsiteT.paramTypes = {...}
	local ParentClass = RandVarFromFunctions(scalarType, sample, logprobfn, propose, ...)
	inheritance.dynamicExtend(ParentClass, RandVarFromCallsiteT)

	-- Need a new constructor that initializes the deepcopy vtable
	local ctor = ParentClass.methods.__construct
	local newctor = nil
	for _,d in ipairs(ctor:getdefinitions()) do
		local syms = {symbol(&RandVarFromCallsiteT)}
		local paramtypes = d:gettype().parameters
		for i=2,#paramtypes do table.insert(syms, symbol(paramtypes[i])) end
		local self = syms[1]
		local def = terra([syms])
			ctor([syms])
			self:init_deepcopyVtable()
		end
		if not newctor then newctor = def else newctor:adddefinition(def:getdefinitions()[1]) end
	end
	RandVarFromCallsiteT.methods.__construct = newctor

	-- Also need a new __templatecopy for this reason.
	RandVarFromCallsiteT.__templatecopy = templatize(function(P)
		local RandVarFromCallsiteP = RandVarFromCallsite(P, computation, id)
		return terra(self: &RandVarFromCallsiteT, other: &RandVarFromCallsiteP)
			[ParentClass.__templatecopy(P, unpack(RandVarFromCallsiteP.paramTypes))](self, other)
			self:init_deepcopyVtable()
		end
	end)

	-- The only real extra functionality provided by this subclass is deepcopy.
	-- We need to know exactly which ERP type to copy into, which requires knowledge of
	--   parameter types, which may vary from callsite to callsite.
	virtualTemplate(RandVarFromCallsiteT, "deepcopy", function(P) return {}->{&RandVar(P)} end, function(P)
		local RandVarFromCallsiteP = RandVarFromCallsite(P, computation, id)
		return terra(self: &RandVarFromCallsiteT)
			var newvar = m.new(RandVarFromCallsiteP)
			[RandVarFromCallsiteP.__templatecopy(scalarType)](newvar, self)
			return newvar
		end
	end)

	-- Finish up
	m.addConstructors(RandVarFromCallsiteT)
	local key = util.stringify(scalarType, computation, id)
	randVarFromCallsiteCache[key] = RandVarFromCallsiteT
	return RandVarFromCallsiteT
end


-- Make a new random primitive
-- This returns a Lua function which performs sampling (the public interface to the 
--   random primitive)
-- The function expects all the parameters expected by 'sample', plus an (optional) struct
--   which carries info such as 'structural', 'constrainTo', etc.
-- NOTE: Any and all parameter/value types must define the __eq operator!
local function makeERP(sample, logprobfn, propose)

	local numparams = #sample(double):gettype().parameters

	-- If we don't have propose function, make a default.
	if not propose then
		propose = templatize(function(V)
			return macro(function(currval, ...)
				local params = {}
				for i=1,select("#",...) do table.insert(params, (select(i,...))) end
				-- Default: sample and score a new value irrespective of the current value
				return quote
					var newval = [sample(V)]([params])
					var fwdPropLP = [logprobfn(V)](newval, [params])
					var rvsPropLP = [logprobfn(V)](currval, [params])
				in
					newval, fwdPropLP, rvsPropLP
				end
			end)
		end)
	end

	-- Generate an overloaded function which does the ERP call
	-- Memoize results for different specializations
	local genErpFunction = spec.specializable(function(...)
		local specParams = spec.paramListToTable(...)
		local V = spec.paramFromTable("scalarType", specParams)
		local computation = spec.paramFromTable("computation", specParams)
		local erpfn = nil
		for _,d in ipairs(sample(V):getdefinitions()) do
			local paramtypes = d:gettype().parameters
			local rettype = d:gettype().returns[1]
			local RandVarType = createRandVarFromCallsite(V, sample, logprobfn, propose, computation, unpack(paramtypes))
			local params = {}
			for _,t in ipairs(paramtypes) do table.insert(params, symbol(t)) end
			-- First the conditioned version
			local def = terra(isstruct: bool, condVal: rettype, [params])
				return [trace.lookupVariableValue(RandVarType, isstruct, true, condVal, params, specParams)]
			end
			if not erpfn then erpfn = def else erpfn:adddefinition(def:getdefinitions()[1]) end
			-- Then the unconditioned version
			def = terra(isstruct: bool, [params])
				return [trace.lookupVariableValue(RandVarType, isstruct, false, nil, params, specParams)]
			end
			erpfn:adddefinition(def:getdefinitions()[1])
		end
		-- The ERP must push an ID to the callsite stack.
		erpfn = trace.pfn(specParams)(erpfn)
		return erpfn
	end)

	local function getisstruct(opstruct)
		if opstruct then
			local t = opstruct:gettype()
			for _,e in ipairs(t.entries) do
				if e.field == "structural"
					then return `opstruct.structural
				end
			end
		end
		-- Defaulting to 'structural=true' is more sensible.
		return true
	end

	local function getcondval(opstruct)
		if opstruct then
			local t = opstruct:gettype()
			for _,e in ipairs(t.entries) do
				if e.field == "constrainTo"
					then return `opstruct.constrainTo
				end
			end
		end
		return nil
	end

	-- Finally, wrap everything in a function that extracts options from the
	-- options table.
	return spec.specializable(function(...)
		local paramTable = spec.paramListToTable(...)
		return macro(function(...)
			local params = {}
			for i=1,numparams do table.insert(params, (select(i,...))) end
			local opstruct = (select(numparams+1, ...))
			local isstruct = getisstruct(opstruct)
			local condval = getcondval(opstruct)
			local erpfn = genErpFunction(paramTable)
			if condval then
				return `erpfn(isstruct, condval, [params])
			else
				return `erpfn(isstruct, [params])
			end
		end)
	end)
end


-- Define some commonly-used ERPs

local erp = {makeERP = makeERP}

erp.flip =
makeERP(random.flip_sample,
		random.flip_logprob,
		erph.overloadOnParams(1, function(V, P)
			return terra(currval: bool, p: P)
				if currval then
				return false, P(0.0), P(0.0)
				else
					return true, P(0.0), P(0.0)
				end
			end
		end))

erp.uniform =
makeERP(random.uniform_sample,
		random.uniform_logprob)

erp.multinomial =
makeERP(random.multinomial_sample,
	    random.multinomial_logprob,
	    erph.overloadOnParams(1, function(V, P)
	    	return terra(currval: int, params: Vector(P))
	    		var newparams = m.copy(params)
		    	newparams:set(currval, 0.0)
		    	var newval = [random.multinomial_sample(V)](newparams)
		    	var fwdPropLP = [random.multinomial_logprob(V)](newval, newparams)
		    	m.destruct(newparams)
		    	newparams = m.copy(params)
		    	newparams:set(newval, 0.0)
		    	var rvsPropLP = [random.multinomial_logprob(V)](currval, newparams)
		    	m.destruct(newparams)
		    	return newval, fwdPropLP, rvsPropLP
	    	end
    	end))

erp.multinomialDraw = spec.specializable(function(...)
	local paramTable = spec.paramListToTable(...)
	return macro(function(items, probs, opstruct)
		opstruct = opstruct or `{}
		return `items:get([erp.multinomial(paramTable)](probs, opstruct))
	end)
end)

erp.uniformDraw = spec.specializable(function(...)
	local paramTable = spec.paramListToTable(...)
	return macro(function(items, opstruct)
		opstruct = opstruct or `{}
		return quote
			var probs = [Vector(double)].stackAlloc(items.size, 1.0/items.size)
			var result = items:get([erp.multinomial(paramTable)](probs, opstruct))
			m.destruct(probs)
		in
			result
		end
	end)
end)

erp.gaussian =
makeERP(random.gaussian_sample,
		random.gaussian_logprob,
		erph.overloadOnParams(2, function(V, P1, P2)
			return terra(currval: V, mu: P1, sigma: P2)
				var newval = [random.gaussian_sample(V)](currval, sigma)
				var fwdPropLP = [random.gaussian_logprob(V)](newval, currval, sigma)
				var rvsPropLP = [random.gaussian_logprob(V)](currval, newval, sigma)
				return newval, fwdPropLP, rvsPropLP
			end
		end))

erp.gamma =
makeERP(random.gamma_sample,
		random.gamma_logprob)

erp.beta = 
makeERP(random.beta_sample,
		random.beta_logprob)

erp.binomial = 
makeERP(random.binomial_sample,
		random.binomial_logprob)

erp.poisson = 
makeERP(random.poisson_sample,
		random.poisson_logprob)

erp.dirichlet =
makeERP(random.dirichlet_sample,
		random.dirichlet_logprob)




return
{
	globals = erp
}









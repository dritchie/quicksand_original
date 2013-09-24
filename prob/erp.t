
local random = terralib.require("prob.random")
local templatize = terralib.require("templatize")
local inheritance = terralib.require("inheritance")
local Vector = terralib.require("vector")
local m = terralib.require("mem")

local erph = terralib.require("prob.erph")
local RandVar = erph.RandVar
local notImplementedError = erph.notImplementedError
local typeToID = erph.typeToID

local trace = terralib.require("prob.trace")


-- Every random variable has some value type; this intermediate
-- class manages that
local RandVarWithVal = templatize(function(ValType)
	local struct RandVarWithValT
	{
		value: ValType
	}
	inheritance.dynamicExtend(RandVar, RandVarWithValT)

	terra RandVarWithValT:__construct(val: ValType, isstruct: bool, iscond: bool)
		RandVar.__construct(self, isstruct, iscond)
		self.value = m.copy(val)
	end

	terra RandVarWithValT:__copy(othervar: &RandVarWithValT)
		RandVar.__copy(self, othervar)
		self.value = m.copy(othervar.value)
	end

	terra RandVarWithValT:__destruct()
		m.destruct(self.value)
	end

	local ValTypeID = typeToID(ValType)
	terra RandVarWithValT:valueTypeID() : uint64
		return ValTypeID
	end
	inheritance.virtual(RandVarWithValT, "valueTypeID")

	terra RandVarWithValT:pointerToValue() : &opaque
		return [&opaque](&self.value)
	end
	inheritance.virtual(RandVarWithValT, "pointerToValue")

	return RandVarWithValT
end)


-- Finally, at the bottom of the hierarchy, we have random primitives defined by a set of functions
--    * A sampling function. It may be overloaded, but all overloads must have the same return type
--    * A log probability function
--    * A proposal function
--    * ... Variadic arguments are the types of the parameters to the ERP (essentially specifying which overload
--          of the provided functions we're using)
local RandVarFromFunctions = templatize(function(sample, logprobfn, propose, ...)
	local paramtypes = {}
	for i=1,select("#", ...) do table.insert(paramtypes, (select(i,...))) end

	-- Can't handle functions with multiple return values
	assert(#sample:getdefinitions()[1]:gettype().returns == 1)
	-- All overloads of the sampling function must have the same return type
	local ValType = sample:getdefinitions()[1]:gettype().returns[1]
	for i=2,#sample:getdefinitions() do assert(sample:getdefinitions()[i]:gettype().returns[1] == ValType) end

	-- Initialize the class we're building
	local struct RandVarFromFunctionsT {}
	RandVarFromFunctionsT.ValType = ValType
	inheritance.dynamicExtend(RandVarWithVal(ValType), RandVarFromFunctionsT)

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

	---- Constructors take extra parameters
	-- Ctor 1: Take in a value argument 
	local paramsyms = {}
	for i,t in ipairs(paramtypes) do table.insert(paramsyms, symbol(t)) end
	terra RandVarFromFunctionsT:__construct(val: ValType, isstruct: bool, iscond: bool, [paramsyms])
		[RandVarWithVal(ValType)].__construct(self, val, isstruct, iscond)
		[genParamFieldsExpList(self)] = [paramsyms]
		self:updateLogprob()
	end
	-- Ctor 2: No value argument, sample one instead
	paramsyms = {}
	for i,t in ipairs(paramtypes) do table.insert(paramsyms, symbol(t)) end
	terra RandVarFromFunctionsT:__construct(isstruct: bool, iscond: bool, [paramsyms])
		var val = sample([paramsyms])
		[RandVarWithVal(ValType)].__construct(self, val, isstruct, iscond)
		[genParamFieldsExpList(self)] = [paramsyms]
		self:updateLogprob()
	end

	-- Exposing the sample function
	paramsyms = {}
	for i,t in ipairs(paramtypes) do table.insert(paramsyms, symbol(t)) end
	RandVarFromFunctionsT.methods.sample = terra([paramsyms])
		return sample([paramsyms])
	end

	-- Copy constructor...
	local function wrapExpListWithCopies(explist)
		local ret = {}
		for _,exp in ipairs(explist) do
			table.insert(ret, `m.copy([exp]))
		end
		return ret
	end
	terra RandVarFromFunctionsT:__copy(othervar: &RandVarFromFunctionsT)
		[RandVarWithVal(ValType)].__copy(self, othervar)
		[genParamFieldsExpList(self)] = [wrapExpListWithCopies(genParamFieldsExpList(othervar))]
	end
	-- ...and implementing the "deepcopy" virtual method
	terra RandVarFromFunctionsT:deepcopy() : &RandVar
		var newvar = m.new(RandVarFromFunctionsT)
		newvar:__copy(self)
		return newvar
	end
	inheritance.virtual(RandVarFromFunctionsT, "deepcopy")

	-- Destructor should clean up any parameters
	local function genDestructBlock(self)
		local statements = {}
		for i,n in ipairs(paramFieldNames) do
			table.insert(statements, `m.destruct(self.[n]))
		end
		return statements
	end
	terra RandVarFromFunctionsT:__destruct()
		[RandVarWithVal(ValType)].__destruct(self)
		[genDestructBlock(self)]
	end

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
	terra RandVarFromFunctionsT:proposeNewValue() : {double, double}
		var newval, fwdPropLP, rvsPropLP = propose(self.value, [genParamFieldsExpList(self)])
		m.destruct(self.value)
		self.value = newval
		self:updateLogprob()
		return fwdPropLP, rvsPropLP
	end
	inheritance.virtual(RandVarFromFunctionsT, "proposeNewValue")

	m.addConstructors(RandVarFromFunctionsT)
	return RandVarFromFunctionsT
end)


-- Make a new random primitive
-- This returns a macro which performs sampling (the public interface to the 
--   random primitive)
-- The macro expects all the parameters expected by 'sample', plus an (optional) anonymous struct
--   which carries info such as 'isStructural', 'conditionedValue', etc.
-- NOTE: Any and all parameter/value types must define the __eq operator!
local function makeERP(sample, logprobfn, propose)

	local numparams = #sample:getdefinitions()[1]:gettype().parameters

	-- If we don't have propose function, make a default.
	if not propose then
		-- Default: sample and score a new value irrespective of the current value (argument 1)
		for _,d in ipairs(sample:getdefinitions()) do
			local valtype = d:gettype().returns[1]
			local params = {}
			for _,t in ipairs(d:gettype().parameters) do
				table.insert(params, symbol(t))
			end
			local fn = terra(currval: valtype, [params])
				var newval = sample([params])
				var fwdPropLP = logprobfn(newval, [params])
				var rvsPropLP = logprobfn(currval, [params])
				return newval, fwdPropLP, rvsPropLP
			end
			if not propose then
				propose = fn
			else
				fn:adddefinition(fn:getdefinitions()[1])
			end
		end
	end

	-- We now default to true, because that seems more sensible
	-- (i.e. it is always correct to call a variable structural, just
	-- possibly less efficient)
	local function getIsStructural(opstruct)
		if opstruct then
			local ostype = opstruct:gettype()
			for _,e in ipairs(ostype:getentries()) do
				if e.field == "isStructural" then
					return `opstruct.isStructural
				end
			end
		end
		return true
	end

	local function getIsConditioned(opstruct)
		if opstruct then
			local ostype = opstruct:gettype()
			for _,e in ipairs(ostype:getentries()) do
				if e.field == "conditionedValue" then
					return true
				end
			end
		end
		return false
	end

	-- Generate an overloaded function which does the ERP call
	local erpfn = nil
	for _,d in ipairs(sample:getdefinitions()) do
		local paramtypes = d:gettype().parameters
		local rettype = d:gettype().returns[1]
		local RandVarType = RandVarFromFunctions(sample, logprobfn, propose, unpack(paramtypes))
		local params = {}
		for _,t in ipairs(paramtypes) do table.insert(params, symbol(t)) end
		-- First the conditioned version
		local def = terra(isstruct: bool, condVal: rettype, [params])
			return [trace.lookupVariableValue(RandVarType, isstruct, true, condVal, params)]
		end
		if not erpfn then erpfn = def else erpfn:adddefinition(def:getdefinitions()[1]) end
		-- Then the unconditioned version
		def = terra(isstruct: bool, [params])
			return [trace.lookupVariableValue(RandVarType, isstruct, false, nil, params)]
		end
		erpfn:adddefinition(def:getdefinitions()[1])
	end
	-- The ERP must push an ID to the callsite stack.
	erpfn = trace.pfn(erpfn)

	-- Finally, wrap everything in a macro that extracts options from the
	-- options struct.
	return macro(function(...)
		local params = {}
		for i=1,numparams do table.insert(params, (select(i,...))) end
		local optstruct = (select(numparams+1, ...))
		local iscond = getIsConditioned(optstruct)
		local isstruct = getIsStructural(optstruct)
		local condVal = iscond and `optstruct.conditionedValue or nil
		if condVal then
			return `erpfn(isstruct, condVal, [params])
		else
			return `erpfn(isstruct, [params])
		end
	end)
end


-- Define some commonly-used ERPs

local erp = {makeERP = makeERP}

erp.flip =
makeERP(random.flip_sample(double),
		random.flip_logprob(double),
		terra(currval: double, p: double)
			if currval == 0 then
				return 1, 0.0, 0.0
			else
				return 0, 0.0, 0.0
			end
		end)

erp.uniform =
makeERP(random.uniform_sample(double),
		random.uniform_logprob(double))

erp.multinomial =
makeERP(random.multinomial_sample(double),
	    random.multinomial_logprob(double),
	    terra(currval: int, params: Vector(double))
	    	var newparams = m.copy(params)
	    	newparams:set(currval, 0.0)
	    	var newval = [random.multinomial_sample(double)](newparams)
	    	var fwdPropLP = [random.multinomial_logprob(double)](newval, newparams)
	    	m.destruct(newparams)
	    	newparams = m.copy(params)
	    	newparams:set(newval, 0.0)
	    	var rvsPropLP = [random.multinomial_logprob(double)](currval, newparams)
	    	m.destruct(newparams)
	    	return newval, fwdPropLP, rvsPropLP
	    end)

erp.multinomialDraw = macro(function(items, probs, opstruct)
	opstruct = opstruct or `{}
	return `items:get(erp.multinomial(probs, opstruct))
end)

erp.uniformDraw = macro(function(items, opstruct)
	opstruct = opstruct or `{}
	return quote
		var probs = [Vector(double)].stackAlloc(items.length, 1.0/items.length)
	in
		items:get(erp.multinomial(probs, opstruct))
	end
end)

erp.gaussian =
makeERP(random.gaussian_sample(double),
		random.gaussian_logprob(double),
		terra(currval: double, mu: double, sigma: double)
			var newval = [random.gaussian_sample(double)](currval, sigma)
			var fwdPropLP = [random.gaussian_logprob(double)](newval, currval, sigma)
			var rvsPropLP = [random.gaussian_logprob(double)](currval, newval, sigma)
			return newval, fwdPropLP, rvsPropLP
		end)

erp.gamma =
makeERP(random.gamma_sample(double),
		random.gamma_logprob(double))

erp.beta = 
makeERP(random.beta_sample(double),
		random.beta_logprob(double))

erp.binomial = 
makeERP(random.binomial_sample(double),
		random.binomial_logprob(double))

erp.poisson = 
makeERP(random.poisson_sample(double),
		random.poisson_logprob(double))

erp.dirichlet =
makeERP(random.dirichlet_sample(double),
		random.dirichlet_logprob(double))




return erp










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
local ad = terralib.require("ad")


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

	terra RandVarWithValT:__construct(val: ValType, isstruct: bool, iscond: bool, depth: uint, mass: double)
		RVar.__construct(self, isstruct, iscond, depth, mass)
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


	-- By default, we understand how to get/set the real components of double, ad.num,
	--    Vector(double), and Vector(ad.num). Any other value types must define the
	--    'getRealComponents' and 'setRealComponents' methods--otherwise, they'll be
	--    treated as having no real components.
	RandVarWithValT.HasRealComponents = (ValType == double or ValType == ad.num or
							   ValType == Vector(double) or ValType == Vector(ad.num) or
							   ValType:getmethod("setRealComponents"))
	local function genGetReals(self, comps)
		if ValType == double or ValType == ad.num then
			return quote
				comps:push(self.value)
			end
		elseif ValType == Vector(double) or ValType == Vector(ad.num) then
			return quote
				for i=0,self.value.size do
					comps:push(self.value:get(i))
				end
			end
		elseif ValType:getmethod("getRealComponents") then
			return quote
				self.value:getRealComponents(comps)
			end
		end
	end
	local function genSetReals(self, comps, index)
		if ValType == double or ValType == ad.num then
			return quote
				self.value = comps:get(@index)
				@index = @index + 1
			end
		elseif ValType == Vector(double) or ValType == Vector(ad.num) then
			return quote
				for i=0,self.value.size do
					self.value:set(i, comps:get(@index+i))
				end
				@index = @index + self.value.size
			end
		elseif ValType:getmethod("setRealComponents") then
			return quote
				self.value:setRealComponents(comps, index)
			end
		end
	end

	terra RandVarWithValT:getRealComponents(comps: &Vector(ProbType)) : {}
		[genGetReals(self, comps)]
	end
	inheritance.virtual(RandVarWithValT, "getRealComponents")

	terra RandVarWithValT:setReals(comps: &Vector(ProbType), index: &uint)
		[genSetReals(self, comps, index)]
	end
	terra RandVarWithValT:setRealComponents(comps: &Vector(ProbType), index: &uint) : {}
		self:setReals(comps, index)
	end
	inheritance.virtual(RandVarWithValT, "setRealComponents")

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
	local paramtypes = {...}

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
	local function genParamCopyBlock(self, other, otherparamtypes)
		local selfparamexprs = genParamFieldsExpList(self)
		local otherparamexprs = genParamFieldsExpList(other)
		local lines = {}
		for i=1,#paramtypes do
			if paramtypes[i] == otherparamtypes[i] then
				table.insert(lines, quote [selfparamexprs[i]] = m.copy([otherparamexprs[i]]) end)
			else
				table.insert(lines, quote [selfparamexprs[i]] = [m.templatecopy(ProbType)]([otherparamexprs[i]]) end)
			end
		end
		return lines
	end

	---- Constructors take extra parameters
	-- Ctor 1: Take in a value argument 
	local paramsyms = {}
	for i,t in ipairs(paramtypes) do table.insert(paramsyms, symbol(t)) end
	terra RandVarFromFunctionsT:__construct(val: ValType, isstruct: bool, iscond: bool, depth: uint, mass: double, [paramsyms])
		ParentClass.__construct(self, val, isstruct, iscond, depth, mass)
		[genParamFieldsExpList(self)] = [wrapExpListWithCopies(paramsyms)]
		self:updateLogprob()
	end
	-- Ctor 2: No value argument, sample one instead
	paramsyms = {}
	for i,t in ipairs(paramtypes) do table.insert(paramsyms, symbol(t)) end
	terra RandVarFromFunctionsT:__construct(isstruct: bool, iscond: bool, depth: uint, mass: double, [paramsyms])
		var val = sample([paramsyms])
		ParentClass.__construct(self, val, isstruct, iscond, depth, mass)
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
		local otherparamtypes = {...}
		return terra(self: &RandVarFromFunctionsT, other: &RVFFP)
			[ParentClass.__templatecopy(P, V)](self, other)
			[genParamCopyBlock(self, other, otherparamtypes)]
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
			-- We must *always* refresh params / updated logprobs
			--   if we're using dual numbers. Otherwise, nums could
			--   become stale after memory pool wipes and we'll end
			--   up with mysterious segfaults.
			if scalarType == ad.num then
				table.insert(checkexps,
				quote
					m.destruct(self.[n])
					self.[n] = m.copy([p])
					hasChanges = true
				end)
			-- Otherwise, only refresh if something has changed.
			else
				table.insert(checkexps,
				quote
					if not (self.[n] == [p]) then
						m.destruct(self.[n])
						self.[n] = m.copy([p])
						hasChanges = true
					end
				end)
			end
		end
		return checkexps
	end
	paramsyms = {}
	for i,t in ipairs(paramtypes) do table.insert(paramsyms, symbol(t)) end
	terra RandVarFromFunctionsT:checkForUpdates(mass: double, [paramsyms])
		var hasChanges = false
		[checkParams(self, hasChanges)]
		if not self.isStructural and (mass ~= self.mass) then
			self.mass = mass
			self.invMass = 1.0/mass
			hasChanges = true
		end
		if hasChanges then
			self:updateLogprob()
		end
	end
	paramsyms = {}
	for i,t in ipairs(paramtypes) do table.insert(paramsyms, symbol(t)) end
	terra RandVarFromFunctionsT:checkForUpdates(val: ValType, mass: double, [paramsyms])
		var hasChanges = false
		[checkParams(self, hasChanges)]
		if not self.isStructural and (mass ~= self.mass) then
			self.mass = mass
			self.invMass = 1.0/mass
			hasChanges = true
		end
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

	-- Setting real components may require us to update the logprob.
	terra RandVarFromFunctionsT:setRealComponents(comps: &Vector(ProbType), index: &uint) : {}
		ParentClass.setReals(self, comps, index)
		[ParentClass.HasRealComponents and (quote self:updateLogprob() end) or (quote end)]
	end
	inheritance.virtual(RandVarFromFunctionsT, "setRealComponents")

	-- Rescore the ERP
	terra RandVarFromFunctionsT:rescore() : {}
		self:updateLogprob()
	end
	inheritance.virtual(RandVarFromFunctionsT, "rescore")

	m.addConstructors(RandVarFromFunctionsT)
	return RandVarFromFunctionsT
end)


-- OK, I lied: this isn't quite the bottom of the hierarchy. We actually need to have a unique
--    type for every ERP callsite, which requires a minor extension of the above class.
-- This does not use the normal templating mechanism, since creation and retrieval of classes
--    need to happen through different interfaces.
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
	if not class then
		print(debug.traceback())
	end
	assert(class)
	return class
end
-- As with RandVarFromFunctions, variadic args are the parameter types for the ERP.
local function createRandVarFromCallsite(scalarType, sample, logprobfn, propose, computation, ...)
	local id = erph.getCurrentERPID()

	local struct RandVarFromCallsiteT {}
	RandVarFromCallsiteT.paramTypes = {...}
	local ParentClass = RandVarFromFunctions(scalarType, sample, logprobfn, propose, ...)
	inheritance.dynamicExtend(ParentClass, RandVarFromCallsiteT)

	-- The only extra functionality provided by this subclass is deepcopy.
	-- We need to know exactly which ERP type to copy into, which requires knowledge of
	--   parameter types, which may vary from callsite to callsite.
	virtualTemplate(RandVarFromCallsiteT, "deepcopy", function(P) return {}->{&RandVar(P)} end, function(P)
		local RandVarFromCallsiteP = RandVarFromCallsite(P, computation, id)
		local RandVarFromFunctionsP = RandVarFromFunctions(P, sample, logprobfn, propose, unpack(RandVarFromCallsiteP.paramTypes))
		return terra(self: &RandVarFromCallsiteT)
			var newvar = m.new(RandVarFromCallsiteP)
			-- Can just call the parent class __templatecopy since there's no new copy functionality added.
			[RandVarFromFunctionsP.__templatecopy(scalarType, unpack(RandVarFromCallsiteT.paramTypes))](newvar, self)
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

		return macro(function(...)
			local args = {...}
			local specSample = sample(V)
			local numParams = #specSample:gettype().parameters
			-- Check whether this is a conditioned or unconditioned ERP
			local paramtypes = {}
			local isCond = false
			if #args == numParams + 3 then
				isCond = true
				for i=4,#args do table.insert(paramtypes, args[i]:gettype()) end
			elseif #args == numParams + 2 then
				for i=3,#args do table.insert(paramtypes, args[i]:gettype()) end
			else
				error("Unexpected number of arguments to ERP function call.")
			end
			local rettype = specSample:gettype().returns[1]	-- All overloads must have same return type, so this is fine.
			local RandVarType = createRandVarFromCallsite(V, sample, logprobfn, propose, computation, unpack(paramtypes))
			local params = {}
			for _,t in ipairs(paramtypes) do table.insert(params, symbol(t)) end
			local erpfn = nil
			if isCond then
				erpfn = terra(isstruct: bool, condVal: rettype, mass: double, [params])
					return [trace.lookupVariableValue(RandVarType, isstruct, true, condVal, mass, params, specParams)]
				end
			else
				erpfn = terra(isstruct: bool, mass: double, [params])
					return [trace.lookupVariableValue(RandVarType, isstruct, false, nil, mass, params, specParams)]
				end
			end
			-- The ERP must push an ID to the callsite stack.
			erpfn = trace.pfn(specParams)(erpfn)
			-- Generate call to function
			return `erpfn([args])
		end)
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

	local function getmass(opstruct)
		if opstruct then
			local t = opstruct:gettype()
			for _,e in ipairs(t.entries) do
				if e.field == "mass"
					then return `opstruct.mass
				end
			end
		end
		return `1.0
	end

	-- Finally, wrap everything in a function that extracts options from the
	-- options struct.
	return spec.specializable(function(...)
		local paramTable = spec.paramListToTable(...)
		return macro(function(...)
			local params = {}
			for i=1,numparams do table.insert(params, (select(i,...))) end
			local opstruct = (select(numparams+1, ...))
			local isstruct = getisstruct(opstruct)
			local condval = getcondval(opstruct)
			local mass = getmass(opstruct)
			local erpfn = genErpFunction(paramTable)
			if condval then
				return `erpfn(isstruct, condval, mass, [params])
			else
				return `erpfn(isstruct, mass, [params])
			end
		end)
	end)
end


-- Define some commonly-used ERPs

local erp = {}

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

erp.uniformWithFalloff = 
makeERP(random.uniform_sample,
		erph.overloadOnParams(2, function(V, P1, P2)
			return terra(val: V, lo: P1, hi: P2)
				var lp = V(-ad.math.log(hi - lo))
				if val > hi then lp = lp - (val-lo)/(hi-lo) end
				if val < lo then lp = lp - (hi-val)/(hi-lo) end
				return lp
			end
		end))

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

-- Parameters are more intuitive, I think
erp.gammaMeanShape = spec.specializable(function(...)
	local paramTable = spec.paramListToTable(...)
	return macro(function(mean, shape, opstruct)
		opstruct = opstruct or `{}
		return `[erp.gamma(paramTable)](shape, mean/shape, opstruct)
	end)
end)

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



-- Public interface to create new ERPs
erp.newERP = function(name, sample, logprobfn, propose)
	local newerp = makeERP(sample, logprobfn, propose)
	spec.registerGlobalSpecializable(name, newerp)
	rawset(_G, name, newerp())
end


return
{
	globals = erp
}










local random = terralib.require("prob.random")
local templatize = terralib.require("templatize")
local inheritance = terralib.require("inheritance")
local m = terralib.require("mem")

local erph = terralib.require("erp.h")
local RandVar = erph.RandVar
local notImplementedError = erph.notImplementedError
local typeToID = erph.typeToID

local trace = terralib.require("trace")



-- Every random variable has some value type; this intermediate
-- class manages that
local RandVarWithVal = templatize(function(ValType)
	local struct RandVarWithValT
	{
		value: ValType
	}

	terra RandVarWithValT:__construct(val: ValType, isstruct: bool, iscond: bool)
		RandVar.__construct(self, isstruct, iscond)
		self.value = val
	end

	terra RandVarWithValT:__destruct()
		m.destruct(self.value)	
	end

	local ValTypeID = typeToID(ValType)
	terra RandVarWithValT:valueTypeID() : uint64
		return ValTypeID
	end
	inheritance.virtual(RandVarWithValT.methods.valueTypeID)

	terra RandVarWithValT:pointerToValue() : &opaque
		return [&opaque](&self.value)
	end
	inheritance.virtual(RandVarWithValT.methods.pointerToValue)

	terra RandVarWithValT:proposeNewValue() : {ValType, double, double}
		notImplementedError("proposeNewValue", [string.format("RandVarWithVal(%s)", tostring(ValType))])
	end
	inheritance.virtual(RandVarWithValT.methods.proposeNewValue)

	inheritance.dynamicExtend(RandVar, RandVarWithValT)
	return RandVarWithValT
end)


-- Finally, at the bottom of the hierarchy, we have random primitives defined by a set of functions
--    * paramtypes: the types of the parameters to the ERP. Essentially tells us which overload of the sample
--      sample function we are using.
--    * A sampling function. It may be overloaded, but all overloads must have the same return type
--    * A log probability function
--    * (Optional) A proposal function
--    * (Optional) A proposal log probability function
local function RandVarFromFunctions(paramtypes, sample, logprobfn, propose, logProposalProb)
	-- Can't handle functions with multiple return values
	assert(#sample:getdefinitions()[1]:gettype().returns == 1)
	-- All overloads of the sampling function must have the same return type
	local ValType = sample:getdefinitions()[1]:gettype().returns[1]
	for i=2,#sample:getdefinitions() do assert(sample:getdefinitions()[i]:gettype().returns[1] == ValType) end

	-- If we don't have propose or logProposalProb functions, make macros for default
	-- versions of these things given the functions we do have
	if not propose then
		-- Default: sample a new value irrespective of the current value (argument 1)
		propose = macro(function(...)
			local numargs = #paramtypes + 1
			local args = {}
			for i=2,numargs do
				table.insert(args, select(i,...))
			end
			return `sample([args])
		end)
	end
	if not logProposalProb then
		-- Default: Score the proposed value irrespective of the current value
		logProposalProb = macro(function(...)
			local numargs = #paramtypes + 2
			local args = {}
			for i=2,numargs do
				table.insert(args, select(i,...))
			end
			return `logprobfn([args])
		end)
	end

	-- Initialize the class we're building
	local struct RandVarFromFunctionsT {}
	RandVarFromFunctionsT.ValType = ValType

	-- Add one field for each parameter
	local paramFieldNames = {}
	for i,t in ipairs(paramtypes) do
		local n = string.format("param%d", i-1),
		table.insert(paramFieldNames, n)
		RandVarFromFunctionsT.entries:insert({ field = n, type = t})
	end

	local function genParamFieldsExpList(self)
		local exps = {}
		for i,n in paramFieldNames do
			table.insert(exps, `self.[n])
		end
		return exps
	end

	-- Constructor takes extra parameters
	local paramsyms = {}
	for i,t in ipairs(paramtypes) do table.insert(paramsyms, symbol(t)) end
	terra RandVarFromFunctionsT:__construct(val: ValType, isstruct: bool, iscond: bool, [paramsyms])
		[RandVarWithVal(ValType)].__construct(self, val, isstruct, iscond)
		[genParamFieldsExpList(self)] = [paramsyms]
	end

	-- Destructor should clean up any parameters
	local function genDestructBlock(self)
		local statements = {}
		for i,n in paramFieldNames do
			table.insert(statements, `m.destruct(self.[n]))
		end
		return statements
	end
	terra RandVarFromFunctionsT:__destruct()
		[RandVarWithVal(ValType)].__destruct(self)
		[genDestructBlock(self)]
	end

	-- Update log probability
	terra RandVarFromFunctionsT:updateLogprob() : {}
		self.logprob = logprobfn(self.value, [genParamFieldsExpList(self)])
	end
	inheritance.virtual(RandVarFromFunctionsT.methods.updateLogprob)

	-- Propose new value
	terra RandVarFromFunctionsT:proposeNewValue() : {ValType, double, double}
		var newval = propose(self.value, [genParamFieldsExpList(self)])
		var fwdPropLP = logProposalProb(self.value, newval, [genParamFieldsExpList(self)])
		var rvsPropLP = logProposalProb(newval, self.value, [genParamFieldsExpList(self)])
		return newval, fwdPropLP, rvsPropLP
	end
	inheritance.virtual(RandVarFromFunctionsT.methods.proposeNewValue)

	inheritance.dynamicExtend(RandVarWithVal(ValType), RandVarFromFunctionsT)
	return RandVarFromFunctionsT
end


-- Make a new random primitive
-- This returns a macro which performs sampling (the public interface to the 
--   random primitive)
-- The macro expects all the parameters expected by 'sample', plus an (optional) anonymous struct
--   which carries info such as 'isStructural', 'conditionedValue', etc.
-- NOTE: Any and all parameter/value types must define the __eq operator!
local function makeERP(sample, logprobfn, propose, logProposalProb)

	local erpFnPair = templatize(function(...)

		local paramtypes = {}
		local paramsyms = {}
		for i=1,select("#",...) do
			local ptype = (select(i,...))
			table.insert(paramtypes, ptype)
			table.insert(paramsyms, symbol(ptype))
		end

		-- TODO: Implement this (retrieve by params)
		local RVType = nil

		local function makeERPfn(iscond)

			local val = iscond and symbol(RVType.ValType) else `sample([paramsyms])
			local isstruct = symbol(bool)

			local function checkParams(self, hasChanges)
				local checkexps = {}
				for i,p in ipairs(paramsyms) do
					local n = string.format("params", i-1)
					table.insert(checkexps,
						quote
							if self.[n] ~= p then
								self.[n] = p
								hasChanges = true
							end
						end)
				end
				return checkexps
			end

			local function checkConditionedValue(self, val, hasChanges)
				if iscond then
					return quote
						if self.value ~= val then
							self.value = val
							hasChanges = true
						end
					end
				else return quote end
			end

			local body = quote
				-- Check if this random variable already exists
				var randvar: &RandVar = nil
				trace.lookupVariable(isstruct)
				if randvar ~= nil then
					-- The variable does exist. The logprob will need to
					--    be updated if the conditioned value or any params
					--    have changed
					var rvart = [&RVType](randvar)
					var hasChanges = false	
					[checkParams(randvar, hasChanges)]
					[checkConditionedValue(randvar, hasChanges)]
				else
					-- This variable doesn't yet exist, so create it
					--    and stick it in the trace
					randvar = RVType.heapAlloc(val, isstruct, iscond, [params])
					trace.addNewVariable(randvar)
				end
				return ([&RVType](randvar)).value
			end

			if iscond then
				return terra(isstruct, val, [paramsyms])
					[body]
				end
			else
				return terra(isstruct, [paramsyms])
					[body]
				end
			end
		end

		local ret = {}
		ret[true] = makeERPfn(true)
		ret[false] = makeERPfn(false)
		return ret

	end)

	local numparams = #sample:getdefinitions()[1]:gettype().params

	-- Finally, return the macro which generates the ERP function call.
	return macro(function(...)
		local params = {}
		for i=1,numparams do table.insert(params, (select(i,...))) end
		local paramtypes = {}
		for _,p in ipairs(params) do table.insert(paramtypes, p:gettype()) end
		local erpfns = erpFnPair(unpack(paramtypes))
		local optstruct = (select(RVType.__numParams+1, ...))
		local iscond = getIsConditioned(opstruct)
		local erpfn = erpfns[iscond]
		local isstruct = getIsStructural(opstruct)
		if iscond then
			local val = `opstruct.conditionedValue
			return `erpfn(isstruct, val, [params])
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
		terra(currval: double, p: double) if currval == 0 then return 1 else return 0 end end,
		terra (currval: double, propval: double, p: double) return 0.0 end)

erp.uniform =
makeERP(random.uniform_sample(double),
		random.uniform_logprob(double))

erp.multinomial =
makeERP(random.multinomial_sample(double),
	    random.multinomial_logprob(double),
	    terra(currval: int, params: Vector(double))
	    	var newparams = m.copy(params)
	    	newparams:set(currval, 0)
	    	return [random.multinomial_sample(double)](newparams)
	    	m.destruct(newparams)
	    end,
	    terra(currval: int, propval: int, params: Vector(double))
	    	var newparams = m.copy(params)
	    	newparams:set(currval, 0)
	    	return [random.multinomial_logprob(double)](propval, newparams)
	    	m.destruct(newparams)
    	end)

erp.gaussian =
makeERP(random.gaussian_sample(double),
		random.gaussian_logprob(double),
		terra(currval: double, mu: double, sigma: double)
			return [random.gaussian_sample(double)](currval, sigma)
		end,
		terra(currval: double, propval: double, mu: double, sigma: double)
			return [random.gaussian_logprob(double)](propval, currval, sigma)
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









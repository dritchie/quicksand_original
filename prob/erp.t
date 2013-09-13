
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


-- Finally, at the bottom of the hierarchy, we have random primitives defined by a set of functions:
--    * A sampling function
--    * A log probability function
--    * (Optional) A proposal function
--    * (Optional) A proposal log probability function
local function RandVarFromFunctions(sample, logprobfn, propose, logProposalProb)
	assert(#sample:getdefinitions()[1]:gettype().returns == 1)	-- Can't handle functions with multiple return values
	local ValType = sample:getdefinitions()[1]:gettype().returns[1]
	local ParamTypes = sample:getdefinitions()[1]:gettype().parameters

	-- If we don't have propose or logProposalProb functions, make macros for default
	-- versions of these things given the functions we do have
	if not propose then
		-- Default: sample a new value irrespective of the current value (argument 1)
		propose = macro(function(...)
			local numargs = #ParamTypes + 1
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
			local numargs = #ParamTypes + 2
			local args = {}
			for i=2,numargs do
				table.insert(args, select(i,...))
			end
			return `logprobfn([args])
		end)
	end

	-- Initialize the class we're building
	local struct RandVarFromFunctionsT {}
	RandVarFromFunctionsT.__numParams = #ParamTypes

	-- Add one field for each parameter
	local paramFieldNames = {}
	for i,t in ipairs(ParamTypes) do
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
	for i,t in ipairs(ParamTypes) do table.insert(paramsyms, symbol(t)) end
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

	local RVType = RandVarFromFunctions(sample, logprobfn, propose, logProposalProb)

	local function getIsStructural(opstruct)
		if opstruct then
			local ostype = optstruct:gettype()
			for _,e in ipairs(ostype:getentries()) do
				if e.field == "isStructural" then
					return `optstruct.isStructural
				end
			end
		end
		return nil
	end

	local function getIsConditioned(opstruct)
		if opstruct then
			local ostype = optstruct:gettype()
			for _,e in ipairs(ostype:getentries()) do
				if e.field == "conditionedValue" then
					return true
				end
			end
		end
		return false
	end

	local function getConditionedValue(opstruct)
		if opstruct then
			local ostype = optstruct:gettype()
			for _,e in ipairs(ostype:getentries()) do
				if e.field == "conditionedValue" then
					return `optstruct.conditionedValue
				end
			end
		end
		return nil
	end

	local function makeERPfn(opstruct)
		local paramtypes = sample:getdefinitions()[1]:gettype().parameters
		local paramsyms = {}
		for _,t in ipairs(paramtypes) do table.inserT(paramsyms, symbol(t)) end

		-- Default values
		local val = getConditionedValue(optstruct) or `sample([paramsyms])
		local iscond = getIsConditioned(optstruct)
		local isstruct = getIsStructural(opstruct)

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

		local function checkConditionedValue(self, hasChanges)
			if iscond then
				return quote
					if self.value ~= [val] then
						self.value = [val]
						hasChanges = true
					end
				end
			else return quote end
		end

		-- Here's the actual function that gets called at runtime.
		-- Note that it's wrapped with trace.pfn, so rand var names can be tracked.
		return trace.pfn(terra([paramsyms])
			var value = [val]
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
				randvar = RVType.heapAlloc([val], isstruct, iscond, [params])
				trace.addNewVariable(randvar)
			end
		end)
	end

	-- For efficiency, we cache generated code. We need to generate code
	--   for isDirectlyConditioned={true|false}
	--   Everything else must be resolved at runtime.
	local erpCache = {}
	local function erpCacheLookup(opstruct)
		local iscond = getIsConditioned(optstruct)
		local cfn = erpCache[iscond]
		if not cfn then
			cfn = makeERPfn(optstruct)
			erpCache[iscond] = cfn
		end
		return cfn
	end

	-- Finally, return the macro which generates the ERP function call.
	return macro(function(...)
		local optstruct = (select(RVType.__numParams+1, ...))
		local erpfn = erpCacheLookup(optstruct)
		local params = {}
		for i=1,RVType.__numParams do table.insert(params, (select(i,...))) end
		return `erpfn([params])
	end)
end


-- Define some commonly-used ERPs

local erp = {makeERP = makeERP}

erp.flip =
makeERP(random.flip_sample(double),
		random.flip_logprob(double, double),
		terra(currval: double, p: double) if currval == 0 then return 1 else return 0 end end,
		terra (currval: double, propval: double, p: double) return 0.0 end)

-- erp.uniform = 
-- makeERP(random.uniform_sample())








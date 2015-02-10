
local random = require("prob.random")
local templatize = require("templatize")
local virtualTemplate = require("vtemplate")
local inheritance = require("inheritance")
local Vector = require("vector")
local m = require("mem")
local util = require("util")
local erph = require("prob.erph")
local RandVar = erph.RandVar
local typeToID = erph.typeToID
local trace = require("prob.trace")
local spec = require("prob.specialize")
local ad = require("ad")

local C = terralib.includec("stdio.h")

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

	terra RandVarWithValT:pointerToRawValue() : &opaque
		return [&opaque](&self.value)
	end
	inheritance.virtual(RandVarWithValT, "pointerToRawValue")


	-- By default, we understand how to get/set the real components of double, ad.num,
	--    Vector(double), and Vector(ad.num). Any other value types must define the
	--    'getRealComponents' and 'setRealComponents' methods--otherwise, they'll be
	--    treated as having no real components.
	RandVarWithValT.HasRealComponents = (ValType == double or ValType == ad.num or
							   ValType == Vector(double) or ValType == Vector(ad.num) or
							   ValType:getmethod("setRealComponents") ~= nil)
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
		else
			return quote end
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
		else
			return quote end
		end
	end

	terra RandVarWithValT:getRealComponents(comps: &Vector(ProbType)) : {}
		[genGetReals(self, comps)]
	end
	inheritance.virtual(RandVarWithValT, "getRealComponents")
	terra RandVarWithValT:getRawRealComponents(comps: &Vector(ProbType)) : {}
		[genGetReals(self, comps)]
	end
	inheritance.virtual(RandVarWithValT, "getRawRealComponents")

	terra RandVarWithValT:setReals(comps: &Vector(ProbType), index: &uint)
		[genSetReals(self, comps, index)]
	end
	terra RandVarWithValT:setRealComponents(comps: &Vector(ProbType), index: &uint) : {}
		[genSetReals(self, comps, index)]
	end
	inheritance.virtual(RandVarWithValT, "setRealComponents")
	terra RandVarWithValT:setRawRealComponents(comps: &Vector(ProbType), index: &uint) : {}
		[genSetReals(self, comps, index)]
	end
	inheritance.virtual(RandVarWithValT, "setRawRealComponents")

	return RandVarWithValT
end)

-- Some utility functions for bounded variable transforms
local logistic = macro(function(x, scale)
	return `1.0 / (1.0 + ad.math.exp(-scale*x))
end)
local invlogistic = macro(function(y, scale)
	return `-ad.math.log(1.0/y - 1.0)/scale
end)

-- Finally, at the bottom of the hierarchy, we have random primitives defined by a set of functions
--    * The type of scalar values (doubles or ad.nums)
--    * A sampling function. It may be overloaded, but all overloads must have the same return type
--    * A log probability function
--    * A proposal function
--    * ... Variadic arguments are the types of the parameters to the ERP (essentially specifying which overload
--          of the provided functions we're using)
local RandVarFromFunctions
RandVarFromFunctions = templatize(function(scalarType, sampleTemplate, logprobTemplate, proposeTemplate, OpstructType, ...)
	local paramtypes = {...}

	local sample = sampleTemplate(scalarType)
	local logprobfn = logprobTemplate(scalarType)
	local propose = proposeTemplate(scalarType)

	-- All overloads of the sampling function must have the same return type
	local ValType = sample:getdefinitions()[1]:gettype().returntype
	for i=2,#sample:getdefinitions() do assert(sample:getdefinitions()[i]:gettype().returntype == ValType) end

	local ProbType = scalarType

	-- Some flags that we need to use repeatedly
	local hasCondVal = erph.opts.hasCondVal(OpstructType)
	local hasLowerBound = erph.opts.hasLowerBound(OpstructType)
	local hasUpperBound = erph.opts.hasUpperBound(OpstructType)
	local hasBoundShapeParam = erph.opts.hasBoundShapeParam(OpstructType)
	local isBounded = hasLowerBound or hasUpperBound

	-- Initialize the class we're building
	local struct RandVarFromFunctionsT { hasChanges: bool }
	RandVarFromFunctionsT.ValType = ValType
	local ParentClass = RandVarWithVal(ProbType, ValType)
	inheritance.dynamicExtend(ParentClass, RandVarFromFunctionsT)

	-- Add upper/lower bounds, if provided
	if hasLowerBound then
		local t = erph.opts.typeOfErpOption(OpstructType, "lowerBound")
		RandVarFromFunctionsT.entries:insert({field = "lowerBound", type = t})
	end
	if hasUpperBound then
		local t = erph.opts.typeOfErpOption(OpstructType, "upperBound")
		RandVarFromFunctionsT.entries:insert({field = "upperBound", type = t})
	end
	-- Add bounding shape param
	if isBounded then
		local t = double
		if hasBoundShapeParam then
			t = erph.opts.typeOfErpOption(OpstructType, "boundShapeParam")
		end
		RandVarFromFunctionsT.entries:insert({field = "boundShapeParam", type = t})
	end

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

	-- Transforms (and their inverses) for bounded variables
	local forwardTransform = macro(function(self, x) return x end)
	local inverseTransform = macro(function(self, y) return y end)
	local priorAdjustment = macro(function(self, x) return `0.0 end)
	if hasLowerBound and hasUpperBound then
		forwardTransform = macro(function(self, x)
			return quote
				var logit = logistic(x, self.boundShapeParam)
				if x > [-math.huge] and logit == 0.0 then logit = 1e-15 end
				if x < [math.huge] and logit == 1.0 then logit = [1.0 - 1e-15] end
				var y = self.lowerBound + (self.upperBound - self.lowerBound) * logit
			in
				y
			end
		end)
		inverseTransform = macro(function(self, y)
			return quote
				var z = ad.math.fmax(ad.math.fmin(y, self.upperBound - 1e-15), self.lowerBound + 1e-15)
				var t = (z - self.lowerBound) / (self.upperBound - self.lowerBound)
				var x = invlogistic(t, self.boundShapeParam)
				-- C.printf("y: %g, z: %g, t: %g, x: %g, lowerBound: %g, upperBound:%g\n",
				-- 	ad.val(y), ad.val(z), ad.val(t), ad.val(x), ad.val(self.lowerBound), ad.val(self.upperBound))
			in
				x
			end
		end)
		priorAdjustment = macro(function(self, x)
			-- Simplification of ad.math.log((self.upperBound - self.lowerBound) * logistic(x) * (1 - logistic(x)))
			return `ad.math.log(self.upperBound - self.lowerBound) - self.boundShapeParam*x - 2.0*ad.math.log(1.0 + ad.math.exp(-self.boundShapeParam*x))
		end)
	elseif hasLowerBound then
		forwardTransform = macro(function(self, x) return `ad.math.exp(self.boundShapeParam*x) + self.lowerBound end)
		inverseTransform = macro(function(self, y)
			return quote
				var z = ad.math.fmax(y, self.lowerBound + 1e-15)
				var x = ad.math.log(z - self.lowerBound)/self.boundShapeParam
				-- C.printf("y: %g, z: %g, x: %g, lowerBound: %g\n",	
				-- 	ad.val(y), ad.val(z), ad.val(x), ad.val(self.lowerBound))
			in
				x
			end
		end)
		priorAdjustment = macro(function(self, x) return `ad.math.log(self.boundShapeParam) + self.boundShapeParam*x end)
	elseif hasUpperBound then
		forwardTransform = macro(function(self, x) return `self.upperBound - ad.math.exp(self.boundShapeParam*x) end)
		inverseTransform = macro(function(self, y)
			return quote
				var z = ad.math.fmin(y, self.upperBound - 1e-15)
				var x = ad.math.log(self.upperBound - z)/self.boundShapeParam
				-- C.printf("y: %g, z: %g, x: %g, upperBound: %g\n",
				-- 	ad.val(y), ad.val(z), ad.val(x), ad.val(self.upperBound))
			in
				x
			end
		end)
		priorAdjustment = macro(function(self, x) return `ad.math.log(self.boundShapeParam) + self.boundShapeParam*x end)
	end
	RandVarFromFunctionsT.methods.forwardTransform = forwardTransform
	RandVarFromFunctionsT.methods.inverseTransform = inverseTransform
	RandVarFromFunctionsT.methods.priorAdjustment = priorAdjustment

	-- Get and set the variable's value, respecting these transforms
	RandVarFromFunctionsT.methods.getValue = macro(function(self)
		return `self:forwardTransform(self.value)
	end)
	RandVarFromFunctionsT.methods.setValue = macro(function(self, val)
		return quote
			m.destruct(self.value)
			self.value = self:inverseTransform(m.copy(val))
		end
	end)

	-- Constructor
	local paramsyms = {}
	local hasInitialVal = erph.opts.hasInitialVal(OpstructType)
	for i,t in ipairs(paramtypes) do table.insert(paramsyms, symbol(t)) end
	terra RandVarFromFunctionsT:__construct(depth: uint, options: OpstructType, [paramsyms])
		var isstruct = [erph.opts.getIsStruct(`options, OpstructType)]
		var mass = [erph.opts.getMass(`options, OpstructType)]
		var iscond = hasCondVal
		var val : ValType
		-- Set the initial value of this variable
		-- It's either constrained to be some value, or has a specified initial value, or
		--    we sample from the prior.
		[util.optionally(hasCondVal, function() return quote
			val = [erph.opts.getCondVal(`options, OpstructType)]
		end end)]
		[util.optionally(not hasCondVal and hasInitialVal, function() return quote
			val = [erph.opts.getInitialVal(`options, OpstructType)]
		end end)]
		[util.optionally(not hasCondVal and (not hasInitialVal), function() return quote
			val = sample([paramsyms])
		end end)]
		-- Record bounds, if they exist
		[util.optionally(hasLowerBound, function() return quote
			self.lowerBound = [erph.opts.getLowerBound(`options, OpstructType)]
		end end)]
		[util.optionally(hasUpperBound, function() return quote
			self.upperBound = [erph.opts.getUpperBound(`options, OpstructType)]
		end end)]
		-- Record bounding shape (or a default, if not provided)
		[util.optionally(isBounded, function() return quote
			self.boundShapeParam = [erph.opts.getBoundShapeParam(`options, OpstructType)]
		end end)]
		-- Finish up
		val = self:inverseTransform(val)
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
	local function genParamCopyBlock(self, other, otherparamtypes)
		local selfparamexprs = genParamFieldsExpList(self)
		local otherparamexprs = genParamFieldsExpList(other)
		local lines = {}
		for i=1,#paramtypes do
			if paramtypes[i] == otherparamtypes[i] then
				table.insert(lines, quote [selfparamexprs[i]] = m.copy([otherparamexprs[i]]) end)
			else
				table.insert(lines, quote [selfparamexprs[i]] = [m.templatecopy(paramtypes[i])]([otherparamexprs[i]]) end)
			end
		end
		return lines
	end
	RandVarFromFunctionsT.__templatecopy = templatize(function(P, ...)
		local RVFFP = RandVarFromFunctions(P, sampleTemplate, logprobTemplate, proposeTemplate, ...)
		local V = RVFFP.ValType
		local otherparamtypes = {...}
		return terra(self: &RandVarFromFunctionsT, other: &RVFFP)
			[ParentClass.__templatecopy(P, V)](self, other)
			[genParamCopyBlock(self, other, otherparamtypes)]
			[util.optionally(hasLowerBound, function() return quote
				self.lowerBound = other.lowerBound
			end end)]
			[util.optionally(hasUpperBound, function() return quote
				self.upperBound = other.upperBound
			end end)]
			[util.optionally(isBounded, function() return quote
				self.boundShapeParam = other.boundShapeParam
			end end)]
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
			-- We must *always* refresh params / update logprobs
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
	terra RandVarFromFunctionsT:checkForUpdates(options: OpstructType, [paramsyms])
		-- We must always rescore if we're using AD, because the dual nums get wiped
		--    out after every run.
		var hasChanges = self.hasChanges or [scalarType == ad.num]
		var mass = [erph.opts.getMass(`options, OpstructType)]
		-- Check for changes in parameters
		[checkParams(self, hasChanges)]
		if not self.isStructural and (mass ~= self.mass) then
			self.mass = mass
			self.invMass = 1.0/mass
			hasChanges = true
		end
		-- Check for change in bounds
		[util.optionally(hasLowerBound, function() return quote
			var lowerBound = [erph.opts.getLowerBound(`options, OpstructType)]
			if [util.istype(ad.num)](lowerBound) or not (self.lowerBound == lowerBound) then
				self.lowerBound = lowerBound
				hasChanges = true
			end
		end end)]
		[util.optionally(hasUpperBound, function() return quote
			var upperBound = [erph.opts.getUpperBound(`options, OpstructType)]
			if [util.istype(ad.num)](upperBound) or not (self.upperBound == upperBound) then
				self.upperBound = upperBound
				hasChanges = true
			end
		end end)]
		[util.optionally(hasBoundShapeParam, function() return quote
			var boundShapeParam = [erph.opts.getBoundShapeParam(`options, OpstructType)]
			if [util.istype(ad.num)](boundShapeParam) or not (self.boundShapeParam == boundShapeParam) then
				self.boundShapeParam = boundShapeParam
				hasChanges = true
			end
		end end)]
		-- Check for change in conditioned value.
		[util.optionally(hasCondVal, function() return quote
			var val = [erph.opts.getCondVal(`options, OpstructType)]
			val = self:inverseTransform(val)
			if [util.istype(ad.num)](val) or not (self.value == val) then
				m.destruct(self.value)
				self.value = m.copy(val)
				hasChanges = true
			end
		end end)]
		if hasChanges then
			self:updateLogprob()
		end
	end

	-- Update log probability
	terra RandVarFromFunctionsT:updateLogprob() : {}
		self.logprob = self:priorAdjustment(self.value) + logprobfn(self:forwardTransform(self.value), [genParamFieldsExpList(self)])
		-- self.logprob = self:priorAdjustment(self.value)
		-- self.logprob = logprobfn(self:forwardTransform(self.value), [genParamFieldsExpList(self)])
		-- self.logprob = 0.0
		self.hasChanges = false
	end

	-- Propose new value
	terra RandVarFromFunctionsT:proposeNewValue() : {ProbType, ProbType}
		var newval, fwdPropLP, rvsPropLP = propose(self:forwardTransform(self.value), [genParamFieldsExpList(self)])
		self:setValue(newval)
		self:updateLogprob()
		return fwdPropLP, rvsPropLP
	end
	inheritance.virtual(RandVarFromFunctionsT, "proposeNewValue")

	-- Set the value directly
	terra RandVarFromFunctionsT:setRawValue(valptr: &opaque) : {}
		m.destruct(self.value)
		self.value = m.copy(@([&ValType](valptr)))
		self.hasChanges = true
	end
	inheritance.virtual(RandVarFromFunctionsT, "setRawValue")

	-- We may (eventually) need to update the logprob.
	terra RandVarFromFunctionsT:setRawRealComponents(comps: &Vector(ProbType), index: &uint) : {}
		ParentClass.setReals(self, comps, index)
		[util.optionally(ParentClass.HasRealComponents, function() return quote
			self.hasChanges = true
		end end)]
	end
	inheritance.virtual(RandVarFromFunctionsT, "setRawRealComponents")

	-- Need to take into account variable transforms
	terra RandVarFromFunctionsT:getRealComponents(comps: &Vector(ProbType)) : {}
		var currsize = comps.size
		self:getRawRealComponents(comps)
		[util.optionally(isBounded, function() return quote
			comps(currsize) = self:forwardTransform(comps(currsize))
		end end)]
	end
	inheritance.virtual(RandVarFromFunctionsT, "getRealComponents")
	terra RandVarFromFunctionsT:setRealComponents(comps: &Vector(ProbType), index: &uint) : {}
		[util.optionally(isBounded, function() return quote
			-- Here, we're using the fact that bounding only works on scalar variables (1 real component)
			var xformedComps = [Vector(ProbType)].stackAlloc(1, self:inverseTransform(comps(@index)))
			var tempIndex = 0U
			self:setRawRealComponents(&xformedComps, &tempIndex)
			m.destruct(xformedComps)
			@index = @index + 1
		end end)]
		[util.optionally(not isBounded, function() return quote
			self:setRawRealComponents(comps, index)
		end end)]
	end
	inheritance.virtual(RandVarFromFunctionsT, "setRealComponents")

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
local function createRandVarFromCallsite(scalarType, sample, logprobfn, propose, computation, OpstructType, ...)
	local id = erph.getCurrentERPID()

	local struct RandVarFromCallsiteT {}
	RandVarFromCallsiteT.OpstructType = OpstructType
	RandVarFromCallsiteT.paramTypes = {...}
	local ParentClass = RandVarFromFunctions(scalarType, sample, logprobfn, propose, OpstructType, ...)
	inheritance.dynamicExtend(ParentClass, RandVarFromCallsiteT)
	RandVarFromCallsiteT.ValType = ParentClass.ValType

	-- The only extra functionality provided by this subclass is deepcopy.
	-- We need to know exactly which ERP type to copy into, which requires knowledge of
	--   parameter types, which may vary from callsite to callsite.
	virtualTemplate(RandVarFromCallsiteT, "deepcopy", function(P) return {}->{&RandVar(P)} end, function(P)
		local RandVarFromCallsiteP = RandVarFromCallsite(P, computation, id)
		local RandVarFromFunctionsP = RandVarFromFunctions(P, sample, logprobfn, propose, RandVarFromCallsiteP.OpstructType,
														   unpack(RandVarFromCallsiteP.paramTypes))
		return terra(self: &RandVarFromCallsiteT)
			var newvar = m.new(RandVarFromCallsiteP)
			-- Can just call the parent class __templatecopy since there's no new copy functionality added.
			[RandVarFromFunctionsP.__templatecopy(scalarType, RandVarFromCallsiteT.OpstructType,
												  unpack(RandVarFromCallsiteT.paramTypes))](newvar, self)
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

	-- When no options struct is provided, we can construct a default one
	--    (and its type) using this function
	local function defaultOpStructAndType()
		local struct OpStruct {}
		return (`OpStruct{}), OpStruct
	end

	-- Generate an overloaded function which does the ERP call
	-- Memoize results for different specializations
	return spec.specializable(function(...)
		local specParams = spec.paramListToTable(...)
		local V = spec.paramFromTable("scalarType", specParams)
		local computation = spec.paramFromTable("computation", specParams)
		local specSample = sample(V)
		local numParams = #specSample:gettype().parameters
		return macro(function(...)
			local params = {}
			for i=1,numParams do table.insert(params, (select(i,...))) end
			local opstruct = nil
			local OpstructType = nil
			if select("#",...) == numParams+1 then
				opstruct = (select(numParams+1, ...))
				OpstructType = opstruct:gettype()
			else
				opstruct, OpstructType = defaultOpStructAndType()
			end
			local paramtypes = {}
			for _,p in ipairs(params) do table.insert(paramtypes, p:gettype()) end 
			local paramsyms = {}
			for _,t in ipairs(paramtypes) do table.insert(paramsyms, symbol(t)) end			
			local RandVarType = createRandVarFromCallsite(V, sample, logprobfn, propose, computation, OpstructType, unpack(paramtypes))
			local terra erpfn(optionStruct: OpstructType, [paramsyms])
				return [trace.lookupVariableValue(RandVarType, `optionStruct, OpstructType, paramsyms, specParams)]
			end
			-- The ERP must push an ID to the callsite stack.
			erpfn = trace.pfn(erpfn)
			-- Generate call to function
			return `erpfn(opstruct, [params])
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

-- This one ends up getting used all the time
erp.boundedUniform = spec.specializable(function(...)
	local paramTable = spec.paramListToTable(...)
	return macro(function(lo, hi, opts)
		local valquote = nil
		if opts then
			local OpsType = opts:gettype()
			local struct NewOpsType {}
			for _,e in ipairs(OpsType.entries) do
				table.insert(NewOpsType.entries, {field=e.field, type=e.type})
			end
			table.insert(NewOpsType.entries, {field="lowerBound", type=lo:gettype()})
			table.insert(NewOpsType.entries, {field="upperBound", type=hi:gettype()})
			table.insert(NewOpsType.entries, {field="structural", type=bool})
			valquote = quote
				var newopts = NewOpsType(opts)
				newopts.structural = false
				newopts.lowerBound = lo
				newopts.upperBound = hi
			in
				[erp.uniform(paramTable)](lo, hi, newopts)
			end
		else
			valquote = `[erp.uniform(paramTable)](lo, hi, {structural=false, lowerBound=lo, upperBound=hi})
		end
		return valquote
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









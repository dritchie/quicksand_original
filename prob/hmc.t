local m = terralib.require("mem")
local inf = terralib.require("prob.inference")
local MCMCKernel = inf.MCMCKernel
local inheritance = terralib.require("inheritance")
local ad = terralib.require("ad")
local trace = terralib.require("prob.trace")
local BaseTraceD = trace.BaseTrace(double)
local BaseTraceAD = trace.BaseTrace(ad.num)
local erp = terralib.require("prob.erph")
local RandVarAD = erp.RandVar(ad.num)



-- Kernel for doing Hamiltonian Monte Carlo proposals
-- Only makes proposals to non-structural variables
-- TODO: lots of places where we can save intermediate variables between calls
--    to 'next' to avoid computation (by detecting that the trace is the same.)
-- TODO: automatically adapt step size? (dual averaging scheme from Stan)
-- TODO: variable mass adjustment based on LARJ annealing schedule
local struct HMCKernel
{
	stepSize: double,
	numSteps: uint,
	proposalsMade: uint,
	proposalsAccepted: uint
}
inheritance.dynamicExtend(MCMCKernel, HMCKernel)

terra HMCKernel:__construct(stepSize: double, numSteps: uint)
	self.stepSize = stepSize
	self.numSteps = numSteps
	self.proposalsMade = 0
	self.proposalsAccepted = 0
end

terra HMCKernel:__destruct() : {} end
inheritance.virtual(HMCKernel, "__destruct")

terra logProbAndGrad(workTrace: &BaseTraceAD, vars: &Vector(&RandVarAD), reals: &Vector(double), grad: &Vector(double))
	var indeps = [Vector(ad.num)].stackAlloc(reals.size, 0.0)
	for i=0,reals.size do
		indeps:set(i, ad.num(reals:get(i)))
	end
	var index = 0
	for i=0,vars.size do
		vars:get(i):setRealComponents(&indeps, &index)
	end
	[trace.traceUpdate(workTrace, {structureChange=false})]
	var lp = workTrace.logprob:val()
	workTrace.logprob:grad(&indeps, grad)
	m.destruct(indeps)
	return lp
end

-- TODO: The updates in this function are a good candidate for vectorized multiply/add...
terra HMCKernel:leapfrog(workTrace: &BaseTraceAD, vars: &Vector(&RandVarAD), reals: &Vector(double))
	var grad = [Vector(double)].stackAlloc()
	var lp : double
	for s=0,self.numSteps do
		lp = logProbAndGrad(workTrace, vars, reals, &grad)
		-- TODO: FINISH! (need momentum variables, too...)
	end
	m.destruct(grad)
end

terra HMCKernel:next(currTrace: &BaseTraceD) : &BaseTraceD
	self.proposalsMade = self.proposalsMade + 1
	var nextTrace = [BaseTraceD.deepcopy(ad.num)](currTrace)
	-- Get the real components of currTrace variables
	var reals = [Vector(double)].stackAlloc()
	var currVars = currTrace:freeVars(false, true)
	for i=0,currVars.size do
		currVars:get(i):getRealComponents(&reals)
	end
	m.destruct(currVars)

	-- Do an HMC step on them
	var newVars = newTrace:freeVars(false, true)
	-- TODO: FINISH!

	m.destruct(newVars)
	m.destruct(reals)

end
inheritance.virtual(HMCKernel, "next")

terra HMCKernel:name() : rawstring return [HMCKernel.name] end
inheritance.virtual(HMCKernel, "name")

terra HMCKernel:stats() : {}
	C.printf("Acceptance ratio: %g (%u/%u)\n",
		[double](self.proposalsAccepted)/self.proposalsMade,
		self.proposalsAccepted,
		self.proposalsMade)
end
inheritance.virtual(HMCKernel, "stats")

m.addConstructors(HMCKernel)


-- Convenience method for generating new HMCKernels
local HMC = inf.makeKernelGenerator(
	terra(stepSize: double, numSteps: uint)
		return HMCKernel.heapAlloc(stepSize, numSteps)
	end,
	-- No way to provide a meaningful default step size...
	{numSteps = 1})
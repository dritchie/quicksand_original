local prob = terralib.require("prob")
local util = terralib.require("util")
local rand = terralib.require("prob.random")
local m = terralib.require("mem")
local Vector = terralib.require("vector")
util.openModule(prob)

local C = terralib.includecstring [[
#include <stdio.h>
#include <math.h>
inline void flush() { fflush(stdout); }
]]

local numsamps = 150
local lag = 20
local runs = 5
local errorTolerance = 0.07

local terra test(name: rawstring, estimates: &Vector(double), trueExpectation: double)
	C.printf("test: %s...", name)
	C.flush()
	var errors = [Vector(double)].stackAlloc()
	for i=0,estimates.size do
		errors:push(C.fabs(estimates:get(i) - trueExpectation))
	end
	var meanAbsErr = [mean(double)](&errors)
	var testMean = [mean(double)](estimates)
	if meanAbsErr > errorTolerance then
		C.printf("failed! True mean: %g | Test mean: %g\n", trueExpectation, testMean)
	else
		C.printf("passed.\n")
	end
end

local function fwdtest(name, computation, trueExpectation)
	return quote
		var estimates = [Vector(double)].stackAlloc()
		for run=0,runs do
			var samps = [Vector(double)].stackAlloc()
			for i=0,numsamps do
				samps:push(computation())
			end
			estimates:push([mean(double)](&samps))
			m.destruct(samps)
		end
		test(name, &estimates, trueExpectation)
		m.destruct(estimates)
	end
end

local function mhtest(name, computation, trueExpectation)
	return quote
		var estimates = [Vector(double)].stackAlloc()
		for run=0,runs do
			var samps = [mcmc(computation, RandomWalk(), {numsamps=numsamps, lag=lag, verbose=false})]
			estimates:push([expectation(double)](&samps))
			m.destruct(samps)
		end
		test(name, &estimates, trueExpectation)
		m.destruct(estimates)
	end
end

local function eqtest(name, estimatedValues, trueValues)
	return quote
		var estvalues = Vector.fromItems([estimatedValues])
		var truevalues = Vector.fromItems([trueValues])
		C.printf("test: %s...", name)
		C.flush()
		for i=0,estvalues.size do
			var ev = estvalues:get(i)
			var tv = truevalues:get(i)
			if C.fabs(ev - tv) > errorTolerance then
				C.printf("failed! True value: %g | Test value: %g\n", tv, ev)
				--return
			end
		end
		C.printf("passed.\n")
		m.destruct(estvalues)
		m.destruct(truevalues)
	end
end

-- Hack to get around Terra not having closures/inner functions
local fns = {}	-- To make sure functions don't get gc'd.
local function lambda(func)
	table.insert(fns, func)
	return func:getdefinitions()[1]:getpointer()
end

-- ...but, I can't do recursive anonymous functions
local terra powerLaw_tailrec(prob: double, x: int) : int
	if [bool](flip(prob)) then
		return x
	else
		return powerLaw_tailrec(prob, x+1)
	end
end
powerLaw_tailrec = pfn(powerLaw_tailrec)
local terra powerLaw(prob: double, x: int) : int
	if [bool](flip(prob)) then
		return x
	else
		return 0 + powerLaw(prob, x+1)
	end
end
powerLaw = pfn(powerLaw)


local terra doTests()

	C.printf("starting tests...\n")

	-- ERP tests

	[fwdtest(
	"flip sample",
	terra() : double
		return flip(0.7)
	end,
	0.7)]

	[mhtest(
	"flip query",
	pfn(terra() : double
		return flip(0.7)
	end),
	0.7)]

	[fwdtest(
	"uniform sample",
	terra() : double
		return uniform(0.1, 0.4)
	end,
	0.5*(.1+.4))]

	[mhtest(
	"uniform query",
	pfn(terra() : double
		return uniform(0.1, 0.4)
	end),
	0.5*(.1+.4))]

	[fwdtest(
	"multinomial sample",
	terra() : double
		var items = Vector.fromItems(.2, .3, .4)
		var probs = Vector.fromItems(.2, .6, .2)
		var ret = multinomialDraw(items, probs)
		m.destruct(items)
		m.destruct(probs)
		return ret
	end,
	0.2*.2 + 0.6*.3 + 0.2*.4)]

	[mhtest(
	"multinomial query",
	pfn(terra() : double
		var items = Vector.fromItems(.2, .3, .4)
		var probs = Vector.fromItems(.2, .6, .2)
		var ret = multinomialDraw(items, probs)
		m.destruct(items)
		m.destruct(probs)
		return ret
	end),
	0.2*.2 + 0.6*.3 + 0.2*.4)]

	[eqtest(
	"multinomial lp",
	{
		rand.multinomial_logprob(double)(0, Vector.fromNums(.2, .6, .2)),
		rand.multinomial_logprob(double)(1, Vector.fromNums(.2, .6, .2)),
		rand.multinomial_logprob(double)(2, Vector.fromNums(.2, .6, .2))
	},
	{math.log(.2), math.log(.6), math.log(.2)})]

	[fwdtest(
	"gaussian sample",
	terra() : double
		return gaussian(0.1, 0.5)
	end,
	0.1)]

	[mhtest(
	"gaussian query",
	pfn(terra() : double
		return gaussian(0.1, 0.5)
	end),
	0.1)]

	[eqtest(
	"gaussian lp",
	{
		rand.gaussian_logprob(double)(0.0, 0.1, 0.5),
		rand.gaussian_logprob(double)(0.25, 0.1, 0.5),
		rand.gaussian_logprob(double)(0.6, 0.1, 0.5),
	},
	{-0.2457913526447274, -0.27079135264472737, -0.7257913526447274})]

	[fwdtest(
	"gamma sample",
	terra() : double
		return gamma(2.0, 2.0)/10.0
	end,
	0.4)]

	[mhtest(
	"gamma query",
	pfn(terra() : double
		return gamma(2.0, 2.0)/10.0
	end),
	0.4)]

	[eqtest(
	"gamma lp",
	{
		rand.gamma_logprob(double)(1.0, 2.0, 2.0),
		rand.gamma_logprob(double)(4.0, 2.0, 2.0),
		rand.gamma_logprob(double)(8.0, 2.0, 2.0)
	},
	{-1.8862944092546166, -2.000000048134726, -3.306852867574781})]

	[fwdtest(
	"beta sample",
	terra() : double
		return beta(2.0, 5.0)
	end,
	2.0/(2+5))]

	[mhtest(
	"beta query",
	pfn(terra() : double
		return beta(2.0, 5.0)
	end),
	2.0/(2+5))]

	[eqtest(
	"beta lp",
	{
		rand.beta_logprob(double)(0.1, 2.0, 5.0),
		rand.beta_logprob(double)(0.2, 2.0, 5.0),
		rand.beta_logprob(double)(0.6, 2.0, 5.0)
	},
	{0.677170196389683, 0.899185234324094, -0.7747911992475776})]

	[fwdtest(
	"binomial sample",
	terra() : double
		return binomial(0.5, 40.0)/40.0
	end,
	0.5)]

	[mhtest(
	"binomial query",
	pfn(terra() : double
		return binomial(0.5, 40.0)/40.0
	end),
	0.5)]

	[eqtest(
	"binomial lp",
	{
		rand.binomial_logprob(double)(15, .5, 40),
		rand.binomial_logprob(double)(20, .5, 40),
		rand.binomial_logprob(double)(30, .5, 40)
	},
	{-3.3234338674089985, -2.0722579911387817, -7.2840211276953575})]

	[fwdtest(
	"poisson sample",
	terra() : double
		return poisson(4.0)/10.0
	end,
	0.4)]

	[mhtest(
	"poisson query",
	pfn(terra() : double
		return poisson(4.0)/10.0
	end),
	0.4)]

	[eqtest(
	"poisson lp",
	{
		rand.poisson_logprob(double)(2, 4),
		rand.poisson_logprob(double)(5, 4),
		rand.poisson_logprob(double)(7, 4)
	},
	{-1.9205584583201643, -1.8560199371825927, -2.821100833226181})]


	-- Tests adapted from Church

	[mhtest(
	"setting a flip",
	pfn(terra() : double
		var a = 1.0/1000
		condition([bool](flip(a)))
		return a
	end),
	1.0/1000)]

	[mhtest(
	"and conditioned on or",
	pfn(terra() : double
		var a = [bool](flip(0.5))
		var b = [bool](flip(0.5))
		condition(a or b)
		return [int](a and b)
	end),
	1.0/3)]

	[mhtest(
	"and conditioned on or, biased flip",
	pfn(terra() : double
		var a = [bool](flip(0.3))
		var b = [bool](flip(0.3))
		condition(a or b)
		return [int](a and b)
	end),
	(0.3*0.3) / (0.3*0.3 + 0.7*0.3 + 0.3*0.7))]

	[mhtest(
	"conditioned flip",
	pfn(terra() : double
		var bitflip = [lambda(pfn(terra(fidelity: double, x: int)
			if [bool](x) then
				return flip(fidelity)
			else
				return flip(1.0-fidelity)
			end
		end))]
		var hyp = flip(0.7)
		condition([bool](bitflip(0.8, hyp)))
		return hyp
	end),
	(0.7*0.8) / (0.7*0.8 + 0.3*0.2))]

	[mhtest(
	"random 'if' with random branches, unconditioned",
	pfn(terra() : double
		if [bool](flip(0.7)) then
			return flip(0.2)
		else
			return flip(0.8)
		end
	end),
	0.7*0.2 + 0.3*0.8)]

	[mhtest(
	"flip with random weight, unconditioned",
	pfn(terra() : double
		var weight: double
		if [bool](flip(0.7)) then
			weight = 0.2
		else
			weight = 0.8
		end
		return flip(weight)
	end),
	0.7*0.2 + 0.3*0.8)]

	[mhtest(
	"random procedure application, unconditioned",
	pfn(terra() : double
		var proc : {} -> {int}
		if [bool](flip(0.7)) then
			proc = [lambda(pfn(terra() return flip(0.2) end))]
		else
			proc = [lambda(pfn(terra() return flip(0.8) end))]
		end
		return proc()
	end),
	0.7*0.2 + 0.3*0.8)]

	[mhtest(
	"conditioned multinomial",
	pfn(terra() : double
		var probs = Vector.fromItems(.1, .6, .3)
		var hyp = multinomial(probs)
		var observe = [lambda(pfn(terra(x: int)
			if [bool](flip(0.8)) then
				return x
			else
				return 0
			end
		end))]
		condition(observe(hyp) == 0)
		m.destruct(probs)
		return [int](hyp == 0)
	end),
	0.357)]

	[mhtest(
	"recursive stochastic fn, unconditioned (tail recursive)",
	pfn(terra() : double
		var a = powerLaw_tailrec(0.3, 1)
		return [int](a < 5)
	end),
	0.7599)]

	[mhtest(
	"recursive stochastic fn, unconditioned",
	pfn(terra() : double
		var a = powerLaw(0.3, 1)
		return [int](a < 5)
	end),
	0.7599)]

	C.printf("tests done!\n")
end

-- local prof = terralib.require("/Users/dritchie/Git/terra/tests/lib/prof")
-- prof.begin()
doTests()
-- prof.finish()






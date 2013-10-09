local rand = terralib.require("prob.random")
local m = terralib.require("mem")
local Vector = terralib.require("vector")
terralib.require("prob")

local C = terralib.includecstring [[
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
inline void flush() { fflush(stdout); }
double currentTimeInSeconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}
void initrand() { srand(time(NULL)); }
]]

C.initrand()

local numsamps = 150
local lag = 20
local larjAnnealSteps = 10
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

local function larjtest(name, computation, trueExpectation)
	local kernel = LARJ(RandomWalk({structural=false}))({intervals=larjAnnealSteps})
	return quote
		var estimates = [Vector(double)].stackAlloc()
		for run=0,runs do
			var samps = [mcmc(computation, kernel, {numsamps=numsamps, lag=lag, verbose=false})]
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

-- Are we doing the nonstructural optimization trick?
-- Turn this on/off to see performance change
local doingNonstructOpt = true
local maybenot = macro(function()
	return not doingNonstructOpt
end)

local terra doTests()

	C.printf("starting tests...\n")

	var t1 = C.currentTimeInSeconds()

	-- -- ERP tests

	[fwdtest(
	"flip sample",
	terra() : double
		return double(flip(0.7, {structural=maybenot()}))
	end,
	0.7)]

	[mhtest(
	"flip query",
	function()
		return terra() : double
			return double(flip(0.7, {structural=maybenot()}))
		end
	end,
	0.7)]

	[fwdtest(
	"uniform sample",
	terra() : double
		return uniform(0.1, 0.4, {structural=maybenot()})
	end,
	0.5*(.1+.4))]

	[mhtest(
	"uniform query",
	function()
		return terra() : double
			return uniform(0.1, 0.4, {structural=maybenot()})
		end
	end,	
	0.5*(.1+.4))]

	[fwdtest(
	"multinomial sample",
	terra() : double
		var items = Vector.fromItems(.2, .3, .4)
		var probs = Vector.fromItems(.2, .6, .2)
		var ret = multinomialDraw(items, probs, {structural=maybenot()})
		m.destruct(items)
		m.destruct(probs)
		return ret
	end,
	0.2*.2 + 0.6*.3 + 0.2*.4)]

	[mhtest(
	"multinomial query",
	function()
		return terra() : double
			var items = Vector.fromItems(.2, .3, .4)
			var probs = Vector.fromItems(.2, .6, .2)
			var ret = multinomialDraw(items, probs, {structural=maybenot()})
			m.destruct(items)
			m.destruct(probs)
			return ret
		end
	end,
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
		return gaussian(0.1, 0.5, {structural=maybenot()})
	end,
	0.1)]

	[mhtest(
	"gaussian query",
	function()
		return terra() : double
			return gaussian(0.1, 0.5, {structural=maybenot()})
		end		
	end,
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
		return gamma(2.0, 2.0, {structural=maybenot()})/10.0
	end,
	0.4)]

	[mhtest(
	"gamma query",
	function()
		return terra() : double
			return gamma(2.0, 2.0, {structural=maybenot()})/10.0
		end
	end,
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
		return beta(2.0, 5.0, {structural=maybenot()})
	end,
	2.0/(2+5))]

	[mhtest(
	"beta query",
	function()
		return terra() : double
			return beta(2.0, 5.0, {structural=maybenot()})
		end
	end,
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
		return binomial(0.5, 40.0, {structural=maybenot()})/40.0
	end,
	0.5)]

	[mhtest(
	"binomial query",
	function()
		return terra() : double
			return binomial(0.5, 40.0, {structural=maybenot()})/40.0
		end
	end,
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
		return poisson(4.0, {structural=maybenot()})/10.0
	end,
	0.4)]

	[mhtest(
	"poisson query",
	function()
		return terra() : double
			return poisson(4.0, {structural=maybenot()})/10.0
		end
	end,
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
	function()
		return terra() : double
			var a = 1.0/1000
			condition(flip(a, {structural=maybenot()}))
			return a
		end
	end,
	1.0/1000)]

	[mhtest(
	"and conditioned on or",
	function()
		return terra() : double
			var a = flip(0.5, {structural=maybenot()})
			var b = flip(0.5, {structural=maybenot()})
			condition(a or b)
			return double(a and b)
		end
	end,
	1.0/3)]

	[mhtest(
	"and conditioned on or, biased flip",
	function()
		return terra() : double
			var a = flip(0.3, {structural=maybenot()})
			var b = flip(0.3, {structural=maybenot()})
			condition(a or b)
			return double(a and b)
		end
	end,
	(0.3*0.3) / (0.3*0.3 + 0.7*0.3 + 0.3*0.7))]

	[mhtest(
	"conditioned flip",
	function()
		local bitflip = pfn(terra(fidelity: double, x: bool)
			var p = fidelity
			if not x then p = 1.0 - fidelity end
			return flip(p, {structural=maybenot()})
		end)
		return terra() : double
			var hyp = flip(0.7, {structural=maybenot()})
			condition(bitflip(0.8, hyp))
			return double(hyp)
		end
	end,
	(0.7*0.8) / (0.7*0.8 + 0.3*0.2))]

	[mhtest(
	"random 'if' with random branches, unconditioned",
	function()
		return terra() : double
			if flip(0.7) then
				return double(flip(0.2, {structural=maybenot()}))
			else
				return double(flip(0.8, {structural=maybenot()}))
			end
		end
	end,
	0.7*0.2 + 0.3*0.8)]

	[mhtest(
	"flip with random weight, unconditioned",
	function()
		return terra() : double
			var weight: double
			if flip(0.7, {structural=maybenot()}) then
				weight = 0.2
			else
				weight = 0.8
			end
			return double(flip(weight, {structural=maybenot()}))
		end
	end,
	0.7*0.2 + 0.3*0.8)]

	[mhtest(
	"random procedure application, unconditioned",
	function()
		local p1 = pfn(terra() return flip(0.2, {structural=maybenot()}) end)
		local p2 = pfn(terra() return flip(0.8, {structural=maybenot()}) end)
		return terra() : double
			if flip(0.7) then
				return double(p1())
			else
				return double(p2())
			end
		end
	end,
	0.7*0.2 + 0.3*0.8)]

	[mhtest(
	"conditioned multinomial",
	function()
		local observe = pfn(terra(x: int)
			if flip(0.8, {structural=maybenot()}) then
				return x
			else
				return 0
			end
		end)
		return terra() : double
			var probs = Vector.fromItems(.1, .6, .3)
			var hyp = multinomial(probs, {structural=maybenot()})
			condition(observe(hyp) == 0)
			m.destruct(probs)
			return double(hyp == 0)
		end
	end,
	0.357)]

	[mhtest(
	"recursive stochastic fn, unconditioned (tail recursive)",
	function()
		local powerLaw = pfn()
		powerLaw:define(terra(prob: double, x: int) : int
			if flip(prob) then
				return x
			else
				return powerLaw(prob, x+1)
			end
		end)
		return terra() : double
			var a = powerLaw(0.3, 1)
			return double(a < 5)
		end
	end,
	0.7599)]

	[mhtest(
	"recursive stochastic fn, unconditioned",
	function()
		local powerLaw = pfn()
		powerLaw:define(terra(prob: double, x: int) : int
			if flip(prob) then
				return x
			else
				return 0 + powerLaw(prob, x+1)
			end
		end)
		return terra() : double
			var a = powerLaw(0.3, 1)
			return double(a < 5)
		end
	end,
	0.7599)]

	[mhtest(
	"memoized flip, unconditioned",
	function()
		return terra() : double
			var proc = [mem(pfn(terra(x: int) return flip(0.8, {structural=maybenot()}) end))]
			var p11 = proc(1)	
			var p21 = proc(2)
			var p12 = proc(1)
			var p22 = proc(2)
			m.destruct(proc)
			return double(p11 and p21 and p22)
		end
	end,
	0.64)]

	[mhtest(
	"memoized flip, conditioned",
	function()
		return terra() : double
			var proc = [mem(pfn(terra(x: int) return flip(0.2, {structural=maybenot()}) end))]
			var p1 = proc(1)	
			var p21 = proc(2)
			var p22 = proc(2)
			var p23 = proc(2)
			condition(p1 or p21 or p22 or p23)
			var ret = double(proc(1))
			m.destruct(proc)
			return ret
		end
	end,
	0.5555555555555555)]

	-- Not quite equivalent to the Church test, but that version is not
	-- expressable in this language (lack of proper closures)
	[mhtest(
	"bound symbol used inside memoizer, unconditioned",
	function()
		return terra() : double
			var a = flip(0.8, {structural=maybenot()})
			var proc = [mem(pfn(terra(a: bool) return a end))]
			var p11 = proc(a)
			var p12 = proc(a)
			m.destruct(proc)
			return double(p11 and p12)
		end
	end,
	0.8)]

	[mhtest(
	"memoized flip with random argument, unconditioned",
	function()
		return terra() : double
			var proc = [mem(pfn(terra(x: int) return flip(0.8, {structural=maybenot()}) end))]
			var items = Vector.fromItems(1, 2, 3)
			var p1 = proc(uniformDraw(items))
			var p2 = proc(uniformDraw(items))
			m.destruct(items)
			m.destruct(proc)
			return double(p1 and p2)
		end
	end,
	0.6933333333333334)]

	[mhtest(
	"memoized random procedure, unconditioned",
	function()
		return terra() : double
			var proc1 = [mem(pfn(terra(x: int) return flip(0.2, {structural=maybenot()}) end))]
			var proc2 = [mem(pfn(terra(x: int) return flip(0.8, {structural=maybenot()}) end))]
			var mp1: bool
			var mp2: bool
			if flip(0.7) then
				mp1 = proc1(1)
				mp2 = proc1(2)
			else
				mp1 = proc2(1)
				mp2 = proc2(2)
			end
			m.destruct(proc1)
			m.destruct(proc2)
			return double(mp1 and mp2)
		end
	end,
	0.22)]

	[mhtest(
	"mh query over rejection query for conditioned flip",
	function()
		local bitflip = pfn(terra(fidelity: double, x: bool)
			var p = fidelity
			if not x then p = 1.0 - fidelity end
			return flip(p, {structural=maybenot()})
		end)
		return terra() : double
			return [rejectionSample(function()
				return terra()
					var a = flip(0.7, {structural=maybenot()})
					condition(bitflip(0.8, a))
					return double(a)
				end
			end)]
		end
	end,
	0.903225806451613)]

	[mhtest(
	"trans-dimensional",
	function()
		return terra() : double
			var a = 0.7
			if flip(0.9) then
				a = beta(1, 5, {structural=maybenot()})
			end
			var b = flip(a, {structural=maybenot()})
			condition(b)
			return a
		end
	end,
	0.417)]

	[larjtest(
	"trans-dimensional (LARJ)",
	function()
		return terra() : double
			var a = 0.7
			if flip(0.9) then
				a = beta(1, 5, {structural=maybenot()})
			end
			var b = flip(a, {structural=maybenot()})
			condition(b)
			return a
		end
	end,
	0.417)]

	[mhtest(
	"memoized flip in if branch (create/destroy memprocs), unconditioned",
	function()
		local nflip = terra(x: double) return flip(x, {structural=maybenot()}) end
		return terra() : double
			var a = [mem(nflip)]
			if flip(0.5, {structural=maybenot()}) then
				m.destruct(a)
				a = [mem(nflip)]
			end
			var b = a(0.5)
			m.destruct(a)
			return double(b)
		end
	end,
	0.5)]


	-- Tests for things specific to new implementation

	[mhtest(
	"native loop",
	function()
		return terra() : double
			var accum = 0
			for i=0,4 do
				accum = accum + int(flip(0.5, {structural=maybenot()}))
			end
			return accum / 4.0
		end
	end,
	0.5)]

	[mhtest(
	"directly conditioning variable value",
	function()
		return terra() : double
			var accum = 0
			for i=0,10 do
				if i < 5 then
					accum = accum + int(flip(0.5, {structural=maybenot(), constrainTo=true}))
				else
					accum = accum + int(flip(0.5, {structural=maybenot()}))
				end
			end
			return accum / 10.0
		end
	end,
	0.75)]


	var t2 = C.currentTimeInSeconds()

	C.printf("tests done!\n")
	C.printf("time: %g\n", t2 - t1)

end

-- local prof = terralib.require("/Users/dritchie/Git/terra/tests/lib/prof")
doTests()
-- prof.begin()
doTests()
-- prof.finish()






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

local str = &int8

local numsamps = 150
local lag = 20
local runs = 5
local errorTolerance = 0.007

local terra test(name: str, estimates: &Vector(double), trueExpectation: double)
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
			var samps = [mcmc(computation, RandomWalk(), {numsamps=numsamps, lag=lag, verbose=true})]
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
			end
		end
		C.printf("passed.\n")
		m.destruct(estvalues)
		m.destruct(truevalues)
	end
end

local terra doTests()

	-- [fwdtest(
	-- "flip sample",
	-- terra() : double
 --    	return flip(0.7)
	-- end,
	-- 0.7)]

	[mhtest(
	"flip query",
	pfn(terra() : double
		return flip(0.7)
	end),
	0.7)]

	-- [fwdtest(
	-- "uniform sample",
	-- terra() : double
 --    	return uniform(0.1, 0.4)
	-- end,
	-- 0.5*(.1+.4))]

	-- [mhtest(
	-- "uniform query",
	-- terra() : double
 --    	return uniform(0.1, 0.4)
	-- end,
	-- 0.5*(.1+.4))]

	-- [fwdtest(
	-- "multinomial sample",
	-- terra() : double
	-- 	var items = Vector.fromItems(.2, .3, .4)
	-- 	var probs = Vector.fromItems(.2, .6, .2)
 --    	var ret = multinomialDraw(items, probs)
 --    	m.destruct(items)
 --    	m.destruct(probs)
 --    	return ret
	-- end,
	-- 0.2*.2 + 0.6*.3 + 0.2*.4)]

	-- [mhtest(
	-- "multinomial query",
	-- terra() : double
	-- 	var items = Vector.fromItems(.2, .3, .4)
	-- 	var probs = Vector.fromItems(.2, .6, .2)
 --    	var ret = multinomialDraw(items, probs)
 --    	m.destruct(items)
 --    	m.destruct(probs)
 --    	return ret
	-- end,
	-- 0.2*.2 + 0.6*.3 + 0.2*.4)]

end

-- local prof = terralib.require("/Users/dritchie/Git/terra/tests/lib/prof")
-- prof.begin()
doTests()
-- prof.finish()





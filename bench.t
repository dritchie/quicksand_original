terralib.require("prob")
local m = terralib.require("mem")


local C = terralib.includecstring [[
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
inline void flush() { fflush(stdout); }
inline double currentTimeInSeconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}
]]

local numsamps = 150
local lag = 20

local function run(computation)
	return quote
		var samps = [mcmc(computation, RandomWalk(), {numsamps=numsamps, lag=lag, verbose=true})]
		m.destruct(samps)
	end
end


-- Are we doing the nonstructural optimization trick?
-- Turn this on/off to see performance change
local doingNonstructOpt = true
local maybenot = macro(function()
	return not doingNonstructOpt
end)


local function genbenchmarks()
	return terra()
		-- -- Flat model with a huge number of variables
		-- [run(function()
		-- 	return terra()
		-- 		var counter = 0
		-- 		for i=0,5000 do
		-- 			counter = counter + int(flip(0.5, {structural=maybenot()}))
		-- 		end
		-- 		return counter
		-- 	end
		-- end)]
		
		-- Highly recursive model
		[run(function()
			local rec = pfn()
			rec:define(terra(depth: int) : int
				if depth < 1000 then
					return int(flip(0.5, {structural=maybenot()})) + rec(depth+1)
				else
					return 0.0
				end
			end)
			return terra()
				var sum = rec(0)
				condition(sum == 500)
				return sum
			end
		end)]
	end
end

doingNonstructOpt = false
print("Without nonstruct optimization:")
local dobenchmarks = genbenchmarks()
dobenchmarks()
doingNonstructOpt = true
print("With nonstruct optimization:")
dobenchmarks = genbenchmarks()
dobenchmarks()
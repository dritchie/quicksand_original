local prob = terralib.require("prob")
local util = terralib.require("util")
util.openModule(prob)
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
local function maybenot()
	return not doingNonstructOpt
end


local function genbenchmarks()
	return terra()
		var t1 = C.currentTimeInSeconds()

		-- Flat model with a huge number of variables
		[run(function()
			return terra()
				var counter = 0
				for i=0,5000 do
					counter = counter + int([flip(0.5, {structural=maybenot()})])
				end
				return counter
			end
		end)]

		var t2 = C.currentTimeInSeconds()
		C.printf("Done! Time: %g\n", t2-t1)
	end
end

doingNonstructOpt = false
print("Without nonstruct optimization:")
local dobenchmarks = genbenchmarks()
dobenchmarks()
dobenchmarks()
doingNonstructOpt = true
print("With nonstruct optimization")
dobenchmarks = genbenchmarks()
dobenchmarks()
dobenchmarks()
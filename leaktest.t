terralib.require("prob")

local Vector = terralib.require("vector")
local AutoPtr = terralib.require("autopointer")


local comp = probcomp(function()
	local numelems = 100000
	return terra()
		var x = gaussian(0.0, 1.0, {structural=false})
		return [Vector(double)].stackAlloc(numelems, x)
		-- return AutoPtr.wrap([Vector(double)].heapAlloc(numelems, x))
	end
end)

local numsamps = 10000
local lag = 1

-- local go = forwardSample(comp, numsamps)
local go = mcmc(comp, GaussianDrift(), {numsamps=numsamps, lag=lag, verbose=true})

local samps = go()
samps:__destruct()
while true do
	--
end
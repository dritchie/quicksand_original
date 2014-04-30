terralib.require("prob")


local testcomp = probcomp(function()
	return terra() : real
		return gaussian(0.1, 0.5, {structural=false})
	end
end)

-- Works fine
local sampler = mcmc(testcomp, HMC({numSteps=1000}), {numsamps=100, verbose=true})
sampler()

-- Segfaults on return
-- mcmc(testcomp, HMC({numSteps=1000}), {numsamps=100, verbose=true})()
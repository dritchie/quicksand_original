local m = terralib.require("mem")
local ad = terralib.require("ad")

-- Univariate dual-averaging optimization (for HMC step size adaptation)
-- Adapted from Stan
local struct DualAverage
{
	gbar: double,
	xbar: double,
	x0: double,
	lastx: double,
	k: int,
	gamma: double,
	adapting: bool,
	minChange: double
}

terra DualAverage:__construct(x0: double, gamma: double, minChange: double) : {}
	self.k = 0
	self.x0 = x0
	self.lastx = x0
	self.gbar = 0.0
	self.xbar = 0.0
	self.gamma = gamma
	self.adapting = true
	self.minChange = minChange
end

terra DualAverage:__construct(x0: double, gamma: double) : {}
	DualAverage.__construct(self, x0, gamma, 0.0001)
end

terra DualAverage:update(g: double)
	if self.adapting then
		self.k = self.k + 1
		var avgeta = 1.0 / (self.k + 10)
		var xbar_avgeta = ad.math.pow(self.k, -0.75)
		var muk = 0.5 * ad.math.sqrt(self.k) / self.gamma
		self.gbar = avgeta*g + (1-avgeta)*self.gbar
		self.lastx = self.x0 - muk*self.gbar
		-- var oldxbar = self.xbar
		-- self.xbar = xbar_avgeta*self.lastx + (1-xbar_avgeta)*self.xbar
		-- if ad.math.fabs(oldxbar - self.xbar) < self.minChange then
		-- 	self.adapting = false
		-- end
	end
	-- return self.xbar
	return self.lastx
end

m.addConstructors(DualAverage)

return DualAverage
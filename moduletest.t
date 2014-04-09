terralib.require("prob")
local m = terralib.require("mem")

-- TEST 1: Use a type from a probmodule
local function test1()
	local mod = probmodule(function(pcomp)
		local struct Foo { x: real }
		terra Foo:setGaussianX(mu: double, sigma: double)
			self.x = gaussian(mu, sigma)
		end
		Foo.methods.setGaussianX = pmethod(Foo.methods.setGaussianX)
		return { Foo = Foo }
	end)

	-- local comp = probcomp(function() return 42 end); comp() 	-- This should error
	-- local comp = probcomp(function() return terra(x: int) return x + 1 end end); comp() 	-- This should error

	local comp = probcomp(function()
		local Foo = mod().Foo
		return terra()
			var foo : Foo
			foo:setGaussianX(0.0, 1.0)
			return foo.x
		end
	end)

	-- Refer to type outside of comp, and before comp has been compiled
	--    (or even specialized)
	local mod_ = mod(comp)
	-- local mod_ = mod()	-- This should error
	-- local mod_ = mod(function() return terra() end end)	-- This should also error
	-- local mod_ = mod(probcomp(function() return terra() end end))	-- This should also error
	local Foo = mod_.Foo
	local terra useFoo()
		var foo : Foo
		foo:setGaussianX(0.0, 1.0)
		return foo.x
	end
	useFoo:compile()
	useFoo()

	local go = mcmc(comp, RandomWalk(), {numsamps=1000, verbose=true})
	go:compile()
	go()
end
test1()


-- TEST 2: Use code from two different probmodules
local function test2()
	local mod1 = probmodule(function(pcomp)
		local struct Foo { x: real }
		terra Foo:setGaussianX(mu: double, sigma: double)
			self.x = gaussian(mu, sigma)
		end
		Foo.methods.setGaussianX = pmethod(Foo.methods.setGaussianX)
		return { Foo = Foo }
	end)

	local mod2 = probmodule(function(pcomp)
		local struct Foo { x: real }
		terra Foo:setGaussianX(mu: double, sigma: double)
			self.x = gaussian(mu, sigma)
		end
		Foo.methods.setGaussianX = pmethod(Foo.methods.setGaussianX)
		return { Foo = Foo }
	end)

	local comp = probcomp(function()
		local Foo1 = mod1().Foo
		local Foo2 = mod2().Foo
		return terra()
			var foo1 : Foo1
			foo1:setGaussianX(0.0, 1.0)
			var foo2 : Foo2
			foo2:setGaussianX(0.0, 1.0)
			return foo1.x + foo2.x
		end
	end)

	-- Refer to type outside of comp, and before comp has been compiled
	--    (or even specialized)
	local Foo1 = mod1(comp).Foo
	local Foo2 = mod2(comp).Foo
	local terra useFoo()
		var foo1 : Foo1
		foo1:setGaussianX(0.0, 1.0)
		var foo2 : Foo2
		foo2:setGaussianX(0.0, 1.0)
		return foo1.x + foo2.x
	end
	useFoo:compile()
	useFoo()

	local go = mcmc(comp, RandomWalk(), {numsamps=1000, verbose=true})
	go:compile()
	go()
end
test2()


-- TEST 3: Use code from nested probmodules
-- (Also check that it's working with HMC / dual num compilation)
local function test3()
	local mod1 = probmodule(function(pcomp)
		local struct Foo { x: real }
		terra Foo:setGaussianX(mu: double, sigma: double)
			self.x = gaussian(mu, sigma, {structural=false})
		end
		Foo.methods.setGaussianX = pmethod(Foo.methods.setGaussianX)
		return { Foo = Foo }
	end)

	local mod2 = probmodule(function(pcomp)
		local Foo1 = mod1(pcomp).Foo
		local struct Foo { x: real }
		terra Foo:setGaussianX(mu: double, sigma: double)
			self.x = gaussian(mu, sigma, {structural=false})
		end
		Foo.methods.setGaussianX = pmethod(Foo.methods.setGaussianX)
		local addRandomFoos = pfn(terra(f: Foo, f1: Foo1)
			f:setGaussianX(0.0, 1.0)
			f1:setGaussianX(0.0, 1.0)
			return f.x + f1.x
		end)
		return { Foo = Foo, addRandomFoos = addRandomFoos }
	end)

	local comp = probcomp(function()
		local Foo1 = mod1().Foo
		local Foo2 = mod2().Foo
		local addRandomFoos = mod2().addRandomFoos
		return terra()
			var foo1 : Foo1
			var foo2 : Foo2
			return addRandomFoos(foo2, foo1)
		end
	end)

	-- Refer to type outside of comp, and before comp has been compiled
	--    (or even specialized)
	local Foo1 = mod1(comp).Foo
	local Foo2 = mod2(comp).Foo
	local addRandomFoos = mod2(comp).addRandomFoos
	local terra useFoo()
		var foo1 : Foo1
		var foo2 : Foo2
		return addRandomFoos(foo2, foo1)
	end
	useFoo:compile()
	useFoo()

	-- local go = mcmc(comp, RandomWalk(), {numsamps=1000, verbose=true})
	local go = mcmc(comp, HMC({numSteps=1}), {numsamps=1000, verbose=true})
	go:compile()
	go()
end
test3()

-- TEST 4: Verify intended type identities
local function test4()
	local mod = probmodule(function(pcomp)
		local struct Foo { x: real }
		terra Foo:setGaussianX(mu: double, sigma: double)
			self.x = gaussian(mu, sigma)
		end
		Foo.methods.setGaussianX = pmethod(Foo.methods.setGaussianX)
		return { Foo = Foo }
	end)

	local comp = probcomp(function()
		local Foo = mod().Foo
		return terra()
			var foo : Foo
			foo:setGaussianX(0.0, 1.0)
			return foo
		end
	end)

	local go = mcmc(comp, RandomWalk(), {numsamps=1000, verbose=true})
	local samps = m.gc(go())

	-- Process resulting samples to verify that types are identical
	local Foo = mod(comp).Foo
	local terra process(samples: &SampleVectorType(comp))
		var firstSamp : Foo
		firstSamp = samples(0).value	-- The critical part
		return firstSamp
	end
	process(samps)
end
test4()










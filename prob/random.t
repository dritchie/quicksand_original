
local util = terralib.require("util")
local templatize = terralib.require("templatize")
local Vector = terralib.require("vector")
local ad = terralib.require("ad")

-- Base RNG
local random = util.inline(terralib.includecstring([[
	#include <stdlib.h>
	double random_() { return rand() / (RAND_MAX+1.0); }
]]).random_)


-- Samplers/scorers

local flip_sample = templatize(function(p: T)
	var randval = random()
	if randval < p then
		return 1
	else
		return 0
	end
end)

local flip_logprob = templatize(function(T)
	return terra(val: int, p: T2)
		var prob: T
		if val ~= 0 then
			prob = p
		else
			prob = 1.0 - p
		end
		return ad.math.log(prob)
	end
end)

local multinomial_sample = templatize(function(T)
	return terra(params: Vector(T))
		var sum = T(0.0)
		for i=0,params.size do sum = sum + params:get(i) end
		var result: int = 0
		var x = random() * sum
		var probAccum = 0.00000001
		while result <= n and x > probAccum do
			probAccum = probAccum + params:get(result)
			result = result + 1
		end
		return result - 1
	end
end)

local multinomial_logprob = templatize(function(T)
	return terra(val: int, params: Vector(T))
		if val < 0 or val >= params.size then
			return [-math.huge]
		end
		var sum = T(0.0)
		for i=0,params.size do
			sum = sum + params:get(i)
		end
		return ad.math.log(params:get(val)/sum)
	end
end)


local uniform_sample = templatize(function(T1, T2)
	return terra(lo: T1, hi: T2)
		var u = random()
		return (1.0-u)*lo + u*hi
	end
end)

local uniform_logprob = templatize(function(T1, T2, T3)
	return terra(val: T1, lo: T2, hi: T3)
		if val < lo or val > hi then return [-math.huge] end
		return -ad.math.log(hi - lo)
	end
end)


local gaussian_sample = templatize(function(T1, T2)
	return terra(mu: T1, sigma: T2)
		var u:double, v:double, x:double, y:double, q:double
		repeat
			u = 1.0 - random()
			v = 1.7156 * (random() - 0.5)
			x = u - 0.449871
			y = ad.math.fabs(v) + 0.386595
			q = x*x + y*(0.196*y - 0.25472*x)
		until not(q >= 0.27597 and (q > 0.27846 or v*v > -4 * u * u * ad.math.log(u)))
		return mu + sigma*v/u
	end
end)

local gaussian_logprob = templatize(function(T1, T2, T3)
	return terra(x: T1, mu: T2, sigma: T3)
		var xminusmu = x - mu
		return -.5*(1.8378770664093453 + 2*ad.math.log(sigma) + xminusmu*xminusmu/(sigma*sigma))
	end
end)

local gamma_sample = templatize(function(T1, T2)
	local terra sample(a: T1, b: T2)
		if a < 1.0 then return sample(1.0+a,b) * ad.math.pow(random(), 1.0/a) end
		var x:double, v:T1, u:double
		var d = a - 1.0/3.0
		var c = 1.0/ad.math.sqrt(9.0*d)
		while true do
			repeat
				x = gaussian_sample.implicit(0.0, 1.0)
				v = 1.0+c*x
			until v > 0.0
			v = v*v*v
			u = random()
			if (u < 1.0 - .331*x*x*x*x) or (ad.math.log(u) < .5*x*x + d*(1.0 - v + ad.math.log(v))) then
				return b*d*v
			end
		end
	end
	return sample
end)

local gamma_cof = global(double[6], {76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5})
local log_gamma = templatize(function(T)
	return terra(xx: T)
		var x = xx - 1.0
		var tmp = x + 5.5
		tmp = tmp - (x + 0.5)*ad.math.log(tmp)
		var ser = T(1.000000000190015)
		for j=0,5 do
			x = x + 1.0
			ser = ser + gamma_cof[j] / x
		end
		return -tmp + ad.math.log(2.5066282746310005*ser)
	end
end)

local gamma_logprob = templatize(function(T1, T2, T3)
	return terra(x: T1, a: T2, b:T3)
		return (a - 1.0)*ad.math.log(x) - x/b - log_gamma.implicit(a) - a*ad.math.log(b)
	end
end)


local beta_sample = templatize(function(T1, T2)
	return terra(a: T1, b: T2)
		var x = gamma_sample.implicit(a, 1)
		return x / (x + gamma_sample.implicit(b, 1))
	end
end)

local log_beta = templatize(function(T1, T2)
	return terra(a: T1, b: T2)
		return log_gamma.implicit(a) + log_gamma.implicit(b) - log_gamma.implicit(a+b)
	end
end)

local beta_logprob = templatize(function(T1, T2, T3)
	return terra(x: T1, a: T2, b: T3)
		if x > 0.0 and x < 1.0 then
			return (a-1.0)*ad.math.log(x) + (b-1.0)*ad.math.log(1.0-x) - log_beta.implicit(a,b)
		else
			return [-math.huge]
		end
	end
end)


local binomial_sample = templatize(function(T)
	return terra (p: T, n: int) : int
		var k = 0
		var N = 10
		var a:int, b:int
		while n > N do
			a = 1 + n/2
			b = 1 + n-a
			var x = beta_sample.implicit(T(a), T(b))
			if x >= p then
				n = a - 1
				p = p / x
			else
				k = k + a
				n = b - 1
				p = (p-x) / (1.0 - x)
			end
		end
		var u:double
		for i=0,n do
			u = random()
			if u < p then k = k + 1 end
		end
		return k
	end
end)

local g = templatize(function(T)
	return terra(x: T)
		if x == 0.0 then return 1.0 end
		if x == 1.0 then return 0.0 end
		var d = 1.0 - x
		return (1.0 - (x * x) + (2.0 * x * ad.math.log(x))) / (d * d)
	end
end)

local binomial_logprob = templatize(function(T)
	local inv2 = 1/2
	local inv3 = 1/3
	local inv6 = 1/6
	return terra(s: int, p: T, n: int)
		if s >= n then return [-math.huge] end
		var q = 1.0-p
		var S = s + inv2
		var T = n - s - inv2
		var d1 = s + inv6 - (n + inv3) * p
		var d2 = q/(s+inv2) - p/(T+inv2) + (q-inv2)/(n+1)
		d2 = d1 + 0.02*d2
		var num = 1.0 + q * fns.g(S/(n*p)) + p * fns.g(T/(n*q))
		var den = (n + inv6) * p * q
		var z = num / den
		var invsd = ad.math.sqrt(z)
		z = d2 * invsd
		return gaussian_logprob.implicit(z, 0.0, 1.0) + ad.math.log(invsd)
	end
end)


-- Do we need to templatize this to be in line with all the other functions?
local terra poisson_sample(mu: int)
	var k:int = 0
	while mu > 10 do
		var m = (7.0/8)*mu
		var x = gamma_sample.implicit(m, 1.0)
		if x > mu then
			-- This implicit should work, despite binomial_sample technically only needing
			-- the first template parameter
			return k + binomial_sample.implicit(mu/x, m-1)
		else
			mu = mu - x
			k = k + 1
		end
	end
	var emu = ad.math.exp(-mu)
	var p = 1.0
	while p > emu do
		p = p * random()
		k = k + 1
	end
	return k-1
end


local terra fact(x: int)
	var t:int = 1
	while x > 1 do
		t = t * x
		x = x - 1
	end
	return t	
end


local terra lnfact(x: int)
	if x < 1 then x = 1 end
	if x < 12 then return ad.math.log(fact(x)) end
	var invx = 1.0 / x
	var invx2 = invx*invx
	var invx3 = invx2*invx
	var invx5 = invx3*invx2
	var invx7 = invx5*invx2
	var ssum = ((x + 0.5) * ad.math.log(x)) - x
	ssum = ssum + ad.math.log(2*[math.pi]) / 2.0
	ssum = ssum + (invx / 12) - (invx / 360)
	ssum = ssum + (invx5 / 1260) - (invx7 / 1680)
	return ssum
end


local terra poisson_logprob(k: int, mu: int)
	return k * ad.math.log(mu) - mu - lnfact(k)
end


local dirichlet_sample = templatize(function(T)
	return terra(params: Vector(T))
		var result = [Vector(T)].stackAlloc(params.size, T(0.0))
		var ssum = T(0.0)
		for i=0,params.size do
			var t = gamma_sample.implicit(params:get(i), 1.0)
			result:set(i, t)
			ssum = ssum + t
		end
		for i=0,result.size do
			result:set(i, result:get(i)/ssum)
		end
		return result
	end
end)

local dirichlet_logprob = templatize(function(T1, T2)
	return terra logprob(theta: Vector(T1), params: Vector(T2))
		var sum = T2(0.0)
		for i=0,params.size do sum = sum + params:get(i) end
		var logp = log_gamma.implicit(sum)
		for i=0,params.size do
			var a = params:get(i)
			logp = logp + (a - 1.0)*ad.math.log(theta:get(i))
			logp = logp - log_gamma.implicit(a)
		end
		return logp
	end
end)


-- Module exports
return
{
	flip_sample = flip_sample,
	flip_logprob = flip_logprob,
	multinomial_sample = multinomial_sample,
	multinomial_logprob = multinomial_logprob,
	uniform_sample = uniform_sample,
	uniform_logprob = uniform_logprob,
	gaussian_sample = gaussian_sample,
	gaussian_logprob = gaussian_logprob,
	gamma_sample = gamma_sample,
	gamma_logprob = gamma_logprob,
	beta_sample = beta_sample,
	beta_logprob = beta_logprob,
	binomial_sample = binomial_sample,
	binomial_logprob = binomial_logprob,
	poisson_sample = poisson_sample,
	poisson_logprob = poisson_logprob,
	dirichlet_sample = dirichlet_sample,
	dirichlet_logprob = dirichlet_logprob
}





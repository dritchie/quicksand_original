
local util = terralib.require("util")
local templatize = terralib.require("templatize")
local Vector = terralib.require("vector")
local ad = terralib.require("ad")
local erph = terralib.require("prob.erph")

-- Base RNG
local random = terralib.includecstring([[
	#include <stdlib.h>
	double random_() { return rand() / (RAND_MAX+1.0); }
]]).random_


-- Turning templated functions into overloaded ones
local fns = {}
local function specialize(name, numparams, fntemplate)
	-- Templatize on the value type
	fns[name] = erph.overloadOnParams(numparams, fntemplate)
end


fns.random = random


-- Samplers/scorers

specialize("flip_sample", 1, function(V, P)
	return terra(p: P)
		var randval = random()
		return (randval < p)
	end
end)

specialize("flip_logprob", 1, function(V, P)
	return terra(val: bool, p: P)
		var prob: P
		if val then
			prob = p
		else
			prob = 1.0 - p
		end
		return ad.math.log(prob)
	end
end)

specialize("multinomial_sample", 1, function(V, P)
	return terra(params: Vector(P))
		var sum = P(0.0)
		for i=0,params.size do sum = sum + params:get(i) end
		var result: int = 0
		var x = random() * sum
		var probAccum = P(0.0)
		repeat
			probAccum = probAccum + params:get(result)
			result = result + 1
		until probAccum > x or result > params.size
		return result - 1
	end
end)

specialize("multinomial_logprob", 1, function(V, P)
	return terra(val: int, params: Vector(P))
		if val < 0 or val >= params.size then
			return [-math.huge]
		end
		var sum = P(0.0)
		for i=0,params.size do
			sum = sum + params:get(i)
		end
		return ad.math.log(params:get(val)/sum)
	end
end)

specialize("uniform_sample", 2, function(V, P1, P2)
	return terra(lo: P1, hi: P2)
		var u = random()
		return V((1.0-u)*lo + u*hi)
	end
end)

specialize("uniform_logprob", 2, function(V, P1, P2)
	return terra(val: V, lo: P1, hi: P2)
		if val < lo or val > hi then return [-math.huge] end
		return -ad.math.log(hi - lo)
	end
end)

terra fns.uniformRandomInt(lo: uint, hi: uint)
	return [uint]([fns.uniform_sample(double)](lo, hi))
end

specialize("gaussian_sample", 2, function(V, P1, P2)
	return terra(mu: P1, sigma: P2)
		var u:double, v:double, x:double, y:double, q:double
		repeat
			u = 1.0 - random()
			v = 1.7156 * (random() - 0.5)
			x = u - 0.449871
			y = ad.math.fabs(v) + 0.386595
			q = x*x + y*(0.196*y - 0.25472*x)
		until not(q >= 0.27597 and (q > 0.27846 or v*v > -4 * u * u * ad.math.log(u)))
		return V(mu + sigma*v/u)
	end
end)

specialize("gaussian_logprob", 2, function(V, P1, P2)
	return terra(x: V, mu: P1, sigma: P2)
		var xminusmu = x - mu
		return -.5*(1.8378770664093453 + 2*ad.math.log(sigma) + xminusmu*xminusmu/(sigma*sigma))
	end
end)

specialize("gamma_sample", 2, function(V, P1, P2)
	local terra sample(a: P1, b: P2) : V
		if a < 1.0 then return V(sample(1.0+a,b) * ad.math.pow(random(), 1.0/a)) end
		var x:double, v:V, u:double
		var d = a - 1.0/3.0
		var c = 1.0/ad.math.sqrt(9.0*d)
		while true do
			repeat
				x = [fns.gaussian_sample(V)](0.0, 1.0)
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

local gamma_cof = global(double[6])
local terra init_gamma_cof()
	gamma_cof = array(76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5)
end
init_gamma_cof()
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

specialize("gamma_logprob", 2, function(V, P1, P2)
	return terra(x: V, a: P1, b: P2)
		return (a - 1.0)*ad.math.log(x) - x/b - log_gamma.implicit(a) - a*ad.math.log(b)
	end
end)

specialize("beta_sample", 2, function(V, P1, P2)
	return terra(a: P1, b: P2)
		var x = [fns.gamma_sample(V)](a, 1)
		return V(x / (x + [fns.gamma_sample(V)](b, 1)))
	end
end)

local log_beta = templatize(function(T1, T2)
	return terra(a: T1, b: T2)
		return log_gamma.implicit(a) + log_gamma.implicit(b) - log_gamma.implicit(a+b)
	end
end)

specialize("beta_logprob", 2, function(V, P1, P2)
	return terra(x: V, a: P1, b: P2)
		if x > 0.0 and x < 1.0 then
			return (a-1.0)*ad.math.log(x) + (b-1.0)*ad.math.log(1.0-x) - log_beta.implicit(a,b)
		else
			return [-math.huge]
		end
	end
end)

specialize("binomial_sample", 1, function(V, P)
	return terra (p: P, n: int) : int
		var k = 0
		var N = 10
		var a:int, b:int
		while n > N do
			a = 1 + n/2
			b = 1 + n-a
			var x = [fns.beta_sample(V)](P(a), P(b))
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

specialize("binomial_logprob", 1, function(V, P)
	local inv2 = 1/2
	local inv3 = 1/3
	local inv6 = 1/6
	return terra(s: int, p: P, n: int)
		if s >= n then return [-math.huge] end
		var q = 1.0-p
		var S = s + inv2
		var T = n - s - inv2
		var d1 = s + inv6 - (n + inv3) * p
		var d2 = q/(s+inv2) - p/(T+inv2) + (q-inv2)/(n+1)
		d2 = d1 + 0.02*d2
		var num = 1.0 + q * g.implicit(S/(n*p)) + p * g.implicit(T/(n*q))
		var den = (n + inv6) * p * q
		var z = num / den
		var invsd = ad.math.sqrt(z)
		z = d2 * invsd
		return [fns.gaussian_logprob(V)](z, 0.0, 1.0) + ad.math.log(invsd)
	end
end)

specialize("poisson_sample", 0, function(V)
	return terra(mu: int)
		var k:int = 0
		while mu > 10 do
			var m = (7.0/8)*mu
			var x = [fns.gamma_sample(V)](m, 1.0)
			if x > mu then
				return k + [fns.binomial_sample(V)](mu/x, m-1)
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
end)

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

specialize("poisson_logprob", 0, function(V)
	return terra(k: int, mu: int)
		return k * ad.math.log(mu) - mu - lnfact(k)
	end	
end)

specialize("dirichlet_sample", 1, function(V, P)
	return terra(params: Vector(P))
		var result = [Vector(V)].stackAlloc(params.size, V(0.0))
		var ssum = V(0.0)
		for i=0,params.size do
			var t = [fns.gamma_sample(V)](params:get(i), 1.0)
			result:set(i, t)
			ssum = ssum + t
		end
		for i=0,result.size do
			result:set(i, result:get(i)/ssum)
		end
		return result
	end
end)

specialize("dirichlet_logprob", 1, function(V, P)
	return terra(theta: Vector(V), params: Vector(P))
		var sum = P(0.0)
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
return fns






local C = terralib.includec("stdio.h")
local Vector = terralib.require("vector")
local m = terralib.require("mem")
local ffi = require("ffi")

local callsiteStack = global(Vector(int))
local terra initGlobals()
	m.init(callsiteStack)
end
initGlobals()

-- Elides a lot of checks, doesn't do recursion, and doesn't do
--    functions with 0 return values.

-- Wrap Terra function 'fn' in probabilistic function exotype
local id = 0
local function pfn(fn)
	
	-- Exotype declaration
	local Pfn = terralib.types.newstruct()

	-- Wrap function call with macro that manages address stack
	Pfn.metamethods.__apply = macro(function(self, ...)
		id = id + 1
		local args = {...}
		local argIntermediates = {}
		local results = {}
		for _,a in ipairs(args) do table.insert(argIntermediates, symbol(a:gettype())) end
		for _,t in ipairs(fn:gettype().returns) do table.insert(results, symbol(t)) end
		return quote
			var [argIntermediates] = [args]
			-- C.printf("callsite #%u\n", id)
			callsiteStack:push(id)
			var [results] = fn([argIntermediates])
			callsiteStack:pop()
		in
			[results]
		end
	end)

	-- Forward all other methods to Terra function fn
	Pfn.metamethods.__getmethod = function(self, methodname)
		return function(...)
			return fn[methodname](fn, ...)
		end
	end

	return terralib.new(Pfn)
end

------------------------------------

local incr = pfn(terra(x: int) return x + 1 end)

print(incr:gettype())

terra test()
	var x = incr(42)
	C.printf("%d\n", x)
end
test()
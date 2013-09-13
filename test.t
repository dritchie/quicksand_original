local C = terralib.includec("stdio.h")

------------------------------

-- local pwhile = macro(function(cond, body)
-- 	return quote
-- 		while cond do
-- 			body
-- 		end
-- 	end
-- end)

-- local terra testwhile()
-- 	var i = 0
-- 	pwhile(i < 10,
-- 	[quote
-- 		C.printf("%d\n", i)
-- 		i = i + 1
-- 	end]
-- 	)
-- end

-- testwhile()

------------------------------

local function forWithStep(indexvar, initval, finalval, step, body)
	local loopindex = symbol(int)
	return quote
		for [loopindex]=initval,finalval,step do
			indexvar = [loopindex]
			body
		end
	end
end

local pfor = macro(function(...)
	if select("#",...) == 5 then
		return forWithStep(...)
	else
		return forWithStep((select(1,...)), (select(2,...)), (select(3,...)), 1, (select(4,...)))
	end
end)

local terra testfor()
	var i: int
	pfor(i, 0, 10,
	[quote
		C.printf("%d\n", i)
	end]
	)
end

testfor()

------------------------------

-- import "/Users/dritchie/Git/terra/tests/lib/sumlanguage"

-- local terra testsum()
-- 	C.printf("%d\n", [sum 1,2,3 done])
-- end

-- testsum()
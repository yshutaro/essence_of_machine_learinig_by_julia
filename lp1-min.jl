using JuMP
using Clp

m = Model(solver=ClpSolver())
@variable(m, x >= 0)
@variable(m, y >= 0)

@objective(m, Min, (-3) * x + (-4) * y )

@constraint(m, x + 4 * y <= 1700)
@constraint(m, 2 * x + 3 * y <= 1400)
@constraint(m, 2 * x + y <= 1000)

status = solve(m)

println(getvalue(x))
println(getvalue(y))
println(getobjectivevalue(m))

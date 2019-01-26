using JuMP
using OSQP

solver = OSQPMathProgBaseInterface.OSQPSolver()
m = Model(solver=solver)

@variable(m, x)
@variable(m, y)

@constraint(m, x + y == 0)
@objective(m, Min, x^2 + x*y + y^2 + 2x + 4y)

status = solve(m)

println(getvalue(x))
println(getvalue(y))
println(getobjectivevalue(m))

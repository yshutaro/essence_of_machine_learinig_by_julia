using JuMP
using OSQP

solver = OSQPMathProgBaseInterface.OSQPSolver()
m = Model(solver=solver)

@variable(m, x[1:2])

A = [1, 1]
b = 0
@constraint(m, A'*x == b)

P = [2 1
     1 2]
q = [2, 4]
@objective(m, Min, 0.5 * x'*P*x + q'*x)

status = solve(m)

println(getvalue(x))
println(getobjectivevalue(m))

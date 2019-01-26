using JuMP
using OSQP

solver = OSQPMathProgBaseInterface.OSQPSolver()
m = Model(solver=solver)

@variable(m, x[1:2])

G = [2, 3]
h = 3
@constraint(m, G'*x <= h)

P = [2 1
     1 2]
q = [2, 4]
@objective(m, Min, 0.5 * x'*P*x + q'*x)

status = solve(m)

println(getvalue(x))
println(getobjectivevalue(m))

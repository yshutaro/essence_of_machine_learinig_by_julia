f(x) = x^3 - 5*x + 1

df(x) = 3*x^2 - 5

function newton1dim(f, df, x0, ϵ=1e-10, max_iter=1000)
  x = x0
   x_new = 0
  iter = 0
  while true
    x_new = x - f(x)/df(x)
    if abs(x - x_new ) < ϵ
      break
    end
    x = x_new
    iter += 1
    if iter == max_iter
      break
    end
  end
  return x_new
end

println(newton1dim(f, df, 2))
println(newton1dim(f, df, 0))
println(newton1dim(f, df, -3))

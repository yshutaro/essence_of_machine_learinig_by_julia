module newton

struct Newton
  ϵ::Float64
  max_iter::Int

  function Newton(ϵ=1e-10, max_iter=1000)
    new(ϵ, max_iter)
  end
end

function solveNewton(newton::Newton, f, df, x0)
  x = x0
  x_new = 0
  iter = 0
  while true
    x_new = x - inv(df(x)) * f(x)
    if sum((x - x_new).^2) < (newton.ϵ)^2
      break
    end
    x = x_new
    iter += 1
    if iter == newton.max_iter
      break
    end
  end
  return x_new
end

end


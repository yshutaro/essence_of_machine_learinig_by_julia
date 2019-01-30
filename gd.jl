module gd

mutable struct GradientDescent
  path_::Array
  x_::Array
  opt_::Float64

  function GradientDescent()
    new([], [], 0.0)
  end
end

function solveGradientDescent(g::GradientDescent, f, df, init::Array, α=0.01, ϵ=1e-6)
  x = init
  path = []
  grad = df(x)
  append!(path, x)
  
  while (sum(grad.^2)) > ϵ^2
    x = x - α * grad
    grad = df(x)
    append!(path, x)
  end

  g.path_ = path
  g.x_ = x
  g.opt_ = f(x)
end

end


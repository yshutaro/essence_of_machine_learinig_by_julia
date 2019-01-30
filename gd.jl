module gd

mutable struct GradientDescent
  path_x_::Array
  path_y_::Array
  x_::Array
  opt_::Float64

  function GradientDescent()
    new([], [], [], 0.0)
  end
end

function solveGradientDescent(g::GradientDescent, f, df, init::Array, α=0.01, ϵ=1e-6)
  x = init
  path_x = []
  path_y = []
  grad = df(x)
  append!(path_x, x[1])
  append!(path_y, x[2])
  
  while (sum(grad.^2)) > ϵ^2
    x = x - α * grad
    grad = df(x)
    append!(path_x, x[1])
    append!(path_y, x[2])
  end

  g.path_x_ = path_x
  g.path_y_ = path_y
  g.x_ = x
  g.opt_ = f(x)
end

end


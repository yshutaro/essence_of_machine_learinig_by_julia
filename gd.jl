module gd

mutable struct GradientDescent
  path_::Array
  x_::Array
  opt_::Float64

  function GradientDescent()
    new([], [], 0.0)
  end
end

function solveGradientDescent(g::GradientDescent, f, df, init::Array, α=0.01, ϵ=eps())
  x = init
  println("x=", x)
  path = []
  ∇ = df(x)
  println("∇=", ∇)
  append!(path, x)
  println("sum=", sum(∇.^2))
  x = x - α .* ∇
  println("x(after)=", x)
  
  #while sum(∇.^2) > ϵ^2
  #  println("∇=", ∇)
  #  println("sum=", sum(∇.^2))
  #  println("x(befor)=", x)
  #  x = x - α .* ∇
  #  println("x(after)=", x)
  #  ∇ = df(x)
  #  println("df[1]=",∇[1])
  #  println("df[2]=",∇[2])
  #  append!(path, x)
  #end


  g.path_ = path
  g.x_ = x
  g.opt_ = f(x)
end

end


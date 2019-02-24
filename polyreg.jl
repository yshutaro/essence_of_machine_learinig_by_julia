include("linearreg.jl")

module polyreg
using ..linearreg

mutable struct PolynomialRegression
  degree::Int
  w_
  function PolynomialRegression(degree::Int)
    new(degree, Nothing)
  end
end

function fit(s::PolynomialRegression, x, y)
  x_pow = []
  xx = vec(x)
  for i in 1:s.degree
    push!(x_pow, xx.^i)
  end
  mat = hcat(x_pow...)
  linreg = linearreg.LinearRegression()
  linearreg.fit(linreg, mat, y)
  s.w_ = linreg.w_
end

function predict(s::PolynomialRegression, x)
  r = 0
  for i in 1:(s.degree + 1)
    r += x^i * s.w_[i]
  end
  return r
end

end

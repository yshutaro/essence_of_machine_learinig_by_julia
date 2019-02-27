module linearreg

mutable struct LinearRegression
  w_
  function LinearRegression()
    new(Nothing)
  end
end

function fit(s::LinearRegression, X, t)
  Xtil = hcat(ones(size(X)[1]), X)
  A = Xtil' * Xtil
  b = Xtil' * t
  s.w_ = A \ b
end

function predict(s::LinearRegression, X)
  #if ndims(X) == 1
  #  X = reshape(X, 1, :)
  #end
  Xtil = hcat(ones(size(X)[1]), X)
  return Xtil * s.w_
end

end

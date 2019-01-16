using Random

mutable struct Dice
  sum_::Float64

  function Dice()
    Random.seed!(0)
    new(0)
  end

end

function throw(x::Dice)
  x.sum_ += rand(1:6)
end

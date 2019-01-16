using Random

mutable struct Dice
  sum_::Float64
  rng::MersenneTwister

  function Dice(random_seed::Int)
    rng = MersenneTwister(random_seed)
    new(0, rng)
  end

end

function throw(x::Dice)
  x.sum_ += rand(x.rng, 1:6)
end

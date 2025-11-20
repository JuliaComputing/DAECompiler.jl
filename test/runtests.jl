include("basic.jl")
include("debugging.jl")
include("reflection.jl")
include("incidence.jl")
include("ipo.jl")
include("ssrm.jl")
include("regression.jl")
include("errors.jl")
include("invalidation.jl")
include("validation.jl")
include("cthulhu.jl")

using Pkg
Pkg.activate(joinpath(dirname(@__DIR__), "benchmark")) do
  include("benchmark.jl")
end

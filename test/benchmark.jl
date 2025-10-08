module _benchmark

using DAECompiler
using DAECompiler: compute_residual_vectors
using SciMLBase, Sundials
using Test

include("../benchmark/thermalfluid.jl")

@testset "Validation" begin
  Benchmark{3}()()

  u = zeros(68)
  du = zeros(68)
  residuals, expanded_residuals = compute_residual_vectors(Benchmark{3}(), u, du)
  @test_broken length(residuals) == length(expanded_residuals)
  @test_broken residuals ≈ expanded_residuals
  # indices = findall(i -> residuals[i] ≉ expanded_residuals[i], eachindex(residuals))
  # residuals[indices]
  # expanded_residuals[indices]
end

let sol = solve(DAECProblem(Benchmark{3}(), [1:9;] .=> 0.), IDA())
    @test_broken sol.retcode == ReturnCode.Success
end

end # module _benchmark

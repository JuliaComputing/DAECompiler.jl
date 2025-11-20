module DAECompiler

    using Graphs
    using StateSelection
    using StateSelection: DiffGraph
    using Diffractor
    using OrderedCollections
    using Compiler
    using Compiler: AbstractLattice, IRCode, IncrementalCompact, DebugInfoStream, NewInstruction, argextype, singleton_type, isexpr, widenconst
    using Core.IR
    using SciMLBase
    using AutoHashEquals
    using LinearAlgebra: LinearAlgebra
    using InteractiveUtils: gen_call_with_extracted_types_and_kwargs

    include("settings.jl")
    include("utils.jl")
    include("intrinsics.jl")
    include("reflection.jl")
    include("analysis/utils.jl")
    include("analysis/lattice.jl")
    include("analysis/ADAnalyzer.jl")
    include("analysis/scopes.jl")
    include("analysis/flattening.jl")
    include("analysis/cache.jl")
    include("analysis/refiner.jl")
    include("analysis/ipoincidence.jl")
    include("analysis/structural.jl")
    include("transform/state_selection.jl")
    include("transform/common.jl")
    include("transform/runtime.jl")
    include("transform/unoptimized.jl")
    include("transform/reconstruct.jl")
    include("transform/tearing/schedule.jl")
    include("transform/codegen/dae_factory.jl")
    include("transform/codegen/ode_factory.jl")
    include("transform/codegen/init_factory.jl")
    include("transform/codegen/rhs.jl")
    include("transform/codegen/init_uncompress.jl")
    include("transform/autodiff/ad_common.jl")
    include("transform/autodiff/ad_runtime.jl")
    include("transform/autodiff/index_lowering.jl")
    include("analysis/consistency.jl")
    include("interface.jl")
    include("problem_interface.jl")

    export dae_provider # use with Cthulhu, `@descend provider=dae_provider() pingpong()`
end

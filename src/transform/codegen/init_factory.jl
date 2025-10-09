
function init_uncompress_gen(result::DAEIPOResult, ci::CodeInstance, init_key::TornCacheKey, diff_key::TornCacheKey, world::UInt, settings::Settings)
    ir_factory = copy(ci.inferred.ir)
    pushfirst!(ir_factory.argtypes, Settings)
    pushfirst!(ir_factory.argtypes, typeof(factory))
    compact = IncrementalCompact(ir_factory)

    new_oc = init_uncompress_gen!(compact, result, ci, init_key, diff_key, world, settings)
    line = result.ir[SSAValue(1)][:line]
    @insert_instruction_here(compact, line, settings, (return new_oc)::Core.OpaqueClosure, true)

    ir_factory = Compiler.finish(compact)
    Compiler.verify_ir(ir_factory)

    return ir_factory
end

function init_uncompress_gen!(compact::Compiler.IncrementalCompact, result::DAEIPOResult, ci::CodeInstance, init_key::TornCacheKey, diff_key::TornCacheKey, world::UInt, settings::Settings)
    torn_ci = find_matching_ci(ci->isa(ci.owner, TornIRSpec) && ci.owner.key == init_key, ci.def, world)
    @assert torn_ci !== nothing
    torn_ir = torn_ci.inferred
    (;ir_sicm) = torn_ir

    local line
    if ir_sicm !== nothing
        sicm_ci = find_matching_ci(ci->isa(ci.owner, SICMSpec) && ci.owner.key == init_key, ci.def, world)
        @assert sicm_ci !== nothing

        line = result.ir[SSAValue(1)][:line]
        callee_argtypes = ci.inferred.ir.argtypes
        callee_argmap = ArgumentMap(callee_argtypes)
        args = Argument.(2 .+ eachindex(callee_argtypes))
        new_args = flatten_arguments_for_callee!(compact, callee_argmap, callee_argtypes, args, line, settings)
        param_list = @insert_instruction_here(compact, line, settings, tuple(new_args...)::Tuple)
        sicm = @insert_instruction_here(compact, line, settings, invoke(param_list, sicm_ci)::Tuple)
    else
        sicm = ()
    end

    # (nlsol,)
    argt = Tuple{Any}
    daef_ci = gen_init_uncompress!(result, ci, init_key, diff_key, world, settings, 1)

    # Create a small opaque closure to adapt from SciML ABI to our own internal
    # ABI

    numstates = zeros(Int, Int(LastEquationStateKind))

    all_states = Int[]
    for var = 1:length(result.var_to_diff)
        kind = classify_var(result.var_to_diff, diff_key, var)
        kind == nothing && continue
        numstates[kind] += 1
        (kind != AlgebraicDerivative) && push!(all_states, var)
    end

    ir_oc = copy(ci.inferred.ir)
    empty!(ir_oc.argtypes)
    push!(ir_oc.argtypes, Tuple)
    push!(ir_oc.argtypes, Any)

    Compiler.verify_ir(ir_oc)
    oc_compact = IncrementalCompact(ir_oc)
    line = ir_oc[SSAValue(1)][:line]

    # Zero the output
    nout = numstates[UnassignedDiff] + numstates[AssignedDiff]
    out_arr = @insert_instruction_here(oc_compact, line, settings, zeros(nout)::Vector{Float64})

    nscratch = numstates[Algebraic] + numstates[AlgebraicDerivative]
    scratch_arr = @insert_instruction_here(oc_compact, line, settings, zeros(nout)::Vector{Float64})

    # Get the solution vector out of the solution object
    in_nlsol_u = @insert_instruction_here(oc_compact, line, settings, getproperty(Argument(2), QuoteNode(:u0))::Vector{Float64})

    # Adapt to DAECompiler ABI
    nassgn = numstates[AssignedDiff]
    ntotalstates = numstates[AssignedDiff] + numstates[UnassignedDiff] + numstates[Algebraic]

    (out_u_mm, out_u_unassgn, out_alg) = sciml_dae_split_u!(oc_compact, line, settings, out_arr, numstates)
    (out_du_unassgn, _) = sciml_dae_split_du!(oc_compact, line, settings, scratch_arr, numstates)

    # Call DAECompiler-generated RHS with internal ABI
    oc_sicm = @insert_instruction_here(oc_compact, line, settings, getfield(Argument(1), 1)::Core.OpaqueClosure)
    @insert_instruction_here(oc_compact, line, settings, (:invoke)(daef_ci, oc_sicm, (), out_u_mm, out_u_unassgn, out_du_unassgn, out_alg, in_nlsol_u, 0.0)::Nothing)

    # Return
    @insert_instruction_here(oc_compact, line, settings, (return out_arr)::Vector{Float64})

    ir_oc = Compiler.finish(oc_compact)
    oc = optimized_opaque_closure(ir_oc, world)

    line = result.ir[SSAValue(1)][:line]

    oc_source_method = oc.source
    # Sketchy, but not clear that we have something better for the time being
    oc_ci = oc_source_method.specializations.cache
    @atomic oc_ci.max_world = @atomic ci.max_world
    @atomic oc_ci.min_world = 1 # @atomic ci.min_world

    new_oc = @insert_instruction_here(compact, line, settings, (:new_opaque_closure)(
        argt, Vector{Float64}, Vector{Float64}, true, oc_source_method, sicm)::Core.OpaqueClosure, true)

    return new_oc
end

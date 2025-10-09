"""
    sciml_dae_split_u!(compact, arg, numstates)

Given an IR value `arg` that corresponds to `u` in SciML's DAE ABI, split it into component pieces for
the DAECompiler internal ABI.
"""
function sciml_dae_split_u!(compact, line, settings, arg, numstates)
    nassgn = numstates[AssignedDiff]
    ntotalstates = numstates[AssignedDiff] + numstates[UnassignedDiff] + numstates[Algebraic]

    u_mm = @insert_instruction_here(compact, line, settings, view(arg, 1:nassgn)::VectorViewType)
    u_unassgn = @insert_instruction_here(compact, line, settings, view(arg, (nassgn+1):(nassgn+numstates[UnassignedDiff]))::VectorViewType)
    alg = @insert_instruction_here(compact, line, settings, view(arg, (nassgn+numstates[UnassignedDiff]+1):ntotalstates)::VectorViewType)

    return (u_mm, u_unassgn, alg)
end

"""
    sciml_dae_split_du!(compact, arg, numstates)

Given an IR value `arg` that corresponds to `du` in SciML's DAE ABI, split it into component pieces for
the DAECompiler internal ABI.
"""
function sciml_dae_split_du!(compact, line, settings, arg, numstates)
    nassgn = numstates[AssignedDiff]
    ntotalstates = numstates[AssignedDiff] + numstates[UnassignedDiff] + numstates[Algebraic]

    in_du_assgn = @insert_instruction_here(compact, line, settings, view(arg, 1:nassgn)::VectorViewType)
    in_du_unassgn = @insert_instruction_here(compact, line, settings, view(arg, (nassgn+1):(nassgn+numstates[UnassignedDiff]))::VectorViewType)

    return (in_du_assgn, in_du_unassgn)
end

function make_daefunction(f)
    DAEFunction(f)
end

function make_daefunction(f, initf)
    DAEFunction(f; initialization_data = SciMLBase.OverrideInitData(NonlinearProblem((args...)->nothing, nothing, nothing), nothing, initf, nothing, nothing, Val{false}()))
end

function continuous_variables(state::TransformationState)
    filter(var -> varkind(state, var) == Intrinsics.Continuous, 1:length(state.result.var_to_diff))
end

const SCIML_ABI = Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, SciMLBase.NullParameters, Float64}

function sciml_to_internal_abi!(ir::IRCode, state::TransformationState, internal_ci::CodeInstance, key::TornCacheKey, var_eq_matching, world::UInt, settings::Settings)
    (; result, structure) = state

    numstates = zeros(Int, Int(LastEquationStateKind))
    for var in continuous_variables(state)
        kind = classify_var(result.var_to_diff, key, var)
        kind == nothing && continue
        numstates[kind] += 1
    end

    empty!(ir.argtypes)
    push!(ir.argtypes, Tuple) # opaque closure captures
    append!(ir.argtypes, fieldtypes(SCIML_ABI))

    compact = IncrementalCompact(ir)

    # Zero the output
    line = ir[SSAValue(1)][:line]
    @insert_instruction_here(compact, line, settings, zero!(Argument(2))::VectorViewType)

    # out_du_mm, out_eq, in_u_mm, in_u_unassgn, in_du_unassgn, in_alg
    nassgn = numstates[AssignedDiff]
    ntotalstates = numstates[AssignedDiff] + numstates[UnassignedDiff] + numstates[Algebraic]
    out_du_mm = @insert_instruction_here(compact, line, settings, view(Argument(2), 1:nassgn)::VectorViewType)
    out_eq = @insert_instruction_here(compact, line, settings, view(Argument(2), (nassgn+1):ntotalstates)::VectorViewType)

    (in_du_assgn, in_du_unassgn) = sciml_dae_split_du!(compact, line, settings, Argument(3), numstates)
    (in_u_mm, in_u_unassgn, in_alg) = sciml_dae_split_u!(compact, line, settings, Argument(4), numstates)

    # Call DAECompiler-generated RHS with internal ABI
    oc_sicm = @insert_instruction_here(compact, line, settings, getfield(Argument(1), 1)::Core.OpaqueClosure)

    # N.B: The ordering of arguments should match the ordering in the StateKind enum
    @insert_instruction_here(compact, line, settings, (:invoke)(internal_ci, oc_sicm, (), in_u_mm, in_u_unassgn, in_du_unassgn, in_alg, out_du_mm, out_eq, Argument(6))::Nothing)

    # Manually apply mass matrix and implicit equations between selected states
    (_, var_assignment, _) = assign_slots(state, key, var_eq_matching)
    for v = 1:ndsts(structure.graph)
        vdiff = structure.var_to_diff[v]
        vdiff === nothing && continue

        if var_eq_matching[v] !== SelectedState() || var_eq_matching[vdiff] !== SelectedState()
            # Solved variables were already handled above
            continue
        end

        (kind, slot) = var_assignment[v]
        (dkind, dslot) = var_assignment[vdiff]
        @assert kind == AssignedDiff
        @assert dkind in (AssignedDiff, UnassignedDiff)

        v_val = @insert_instruction_here(compact, line, settings, getindex(dkind == AssignedDiff ? in_u_mm : in_u_unassgn, dslot)::Any)
        @insert_instruction_here(compact, line, settings, setindex!(out_du_mm, v_val, slot)::Any)
    end

    bc = @insert_instruction_here(compact, line, settings, Base.Broadcast.broadcasted(-, out_du_mm, in_du_assgn)::Any)
    @insert_instruction_here(compact, line, settings, Base.Broadcast.materialize!(out_du_mm, bc)::Nothing)

    # Return
    @insert_instruction_here(compact, line, settings, (return nothing)::Union{})

    ir = Compiler.finish(compact)
    maybe_rewrite_debuginfo!(ir, settings)
    resize!(ir.cfg.blocks, 1)
    empty!(ir.cfg.blocks[1].succs)
    Compiler.verify_ir(ir)
    return optimized_opaque_closure(ir, world; slotnames = [:captures, :out, :du, :u, :p, :t])
end

"""
    dae_factory_gen(ci, key)

Generate the `factory` function for CodeInstance `ci`, returning a DAEFunction.
The resulting function is roughly:

```
function factory(settings, f)
    # Run all parts of `f` that do not depend on state
    state_invariant_pieces = f_state_invariant()
    f! = %new_opaque_closure(f_rhs, state_invariant_pieces)
    DAEFunction(f!), differential_vars
end
```

"""
function dae_factory_gen(state::TransformationState, ci::CodeInstance, key::Union{TornCacheKey, UnoptimizedKey}, world::UInt, settings::Settings, init_key::Union{TornCacheKey, Nothing})
    (; result, structure) = state

    ir_factory = copy(ci.inferred.ir)
    pushfirst!(ir_factory.argtypes, Settings)
    pushfirst!(ir_factory.argtypes, typeof(factory))
    compact = IncrementalCompact(ir_factory)

    # Create a small opaque closure to adapt from SciML ABI to our own internal ABI
    argt = Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, SciMLBase.NullParameters, Float64}
    sicm = ()
    if settings.skip_optimizations
        daef_ci = rhs_finish_noopt!(state, ci, key, world, settings; opaque_closure = true)
        oc = sciml_to_internal_abi_noopt!(copy(ci.inferred.ir), state, daef_ci, settings)
    else
        # TODO: We should not have to recompute this here
        var_eq_matching = matching_for_key(state, key)

        torn_ci = find_matching_ci(ci->isa(ci.owner, TornIRSpec) && ci.owner.key == key, ci.def, world)
        torn_ir = torn_ci.inferred

        (; ir_sicm) = torn_ir

        local line
        if ir_sicm !== nothing
            sicm_ci = find_matching_ci(ci->isa(ci.owner, SICMSpec) && ci.owner.key == key, ci.def, world)
            @assert sicm_ci !== nothing

            line = result.ir[SSAValue(1)][:line]
            callee_argtypes = ci.inferred.ir.argtypes
            callee_argmap = ArgumentMap(callee_argtypes)
            args = Argument.(2 .+ eachindex(callee_argtypes))
            new_args = flatten_arguments_for_callee!(compact, callee_argmap, callee_argtypes, args, line, settings)
            list = @insert_instruction_here(compact, line, settings, tuple(new_args...)::Tuple)
            sicm = @insert_instruction_here(compact, line, settings, invoke(list, sicm_ci)::Tuple)
        end

        daef_ci = rhs_finish!(state, ci, key, world, settings, 1)
        oc = sciml_to_internal_abi!(copy(ci.inferred.ir), state, daef_ci, key, var_eq_matching, world, settings)
    end

    line = result.ir[SSAValue(1)][:line]

    oc_source_method = oc.source
    # Sketchy, but not clear that we have something better for the time being
    oc_ci = oc_source_method.specializations.cache
    @atomic oc_ci.max_world = @atomic ci.max_world
    @atomic oc_ci.min_world = 1 # @atomic ci.min_world

    new_oc = @insert_instruction_here(compact, line, settings, (:new_opaque_closure)(argt, Union{}, Nothing, true, oc_source_method, sicm)::Core.OpaqueClosure, true)

    if settings.skip_optimizations
        differential_states = Bool[structure.var_to_diff[v] !== nothing for v in continuous_variables(state)]
    else
        all_states = filter(continuous_variables(state)) do var
            kind = classify_var(result, key, var)
            kind === nothing && return false
            return kind !== AlgebraicDerivative
        end
        differential_states = Bool[v in key.diff_states for v in all_states]
    end

    if init_key !== nothing
        initf = init_uncompress_gen!(compact, result, ci, init_key, key, world, settings)
        daef = @insert_instruction_here(compact, line, settings, make_daefunction(new_oc, initf)::DAEFunction, true)
    else
        daef = @insert_instruction_here(compact, line, settings, make_daefunction(new_oc)::DAEFunction, true)
    end

    # TODO: Ideally, this'd be in DAEFunction
    daef_and_diff = @insert_instruction_here(compact, line, settings, tuple(daef, differential_states)::Tuple, true)

    @insert_instruction_here(compact, line, settings, (return daef_and_diff)::Tuple, true)

    ir_factory = Compiler.finish(compact)
    resize!(ir_factory.cfg.blocks, 1)
    empty!(ir_factory.cfg.blocks[1].succs)
    Compiler.verify_ir(ir_factory)

    slotnames = [:factory, :settings, :f]
    return ir_factory, slotnames
end

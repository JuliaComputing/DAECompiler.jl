"""
    sciml_ode_split_u!(compact, arg, numstates)

Given an IR value `arg` that corresponds to `u` in SciML's ODE ABI, split it into component pieces for
the DAECompiler internal ABI.
"""
function sciml_ode_split_u!(compact, line, settings, arg, numstates)
    ntotalstates = numstates[AssignedDiff] + numstates[UnassignedDiff] + numstates[Algebraic] + numstates[AlgebraicDerivative]

    u_mm = @insert_instruction_here(compact, line, settings, view(arg,
        1:numstates[AssignedDiff])::VectorViewType)
    u_unassgn = @insert_instruction_here(compact, line, settings, view(arg,
        (numstates[AssignedDiff] + 1):(numstates[AssignedDiff] + numstates[UnassignedDiff]))::VectorViewType)
    alg = @insert_instruction_here(compact, line, settings, view(arg,
        (numstates[AssignedDiff] + numstates[UnassignedDiff] + 1):(numstates[AssignedDiff] + numstates[UnassignedDiff] + numstates[Algebraic]))::VectorViewType)
    alg_derv = @insert_instruction_here(compact, line, settings, view(arg,
        (numstates[AssignedDiff] + numstates[UnassignedDiff] + numstates[Algebraic] + 1):ntotalstates)::VectorViewType)

    return (u_mm, u_unassgn, alg, alg_derv)
end

function generate_ode_mass_matrix(nd, na)
    n = nd + na
    mass_matrix = zeros(Float64, n, n)
    for i in 1:nd
        mass_matrix[i, i] = 1.0
    end
    return mass_matrix
end

function make_odefunction(f, mass_matrix = LinearAlgebra.I, initf = nothing)
    ODEFunction(f; mass_matrix, initialization_data = (initf === nothing ? nothing : initialization_data_ode(initf)))
end

function initialization_data_ode(initf)
    return SciMLBase.OverrideInitData(NonlinearProblem((args...)->nothing, nothing, nothing), nothing, initf, nothing, nothing, Val{false}())
end

"""
    ode_factory_gen(ci, key)

Generate the `factory` function for CodeInstance `ci`, returning a ODEFunction.
The resulting function is roughly:

```
function factory(settings, f)
    # Run all parts of `f` that do not depend on state
    state_invariant_pieces = f_state_invariant()
    f! = %new_opaque_closure(f_rhs, state_invariant_pieces)
    ODEFunction(f!)
end
```

"""
function ode_factory_gen(state::TransformationState, ci::CodeInstance, key::TornCacheKey, world::UInt, settings::Settings, init_key::Union{TornCacheKey, Nothing})
    result = state.result
    torn_ci = find_matching_ci(ci->isa(ci.owner, TornIRSpec) && ci.owner.key == key, ci.def, world)
    torn_ir = torn_ci.inferred

    sicm_ir = torn_ir.ir_sicm

    returned_ir = copy(ci.inferred.ir)
    pushfirst!(returned_ir.argtypes, Settings)
    pushfirst!(returned_ir.argtypes, typeof(factory))
    returned_ic = IncrementalCompact(returned_ir)

    local line
    if sicm_ir !== nothing
        sicm_ci = find_matching_ci(ci->isa(ci.owner, SICMSpec) && ci.owner.key == key, ci.def, world)
        @assert sicm_ci !== nothing

        line = result.ir[SSAValue(1)][:line]
        callee_argtypes = ci.inferred.ir.argtypes
        callee_argmap = ArgumentMap(callee_argtypes)
        args = Argument.(2 .+ eachindex(callee_argtypes))
        new_args = flatten_arguments_for_callee!(returned_ic, callee_argmap, callee_argtypes, args, line, settings)
        param_list = @insert_instruction_here(returned_ic, line, settings, tuple(new_args...)::Tuple)
        sicm_state = @insert_instruction_here(returned_ic, line, settings, invoke(param_list, sicm_ci)::Tuple)
    else
        sicm_state = ()
    end

    odef_ci = rhs_finish!(state, ci, key, world, settings, 1)

    # Create a small opaque closure to adapt from SciML ABI to our own internal ABI

    numstates = zeros(Int, Int(LastEquationStateKind))

    all_states = Int[]
    for var = 1:length(result.var_to_diff)
        varkind(state, var) == Intrinsics.Continuous || continue
        kind = classify_var(result.var_to_diff, key, var)
        kind == nothing && continue
        numstates[kind] += 1
        (kind != AlgebraicDerivative) && push!(all_states, var)
    end

    interface_ir = copy(ci.inferred.ir)
    empty!(interface_ir.argtypes)
    argt = Tuple{Vector{Float64}, Vector{Float64}, SciMLBase.NullParameters, Float64}
    push!(interface_ir.argtypes, Tuple)
    append!(interface_ir.argtypes, fieldtypes(argt))
    Compiler.verify_ir(interface_ir)

    interface_ic = IncrementalCompact(interface_ir)
    self = Argument(1)
    du = Argument(2)
    u = Argument(3)
    p = Argument(4)
    t = Argument(5)

    line = interface_ir[SSAValue(1)][:line]

    # Zero the output
    @insert_instruction_here(interface_ic, line, settings, zero!(du)::VectorViewType)

    nassgn = numstates[AssignedDiff]
    nunassgn = numstates[UnassignedDiff]
    ntotalstates = numstates[AssignedDiff] + numstates[UnassignedDiff] + numstates[Algebraic] + numstates[AlgebraicDerivative]

    (in_u_mm, in_u_unassgn, in_alg, in_alg_derv) = sciml_ode_split_u!(interface_ic, line, settings, u, numstates)
    out_du_mm = @insert_instruction_here(interface_ic, line, settings, view(du, 1:nassgn)::VectorViewType)
    out_du_unassgn = @insert_instruction_here(interface_ic, line, settings, view(du, (nassgn+1):(nassgn+nunassgn))::VectorViewType)
    out_eq = @insert_instruction_here(interface_ic, line, settings, view(du, (nassgn+nunassgn+1):ntotalstates)::VectorViewType)

    # Call DAECompiler-generated RHS with internal ABI
    sicm_oc = @insert_instruction_here(interface_ic, line, settings, getfield(self, 1)::Core.OpaqueClosure)

    # N.B: The ordering of arguments should match the ordering in the StateKind enum
    @insert_instruction_here(interface_ic, line, settings, (:invoke)(odef_ci, sicm_oc, (), in_u_mm, in_u_unassgn, in_alg_derv, in_alg, out_du_mm, out_eq, t)::Nothing)

    # Assign the algebraic derivatives to the their corresponding variables
    bc = @insert_instruction_here(interface_ic, line, settings, Base.Broadcast.broadcasted(identity, in_alg_derv)::Any)
    @insert_instruction_here(interface_ic, line, settings, Base.Broadcast.materialize!(out_du_unassgn, bc)::Nothing)

    # Return
    @insert_instruction_here(interface_ic, line, settings, (return)::Union{})

    interface_ir = Compiler.finish(interface_ic)
    maybe_rewrite_debuginfo!(interface_ir, settings)
    Compiler.verify_ir(interface_ir)
    interface_oc = optimized_opaque_closure(interface_ir, world; slotnames = [:self, :du, :u, :p, :t])

    line = result.ir[SSAValue(1)][:line]

    interface_method = interface_oc.source
    # Sketchy, but not clear that we have something better for the time being
    interface_ci = interface_method.specializations.cache
    @atomic interface_ci.max_world = @atomic ci.max_world
    @atomic interface_ci.min_world = 1 # @atomic ci.min_world

    new_oc = @insert_instruction_here(returned_ic, line, settings, (:new_opaque_closure)(argt, Union{}, Nothing, true, interface_method, sicm_state)::Core.OpaqueClosure, true)

    nd = numstates[AssignedDiff] + numstates[UnassignedDiff]
    na = numstates[Algebraic] + numstates[AlgebraicDerivative]
    mass_matrix = na == 0 ? GlobalRef(LinearAlgebra, :I) : @insert_instruction_here(returned_ic, line, settings, generate_ode_mass_matrix(nd, na)::Matrix{Float64})
    initf = init_key !== nothing ? init_uncompress_gen!(returned_ic, result, ci, init_key, key, world, settings) : nothing
    odef = @insert_instruction_here(returned_ic, line, settings, make_odefunction(new_oc, mass_matrix, initf)::ODEFunction, true)
    odef_and_n = @insert_instruction_here(returned_ic, line, settings, tuple(odef, nd + na)::Tuple, true)
    @insert_instruction_here(returned_ic, line, settings, (return odef_and_n)::Core.OpaqueClosure, true)

    returned_ir = Compiler.finish(returned_ic)
    Compiler.verify_ir(returned_ir)

    slotnames = [[:factory, :settings]; Symbol.(:arg, 1:(length(returned_ir.argtypes) - 2))]
    return returned_ir, slotnames
end

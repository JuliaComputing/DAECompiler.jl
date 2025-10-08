"""
    struct InitUncompressSpec

Cache partition for the RHS
"""
struct InitUncompressSpec
    init_key::TornCacheKey
    diff_key::TornCacheKey
    ordinal::Int
end

function gen_init_uncompress!(result::DAEIPOResult, ci::CodeInstance, init_key::TornCacheKey, diff_key::TornCacheKey, world::UInt, settings::Settings, ordinal::Int, indexT=Int)
    structure = make_structure_from_ipo(result)
    tstate = TransformationState(result, structure)
    return gen_init_uncompress!(tstate, ci, init_key, diff_key, world, settings, ordinal, indexT)
end

function gen_init_uncompress!(
        state::TransformationState,
        ci::CodeInstance,
        init_key::TornCacheKey,
        diff_key::TornCacheKey,
        world::UInt,
        settings::Settings,
        ordinal::Int,
        indexT=Int)

    (; result, structure) = state

    result_ci = find_matching_ci(ci->isa(ci.owner, InitUncompressSpec) && ci.owner.init_key == init_key && ci.owner.diff_key == diff_key && ci.owner.ordinal == ordinal, ci.def, world)
    if result_ci !== nothing
        return result_ci
    end

    allow_unassigned = false

    torn_ci = find_matching_ci(ci->isa(ci.owner, TornIRSpec) && ci.owner.key == init_key, ci.def, world)
    torn = torn_ci.inferred
    rhs_ms = nothing
    old_daef_mi = nothing
    assigned_slots = falses(length(result.total_incidence))
    diff_var_eq_matching = matching_for_key(state, diff_key)

    (diff_slot_assignments, diff_var_assignment, diff_eq_assignment) = assign_slots(state, diff_key, diff_var_eq_matching)


    cis = Vector{CodeInstance}()
    for (ir_ordinal, ir) in enumerate(torn.ir_seq)
        # Read in from the last level before any DAE or ODE-specific `ir_levels`
        # We assume this is named `tearing_schedule!`
        ir = copy(ir)
        empty!(ir.argtypes)
        push!(ir.argtypes, Tuple)  # SICM State
        push!(ir.argtypes, Tuple)  # in vars

        arg_range = 3:6
        out_u_mm, out_u_unassgn, out_du_unassgn, out_alg  =
            Argument.(arg_range)
        for arg in arg_range
            push!(ir.argtypes, VectorViewType)
        end

        in_nlsolve_u = Argument(last(arg_range)+1)
        push!(ir.argtypes, Vector{Float64})  # nlsolve u in

        t = Argument(last(arg_range)+2)
        push!(ir.argtypes, Float64)  #  t

        processed_variables = BitSet()
        var_assignment = Vector{Union{Nothing, Int}}(nothing, length(result.var_to_diff))

        diff_states_in_callee = BitSet()

        for i = 1:length(ir.stmts)
            inst = ir[SSAValue(i)]
            stmt = inst[:stmt]
            info = inst[:info]

            if isa(info, Compiler.ConstCallInfo) && any(result->isa(result, Compiler.SemiConcreteResult), info.results)
                # Drop any semi-concrete results from the DAE-interpreter. We will redo
                # them with the native interpreter to avoid getting suboptimal codegen.
                ir[SSAValue(i)][:info] = info.call
                ir[SSAValue(i)][:flag] |= Compiler.IR_FLAG_REFINED
            end

            if isexpr(stmt, :invoke) && isa(stmt.args[1], Tuple)
                info::MappingInfo
                callee_ci = stmt.args[1][1]
                closure_env = stmt.args[2]
                in_vars = stmt.args[3]
                if isa(callee_ci, MethodInstance)
                    callee_ci = Compiler.get(Compiler.code_cache(interp), callee_ci, nothing)
                end

                @assert callee_ci !== nothing

                spec_data = stmt.args[1]
                callee_init_key = stmt.args[1][2]
                callee_diff_key = project_mapping(state, info, diff_key, diff_var_eq_matching, state.total_incidence)
                callee_ordinal = stmt.args[1][end]::Int
                callee_result = structural_analysis!(callee_ci, world, settings)
                callee_daef_ci = gen_init_uncompress!(callee_result, callee_ci, callee_init_key, callee_diff_key, world, settings, callee_ordinal)
                # Allocate a continuous block of variables for all callee alg and diff states

                empty!(stmt.args)
                push!(stmt.args, callee_daef_ci)
                push!(stmt.args, closure_env)
                push!(stmt.args, in_vars)

                # Ordering from tearing is (AssignedDiff, UnassignedDiff, Algebraic, Explicit)
                slot_ranges = compute_slot_ranges(state, info, callee_diff_key, diff_var_assignment, diff_eq_assignment)
                for (arg, range) in zip(arg_range, slot_ranges)
                    push!(stmt.args, insert_node!(ir, SSAValue(i),
                        NewInstruction(inst;
                        stmt=Expr(:call, view, Argument(arg), range),
                        type=VectorViewType)))
                end

                # TODO: Needs to subset
                push!(stmt.args, in_nlsolve_u)

                # TODO: Track whether the system is autonomous
                push!(stmt.args, t)
            end

            if is_known_invoke(stmt, variable, ir) || is_equation_call(stmt, ir)
                display(ir)
                @sshow stmt
                error()
            elseif is_known_invoke(stmt, equation, ir)
                # Equation - used, but only as an arg to equation call, which will all get
                # eliminated by the end of this loop, so we can delete this statement, as
                # long as we don't touch the type yet.
                ir[SSAValue(i)][:inst] = Intrinsics.placeholder_equation
            elseif is_solved_variable(stmt)
                (varnum, argval) = stmt.args[(end-1):end]
                slot = diff_var_assignment[varnum]
                if slot === nothing
                    ir[SSAValue(i)] = nothing
                else
                    (kind, slotidx) = slot
                    which = kind == AssignedDiff ? out_u_mm : error()
                    replace_call!(ir, SSAValue(i), Expr(:call, Base.setindex!, which, argval, slotidx), settings, @__SOURCE__)
                end
            else
                replace_if_intrinsic!(ir, settings, SSAValue(i), nothing, nothing, Argument(1), t, var_assignment)
            end
        end

        # Just before the end of the function
        ir = Compiler.compact!(ir)
        Compiler.verify_ir(ir)

        widen_extra_info!(ir)
        src = ir_to_src(ir, settings)

        abi = Tuple{Tuple, Tuple, (VectorViewType for _ in arg_range)..., Vector{Float64}, Float64}
        daef_ci = cache_dae_ci!(ci, src, src.debuginfo, abi, InitUncompressSpec(init_key, diff_key, ir_ordinal))
        ccall(:jl_add_codeinst_to_jit, Cvoid, (Any, Any), daef_ci, src)

        push!(cis, daef_ci)
    end

    return cis[ordinal]
end

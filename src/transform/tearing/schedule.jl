
using StateSelection.BipartiteGraphs: nsrcs, ndsts, DiCMOBiGraph, invview
using Compiler: userefs, insert_node!, insert_node_here!, NewInstruction, NewSSAValue

struct SolvedVariable
    ssa::Union{SSAValue, Float64}
end

"""
Scan `compact` for all variables and equations.
"""
function find_eqs_vars(graph, compact::IncrementalCompact)
    eqs = Pair{SSAValue, Vector{SSAValue}}[ SSAValue(0) => SSAValue[] for _ in 1:nsrcs(graph)]
    vars = Union{SSAValue, NewSSAValue, SolvedVariable}[SSAValue(0) for _ in 1:ndsts(graph)]
    for ((_, i), stmt) in compact
        if is_known_invoke_or_call(stmt, equation, compact)
            eq = compact[SSAValue(i)][:type].id
            eqs[eq] = SSAValue(i) => eqs[eq][2]
        elseif is_known_invoke_or_call(stmt, variable, compact)
            var = only(rowvals(compact[SSAValue(i)][:type].row)) - 1
            vars[var] = SSAValue(i)
        elseif is_equation_call(stmt, compact)
            push!(eqs[idnum(argextype(_eq_function_arg(stmt), compact))][2], SSAValue(i))
        elseif is_known_invoke_or_call(stmt, solved_variable, compact)
            vars[stmt.args[end-1]] = SolvedVariable(stmt.args[end])
        end
    end
    (eqs, vars)
end

function find_eqs_vars(state::TransformationState)
    compact = IncrementalCompact(copy(state.result.ir))
    find_eqs_vars(state.structure.graph, compact)
end

function ir_add!(compact::IncrementalCompact, line, settings::Settings, @nospecialize(_a), @nospecialize(_b), source = @__SOURCE__)
    a, b = _a, _b
    (b === nothing || b === 0.) && return _a
    (a === nothing || b === 0.) && return _b
    idx = insert_instruction_here!(compact, line, settings, source, :($a + $b), Any)
    compact[idx][:flag] |= Compiler.IR_FLAG_REFINED
    idx
end

function ir_mul_const!(compact, line, settings, coeff::Float64, _a, source = @__SOURCE__)
    if isone(coeff)
        return _a
    end
    idx = insert_instruction_here!(compact, line, settings, source, :($coeff * $_a), Any)
    compact[idx][:flag] |= Compiler.IR_FLAG_REFINED
    return idx
end

Base.IteratorSize(::Type{Compiler.UseRefIterator}) = Base.SizeUnknown()

function schedule_incidence!(compact, curval, ::Type, var, line, settings; vars=nothing, schedule_missing_var! = nothing)
    # This just needs the linear part, which is `0` in `Type`
    return (curval, nothing)
end

function schedule_incidence!(compact, curval, incT::Const, var, line, settings; vars=nothing, schedule_missing_var! = nothing)
    if curval !== nothing
        return (ir_add!(compact, line, settings, curval, incT.val, @__SOURCE__), nothing)
    end
    return (incT.val, nothing)
end

function schedule_incidence!(compact, curval, incT::Incidence, var, line, settings; vars=nothing, schedule_missing_var! = nothing)
    thiscoeff = nothing

    # We do need to materialize the linear parts of the incidence here
    for (lin_var_offset, coeff) in zip(rowvals(incT.row), nonzeros(incT.row))
        lin_var = lin_var_offset - 1

        if lin_var == var
            # This is the value we're solving for
            thiscoeff = -coeff
            continue
        end

        isa(coeff, Float64) || continue

        if lin_var == 0
            lin_var_ssa = @insert_instruction_here(compact, line, settings, (:invoke)(nothing, Intrinsics.sim_time)::Incidence(0))
        else
            if vars === nothing || !isassigned(vars, lin_var)
                lin_var_ssa = schedule_missing_var!(lin_var)
            else
                lin_var_ssa = vars[lin_var]
            end
            if lin_var_ssa === 0.0
                continue
            end
        end

        acc = ir_mul_const!(compact, line, settings, coeff, lin_var_ssa, @__SOURCE__)
        curval = curval === nothing ? acc : ir_add!(compact, line, settings, curval, acc, @__SOURCE__)
    end
    (curval, _) = schedule_incidence!(compact, curval, incT.typ, var, line, settings; vars, schedule_missing_var!)
    return (curval, thiscoeff)
end

struct RenameOverlayVector <: AbstractVector{Any}
    base::Vector{Any}
    overlay::Vector{Any}
end

function Base.getindex(rno::RenameOverlayVector, i::Int)
    val = rno.overlay[i]
    val !== nothing && return val
    return rno.base[i]
end
Base.setindex!(rno::RenameOverlayVector, @nospecialize(val), i::Int) =
    Base.setindex!(rno.overlay, val, i)
Base.size(rno::RenameOverlayVector) = Base.size(rno.base)

is_var_part_known_linear(incT::Const) = true
is_var_part_known_linear(::Type) = false
is_var_part_known_linear(incT::Incidence) = all(x -> isa(x, Float64), incT.row)
is_const_plus_var_known_linear(incT) = is_var_part_known_linear(incT) && isa(incT.typ, Const)
is_const_plus_var_known_linear(incT::Const) = true

is_fully_state_linear(incT, param_vars) = is_const_plus_var_known_linear(incT) && is_fully_state_linear(incT.typ, param_vars)
is_fully_state_linear(incT::Const, param_vars) = iszero(incT.val)

function schedule_nonlinear!(compact, settings, param_vars, var_eq_matching, ir, ordinal, val::Union{SSAValue, Argument}, ssa_rename::AbstractVector{Any}; vars, schedule_missing_var! = nothing)
    isa(val, Argument) && return vars[idnum(argextype(val, ir))]

    if isassigned(ssa_rename, val.id)
        return ssa_rename[val.id]
    end

    inst = ir[val]

    #=
    if all(x->((x-1) in param_vars), rowvals(inst[:type].row))
        @assert (inst[:stmt]::Expr).args[1] === getfield
        this = insert_node_here!(compact, NewInstruction(inst))
        # No state dependence, we're done
        ssa_rename[val.id] = this
        return this
    end
    =#

    stmt = inst[:stmt]
    info = inst[:info]
    incT = inst[:type]::Incidence
    if isa(info, Diffractor.FRuleCallInfo)
        info = info.info
    end
    call_is_linear = call_is_omittable = false
    if isa(info, MappingInfo)
        result = info.result
        extended_rt = result.extended_rt
        call_is_linear = is_fully_state_linear(extended_rt, nothing)
    else
        @assert isexpr(stmt, :call)
        f = argextype(stmt.args[1], ir)
        @assert isa(f, Const)
        f = f.val
        @assert f in (Core.Intrinsics.sub_float, Core.Intrinsics.add_float,
                      Core.Intrinsics.mul_float, Core.Intrinsics.copysign_float,
                      Core.ifelse, Core.Intrinsics.or_int, Core.Intrinsics.and_int,
                      Core.Intrinsics.fma_float, Core.Intrinsics.muladd_float,
                      Core.Intrinsics.have_fma, Core.getfield,
                      InternalIntrinsics.assign_var,
                      Core.Intrinsics.sitofp)
        # TODO: or_int is linear in Bool
        # TODO: {fma, muladd}_float is linear in one of its arguments
        call_is_linear = f in (Core.Intrinsics.sub_float, Core.Intrinsics.add_float,
            InternalIntrinsics.assign_var)
        call_is_omittable = f in (Core.Intrinsics.add_float, InternalIntrinsics.assign_var)
    end

    args = map(enumerate(Iterators.drop(userefs(stmt), 1))) do (i, ur)
        arg = ur[]
        typ = argextype(arg, ir)

        if isa(typ, Const)
            this_nonlinear = nothing
        elseif !is_const_plus_var_known_linear(typ::Incidence)
            this_nonlinear = schedule_nonlinear!(compact, settings, param_vars, var_eq_matching, ir, ordinal, arg, ssa_rename; vars, schedule_missing_var!)
        else
            if @isdefined(result)
                # This relies on the flattening transform
                template_v = i
                if !isa((extended_rt::Incidence).row[template_v+1], Linearity)
                    return nothing
                end
            end
            this_nonlinear = nothing
        end

        if call_is_linear
            return this_nonlinear
        end

        argval = schedule_incidence!(compact, this_nonlinear, typ, -1, inst[:line], settings; vars, schedule_missing_var!)[1]
        if argval === nothing
            display(ir)
        end
        @assert argval !== nothing
        return argval
    end

    if call_is_omittable
        inds = findall(!=(nothing), args)
        length(inds) == 1 && return args[only(inds)]
    end

    if is_const_plus_var_known_linear(incT)
        ret = schedule_incidence!(compact, nothing, info.result.extended_rt, -1, inst[:line], settings; vars=
            [arg === nothing ? 0.0 : arg for arg in args], schedule_missing_var! = var->error((var, incT, args)))[1]
    else
        new_stmt = copy(stmt)
        for (i, arg) in enumerate(args)
            i += 1
            if arg == nothing
                if isa(new_stmt.args[i], NewSSAValue)
                    new_stmt.args[i] = SSAValue(new_stmt.args[i].id)
                elseif !isa(argextype(new_stmt.args[i], ir), Const)
                    new_stmt.args[i] = 0.0
                end
                continue
            end
            new_stmt.args[i] = arg
        end

        ret = insert_instruction_here!(compact, settings, @__SOURCE__, NewInstruction(inst; stmt=new_stmt))
    end

    ssa_rename[val.id] = isa(ret, SSAValue) ? CarriedSSAValue(ordinal, ret.id) : ret
    return ret
end

struct CarriedSSAValue
    ordinal::Int
    id::Int
end

function is_fully_ready(info::MappingInfo, available::BitSet)
    callee_vars = BitSet()

    # TODO: this needs to count returned vars as external
    # TODO: We only care about variables used non-linearly
    for i = 1:info.result.nexternalargvars
        caller_map = info.mapping.var_coeffs[i]
        isa(caller_map, Const) && continue
        union!(callee_vars, rowvals(caller_map.row) .- 1)
    end

    all(cv->(cv in available), callee_vars)
end

function compute_eq_schedule(key::TornCacheKey, total_incidence, result, mss::StateSelection.MatchedSystemStructure)
    (; diff_states, alg_states, var_schedule) = key
    (; structure, var_eq_matching) = mss

    eq_orders  = Vector{Union{Int, StructuralSSARef}}[]
    callee_schedules = Dict{StructuralSSARef, Vector{Pair{BitSet, BitSet}}}()

    available = BitSet()
    # Add all selected states to available
    if diff_states !== nothing
        union!(available, diff_states) # Computed by integrator
    end
    union!(available, alg_states)  # Computed by NL solver
    union!(available, key.param_vars)  # Computed by SICM hoist

    frontier = BitSet()

    isempty(var_schedule) && (var_schedule = Pair{BitSet, BitSet}[available=>BitSet()])
    vargraph = DiCMOBiGraph{true}(structure.graph, var_eq_matching)

    function may_enqueue_frontier_var!(var)
        isa(var_eq_matching[var], Int) || return
        (var in available) && return # Already processed
        (var in frontier) && return # Already queued
        # Check if this neighbor is ready
        if all(inn->(inn in available), inneighbors(vargraph, var))
            push!(frontier, var)
        end
    end

    for (ordinal, sched) in enumerate(var_schedule)
        eq_order = Union{Int, StructuralSSARef}[]
        push!(eq_orders, eq_order)

        (in_vars, out_eqs) = sched
        union!(available, in_vars)

        for neighbor in 1:ndsts(structure.graph)
            may_enqueue_frontier_var!(neighbor)
        end

        new_available = BitSet()

        function schedule_callee_now!(ssa::StructuralSSARef)
            schedule = get!(callee_schedules, ssa, Vector{Pair{BitSet, BitSet}}())
            callee_info = result.ir[SSAValue(ssa.id)][:info]::MappingInfo

            # Determine which of this callee's external variables we have
            this_callee_in_vars = BitSet()
            for i = 1:length(callee_info.mapping.var_coeffs)
                caller_map = callee_info.mapping.var_coeffs[i]
                isa(caller_map, Const) && continue
                caller_vars_for_this_callee_var = rowvals(caller_map.row) .- 1
                if length(caller_vars_for_this_callee_var) == 1
                    caller_var = only(caller_vars_for_this_callee_var)
                    if ((key.diff_states !== nothing && caller_var in key.diff_states) || caller_var in key.alg_states) &&
                       varclassification(result, structure, caller_var) == CalleeInternal
                        continue
                    end
                end
                if all(x->(x == 0 || x in available || x in key.param_vars), caller_vars_for_this_callee_var) &&
                # If these are all param_vars, the var itself becomes a param var,
                # not an in var.
                !all(in(key.param_vars), caller_vars_for_this_callee_var)
                    push!(this_callee_in_vars, i)
                end
            end
            if !isempty(schedule)
                @assert this_callee_in_vars != schedule[end][1]
            end

            previously_scheduled_or_ignored = isempty(schedule) ? BitSet() : mapreduce(x->copy(x[2]), union, schedule)

            # Schedule all equations that do not depend non-linearily on a variable we do not have
            this_callee_eqs = BitSet()
            found_any = true
            while found_any
                found_any = false
                for i = 1:length(callee_info.result.total_incidence)
                    i in previously_scheduled_or_ignored && continue # We scheduled this previously
                    i in this_callee_eqs && continue # We already scheduled this
                    # Skip equations that the callee defines but does not apply.
                    !isassigned(callee_info.result.total_incidence, i) && continue
                    callee_incidence = callee_info.result.total_incidence[i]
                    incidence = apply_linear_incidence!(callee_info.mapping, nothing, callee_incidence, nothing)
                    if is_const_plus_var_known_linear(incidence)
                        # No non-linear components - skip it
                        push!(previously_scheduled_or_ignored, i)
                        continue
                    end
                    for (caller_var_offset, coeff) in zip(rowvals(incidence.row), nonzeros(incidence.row))
                        caller_var = caller_var_offset - 1
                        if caller_var == 0 || coeff !== nonlinear
                            continue
                        end
                        if !(caller_var in available) #&& !(caller_var in new_available)
                            # Can't schedule this yet - we don't have a variable that is used non-linearly
                            @goto outer_continue
                        end
                    end
                    # We have all variables for this incidence, schedule it
                    mapped_eq = callee_info.mapping.eqs[i]
                    assigned_var = invview(var_eq_matching)[mapped_eq]
                    @assert assigned_var !== unassigned
                    if assigned_var === AlgebraicState()
                        push!(previously_scheduled_or_ignored, i)
                        continue
                    # TODO: Equations that stay entirely within the callee can be assigned here
                    end
                    push!(this_callee_eqs, i)
                    @label outer_continue
                end
            end

            @assert isempty(eq_order) || eq_order[end] != ssa
            push!(eq_order, ssa)
            push!(schedule, this_callee_in_vars => this_callee_eqs)
        end

        function schedule_eq!(eq; force=false)
            if !isassigned(total_incidence, eq) || is_const_plus_var_known_linear(total_incidence[eq])
                # This is a linear equation, we can schedule it now
                return true
            end

            # Otherwise, we need to compute some non-linear component
            eb = baseeq(result, structure, eq)
            mapping = result.eq_callee_mapping[eb]
            if mapping === nothing
                # Eq is toplevel, can be scheduled now
                return true
            end
            @assert eq == eb # TODO

            for (ssa, callee_eq) in mapping
                schedule = get!(callee_schedules, ssa, Vector{Pair{BitSet, BitSet}}())
                callee_info = result.ir[SSAValue(ssa.id)][:info]::MappingInfo

                if is_const_plus_var_known_linear(callee_info.result.total_incidence[callee_eq])
                    # This portion of the calle is linear, we can schedule it
                    continue
                end

                # Check if this equation was already scheduled
                any(((_,eqs),)->(callee_eq in eqs), schedule) && continue

                # Check if this callee is ready to be fully scheduled
                fully_ready = is_fully_ready(callee_info, available)
                if !force && !fully_ready
                    return false
                end

                # Schedule (a portion of) this callee now
                schedule_callee_now!(ssa)
                if !(callee_eq in schedule[end][2])
                    display(mss)
                    error("Failed to schedule callee eq $callee_eq (caller eq $eq) for $ssa")
                end
            end
            return true
        end

        function schedule_frontier_var!(var; force=false)
            # Find all frontier variables that in a callee that is fully available
            eq = var_eq_matching[var]::Int
            if schedule_eq!(eq; force) && !(var in new_available)
                push!(new_available, var)
                push!(eq_order, eq)
            end
        end

        while !isempty(frontier)
            worklist = copy(frontier)
            found_any = false
            for var in worklist
                var in new_available && continue
                schedule_frontier_var!(var)

                if !isempty(new_available)
                    found_any = true
                    union!(available, new_available)
                    for var in new_available
                        for neighbor in outneighbors(vargraph, var)
                            may_enqueue_frontier_var!(neighbor)
                        end
                    end
                    setdiff!(frontier, new_available)
                    empty!(new_available)
                    continue
                end
            end

            found_any && continue

            # TODO: We should have a heuristic for which callee to partition here rather
            # than just picking the first one
            schedule_frontier_var!(first(frontier); force=true)
            @assert !isempty(new_available)
            union!(available, new_available)
            for var in new_available
                for neighbor in outneighbors(vargraph, var)
                    may_enqueue_frontier_var!(neighbor)
                end
            end
            setdiff!(frontier, new_available)
            empty!(new_available)
        end

        for eq in out_eqs
            did_sched = schedule_eq!(eq; force=true)
            @assert did_sched
        end

        if ordinal == length(var_schedule)
            for eq in key.explicit_eqs
                callees = result.eq_callee_mapping[eq]
                eqclassification(result, structure, eq) === CalleeInternal && continue
                did_sched = schedule_eq!(eq; force=true)
                @assert did_sched
                push!(eq_order, eq)
            end
        end
    end

    return (eq_orders, callee_schedules)
end

function invert_eq_callee_mapping(eq_callee_mapping)
    callee_eq_mapping = Dict{StructuralSSARef, Vector{Int}}()

    for (ieq, mapping) in pairs(eq_callee_mapping)
        mapping === nothing && continue
        for val in mapping
            val === nothing && continue
            (eq, callee_eq) = val
            vec = get!(callee_eq_mapping, eq, Vector{Int}())
            n = length(vec)
            if n < callee_eq
                resize!(vec, callee_eq)
                vec[(n+1):callee_eq] .= 0
            end
            vec[callee_eq] = ieq
        end
    end

    return callee_eq_mapping
end

classify_var(structure::DAESystemStructure, key::TornCacheKey, var) = classify_var(structure.var_to_diff, key, var)
classify_var(result::DAEIPOResult, key::TornCacheKey, var) = classify_var(result.var_to_diff, key, var)
function classify_var(var_to_diff, key::TornCacheKey, var)
    if var in key.alg_states
        vint = invview(var_to_diff)[var]
        if vint in key.diff_states
            kind = AlgebraicDerivative
        else
            kind = Algebraic
        end
    elseif key.diff_states !== nothing && (var in key.diff_states)
        vdiff = var_to_diff[var]
        if vdiff in key.alg_states
            kind = UnassignedDiff
        else
            kind = AssignedDiff
        end
    else
        return nothing
    end
    return kind
end

function assign_slots(state::TransformationState, key::TornCacheKey, var_eq_matching::Union{StateSelection.Matching, Nothing})
    (; result, structure) = state
    slot_assignments = zeros(Int, Int(LastEquationStateKind))
    var_assignment = Vector{Union{Nothing, Pair{StateKind, Int}}}(nothing, length(structure.var_to_diff))
    eq_assignment = var_eq_matching === nothing ? nothing : Vector{Union{Nothing, Pair{EquationStateKind, Int}}}(nothing, length(state.total_incidence))

    function assign_slot!(kind, varnum)
        @assert kind != AlgebraicDerivative # Always the same assignment as the UnassignedDiff of the vint
        @assert kind != StateDiff # Always the same as the assignment as the AssignedDiff of the var
        iseq = kind in (StateDiff, Explicit)
        arr = (iseq ? eq_assignment : var_assignment)
        if arr[varnum] === nothing
            slot_assignments[kind] += 1
            arr[varnum] = kind => slot_assignments[kind]
        end
        (skind, slot) = arr[varnum]
        @assert skind == kind
        return slot
    end

    for i = 1:length(var_assignment)
        varclassification(result, structure, i) == External && continue
        kind = classify_var(state.structure.var_to_diff, key, i)
        kind === nothing && continue
        cache_kind = kind
        varnum = i
        if kind == AlgebraicDerivative
            cache_kind = UnassignedDiff
            varnum = invview(state.structure.var_to_diff)[varnum]::Int
        end
        slot = assign_slot!(cache_kind, varnum)
        if kind == AlgebraicDerivative
            var_assignment[i] = kind => slot
        elseif kind == AssignedDiff && var_eq_matching !== nothing
            eq = var_eq_matching[state.structure.var_to_diff[i]]
            if isa(eq, Int)
                eq_assignment[eq] = StateDiff => slot
            end
        end
    end

    if var_eq_matching !== nothing
        for eq = 1:length(state.total_incidence)
            eqclassification(result, structure, eq) == External && continue
            eqkind(result, structure, eq) == Intrinsics.Always || continue
            assgn = invview(var_eq_matching)[eq]
            @assert assgn !== unassigned
            (assgn === AlgebraicState()) || continue
            assign_slot!(Explicit, eq)
        end
    end

    (slot_assignments, var_assignment, eq_assignment)
end

"""
    struct SICMSpec

Cache partition for the state-invariant prologue
"""
struct SICMSpec
    key::TornCacheKey
end

function Base.StackTraces.show_custom_spec_sig(io::IO, owner::SICMSpec, linfo::CodeInstance, frame::Base.StackTraces.StackFrame)
    print(io, "SICM Partition for ")
    mi = Base.get_ci_mi(linfo)
    return Base.StackTraces.show_spec_sig(io, mi.def, mi.specTypes)
end

struct TornIRSpec
    key::TornCacheKey
end

function matching_for_key(state::TransformationState, key::TornCacheKey)
    (; diff_states, alg_states, explicit_eqs, var_schedule) = key
    (; result, structure, total_incidence) = state

    allow_init_eqs = key.diff_states === nothing

    may_use_var(var) = varclassification(state, var) != External && (diff_states === nothing || !(var in diff_states)) && !(var in alg_states) && varkind(state, var) == Intrinsics.Continuous
    may_use_eq(eq) = !(eq in explicit_eqs) && eqclassification(state, eq) != External && eqkind(state, eq) in (allow_init_eqs ? (Intrinsics.Initial, Intrinsics.Always) : (Intrinsics.Always,))

    # Max match is the (unique) tearing result given the choice of states
    var_eq_matching = StateSelection.complete(StateSelection.maximal_matching(structure.solvable_graph, IPOMatches;
        dstfilter = may_use_var, srcfilter = may_use_eq), nsrcs(structure.solvable_graph))

    if diff_states !== nothing
        for var in diff_states
            var_eq_matching[var] = SelectedState()
        end
    end

    for var in key.param_vars
        var_eq_matching[var] = StateInvariant()
    end

    for var in alg_states
        @assert var_eq_matching[var] === unassigned
        var_eq_matching[var] = AlgebraicState()
    end

    for (ordinal, (in_vars, out_eqs)) in enumerate(key.var_schedule)
        for in_var in in_vars
            isa(var_eq_matching[in_var], InOut) && continue
            #@assert var_eq_matching[in_var] === unassigned
            var_eq_matching[in_var] = InOut(ordinal)
        end

        for out_eq in out_eqs
            invview(var_eq_matching)[out_eq] = InOut(ordinal)
        end
    end

    for eq in explicit_eqs
        invview(var_eq_matching)[eq] = AlgebraicState()
    end

    for eq = 1:length(invview(var_eq_matching))
        if diff_states !== nothing && eqkind(state, eq) != Intrinsics.Always
            invview(var_eq_matching)[eq] = WrongEquation()
            continue
        end
        if !isassigned(total_incidence, eq) || (is_const_plus_var_known_linear(total_incidence[eq]) && invview(var_eq_matching)[eq] === unassigned)
            invview(var_eq_matching)[eq] = FullyLinear()
        end
    end

    return var_eq_matching
end

function tearing_schedule!(result::DAEIPOResult, ci::CodeInstance, key::TornCacheKey, world::UInt, settings::Settings)
    structure = make_structure_from_ipo(result)
    tstate = TransformationState(result, structure)
    return tearing_schedule!(tstate, ci, key, world, settings)
end

struct DummyOptInterp <: Compiler.AbstractInterpreter;
    world::UInt
end
Compiler.optimizer_lattice(::DummyOptInterp) = Compiler.PartialsLattice(EqStructureLattice())
Compiler.get_inference_world(interp::DummyOptInterp) = interp.world

function StateSelection.SSAUses(result::DAEIPOResult)
    eq_callees = Union{Nothing, Vector{StructuralSSARef}}[]
    var_callees = Vector{Union{Nothing, Vector{StructuralSSARef}}}(nothing, length(result.varclassification))
    for value in result.eq_callee_mapping
        if value === nothing
            push!(eq_callees, nothing)
            continue
        end
        callee = collect(unique(first.(value)))
        push!(eq_callees, callee)
    end
    for i = 1:length(result.ir.stmts)
        info = result.ir[SSAValue(i)][:info]
        info isa MappingInfo || continue
        for (callee_var, caller_mapping) in enumerate(info.mapping.var_coeffs)
            isa(caller_mapping, Incidence) || continue
            if length(rowvals(caller_mapping.row)) == 1
                caller_var = only(rowvals(caller_mapping.row)) - 1
                caller_var === 0 && continue
                result.varclassification[caller_var] == CalleeInternal || continue
                var_callees[caller_var] !== nothing || (var_callees[caller_var] = Vector{StructuralSSARef}())
                push!(var_callees[caller_var], StructuralSSARef(i))
            end
        end
    end
    return StateSelection.SSAUses(CalleeInfo.(eq_callees), CalleeInfo.(var_callees))
end

function StateSelection.MatchedSystemStructure(result::DAEIPOResult, structure, var_eq_matching)
    StateSelection.MatchedSystemStructure(structure, var_eq_matching, StateSelection.SSAUses(result))
end

function project_mapping(state::TransformationState, info::MappingInfo, key::TornCacheKey, var_eq_matching::StateSelection.Matching, total_incidence; callee_var_schedule = nothing)
    (; diff_states, alg_states, var_schedule) = key

    callee_diff_vars = BitSet()
    callee_alg_vars = BitSet()
    callee_param_vars = BitSet()
    callee_explicit_eqs = BitSet()
    externally_solved_vars = BitSet()

    callee_result = (info::MappingInfo).result

    for callee_var = 1:length(callee_result.var_to_diff)
        caller_map = info.mapping.var_coeffs[callee_var]
        if isa(caller_map, Const)
            push!(callee_param_vars, callee_var)
        else
            if varclassification(callee_result, callee_var) == External
                if all(x->(x in key.param_vars), rowvals(caller_map.row) .- 1)
                    push!(callee_param_vars, callee_var)
                end
                continue
            end
            caller_var = only(rowvals(caller_map.row))-1
            if diff_states !== nothing && caller_var in diff_states
                push!(callee_diff_vars, callee_var)
            elseif caller_var in alg_states
                push!(callee_alg_vars, callee_var)
            elseif caller_var in key.param_vars
                push!(callee_param_vars, callee_var)
            end
            eq = var_eq_matching[caller_var]
        end
    end

    for (callee_eq, caller_eq) in enumerate(info.mapping.eqs)
        caller_eq == 0 && continue
        if caller_eq in key.explicit_eqs && eqclassification(state, caller_eq) == CalleeInternal
            push!(callee_explicit_eqs, callee_eq)
        else
            var = invview(var_eq_matching)[caller_eq]
            (!isassigned(callee_result.total_incidence, callee_eq) ||
                is_const_plus_var_known_linear(callee_result.total_incidence[callee_eq]) ||
                !isassigned(total_incidence, caller_eq) ||
                is_const_plus_var_known_linear(total_incidence[caller_eq])) && continue
            @assert var !== unassigned
            if callee_var_schedule !== nothing && !any(out->callee_eq in out[2], callee_var_schedule)
                display(mss)
                cstructure = make_structure_from_ipo(callee_result)
                cvar_eq_matching = matching_for_key(callee_result, callee_key)
                display(StateSelection.MatchedSystemStructure(callee_result, cstructure, cvar_eq_matching))
                @sshow eq_orders
                @sshow callee_result.total_incidence[callee_eq]
                @sshow total_incidence[caller_eq]
                @sshow callee_var_schedule
                error("Caller Equation $(caller_eq) (callee equation $(callee_eq)) is not in the callee schedule ($sref)")
            end
        end
    end

    return TornCacheKey(callee_diff_vars, callee_alg_vars, callee_param_vars, callee_explicit_eqs,
        callee_var_schedule === nothing ? Vector{Pair{BitSet, BitSet}}() : callee_var_schedule)
end

function tearing_schedule!(state::TransformationState, ci::CodeInstance, key::TornCacheKey, world::UInt, settings::Settings)
    result_ci = find_matching_ci(ci->isa(ci.owner, SICMSpec) && ci.owner.key == key, ci.def, world)
    if result_ci !== nothing
        return result_ci
    end

    (; diff_states, alg_states, var_schedule) = key
    (; result, structure, total_incidence ) = state

    var_eq_matching = matching_for_key(state, key)

    mss = StateSelection.MatchedSystemStructure(result, structure, var_eq_matching)
    (eq_orders, callee_schedules) = compute_eq_schedule(key, total_incidence, result, mss)

    ir = index_lowering_ad!(state, key, settings)
    ir = Compiler.sroa_pass!(ir, Compiler.InliningState(DummyOptInterp(world)))
    ir = Compiler.compact!(ir)

    # First, schedule any statements that do not have state dependence
    sicm_rename = Vector{Any}(undef, length(ir.stmts))
    nonlin_sicm_rename = Vector{Any}(undef, length(ir.stmts))

    compact = IncrementalCompact(copy(ir))
    line = ir[SSAValue(1)][:line]
    #insert_node_here!(compact, NewInstruction(Expr(:call, println, "Trace: SICM"), Cvoid, line))

    isready(ssa::SSAValue) = isassigned(sicm_rename, ssa.id)
    function isready(inst::Compiler.Instruction)
        for arg in userefs(inst[:stmt])
            isa(arg[], SSAValue) || continue
            isready(arg[]) || return false
        end
        return true
    end

    non_sicm_use_counter = zeros(Int64, length(ir.stmts))

    function maybe_schedule_sicm_inst!(inst)
        return false
    end

    slot_assignments = zeros(Int, Int(LastEquationStateKind))

    callee_eq_mapping = invert_eq_callee_mapping(result.eq_callee_mapping)

    diff_states_in_callee = BitSet()
    processed_variables = Set{Int}()

    var_sols = Vector{Any}(undef, length(structure.var_to_diff))

    for (idx, var) in enumerate(key.param_vars)
        var_sols[var] = @insert_instruction_here(compact, line, settings, getfield(Argument(1), idx)::Any)
    end

    carried_states = Dict{StructuralSSARef, Any}()
    callee_position_map = Dict{StructuralSSARef, SSAValue}()

    # Generate SICM partition
    for i = 1:length(ir.stmts)
        inst = ir[SSAValue(i)]
        # TODO: This ignores all control flow
        if !maybe_schedule_sicm_inst!(inst)
            stmt = inst[:stmt]
            # Return value with state-independent part or contained equations with state-invariant non-linear component
            if is_equation_call(stmt, ir)
                #=
                eqnum = idnum(argextype(_eq_function_arg(stmt), compact))
                val = _eq_val_arg(stmt)

                display(ir)
                nonlin = schedule_nonlinear!(compact, key.param_vars, nothing, ir, val, RenameOverlayVector(sicm_rename, nonlin_sicm_rename); vars=param_arg_rename)
                val = schedule_incidence!(compact, nothing, nonlin, argextype(val, ir), -1, inst[:line]; vars=param_arg_rename)[1]
                =#
                #ir_add!(compact, inst[:line], eq_resid.args[eqnum+1], val)
                # inst[:type] = Any # Make sure this shows up in rename
            elseif is_known_invoke(stmt, equation, ir)
                # Retain this in both the SICM (to be able to make additional vars parameters) and
                # the RHS
                # sicm_rename[inst.idx] = insert_node_here!(compact, NewInstruction(inst))
                continue
            elseif is_known_invoke(stmt, variable, ir)
                # Receive the variable number based on the incidence -- we know it will be there
                varnum = idnum(ir.stmts.type[i])

                # Ensure that we only process each variable once
                if varnum ∈ processed_variables
                    record_ir!(state, "error", ir)
                    throw(UnsupportedIRException("Duplicate variable ($(varnum))", ir))
                end
                push!(processed_variables, varnum)
            elseif isexpr(stmt, :invoke)
                # Project the state mapping
                info = inst[:info]
                isa(info, MappingInfo) || continue

                sref = stmt.args[1][1]
                callee_var_schedule = get!(callee_schedules, sref, Vector{Pair{BitSet, BitSet}}())
                callee_position_map[sref] = SSAValue(i)
                callee_key = project_mapping(state, info, key, var_eq_matching, total_incidence; callee_var_schedule)

                callee_codeinst = stmt.args[1][2]
                if isa(callee_codeinst, MethodInstance)
                    callee_codeinst = Compiler.get(Compiler.code_cache(interp), callee_codeinst, nothing)
                end
                callee_result = structural_analysis!(callee_codeinst, world, settings)

                if !isempty(callee_key.explicit_eqs)
                    vars_for_final = BitSet()
                    # TODO: This is a very expensive way to compute this - we should be able to do this cheaper
                    cstructure = make_structure_from_ipo(callee_result)
                    tstate = TransformationState(callee_result, cstructure)
                    cvar_eq_matching = matching_for_key(tstate, callee_key)
                    for callee_var in 1:length(cvar_eq_matching)
                        if cvar_eq_matching[callee_var] !== unassigned
                            continue
                        end
                        push!(vars_for_final, callee_var)
                    end
                    if !isempty(vars_for_final)
                        push!(callee_var_schedule, vars_for_final=>BitSet())
                        push!(eq_orders[end], sref)
                    end
                end
                if isempty(callee_var_schedule) && isempty(callee_key.explicit_eqs)
                    # Apparently we just don't need this callee at all (e.g. it only has linear equations)
                    continue
                end

                callee_sicm_ci = tearing_schedule!(callee_result, callee_codeinst, callee_key, world, settings)

                inst[:type] = Any
                inst[:flag] = UInt32(0)
                new_stmt = copy(stmt)
                stmt.args[1] = (callee_codeinst, callee_key)

                if !isdefined(callee_sicm_ci, :rettype_const) && callee_sicm_ci.rettype !== Tuple{}
                    resize!(new_stmt.args, 2)
                    new_stmt.args[1] = callee_sicm_ci

                    in_param_vars = Expr(:call, tuple)
                    for var in callee_key.param_vars
                        varmap = info.mapping.var_coeffs[var]
                        nonlin = nothing
                        if !is_const_plus_var_known_linear(varmap)
                            nonlin = schedule_nonlinear!(compact, settings, key.param_vars, var_eq_matching, ir, 0, stmt.args[1+var], sicm_rename; vars=var_sols)
                        end
                        (argval, _) = schedule_incidence!(compact,
                            nonlin, info.mapping.var_coeffs[var], -1, line, settings; vars=var_sols)
                        @assert argval !== nothing
                        push!(in_param_vars.args, argval)
                    end

                    new_stmt.args[2] = insert_instruction_here!(compact, settings, @__SOURCE__, NewInstruction(inst; stmt=in_param_vars, type=Tuple, flag=UInt32(0)))
                    sstate = insert_instruction_here!(compact, settings, @__SOURCE__, NewInstruction(inst; stmt=new_stmt, type=Tuple, flag=UInt32(0)))
                    carried_states[sref] = CarriedSSAValue(0, sstate.id)
                else
                    carried_states[sref] = isdefined(callee_sicm_ci, :rettype_const) ? callee_sicm_ci.rettype_const : callee_sicm_ci.rettype.instance
                end
            elseif stmt === nothing || isa(stmt, ReturnNode)
                continue
            elseif isexpr(stmt, :call) || isexpr(stmt, :new) || isa(stmt, GotoNode) || isexpr(stmt, :boundscheck)
                # TODO: Pull this up, if arguments are state-independent
                continue
            else
                @sshow stmt
                error()
            end
        else
            ir[SSAValue(i)][:stmt] = CarriedSSAValue(0, sicm_rename[i].id)
        end
    end

    for (idx, var) in enumerate(key.param_vars)
        var_sols[var] = CarriedSSAValue(0, var_sols[var].id)
    end

    # Now make sure to schedule all unassigned equations that were not needed
    # to satisfy any variable/out requests.
    # TODO: This schedules them in the very last partition - would it be better
    # to schedule them as soon as they're available?
    scheduled_eqs = BitSet()
    for eq_order in eq_orders
        for eq in eq_order
            isa(eq, StructuralSSARef) && continue
            push!(scheduled_eqs, eq::Int)
        end
    end

    for (_, out_eqs) in key.var_schedule
        union!(scheduled_eqs, out_eqs)
    end

    compact1 = IncrementalCompact(ir)
    foreach(_->nothing, compact1)
    rename_hack = copy(compact1.ssa_rename)
    ir = Compiler.finish(compact1)

    # Now schedule the state-invariant non-linear part of all contained equations
    compact1 = IncrementalCompact(ir)
    (eqs, vars) = find_eqs_vars(structure.graph, compact1)
    eqs = convert(Vector{Pair{SSAValue, Vector{Union{SSAValue, NewSSAValue}}}}, eqs)
    rename_hack2 = copy(compact1.ssa_rename)

    ir = Compiler.finish(compact1)

    ssa_rename = Vector{Any}(undef, length(result.ir.stmts))

    function insert_solved_var_here!(compact1, var, curval, line)
        @insert_instruction_here(compact1, line, settings, solved_variable(var, curval)::Nothing)
    end

    isempty(var_schedule) && (var_schedule = Pair{BitSet, BitSet}[BitSet()=>BitSet()])

    callee_ordinals = Dict{StructuralSSARef, Int}()

    irs = IRCode[]
    resids = Vector{Tuple{IncrementalCompact, Expr, Union{Tuple{}, SSAValue}}}(undef, length(var_schedule))
    for (ordinal, (eq_order, sched)) in enumerate(zip(eq_orders, var_schedule))
        function schedule_missing_var!(lin_var)
            if lin_var in key.param_vars
                error()
            else
                if !((key.diff_states !== nothing && lin_var in key.diff_states) ||
                     (lin_var in key.alg_states))
                    @sshow lin_var
                    @sshow ordinal
                    @sshow eq_order
                    @sshow result.ir
                    error("Tried to schedule variable $(lin_var) that we do not have a solution to (but our scheduling should have ensured that we do)")
                end
                var_sols[lin_var] = CarriedSSAValue(ordinal, (@insert_instruction_here(compact1, line, settings, (:invoke)(nothing, Intrinsics.variable)::Incidence(lin_var)).id))
            end
        end

        # Schedule internal var-eq pairs
        nir = copy(ir)
        empty!(nir.argtypes)
        push!(nir.argtypes, Tuple)
        push!(nir.argtypes, Tuple)
        compact1 = IncrementalCompact(nir)
        line = nir[SSAValue(1)][:line]

        #insert_node_here!(compact1, NewInstruction(Expr(:call, println, "Trace: Ordinal $ordinal"), Cvoid, line))

        (in_vars, out_eqs) = sched
        for (idx, var) in enumerate(in_vars)
            var_sols[var] = CarriedSSAValue(ordinal, (@insert_instruction_here(compact1, line, settings, getfield(Argument(2), idx)::Any).id))
            insert_solved_var_here!(compact1, var, var_sols[var], line)
        end

        for eq in eq_order
            if isa(eq, StructuralSSARef)
                eqinst = ir[callee_position_map[eq]]
                @assert isexpr(eqinst[:stmt], :invoke)

                info = eqinst[:info]::MappingInfo
                callee_result = info.result

                callee_ordinal = get!(callee_ordinals, eq, 1)
                callee_sched = callee_schedules[eq][callee_ordinal]
                (callee_in_vars, callee_out_eqs) = callee_sched

                in_vars = Expr(:call, tuple)
                for var in callee_in_vars
                    nonlin = nothing
                    varmap = info.mapping.var_coeffs[var]
                    if !is_const_plus_var_known_linear(varmap)
                        nonlin = schedule_nonlinear!(compact1, settings, key.param_vars, var_eq_matching, ir, ordinal, eqinst[:stmt].args[1+var], ssa_rename; vars=var_sols, schedule_missing_var!)
                    end
                    (argval, _) = schedule_incidence!(compact1,
                        nonlin, info.mapping.var_coeffs[var], -1, line, settings; vars=var_sols, schedule_missing_var!)

                    @assert argval !== nothing
                    push!(in_vars.args, argval)
                end

                in_vars_ssa = insert_instruction_here!(compact1, settings, @__SOURCE__, NewInstruction(eqinst; stmt=in_vars, type=Tuple))

                new_stmt = copy(eqinst[:stmt])
                resize!(new_stmt.args, 2)
                spec = new_stmt.args[1]
                @assert isa(spec, NTuple{2, Any})
                new_stmt.args[1] = (spec..., callee_ordinal)
                push!(new_stmt.args, in_vars_ssa)
                new_stmt.args[2] = carried_states[eq]

                urs = userefs(new_stmt)
                for ur in urs
                    isa(ur[], NewSSAValue) || continue
                    ur[] = SSAValue(ur[].id)
                end

                callee_ordinals[eq] = callee_ordinal+1

                this_call = insert_instruction_here!(compact1, settings, @__SOURCE__, NewInstruction(eqinst; stmt=urs[]))

                this_eqresids = insert_instruction_here!(compact1, settings, @__SOURCE__, NewInstruction(eqinst; stmt=Expr(:call, getfield, this_call, 1), type=Any))

                new_state = insert_instruction_here!(compact1, settings, @__SOURCE__, NewInstruction(eqinst; stmt=Expr(:call, getfield, this_call, 2), type=Any))

                carried_states[eq] = CarriedSSAValue(ordinal, new_state.id)

                for (idx, this_callee_eq) in enumerate(callee_out_eqs)
                    this_eq = callee_eq_mapping[eq][this_callee_eq]
                    curval = insert_instruction_here!(compact1, settings, @__SOURCE__, NewInstruction(eqinst; stmt=Expr(:call, getfield, this_eqresids, idx), type=Any))
                    push!(eqs[this_eq][2], NewSSAValue(curval.id))
                end
            else
                var = invview(var_eq_matching)[eq]

                incT = state.total_incidence[eq]
                anynonlinear = !is_const_plus_var_known_linear(incT)
                nonlinearssa = nothing
                if anynonlinear
                    if isa(var, Int) && isa(vars[var], SolvedVariable)
                        nonlinearssa = schedule_nonlinear!(compact1, settings, key.param_vars, var_eq_matching, ir, ordinal, vars[var].ssa, ssa_rename; vars=var_sols, schedule_missing_var!)
                    else
                        for eqcallssa in eqs[eq][2]
                            if !isa(eqcallssa, NewSSAValue)
                                inst = ir[eqcallssa]
                                this_nonlinearssa = schedule_nonlinear!(compact1, settings, key.param_vars, var_eq_matching, ir, ordinal, inst[:stmt].args[end], ssa_rename; vars=var_sols, schedule_missing_var!)
                                line = ir[eqcallssa][:line]
                            else
                                # From getfield from a callee
                                this_nonlinearssa = SSAValue(eqcallssa.id)
                                line = compact1[eqcallssa][:line]
                            end
                            nonlinearssa = nonlinearssa === nothing ? this_nonlinearssa : ir_add!(compact1, line, settings, this_nonlinearssa, nonlinearssa, @__SOURCE__)
                        end
                        mapping = result.eq_callee_mapping[eq]
                        if mapping !== nothing
                            # Schedule the portions that were linear in the callee, but non-linear in terms of caller vars
                            for (ssa, callee_eq) in mapping
                                eqinst = ir[callee_position_map[ssa]]
                                callee_info = eqinst[:info]::MappingInfo
                                callee_var_incidence = callee_info.result.total_incidence[callee_eq]
                                function schedule_argument(var)
                                    vc = callee_info.mapping.var_coeffs[var]
                                    is_fully_state_linear(vc, nothing) && return 0.
                                    return schedule_nonlinear!(compact1, settings, key.param_vars, var_eq_matching, ir, ordinal, eqinst[:stmt].args[var+1], ssa_rename; vars=var_sols, schedule_missing_var!)
                                end
                                nonlinearssa = schedule_incidence!(compact1, nonlinearssa, callee_var_incidence, -1, line, settings; schedule_missing_var! = schedule_argument)[1]
                            end
                        end
                    end
                    @assert nonlinearssa !== nothing
                end

                eqssa = eqs[eq][1]
                line = ir[eqssa == SSAValue(0) ? SSAValue(1) : eqssa][:line]

                if isa(var, Int)
                    curval = nonlinearssa
                    (curval, thiscoeff) = schedule_incidence!(compact1, curval, incT, var, line, settings; vars=var_sols, schedule_missing_var!)
                    @assert isa(thiscoeff, Float64)
                    curval = ir_mul_const!(compact1, line, settings, 1/thiscoeff, curval, @__SOURCE__)
                    var_sols[var] = isa(curval, SSAValue) ? CarriedSSAValue(ordinal, curval.id) : curval
                    insert_solved_var_here!(compact1, var, curval, line)
                else
                    curval = nonlinearssa
                    (curval, thiscoeff) = schedule_incidence!(compact1, curval, incT, -1, line, settings; vars=var_sols, schedule_missing_var!)
                    @insert_instruction_here(compact1, line, settings, InternalIntrinsics.contribution!(eq, Explicit, curval)::Nothing)
                end
            end
        end

        # Schedule non-linear part of equations that are returned
        eq_resids = Expr(:call, tuple)
        for eq in out_eqs
            var = invview(var_eq_matching)[eq]
            if isa(var, Int) && isa(vars[var], SolvedVariable)
                nonlinearssa = schedule_nonlinear!(compact1, settings, key.param_vars, var_eq_matching, ir, ordinal, vars[var].ssa, ssa_rename; vars=var_sols, schedule_missing_var!)
            else
                if isempty(eqs[eq][2])
                    nonlinearssa = nothing
                else
                    eqcallssa = only(eqs[eq][2])
                    if isa(eqcallssa, NewSSAValue)
                        nonlinearssa = SSAValue(eqcallssa.id)
                    else
                        nonlinearssa = schedule_nonlinear!(compact1, settings, key.param_vars, var_eq_matching, ir, ordinal, ir[eqcallssa][:stmt].args[3], ssa_rename; vars=var_sols, schedule_missing_var!)
                    end
                end
            end
            push!(eq_resids.args, nonlinearssa === nothing ? 0.0 : nonlinearssa)
        end

        line = ir[SSAValue(length(ir.stmts))][:line]
        eq_resid_ssa = isempty(out_eqs) ? () : @insert_instruction_here(compact1, line, settings,  eq_resids::Tuple)

        state_resid = Expr(:call, tuple)
        resids[ordinal] = (compact1, state_resid, eq_resid_ssa)
    end

    sicm_resid = Expr(:call, tuple)
    sicm_resid_rename = Dict{CarriedSSAValue, Dict{Int, Union{SSAValue, NewSSAValue}}}()
    for i = length(resids):-1:1
        (this_compact, this_resid, eq_resid_ssa) = resids[i]
        line = ir[SSAValue(length(ir.stmts))][:line]
        state_resid_ssa = @insert_instruction_here(this_compact, line, settings, this_resid::Tuple)
        tup_resid_ssa = @insert_instruction_here(this_compact, line, settings, tuple(eq_resid_ssa, state_resid_ssa)::Tuple{Tuple, Tuple})
        @insert_instruction_here(this_compact, line, settings, (return tup_resid_ssa)::Union{})

        # Rewrite SICM to state references
        line = this_compact[SSAValue(1)][:line]
        resid_recv = Argument(1)
        for j = 1:(this_compact.result_idx-1)
            inst = this_compact[SSAValue(j)]
            stmt = inst[:stmt]
            @assert !isa(stmt, CarriedSSAValue)
            urs = userefs(stmt)
            any = false
            for ur in urs
                isa(ur[], CarriedSSAValue) || continue
                rename_dict = get!(sicm_resid_rename, ur[], Dict{Int, Union{SSAValue, NewSSAValue}}(ur[].ordinal => SSAValue(ur[].id)))
                for k = (ur[].ordinal+1):i
                    haskey(rename_dict, k) && continue
                    (_, oldresid) = (k-1 == 0) ? (compact, sicm_resid) : resids[k-1]
                    push!(oldresid.args, rename_dict[k-1])
                    inserted = insert_node!(i == k ? this_compact : resids[k][1], SSAValue(1),
                        NewInstruction(Expr(:call, getfield, resid_recv, length(oldresid.args)-1), Incidence(Any), line))
                    rename_dict[k] = inserted
                end
                # Temporarily remove this stmt from compact during modification
                any || (this_compact[SSAValue(j)] = nothing)
                ur[] = rename_dict[i]
                any = true
            end
            any || continue
            this_compact[SSAValue(j)] = urs[]
        end

        this_ir = Compiler.finish(this_compact)
        this_ir = Compiler.compact!(this_ir)

        push!(irs, this_ir)
    end
    reverse!(irs)

    if compact.result_idx == 1
        ir_sicm = nothing
        src = nothing
        sig = Tuple
        debuginfo = Core.DebugInfo(:sicm)
        sicm_rettype = Tuple{}
    else
        resid_ssa = @insert_instruction_here(compact, line, settings, sicm_resid::Tuple)
        @insert_instruction_here(compact, line, settings, (return resid_ssa)::Union{})
        ir_sicm = Compiler.finish(compact)
        resize!(ir_sicm.cfg.blocks, 1)
        empty!(ir_sicm.cfg.blocks[1].succs)
        widen_extra_info!(ir_sicm)
        empty!(ir_sicm.argtypes)
        push!(ir_sicm.argtypes, Tuple)
        Compiler.verify_ir(ir_sicm)
        src = ir_to_src(ir_sicm, settings)
        sig = Tuple{Tuple}
        debuginfo = src.debuginfo
        sicm_rettype = Tuple
    end

    sicm_ci = cache_dae_ci!(ci, src, debuginfo, sig, SICMSpec(key); rettype=sicm_rettype)
    if src !== nothing
        ccall(:jl_add_codeinst_to_jit, Cvoid, (Any, Any), sicm_ci, src)
    end

    torn_ci = cache_dae_ci!(ci, TornIR(ir_sicm, irs), nothing, sig, TornIRSpec(key))

    return sicm_ci
end

function remap_info(remap_ir!, info)
    # TODO: This is pretty aweful, but it works for now.
    # It'll go away when we switch to IPO.
    if isa(info, Diffractor.FRuleCallInfo) && info.frule_call.rt === Const(nothing)
        info = info.info
    end
    isa(info, Compiler.ConstCallInfo) || return info
    results = map(info.results) do result
        result === nothing && return result
        if isa(result, Compiler.SemiConcreteResult)
            let ir = copy(result.ir)
                remap_ir!(ir)
                Compiler.SemiConcreteResult(result.edge, ir, result.effects, result.spec_info)
            end
        elseif isa(result, Compiler.ConstPropResult)
            if isa(result.result.src, AnalyzedSource)
                # Result could have been discarded (e.g. by limited_src)
                remap_ir!(result.result.src.ir)
            end
            return result
        else
            return result
        end
    end
    return Compiler.ConstCallInfo(info.call, results)
end

function widen_extra_info!(ir)
    for i = 1:length(ir.stmts)
        info = ir.stmts[i][:info]
        if isa(info, Diffractor.FRuleCallInfo)
            info = info.info
        end
        ir.stmts[i][:info] = remap_info(widen_extra_info!, info)
        inst = ir.stmts[i][:inst]
        if isa(inst, PiNode)
            ir.stmts[i][:inst] = PiNode(inst.val, widenconst(inst.typ))
        end
        ir.stmts[i][:type] = widenconst(ir.stmts[i][:type])
    end
end

function ir_to_src(ir::IRCode, settings::Settings; slotnames = nothing, widen = true, isva = false)
    ir.debuginfo.def === nothing && (ir.debuginfo.def = :var"generated IR for OpaqueClosure")
    maybe_rewrite_debuginfo!(ir, settings)
    nargtypes = length(ir.argtypes)
    src = ccall(:jl_new_code_info_uninit, Ref{CodeInfo}, ())
    if slotnames === nothing
        src.slotnames = Symbol[Symbol("arg$i") for i = 1:nargtypes]
    else
        length(slotnames) == nargtypes || error("mismatched `argtypes` and `slotnames`")
        src.slotnames = slotnames
    end
    src.nargs = nargtypes
    src.isva = isva
    src.slotflags = fill(zero(UInt8), nargtypes)
    src.slottypes = copy(ir.argtypes)
    Compiler.replace_code_newstyle!(src, ir)
    widen && Compiler.widen_all_consts!(src)
    return src
end

function maybe_rewrite_debuginfo!(ir::IRCode, settings::Settings)
    settings.insert_ssa_debuginfo && rewrite_debuginfo!(ir)
    return ir
end

function rewrite_debuginfo!(ir::IRCode)
    for (i, stmt) in enumerate(ir.stmts)
        type = stmt[:type]
        annotation = type === nothing ? "" : " (inferred type: $type)"
        # Work around `show` functions requiring `invokelatest` queries
        # that may be problematic to execute from within generated functions.
        local inst
        try
            inst = string(stmt[:inst])
        catch e
            isa(e, UndefVarError) && continue
            rethrow()
        end
        filename = Symbol("%$i = $inst", annotation)
        lineno = LineNumberNode(1, filename)
        stmt[:line] = insert_debuginfo!(ir.debuginfo, i, lineno, stmt[:line])
    end
end

function cache_dae_ci!(old_ci, src, debuginfo, abi, owner; rettype=Tuple)
    mi = old_ci.def
    edges = Core.svec(old_ci)
    daef_ci = CodeInstance(abi === nothing ? old_ci.def : Core.ABIOverride(abi, old_ci.def), owner, rettype, Union{}, nothing, src, Int32(0),
        old_ci.min_world, old_ci.max_world, old_ci.ipo_purity_bits,
        nothing, debuginfo, edges)
    Compiler.store_backedges(daef_ci, edges)
    ccall(:jl_mi_cache_insert, Cvoid, (Any, Any), mi, daef_ci)
    return daef_ci
end

function replace_call!(ir::Union{IRCode,IncrementalCompact}, idx::SSAValue, @nospecialize(new_call), settings::Settings, source)
    replace_call!(ir, idx, new_call)
    settings.insert_stmt_debuginfo || return new_call
    debuginfo = isa(ir, IncrementalCompact) ? ir.ir.debuginfo : ir.debuginfo
    if isa(source, Tuple)
        ir[idx][:line] = source
    else
        maybe_insert_debuginfo!(debuginfo, settings, idx.id, source, ir[idx][:line])
    end
    return new_call
end

function replace_call!(ir::Union{IRCode,IncrementalCompact}, idx::SSAValue, @nospecialize(new_call))
    @assert !isa(ir[idx][:inst], PhiNode)
    ir[idx][:inst] = new_call
    ir[idx][:type] = Any
    ir[idx][:info] = Compiler.NoCallInfo()
    ir[idx][:flag] |= Compiler.IR_FLAG_REFINED
    return idx
end

function maybe_insert_debuginfo!(compact::IncrementalCompact, settings::Settings, source::LineNumberNode, previous = nothing, i = compact.result_idx)
    maybe_insert_debuginfo!(compact.ir.debuginfo, settings, i, source, previous)
end

function maybe_insert_debuginfo!(debuginfo::DebugInfoStream, settings::Settings, i::Integer, source::LineNumberNode, previous)
    settings.insert_stmt_debuginfo || return previous
    insert_debuginfo!(debuginfo, i, source, previous)
end

function insert_debuginfo!(debuginfo::DebugInfoStream, i::Integer, source::LineNumberNode, previous)
    prev_edge_index = prev_edge_line = nothing
    if previous !== nothing && isa(previous, Tuple)
        prev_edge_index, prev_edge_line = previous[2], previous[3]
    end
    prev_edge = prev_edge_index === nothing ? nothing : get(debuginfo.edges, prev_edge_index, nothing)
    edge = new_debuginfo_edge(source, prev_edge, prev_edge_line)
    push!(debuginfo.edges, edge)
    edge_index = length(debuginfo.edges)
    line = Int32.((i, edge_index, 1))
    length(debuginfo.codelocs) ≥ 3i || resize!(debuginfo.codelocs, 3i)
    debuginfo.codelocs[3(i - 1) + 1] = line[1]
    debuginfo.codelocs[3(i - 1) + 2] = line[2]
    debuginfo.codelocs[3(i - 1) + 3] = line[3]
    return line
end

function new_debuginfo_edge((; file, line)::LineNumberNode, prev_edge, prev_edge_line)
    if prev_edge !== nothing && prev_edge_line !== nothing
        codelocs = Int32[line, 1, prev_edge_line]
        edges = Core.svec(prev_edge)
    else
        codelocs = [line, 0, 0]
        edges = Core.svec()
    end
    firstline = codelocs[1]
    compressed = ccall(:jl_compress_codelocs, Any, (Int32, Any, Int), firstline, codelocs, 1)
    DebugInfo(@something(file, :none), nothing, edges, compressed)
end

is_solved_variable(stmt) = isexpr(stmt, :call) && stmt.args[1] == solved_variable ||
    isexpr(stmt, :invoke) && stmt.args[2] == solved_variable


"""
    replace_if_intrinsic!(compact, ssa, du, u, p, t, var_assignment)

In the DAE IR, there are a number of intrinsics such as `variable()`, `state_ddt()`, or `equation!()`.
When converting these to normal julia functions those intrinsics need to be mapped to concrete values;
generally one of the inputs to the function (e.g. the `u`, `p` and `t` arguments) which we typically address as `Argument`s, but literals or `SSAValues` can also be used.

This replaces:
 - `sim_time()` with `t`
 - `variable(...)`/`state_ddt` for selected states with `u[i]` or `du[i]` as appropriate
 - `variable(...)`/`state_ddt` for nonselected states with an unusued marker
 - `_1` with `p` (as in DAE IR the parameters are fields of a functor)
 - `equation!`, `observed!`, `singularity_root!`, `time_periodic_singularity!` are deleted.
 - `epsilon(...)` is replaced with `0.0`

If doing an ODE, then can put `nothing` for `du` argument as we know it will not be used

If `var_assignment` is `nothing`, all variables are assumed unassigned. In this
case `u` and `du` may be `nothing` as well.
"""
function replace_if_intrinsic!(compact, settings, ssa, du, u, p, t, var_assignment)
    inst = compact[ssa]
    stmt = inst[:inst]
    # Transform references to `Argument(1)` into `p`
    if isexpr(stmt, :invoke) || isexpr(stmt, :call)
        for (arg_ii, arg) in enumerate(stmt.args)
            if arg == Argument(1)
                stmt.args[arg_ii] = p
            end
        end
    end

    # Transform calls from `variable()` or `state_ddt()` into `u[var_idx]`
    if is_known_invoke_or_call(stmt, variable, compact)
        var = idnum(inst[:type])
        if var_assignment === nothing
            var_idx = 0
        else
            kind, var_idx = var_assignment[var]
            @assert kind !== AssignedDiff || du !== nothing
        end

        if iszero(var_idx)
            # This is some `variable` or `state_ddt` that is not a selected state,
            # but for some reason, wasn't deleted in any prior pass.
            inst[:inst] = GlobalRef(DAECompiler.Intrinsics, :_VARIABLE_UNASSIGNED)
        else
            source = kind === AssignedDiff ? du : u
            replace_call!(compact, ssa, Expr(:call, getindex, source, var_idx), settings, @__SOURCE__)
        end
    elseif is_known_invoke_or_call(stmt, sim_time, compact)
        inst[:inst] = t
    elseif is_known_invoke_or_call(stmt, equation, compact)
        inst[:inst] = Intrinsics.placeholder_equation
    elseif is_equation_call(stmt, compact) ||
        is_solved_variable(stmt)
        # these have no meaning outside of DAE IR.
        # Its likely the transform has already converted these to something else
        # but if not then we are allowed to remove them
        inst[:inst] = nothing
    elseif is_known_invoke_or_call(stmt, epsilon, compact)
        # It is 0.0 unless we are running `epsjac` in which case it should have already been handled
        inst[:inst] = 0.0
    end
end

struct UnexpectedIntrinsicException <: Exception
    inst
end
function Base.showerror(io::IO, x::UnexpectedIntrinsicException)
    println(io, "It is expected that all DAECompiler intrinsics are removed before using the native compiler. But found $(x.inst).")
end

"""
Checks for any DAECompiler intrinstic (using the rule that any function defined in DAECompiler.Intrinstics is an intrinstic)
and if it finds one, throws an error.
"""
function check_for_daecompiler_intrinstics(ir::IRCode)
    for i in 1:length(ir.stmts)
        inst = ir[SSAValue(i)][:inst]
        isexpr(inst, :invoke) || continue
        mi = inst.args[1]
        if isa(mi, CodeInstance)
            mi = mi.def
        end
        if mi.def.module == DAECompiler.Intrinsics
            throw(UnexpectedIntrinsicException(inst))
        end
    end
end

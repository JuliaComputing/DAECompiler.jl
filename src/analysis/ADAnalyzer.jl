# TODO: This should possibly be in Diffractor

using Compiler
using Diffractor
using Core: SimpleVector, CodeInstance, Const
using Compiler: ArgInfo, StmtInfo, AbstractInterpreter, InferenceParams, OptimizationParams,
    AbsIntState, CallInfo, InferenceResult, InferenceState

struct ADCache; end

"""
    struct ADAnalyzer

A basic abstract interpreter that considers the possibility that any call may later be
AD'd using Diffractor.
"""
struct ADAnalyzer <: Compiler.AbstractInterpreter
    world::UInt
    inf_cache::Vector{Compiler.InferenceResult}
    edges::SimpleVector # additional edges
    function ADAnalyzer(;
            world::UInt = Base.get_world_counter(),
            inf_cache::Vector{Compiler.InferenceResult} = Compiler.InferenceResult[],
            edges = Compiler.empty_edges)
        new(world, inf_cache, edges)
    end
end

Compiler.InferenceParams(interp::ADAnalyzer) = Compiler.InferenceParams()
Compiler.OptimizationParams(interp::ADAnalyzer) = Compiler.OptimizationParams(;
        assume_fatal_throw = true,
        compilesig_invokes = false
    )
Compiler.get_inference_world(interp::ADAnalyzer) = interp.world
Compiler.get_inference_cache(interp::ADAnalyzer) = interp.inf_cache
Compiler.cache_owner(::ADAnalyzer) = ADCache()

Diffractor.disable_forward(interp::ADAnalyzer) = Compiler.NativeInterpreter(interp.world)

@override function Compiler.abstract_call_known(interp::ADAnalyzer, @nospecialize(f),
    arginfo::ArgInfo, si::StmtInfo, sv::AbsIntState, max_methods::Int)
    ret = @invoke Compiler.abstract_call_known(interp::AbstractInterpreter, f::Any,
        arginfo::ArgInfo, si::StmtInfo, sv::AbsIntState, max_methods::Int)
    return Diffractor.fwd_abstract_call_gf_by_type(interp, f, arginfo, si, sv, ret)
end

@override function Compiler.src_inlining_policy(interp::ADAnalyzer,
        @nospecialize(src), @nospecialize(info::CallInfo), stmt_flag::UInt32)
    if isa(info, Diffractor.FRuleCallInfo) && info.frule_call.rt !== Const(nothing)
        return false
    end
    if isa(src, Compiler.OptimizationState)
        @sshow src.linfo
        @sshow src.src
        error()
    end
    if isa(src, AnalyzedSource)
        return src.inline_cost != Compiler.MAX_INLINE_COST
    end
    return @invoke Compiler.src_inlining_policy(interp::AbstractInterpreter, src::Any, info, stmt_flag)
end

struct AnalyzedSource
    ir::Compiler.IRCode
    slotnames::Vector{Any}
    inline_cost::Compiler.InlineCostType
    nargs::UInt
    isva::Bool
end

@override function Compiler.result_edges(interp::ADAnalyzer, caller::InferenceState)
    edges = @invoke Compiler.result_edges(interp::AbstractInterpreter, caller::InferenceState)
    Core.svec(edges..., interp.edges...)
end

function get_slotnames(def::Method)
    names = split(def.slot_syms, '\0')
    return map(Symbol, names)
end

@override function Compiler.transform_result_for_cache(interp::ADAnalyzer, result::InferenceResult, edges::SimpleVector)
    ir = result.src.optresult.ir
    slotnames = get_slotnames(result.linfo.def)
    params = Compiler.OptimizationParams(interp)
    return AnalyzedSource(ir, slotnames, Compiler.compute_inlining_cost(interp, result), result.src.src.nargs, result.src.src.isva)
end

@override function Compiler.transform_result_for_local_cache(interp::ADAnalyzer, result::InferenceResult, edges::SimpleVector)
    if Compiler.result_is_constabi(interp, result)
        return nothing
    end
    ir = result.src.optresult.ir
    slotnames = get_slotnames(result.linfo.def)
    params = Compiler.OptimizationParams(interp)
    return AnalyzedSource(ir, slotnames, Compiler.compute_inlining_cost(interp, result), result.src.src.nargs, result.src.src.isva)
end

function Compiler.retrieve_ir_for_inlining(ci::CodeInstance, result::AnalyzedSource)
    return Compiler.retrieve_ir_for_inlining(Compiler.get_ci_mi(ci), result.ir, true)
end

function Compiler.retrieve_ir_for_inlining(mi::MethodInstance, result::AnalyzedSource, preserve_local_sources::Bool)
    return Compiler.retrieve_ir_for_inlining(mi, result.ir, true)
end

@noinline function single_match_error(@nospecialize tt)
    sig = sprint(Base.show_tuple_as_call, Symbol(""), tt)
    error(lazy"Could not find single target method for `$sig`")
end

function get_method_instance(@nospecialize(tt), world)
    match = Base._methods_by_ftype(tt, 1, world)
    isempty(match) && single_match_error(tt)
    match = only(match)
    mi = Compiler.specialize_method(match)
end

function ad_typeinf(world, tt; force_inline_all=false, edges=Compiler.empty_edges)
    @assert !force_inline_all
    interp = ADAnalyzer(; world, edges)
    mi = get_method_instance(tt, world)
    ci = Compiler.typeinf_ext(interp, mi, Compiler.SOURCE_MODE_ABI)
end

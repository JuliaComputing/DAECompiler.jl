module DAECompilerCthulhuExt

using Core.IR
using DAECompiler: DAECompiler, DAEIPOResult, UncompilableIPOResult, Settings, ADAnalyzer, structural_analysis!, find_matching_ci, matched_system_structure, StructureCache, ir_to_src, get_method_instance, MappingInfo, AnalyzedSource
using Compiler: Compiler, InferenceResult, NativeInterpreter, SOURCE_MODE_GET_SOURCE, typeinf_ext, Effects, get_ci_mi, NoCallInfo
using Accessors: setproperties
using Diffractor: FRuleCallInfo

using Cthulhu: Cthulhu, get_module_for_compiler_integration, CthulhuState, AbstractProvider, Command, generate_code_instance, lookup, value_for_default_command, perform_action, get_inference_world, cached_return_type, cached_exception_type
const CompilerIntegration = get_module_for_compiler_integration(; use_compiler_stdlib = true)
using .CompilerIntegration: LookupResult, get_effects, InferenceDict, PC2Remarks, PC2CallMeta, PC2Effects, PC2Excts, ConstPropCallInfo, SemiConcreteCallInfo, OCCallInfo

mutable struct DAEProvider <: AbstractProvider
    world::UInt
    settings::Settings
    remarks::InferenceDict{PC2Remarks}
    calls::InferenceDict{PC2CallMeta}
    effects::InferenceDict{PC2Effects}
    exception_types::InferenceDict{PC2Excts}
end
DAEProvider(; world = Base.tls_world_age(), settings = Settings()) = DAEProvider(world, settings, InferenceDict{PC2Remarks}(), InferenceDict{PC2CallMeta}(), InferenceDict{PC2Effects}(), InferenceDict{PC2Excts}())

Cthulhu.get_inference_world(provider::DAEProvider) = provider.world

function Cthulhu.find_method_instance(provider::DAEProvider, @nospecialize(tt::Type{<:Tuple}), world::UInt)
    return get_method_instance(tt, world)
end

function check_result(ci::CodeInstance)
    isa(ci.inferred, UncompilableIPOResult) && throw(ci.inferred.error)
    return true
end

function Cthulhu.generate_code_instance(provider::DAEProvider, mi::MethodInstance)
    world = get_inference_world(provider)
    ci = find_matching_ci(ci->ci.owner == StructureCache(), mi, world)
    # XXX: We should not cache the CodeInstance this way, or at least invalidate in the provider in `toggle_setting!`.
    if ci !== nothing
        haskey(provider.remarks, ci) && return ci
    else
        provider.settings.force_inline_all && @warn "`force_inline_all=true` is not supported yet; this setting will be ignored"
        analyzer = ADAnalyzer(; world)
        ci_pre = typeinf_ext(analyzer, mi, SOURCE_MODE_GET_SOURCE)
        result = structural_analysis!(ci_pre, world, provider.settings)
        ci = find_matching_ci(ci->ci.owner == StructureCache(), mi, world)::CodeInstance
    end

    check_result(ci)
    provider.remarks[ci] = PC2Remarks()
    provider.calls[ci] = PC2CallMeta()
    provider.effects[ci] = PC2Effects()
    provider.exception_types[ci] = PC2Excts()

    return ci
end

get_override(provider::DAEProvider, info::ConstPropCallInfo) = nothing
get_override(provider::DAEProvider, info::SemiConcreteCallInfo) = nothing
get_override(provider::DAEProvider, info::OCCallInfo) = nothing

Cthulhu.get_pc_remarks(provider::DAEProvider, key::CodeInstance) = get(provider.remarks, key, nothing)
Cthulhu.get_pc_effects(provider::DAEProvider, key::CodeInstance) = get(provider.effects, key, nothing)
Cthulhu.get_pc_excts(provider::DAEProvider, key::CodeInstance) = get(provider.exception_types, key, nothing)

Cthulhu.lookup(provider::DAEProvider, result::InferenceResult, optimize::Bool) = nothing
function Cthulhu.lookup(provider::DAEProvider, ci::CodeInstance, optimize::Bool)
    if isa(ci.inferred, AnalyzedSource)
        mi = get_ci_mi(ci)
        new_ci = generate_code_instance(provider, mi)
        check_result(new_ci)
        @assert isa(new_ci.inferred, DAEIPOResult) "Inferred type of newly generated `CodeInstance` must be `DAEIPOResult`, got `$(typeof(new_ci.inferred))`"
        return lookup(provider, new_ci, optimize)
    end
    result = ci.inferred::DAEIPOResult
    ir = copy(result.ir)
    pushfirst!(ir.argtypes, Tuple)
    src = ir_to_src(ir, provider.settings; widen = false)
    src.ssavaluetypes = copy(ir.stmts.type)
    src.min_world = @atomic ci.min_world
    src.max_world = @atomic ci.max_world
    optimized = true
    rt = cached_return_type(ci)
    exct = cached_exception_type(ci)
    infos = widen_call_infos(ir.stmts.info)
    return LookupResult(ir, src, rt, exct, infos, src.slottypes, get_effects(ci), optimized)
end

function widen_call_infos(infos)
    infos = copy(infos)
    for (i, info) in enumerate(infos)
        while true
            isa(info, FRuleCallInfo) && (info = info.info; continue)
            isa(info, MappingInfo) && (info = info.info; continue)
            break
        end
        infos[i] = info
    end
    return infos
end

function toggle_setting(provider::DAEProvider, setting::Symbol, value)
    return setproperties(provider.settings, NamedTuple((setting => value,)))
end

function Cthulhu.menu_commands(provider::DAEProvider)
    commands = Cthulhu.default_menu_commands(provider)
    filter!(x -> !in(x.name, (:optimize, :dump_params, :llvm, :native, :inlining_costs)), commands)
    push!(commands, toggle_setting(provider, 'f', :force_inline_all, "force inline all"))
    push!(commands, perform_action(show_mss, 'm', :show_mss, :actions, "Show system structure"))
    return commands
end

function show_mss(state::CthulhuState)
    result = state.ci.inferred::DAEIPOResult
    terminal = state.terminal
    io = terminal.out_stream::IO
    mss = matched_system_structure(result, state.provider.settings.mode)
    (_, width) = displaysize(terminal)
    printstyled(io, '\n', '-'^((width - 26) ÷ 2), " Showing system structure ", '-'^((width - 26) ÷ 2), '\n'; color = :light_black)
    show(io, MIME"text/plain"(), mss)
    printstyled(io, '\n', '-'^width, "\n\n"; color = :light_black)
end

function toggle_setting(provider::DAEProvider, key::Char, name::Symbol, description::String = string(name))
  callback = state -> toggle_setting!(state, name)
  Command(callback, key, name, description, :toggles)
end

function Cthulhu.value_for_command(provider::DAEProvider, state::CthulhuState, command::Command)
    hasproperty(provider.settings, command.name) &&
        return getproperty(provider.settings, command.name)
    return value_for_default_command(provider, state, command)
end

function toggle_setting!(state::CthulhuState, name::Symbol)
  (; provider) = state
  (; settings) = provider
  value = !getproperty(settings, name)::Bool
  provider.settings = setproperties(settings, NamedTuple((name => value,)))
  state.display_code = true
end

DAECompiler.dae_provider(args...; kwargs...) = DAEProvider(args...; kwargs...)

end # module

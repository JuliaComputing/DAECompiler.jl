"""
    factory_gen(world, source, settings, f)

This is the compile-time entry point for DAECompiler code generation. It drives all other pieces.
"""
function factory_gen(@nospecialize(fT), settings::Settings, world::UInt = Base.get_world_counter())
    factory_mi = get_method_instance(Tuple{typeof(factory),Val{settings},typeof(fT)}, world)

    # First, perform ordinary type inference, under the assumption that we may need to AD
    # parts of the function later.
    ci = ad_typeinf(world, Tuple{fT}; force_inline_all=settings.force_inline_all, edges=Core.svec(factory_mi))

    # Perform or lookup DAECompiler specific analysis for this system.
    result = structural_analysis!(ci, world, settings)

    if isa(result, UncompilableIPOResult)
        if isa(result.error, FunctionErrorsException)
            # If the issue is that the function always errors, just call the function itself in the factory -
            # we'll probably see what the error is.
            # TODO: We may still want to transform the function to drop any errors that may be conditional on
            # variable state.
            return Base.generated_body_to_codeinfo(
                Expr(:lambda, Any[:var"#self", :settings, :f], Expr(:block, Expr(:return, Expr(:call, :f)))),
                @__MODULE__, false)
        else
            return Base.generated_body_to_codeinfo(
                Expr(:lambda, Any[:var"#self", :settings, :f], Expr(:block, Expr(:return, Expr(:call, throw, result.error)))),
                @__MODULE__, false)
        end
    end

    structure = make_structure_from_ipo(result)
    tstate = TransformationState(result, structure)

    # Ensure that the system is a consistent DAE system
    err = StateSelection.check_consistency(tstate, nothing)
    if err !== nothing
        return Base.generated_body_to_codeinfo(
            Expr(:lambda, Any[:var"#self", :settings, :f], Expr(:block, Expr(:return, Expr(:call, throw, err)))),
            @__MODULE__, false)
    end

    # Select differential and algebraic states
    if settings.skip_optimizations
        diff_key = UnoptimizedKey()
    else
        ret = top_level_state_selection!(tstate)
        if isa(ret, UncompilableIPOResult)
            return Base.generated_body_to_codeinfo(
                Expr(:lambda, Any[:var"#self", :settings, :f], Expr(:block, Expr(:return, Expr(:call, throw, ret.error)))),
                @__MODULE__, false)
        end
        (diff_key, init_key) = ret
        if settings.mode in (DAE, DAENoInit, ODE, ODENoInit)
            tearing_schedule!(tstate, ci, diff_key, world, settings)
        end
        if settings.mode in (InitUncompress, DAE, ODE)
            tearing_schedule!(tstate, ci, init_key, world, settings)
        end
    end

    # Generate the IR implementation of `factory`, returning the DAEFunction/ODEFunction
    slotnames = nothing
    if settings.mode in (DAE, DAENoInit)
        ir_factory, slotnames = dae_factory_gen(tstate, ci, diff_key, world, settings, settings.mode == DAE ? init_key : nothing)
    elseif settings.mode in (ODE, ODENoInit)
        ir_factory, slotnames = ode_factory_gen(tstate, ci, diff_key, world, settings, settings.mode == ODE ? init_key : nothing)
    elseif settings.mode == InitUncompress
        ir_factory = init_uncompress_gen(result, ci, init_key, diff_key, world, settings)
    else
        return :(error("Unknown generation mode: $(settings.mode)"))
    end

    src = ir_to_src(ir_factory, settings; slotnames)
    src.ssavaluetypes = length(src.code)
    src.min_world = @atomic ci.min_world
    src.max_world = @atomic ci.max_world
    src.edges = Core.svec(ci.def)

    return src
end

function factory_gen(world::UInt, source::Method, @nospecialize(_gen), settings, @nospecialize(fT))
    settings = settings.parameters[1]::Settings
    factory_gen(fT, settings, world)
end

"""
    factory(::Val{::Settings}, f)

Given Julia function `f` compatible with DAECompiler's model representation, return a representation
suitable for solving using numerical integrators. The particular representation depends on the passed
settings.

For `dae`, return a `DAEFunction` suitable for use with DAEProblem.
The DAEFunction will be specific to the parameterization of `f`.

For `ode`, return an `ODEFunction` suitable for use with ODEProblem.
The ODEFunction will be specific to the parameterization of `f`.

To obtain a new parameterization, re-run this function. The runtime complexity of this function is
at most equivalent to one ordinary evaluation of `f`, but this function may have significant
compile-time cost (cached as usual for Julia code).
"""
function factory end
factory(f) = factory(Val(Settings()), f)

function refresh()
    @eval function factory(settings::Val #= {::Settings} =#, f)
        $(Expr(:meta, :generated_only))
        $(Expr(:meta, :generated, factory_gen))
    end
    return nothing
end
refresh()

# methods are to be added via the Cthulhu extension
function dae_provider end

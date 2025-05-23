
struct CalleeMapping
    var_coeffs::Vector{Any}
    eqs::Vector{Int}
    applied_scopes::Vector{Any}
end

struct CallerMappingState
    result::DAEIPOResult
    caller_var_to_diff::DiffGraph
    caller_varclassification::Vector{VarEqClassification}
    caller_varkind::Union{Vector{Intrinsics.VarKind}, Nothing}
    caller_eqclassification::Vector{VarEqClassification}
end

function compute_missing_coeff!(coeffs, (;result, caller_var_to_diff, caller_varclassification, caller_varkind)::CallerMappingState, v)
    # First find the rootvar, and if we already have a coeff for it
    # apply the derivatives.
    ndiffs = 0
    calle_inv = invview(result.var_to_diff)
    while calle_inv[v] !== nothing && !isassigned(coeffs, v)
        ndiffs += 1
        v = calle_inv[v]
    end

    if !isassigned(coeffs, v)
        @assert v > result.nexternalargvars # Arg vars should have already been mapped
        # Reached the root and it's an internal variable. We need to allocate
        # it in the caller now
        coeffs[v] = Incidence(add_vertex!(caller_var_to_diff))
        push!(caller_varclassification, result.varclassification[v] == External ? Owned : CalleeInternal)
        push!(caller_varkind, result.varkinds[v])
    end
    thisinc = coeffs[v]

    for _ = 1:ndiffs
        dv = result.var_to_diff[v]
        coeffs[dv] = structural_inc_ddt(caller_var_to_diff, caller_varclassification, caller_varkind, thisinc)
        v = dv
    end

    return nothing
end

apply_linear_incidence(𝕃, ret::Type, caller::CallerMappingState, mapping::CalleeMapping) = ret
apply_linear_incidence(𝕃, ret::Const, caller::CallerMappingState, mapping::CalleeMapping) = ret
function apply_linear_incidence(𝕃, ret::Incidence, caller::Union{CallerMappingState, Nothing}, mapping::CalleeMapping)
    coeffs = mapping.var_coeffs

    const_val = ret.typ
    new_row = _zero_row()

    for (v_offset, coeff) in zip(rowvals(ret.row), nonzeros(ret.row))
        v = v_offset - 1

        # Time dependence persists as itself
        if v == 0
            new_row[v_offset] += coeff
            continue
        end

        if !isassigned(coeffs, v)
            @assert caller !== nothing
            compute_missing_coeff!(coeffs, caller, v)
        end

        replacement = coeffs[v]
        if isa(replacement, Incidence)
            new_row .+= replacement.row .* coeff
        else
            if isa(replacement, Const)
                if isa(const_val, Const)
                    new_const_val = const_val.val + replacement.val * coeff
                    if isa(new_const_val, Float64)
                        const_val = Const(new_const_val)
                    else
                        const_val = widenconst(const_val)
                    end
                else
                    const_val = widenconst(const_val)
                end
            else
                # The replacement has some unknown type - we need to widen
                # all the way here.
                return widenconst(const_val)
            end
        end
    end

    return Incidence(const_val, new_row)
end

function apply_linear_incidence(𝕃, ret::Eq, caller::CallerMappingState, mapping::CalleeMapping)
    eq_mapping = mapping.eqs[ret.id]
    if eq_mapping == 0
        error("I removed these from StructuralRefiner for conceptual reasons - if we hit these, lets revisit")
        #push!(caller_eqclassification, Owned)
        #push!(caller_eqkinds, result.eqkinds[ret.id])
        mapping.eqs[ret.id] = eq_mapping = length(caller_eqclassification)
    end
    return Eq(eq_mapping)
end

function apply_linear_incidence(𝕃, ret::PartialStruct, caller::CallerMappingState, mapping::CalleeMapping)
    return PartialStruct(𝕃, ret.typ, Any[apply_linear_incidence(𝕃, f, caller, mapping) for f in ret.fields])
end

function CalleeMapping(𝕃::Compiler.AbstractLattice, argtypes::Vector{Any}, callee_result::DAEIPOResult, template_argtypes)
    applied_scopes = Any[]
    coeffs = Vector{Any}(undef, length(callee_result.var_to_diff))
    eq_mapping = fill(0, length(callee_result.total_incidence))

    process_template!(𝕃, coeffs, eq_mapping, applied_scopes, argtypes, template_argtypes)

    return CalleeMapping(coeffs, eq_mapping, applied_scopes)
end


struct MappingInfo <: Compiler.CallInfo
    info::Any
    result::DAEIPOResult
    mapping::CalleeMapping
end

function _make_argument_lattice_elem(𝕃, which::Argument, @nospecialize(argt), add_variable!, add_equation!, add_scope!)
    if isa(argt, Const)
        #@assert !isa(argt.val, Scope) # Shouldn't have been forwarded
        return argt
    elseif isa(argt, Type) && argt <: Intrinsics.AbstractScope
        return PartialScope(add_scope!(which))
    elseif isa(argt, Type) && argt == equation
        return Eq(add_equation!(which))
    elseif is_non_incidence_type(argt)
        return argt
    elseif Compiler.isprimitivetype(argt)
        inc = Incidence(add_variable!(which))
        return argt === Float64 ? inc : Incidence(argt, inc.row)
    elseif isa(argt, PartialStruct)
        return PartialStruct(𝕃, argt.typ, Any[make_argument_lattice_elem(𝕃, which, f, add_variable!, add_equation!, add_scope!) for f in argt.fields])
    elseif isabstracttype(argt) || ismutabletype(argt) || !isa(argt, DataType)
        return nothing
    else
        fields = Any[]
        any = false
        # TODO: This doesn't handle recursion
        if Base.datatype_fieldcount(argt) === nothing
            return nothing
        end
        for i = 1:length(fieldtypes(argt))
            # TODO: Can we make this lazy?
            ft = fieldtype(argt, i)
            mft = _make_argument_lattice_elem(𝕃, which, ft, add_variable!, add_equation!, add_scope!)
            if mft === nothing
                push!(fields, Incidence(ft))
            else
                any = true
                push!(fields, mft)
            end
        end
        return any ? PartialStruct(𝕃, argt, fields) : nothing
    end
end

function make_argument_lattice_elem(𝕃, which::Argument, @nospecialize(argt), add_variable!, add_equation!, add_scope!)
    mft = _make_argument_lattice_elem(𝕃, which, argt, add_variable!, add_equation!, add_scope!)
    mft === nothing ? Incidence(argt) : mft
end

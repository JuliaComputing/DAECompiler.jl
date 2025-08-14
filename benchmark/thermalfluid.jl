using DAECompiler
using DAECompiler.Intrinsics
using Polynomials: fit
using XSteam: my_pT, rho_pT, Cp_pT, tc_pT

struct Sink; end
function (::Sink)(inlet)
    always!(inlet.T - inlet.T_instream)
end

struct FluidPort
    p::Float64
    m::Float64
    T::Float64
    T_instream::Float64
end
FluidPort() = FluidPort(ntuple(_->continuous(), 4)...)

function connect!(
    inner::NTuple{N,FluidPort},
    outer::NTuple{M,FluidPort},
) where {N,M}
    if N == 2 && M == 0
        (a, b) = inner
        always!(a.T_instream - b.T) # stream
        always!(a.T - b.T_instream) # stream
        always!(a.m + b.m) # flow
    elseif N == 1 && M == 1
        a = only(inner)
        b = only(outer)
        always!(a.T_instream - b.T_instream) # stream
        always!(a.T - b.T) # stream
        always!(a.m - b.m) # flow
    else @assert false "unimplemented" end
    always!(a.p - b.p) # potential

    return nothing
end

struct HeatPort
    Q::Float64
    T::Float64
end
HeatPort() = HeatPort(ntuple(_->continuous(), 2)...)
function connect!(
    inner::NTuple{N,HeatPort},
    outer::NTuple{M,HeatPort},
) where {N,M}
    if N == 2 && M == 0
        (a, b) = inner
        always!(a.T - b.T)
        always!(a.Q + b.Q)
    else @assert false "unimplemented" end

    return nothing
end

@kwdef struct Source
    p_feed::Float64 = 100000.0
end

function (this::Source)(outlet)
    flow(t) = 2.75
    T(t) = (t > 12 * 3600) * 56.0 + 12.0

    t = sim_time()

    m_flow = continuous()

    always!(m_flow - flow(t))
    always!(outlet.m + m_flow)
    always!(outlet.p - this.p_feed)
    always!(outlet.T - T(t))
end

@kwdef struct CircularWall{Ne, Nn}
    T0::Float64 = 0.
    d::Float64  = 0.05
    dx::Float64 = 0.01

    t_layer::NTuple{Ne,Float64} = (0.002,)
    cp::NTuple{Ne, Float64} = (500.,)
    ρ::NTuple{Ne, Float64}  = (7850.,)
    λ::NTuple{Ne, Float64}  = (50.,)
end

function Cn_circular_wall_inner(d, D, cp, ρ)
    C = pi / 4 * (D^2 - d^2) * cp * ρ
    return C / 2
  end

function Cn_circular_wall_outer(d, D, cp, ρ)
    C = pi / 4 * (D^2 - d^2) * cp * ρ
    return C / 2
end

function Ke_circular_wall(d, D, λ)
    2 * pi * λ / log(D / d)
end

function (this::CircularWall{Ne, Nn})(inner_heatport, outer_heatport) where {Ne, Nn}
    Tn = ntuple(_->continuous(), Nn)
    Qe = ntuple(_->continuous(), Ne)

    ntuple(i->initial!(Tn[i] - this.T0), Nn)
    ntuple(i->initial!(Qe[i] - this.T0), Ne)

    dn = (this.d, (this.d .+ 2.0 .* cumsum(this.t_layer))...)
    Cni = Cn_circular_wall_inner.(dn[1:Ne], dn[2:Nn], this.cp, this.ρ)
    Cno = Cn_circular_wall_outer.(dn[1:Ne], dn[2:Nn], this.cp, this.ρ)
    Cn = ntuple(Nn) do i
        Base.@_inline_meta
        (if i == 1
            Cni[i]
        elseif i == Nn
            Cno[i-1]
        else
            Cni[i] + Cno[i-1]
        end) * this.dx
    end
    Ke = Ke_circular_wall.(dn[1:Ne], dn[2:Nn], this.λ) .* this.dx

    always!(inner_heatport.T - Tn[1])
    always!(outer_heatport.T - Tn[Nn])

    ntuple(Ne) do i
        Base.@_inline_meta
        always!(Qe[i] - Ke[i] * (Tn[i] - Tn[i+1]))
        nothing
    end

    always!(ddt(Tn[1]) * Cn[1] - (inner_heatport.Q - Qe[1]))
    ntuple(Nn-2) do j
        Base.@_inline_meta
        i = j + 1
        always!(ddt(Tn[i]) * Cn[i] - (Qe[i] - Qe[i-1]))
        nothing
    end
    always!(ddt(Tn[Nn]) * Cn[Nn] - (outer_heatport.Q - Qe[Ne]))
end

@kwdef struct CylindricalSurfaceConvection
    d::Float64  = 1.0
    dx::Float64 = 0.01
    α::Float64  = 5.0
    Tenv::Float64 = 18.0
end

function (this::CylindricalSurfaceConvection)(heatport)
    always!(heatport.Q - this.α*pi*this.d*this.dx*(heatport.T-this.Tenv))
end

# der(T[3:N - 1]) = {u / dx * (c[1] * T[i - 2] - sum(c) * T[i - 1] + c[2] * T[i] + c[3] * T[i + 1]) + Dxx * (T[i - 1] - 2 * T[i] + T[i + 1]) / dx ^ 2 + S[i] / (C[i] - C_shift) for i in 3:N - 1};

p_l = 5 #bar
T_vec = collect(1:1:150);

# TODO: rewrite these to be less eye gorey
@generated kin_visc_T(t) = :(Base.evalpoly(t, $(fit(T_vec, my_pT.(p_l, T_vec) ./ rho_pT.(p_l, T_vec), 5).coeffs...,)))
@generated lambda_T(t) = :(Base.evalpoly(t, $(fit(T_vec, tc_pT.(p_l, T_vec), 3).coeffs...,)))
@generated Pr_T(t) = :(Base.evalpoly(t, $(fit(T_vec, 1e3 * Cp_pT.(p_l, T_vec) .* my_pT.(p_l, T_vec) ./ tc_pT.(p_l, T_vec), 5).coeffs...,)))
@generated rho_T(t) = :(Base.evalpoly(t, $(fit(T_vec, rho_pT.(p_l, T_vec), 4).coeffs...,)))
@generated rhocp_T(t) = :(Base.evalpoly(t, $(fit(T_vec, 1000 * rho_pT.(p_l, T_vec) .* Cp_pT.(p_l, T_vec), 5).coeffs...,)))

function Churchill_f(Re, epsilon, d)
    theta_1 = (-2.457 * log(((7 / Re)^0.9) + (0.27 * (epsilon / d))))^16
    theta_2 = (37530 / Re)^16
    8 * ((((8 / Re)^12) + (1 / ((theta_1 + theta_2)^1.5)))^(1 / 12))
end

function Nusselt(Re, Pr, f)
    if Re <= 2300.0
        3.66
    elseif Re <= 3100.0
        3.5239 * (Re / 1000)^4 - 45.158 * (Re / 1000)^3 + 212.13 * (Re / 1000)^2 - 427.45 * (Re / 1000) + 316.08
    else
        f / 8 * ((Re - 1000) * Pr) / (1 + 12.7 * (f / 8)^(1 / 2) * (Pr^(2 / 3) - 1))
    end
end

function Dxx_coeff(u, d, T)
    Re = abs(u) * d / kin_visc_T(T) + 0.1
    if Re < 1000.0
        (d^2 / 4) * u^2 / 48 / 0.14e-6
    else
        d * u * (1.17e9 * Re^(-2.5) + 0.41)
    end
end

struct FluidRegionElement{H}
    wall::H
    dx::Float64
    dn::Float64
    Rw::Float64
    C_shift::Float64
end

function (this::FluidRegionElement)((; u, alpha, Dxx), T, (u_dx_coeff, Dxx_coeff)::NTuple{2, Float64})
    S = continuous()
    C = continuous()
    Twall = continuous()
    heatport = HeatPort()
    A = pi * this.dn^2 / 4
    #         :(always!(S[$i] - (1 / (1 / (alpha * SP.dn * pi * dx) + abs(p.Rw / 1000))) * (Twall[$i] - T[$i])))
    always!(S - (1 / (1 / (alpha * this.dn * pi * this.dx) + abs(this.Rw / 1000))) * (Twall - T))
    always!(ddt(T) - u / this.dx * u_dx_coeff + Dxx * Dxx_coeff / this.dx ^ 2 + S / (C - this.C_shift))
    always!(C - this.dx * A * rhocp_T(T))
    always!(S - heatport.Q)
    always!(Twall - heatport.T)
    this.wall(heatport)
end

function (this::FluidRegionElement)(vars::NamedTuple, Ts::NTuple{4, Float64})
    (Tᵢ₋₂, Tᵢ₋₁, Tᵢ, Tᵢ₊₁) = Ts
    c = (-1 / 8, -3 / 8, -3 / 8)
    this(vars, Tᵢ, (c[1] * Tᵢ₋₂ - sum(c) * Tᵢ₋₁ + c[2] * Tᵢ + c[3] * Tᵢ₊₁, Tᵢ₋₁ - 2 * Tᵢ + Tᵢ₊₁))
end

@kwdef struct FluidRegion{N, H}
    element::FluidRegionElement{H}
    lumped_T::Float64 = 50.
    e::Float64 = 1e-4
    diffusion::Bool = true
end
FluidRegion{N}(; element::FluidRegionElement{H}, kwargs...) where {H, N} = FluidRegion{N, H}(element=element, kwargs...)

function (this::FluidRegion{N})(inlet::FluidPort, outlet::FluidPort) where {N}
    T = ntuple(_->continuous(), N)

    A       = pi * this.element.dn^2 / 4
    u       = inlet.m / rho_T(inlet.T) / A
    Re      = (0.1 + this.element.dn * abs(u)) / kin_visc_T(this.lumped_T)
    Pr      = Pr_T(this.lumped_T)
    f       = Churchill_f(Re, this.e, this.element.dn)
    alpha   = Nusselt(Re, Pr, f) * lambda_T(this.lumped_T) / this.element.dn
    Dxx     = this.diffusion * Dxx_coeff(u, this.element.dn, this.lumped_T)

    vars = (; u, alpha, Dxx)

    this.element(vars, T[1], (inlet.T - T[1], T[2] - T[1]))
    this.element(vars, (inlet.T, T[1], T[2], T[3]))
    ntuple(N-3) do j
        i = j + 2
        # TODO: log tree
        this.element(vars, (T[i-2], T[i-1], T[i], T[i+1]))
        nothing
    end
    this.element(vars, T[N], ((T[N-1] - T[N]), (T[N-1] - T[N])))

    always!(inlet.m + outlet.m)
    always!(inlet.p - outlet.p)
    always!(inlet.T - inlet.T_instream)
    always!(outlet.T - T[N])
end

@kwdef struct PreinsulatedPipe{N}
    L::Float64 = 1.0
end

function (this::PreinsulatedPipe{N})(inlet::FluidPort, outlet::FluidPort) where {N}
    dx = this.L / N
    element = FluidRegionElement(dx, 0.05, 0., 0.) do inner_heatport
        heatports = ntuple(_->HeatPort(), 3)
        CircularWall{1,2}(;dx)(heatports[1], heatports[2])
        CylindricalSurfaceConvection()(heatports[3])
        connect!((inner_heatport, heatports[1]), ())
        connect!((heatports[2], heatports[3]), ())
    end
    FluidRegion{N}(;element)(inlet, outlet)
end

struct Benchmark{N}
end

function (::Benchmark{N})() where {N}
    in = ntuple(_->FluidPort(), 2)
    out = ntuple(_->FluidPort(), 2)
    connect!(in, ())
    connect!(out, ())
    Source()(in[1])
    PreinsulatedPipe{N}()(in[2], out[1])
    Sink()(out[2])
end

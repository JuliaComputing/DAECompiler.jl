module ipo

using Test
using DAECompiler
using DAECompiler: refresh
using DAECompiler.Intrinsics
using Sundials
using SciMLBase
using OrdinaryDiffEq

#= Basic IPO: We need to read the incidence of the contained `-` =#
@noinline function onecall!()
    x = continuous()
    initial!(x - 1.0)
    always!(ddt(x) - x)
end

onecall!()
sol = solve(DAECProblem(onecall!), DFBDF(autodiff=false))
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[1, :], exp.(sol.t)))
sol = solve(ODECProblem(onecall!), Rodas5(autodiff=false))
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[1, :], exp.(sol.t)))

#==== Contained Equations ============#
function twocall!()
    onecall!(); onecall!();
    return nothing
end

twocall!()
dae_sol = solve(DAECProblem(twocall!), DFBDF(autodiff=false))
ode_sol = solve(ODECProblem(twocall!), Rodas5(autodiff=false))
for (sol, i) in Iterators.product((dae_sol, ode_sol), 1:2)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[i, :], exp.(sol.t)))
end

#============== NonLinear ============#
@noinline function sin!()
    x = continuous()
    initial!(x - 1.0)
    always!(ddt(x) - sin(x))
end
function sin2!()
    sin!(); sin!();
    return nothing
end
dae_sol = solve(DAECProblem(sin2!), DFBDF(autodiff=false))
ode_sol = solve(ODECProblem(sin2!), Rodas5(autodiff=false))
for (sol, i) in Iterators.product((dae_sol, ode_sol), 1:2)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[i, :], 2*acot.(exp.(-sol.t).*cot(1/2))))
end

#============== NonLinear argument ============#
@noinline sink!(x, v) = always!(x - v)
function sinsink!()
    x = continuous()
    initial!(x - 1.0)
    sink!(ddt(x), sin(x))
end
dae_sol = solve(DAECProblem(sinsink!), DFBDF(autodiff=false))
ode_sol = solve(ODECProblem(sinsink!), Rodas5(autodiff=false))
for sol in (dae_sol, ode_sol)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[1, :], 2*acot.(exp.(-sol.t).*cot(1/2))))
end

#============== NonLinear argument + derivative ============#
@noinline sink2!(x, v) = always!(ddt(x) - v)
function sinsink2!()
    x = continuous()
    initial!(x - 1.0)
    sink2!(x, sin(x))
end
dae_sol = solve(DAECProblem(sinsink2!), DFBDF(autodiff=false))
ode_sol = solve(ODECProblem(sinsink2!), Rodas5(autodiff=false))
for sol in (dae_sol, ode_sol)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[1, :], 2*acot.(exp.(-sol.t).*cot(1/2))))
end

#============== SICM ============#
struct sicm!
    arg::Float64
end

@noinline function (this::sicm!)()
    x = continuous()
    initial!(x - 1.0)
    always!(ddt(x) - this.arg)
end

struct sicm2!
    a::Float64
    b::Float64
end

function (this::sicm2!)()
    sicm!(this.a)(); sicm!(this.b)();
    return nothing
end
dae_sol = solve(DAECProblem(sicm2!(1., 1.)), DFBDF(autodiff=false))
ode_sol = solve(ODECProblem(sicm2!(1., 1.)), Rodas5(autodiff=false))
for (sol, i) in Iterators.product((dae_sol, ode_sol), 1:2)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[i, :], 1. .+ sol.t))
end

#========== NonLinear SICM ===========#
struct nlsicm!
    arg::Float64
end

@noinline function (this::nlsicm!)()
    x = continuous()
    initial!(x - 1.0)
    always!(ddt(x) - sin(this.arg))
end

struct nlsicm2!
    a::Float64
    b::Float64
end

function (this::nlsicm2!)()
    nlsicm!(this.a)(); nlsicm!(this.b)();
    return nothing
end
dae_sol = solve(DAECProblem(nlsicm2!(1., 1.)), DFBDF(autodiff=false))
ode_sol = solve(ODECProblem(nlsicm2!(1., 1.)), Rodas5(autodiff=false))
for (sol, i) in Iterators.product((dae_sol, ode_sol), 1:2)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[i, :], 1. .+ sin(1.)*sol.t))
end

#============== Ping Pong ============#
@noinline function ping(a, b, c, d)
    always!(b - sin(a))
    always!(d - sin(c))
end

@noinline function pong(a, b, c, d)
    always!(b - asin(a))
    always!(ddt(d) - asin(c))
end

function pingpong()
    # N.B.: Deliberate not using variables, which requires
    # scope, etc handling
    a = continuous()
    b = continuous()
    c = continuous()
    d = continuous()
    ping(a, b, c, d)
    pong(b, c, d, a)
end
dae_sol = solve(DAECProblem(pingpong, (1,) .=> 0.1), IDA())
ode_sol = solve(ODECProblem(pingpong, (1,) .=> 0.1), Rodas5(autodiff=false))

# asin(sin) are inverses in [-pi/2, pi/2]
for sol in (dae_sol, ode_sol)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[1, :], 0.1exp.(sol.t)))
end

#========== Implicit External ===========#
@noinline intro() = ddt(continuous())
@noinline outro!(x) = always!(x-1)

implicit() = outro!(intro())

dae_sol = solve(DAECProblem(implicit, (1,) .=> 1), IDA())
ode_sol = solve(ODECProblem(implicit, (1,) .=> 1), Rodas5(autodiff=false))

for sol in (dae_sol, ode_sol)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[1, :], 1 .+ sol.t))
end

#========== Structured SICM ===========#
@noinline function add4((a, b, c, d)::NTuple{4, Float64})
    always!((a + b) + (c + d))
end

function structured_sicm()
    x = continuous()
    add4((1., 2., ddt(x), -x))
end

dae_sol = solve(DAECProblem(structured_sicm, (1,) .=> 1), IDA())
ode_sol = solve(ODECProblem(structured_sicm, (1,) .=> 1), Rodas5(autodiff=false))

for sol in (dae_sol, ode_sol)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[1, :], 3 .- 2exp.(sol.t)))
end

#========= Nonlinear Implicit External =========#
@noinline tanhdu() = tanh(ddt(continuous()))
@noinline implicitext_nl(x) = always!(x - 0.5)

implicitext_nl() = implicitext_nl(tanhdu())

dae_sol = solve(DAECProblem(implicitext_nl, (1,) .=> 1), IDA())
ode_sol = solve(ODECProblem(implicitext_nl, (1,) .=> 1), Rodas5(autodiff=false))

for sol in (dae_sol, ode_sol)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[1, :], 1 .+ atanh(0.5)sol.t))
end

#========= External Equation =========#
@noinline external() = always()
@noinline function applyeq(eq)
    x = continuous()
    eq(ddt(x) - x)
end
implicitexteq() = applyeq(external())

dae_sol = solve(DAECProblem(implicitexteq, (1,) .=> 1), IDA())
ode_sol = solve(ODECProblem(implicitexteq, (1,) .=> 1), Rodas5(autodiff=false))

for sol in (dae_sol, ode_sol)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[1, :], exp.(sol.t)))
end

#========= External Equation (nonlinear) =========#
@noinline external2() = always()
@noinline function applyeq_nl(eq)
    x = continuous()
    eq(ddt(x) - sin(x))
end
implicitexteq_nl() = applyeq_nl(external2())

dae_sol = solve(DAECProblem(implicitexteq_nl, (1,) .=> 1), IDA())
ode_sol = solve(ODECProblem(implicitexteq_nl, (1,) .=> 1), Rodas5(autodiff=false))

for sol in (dae_sol, ode_sol)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[1, :], 2acot.(exp.(log(cot(0.5)) .- sol.t))))
end

#========= External Equation (multiple equations) =========#
@noinline external3() = (always(), continuous())
@noinline applyeq2(eq, x) = eq(ddt(x) - 1.)
@noinline function impliciteqvar()
    (eq, x) = external3()
    applyeq2(eq, x)
    applyeq2(eq, x)
end

dae_sol = solve(DAECProblem(impliciteqvar, (1,) .=> 1), IDA())
ode_sol = solve(ODECProblem(impliciteqvar, (1,) .=> 1), Rodas5(autodiff=false))

for sol in (dae_sol, ode_sol)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[1, :], 1 .+ sol.t))
end

#========= External Equation (multiple equations + constant) =========#
@noinline external4() = (always(), ddt(continuous()))
@noinline applyeq3(eq, var) = eq(var)
@noinline function impliciteqvar2()
    (eq, var) = external4()
    applyeq3(eq, var)
    applyeq3(eq, var)
    applyeq3(eq, -4.)
end

dae_sol = solve(DAECProblem(impliciteqvar2, (1,) .=> 1), IDA())
ode_sol = solve(ODECProblem(impliciteqvar2, (1,) .=> 1), Rodas5(autodiff=false))

for sol in (dae_sol, ode_sol)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[1, :], 1 .+ 2sol.t))
end

#========= External Equation (multiple equations + external nonlinear) =========#
@noinline function impliciteqvar3()
    (eq, var) = external3()
    applyeq3(eq, 0.5ddt(var))
    applyeq3(eq, 0.5ddt(var))
    applyeq3(eq, -sin(var))
end

dae_sol = solve(DAECProblem(impliciteqvar3, (1,) .=> 1), IDA())
ode_sol = solve(ODECProblem(impliciteqvar3, (1,) .=> 1), Rodas5(autodiff=false))

for sol in (dae_sol, ode_sol)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[1, :], 2acot.(exp.(-sol.t).*cot(1/2))))
end

#=================== Scope handling ==================#
@noinline function scope!(scope)
    x = continuous(scope)
    always!(ddt(x) - x + epsilon(scope), scope)
end
function scope_outer()
    scope!(Scope(Scope(), :x1));
    scope!(Scope(Scope(), :x2));
    scope!(Scope(Scope(Scope(), :x3), :x4));
    return nothing
end

# ERROR: UndefVarError: `is_valid_partial_scope` not defined in `DAECompiler`
@test_broken begin
    result = @code_structure result = true scope_outer()
    @test length(result.varkinds) == 6 # 3 states + their differentials
    @test length(result.eqkinds) == 3
    # add test for `result.names`
end

#=================== GenScope ==================#
@noinline function genscope!(scope)
    scope = GenScope(scope, :g)
    x = continuous(scope)
    always!(ddt(x) - x + epsilon(scope), scope)
end
function genscope()
    genscope!(Scope(Scope(), :x1))
    genscope!(Scope(Scope(), :x1))
    return nothing
end

# ERROR: AssertionError: ivar == var_num
@test_broken begin
    result = @code_structure result = true genscope()
    @test length(result.varkinds) == 4 # 2 states + their differentials
    @test length(result.eqkinds) == 2
    # add test for `result.names`
end

#================= Derived Scope ===================#
@noinline function derived_scope!(scope)
    vscope = scope(:x)
    x = continuous(vscope)
    observed!(x, scope(:xo))
    always!(ddt(x) - x + epsilon(vscope), vscope)
end
function derived_scope()
    derived_scope!(Scope(Scope(), :x1))
    derived_scope!(Scope(Scope(), :x2))
    return nothing
end

# ERROR: UndefVarError: `is_valid_partial_scope` not defined in `DAECompiler`
@test_broken begin
    result = @code_structure result = true derived_scope()
    @test length(result.varkinds) == 4 # 2 states + their differentials
    @test length(result.eqkinds) == 2
    # add test for `result.names`
end

#================= Equation & scope arguments ===================#
@noinline function eqscope!(eq, scope)
    vscope = scope(:x)
    x = continuous(vscope)
    observed!(x, scope(:xo))
    eq(ddt(x) - x + epsilon(vscope))
end
function eqscope()
    e1 = always(Scope(Scope(), :e1))
    e2 = always(Scope(Scope(), :e2))
    eqscope!(e1, Scope(Scope(), :x1))
    eqscope!(e2, Scope(Scope(), :x2))
    return nothing
end

# ERROR: UndefVarError: `is_valid_partial_scope` not defined in `DAECompiler`
@test_broken begin
    result = @code_structure result = true eqscope()
    @test length(result.varkinds) == 4 # 2 states + their differentials
    @test length(result.eqkinds) == 2
    # add test for `result.names`
end

#============ ScopedValue ==============#
using Base.ScopedValues
const debug_scope = ScopedValue{DAECompiler.Intrinsics.AbstractScope}()

@noinline function scoped_equation()
    scope = debug_scope[]
    x = continuous(scope)
    always!(ddt(x) - x + epsilon(scope), scope)
end
function scoped_equation_outer()
    with(scoped_equation, debug_scope => Scope(Scope(), :x1))
    with(scoped_equation, debug_scope => Scope(Scope(), :x2))
    return nothing
end

# ERROR: UndefVarError: `cur_scope_lattice` not defined in `DAECompiler`
@test_broken begin
    result = @code_structure result = true scoped_equation_outer()
    @test length(result.varkinds) == 4 # 2 states + their differentials
    @test length(result.eqkinds) == 2
    # add test for `result.names`
end

#========= Varargs =========#
@noinline function varargs_inner!(args...)
    map(args) do (x, val)
        always!(ddt(x) - val)
    end
end

@noinline function varargs_middle!(args...)
    # Extra tuple to exercise some of the deeper nesting code paths
    varargs_inner!(map(x -> (x + epsilon(), 2.0), args)...)
end

@noinline function varargs_outer!()
    a = continuous()
    b = continuous()
    c = continuous()
    varargs_middle!(a, b, c)
end

result = @code_structure result = true varargs_outer!()
@test length(result.varkinds) == 12 # (3 variables + 3 epsilon) + their differentials
@test length(result.eqkinds) == 3
dae_sol = solve(DAECProblem(varargs_outer!, (1, 2, 3) .=> 1.), IDA())
ode_sol = solve(ODECProblem(varargs_outer!, (1, 2, 3) .=> 1.), Rodas5(autodiff=false))
for (sol, i) in Iterators.product((dae_sol, ode_sol), 1:3)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[i, :], 1. .+ 2sol.t))
end

#========= Internal variable leaking =========#
@noinline new_variable() = continuous()

@noinline function variable_and_equation()
    x = new_variable()
    always!(ddt(x) - x)
end
function internal_variable_leaking()
    variable_and_equation()
    variable_and_equation()
    return nothing
end

result = @code_structure result = true internal_variable_leaking()
@test length(result.varkinds) == 4 # 2 states + their differentials
@test length(result.eqkinds) == 2

dae_sol = solve(DAECProblem(internal_variable_leaking, (1, 2) .=> 1.), IDA())
ode_sol = solve(ODECProblem(internal_variable_leaking, (1, 2) .=> 1.), Rodas5(autodiff=false))
for (sol, i) in Iterators.product((dae_sol, ode_sol), 1:2)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[i, :], exp.(sol.t)))
end

end

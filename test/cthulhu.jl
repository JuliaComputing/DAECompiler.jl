module CthulhuExt

using Test
using DAECompiler
using DAECompiler.Intrinsics
using Cthulhu
using Cthulhu.Testing
using Cthulhu.Testing: wait_for

function test_descend_for_provider(provider, args...; io = nothing, callsite = 1)
    terminal = VirtualTerminal()
    harness = @run terminal descend(args...; terminal, provider)
    write(terminal, 'A')
    write(terminal, 'T')
    write(terminal, 'L') # should be a no-op because we disable the LLVM view (which otherwise segfaults)
    write(terminal, 'd') # debuginfo: :source
    for _ in 2:callsite write(terminal, :down) end
    write(terminal, :enter)
    write(terminal, 'i') # inlining costs: on
    write(terminal, 'S')
    write(terminal, :up)
    write(terminal, :enter)
    write(terminal, 'q')
    if io !== nothing
        wait_for(harness.task)
        displayed = String(readavailable(harness.io))
        println(io, displayed)
    end
    @test end_terminal_session(harness)
end

@noinline function ping(a, b, c, d)
    always!(b - sin(a))
    always!(d - sin(c))
end

@noinline function pong(a, b, c, d)
    always!(b - asin(a))
    always!(ddt(d) - asin(c))
end

function pingpong()
    a = continuous()
    b = continuous()
    c = continuous()
    d = continuous()
    ping(a, b, c, d)
    pong(b, c, d, a)
end

io = IOBuffer()
test_descend_for_provider(dae_provider(), pingpong; io, callsite = 5)
text = String(take!(io))
@test contains(text, "Incidence(u₄)")
@test contains(text, "Eq(1)")

end

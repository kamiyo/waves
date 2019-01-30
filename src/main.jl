using PyPlot
using PyCall
using LinearAlgebra
using Printf
using WAV
@pyimport matplotlib.animation as anim
pygui(true)

struct Str
    length::Float64
    c2::Float64
    tension::Float64
    mpu::Float64
    mpu_inv::Float64
end

Str(length::Float64, c::Float64, tension::Float64) = (
    c2::Float64 = c ^ 2;
    mpu::Float64 = tension / c2;
    Str(length, c2, tension, mpu, 1. / mpu)
)

mutable struct StrIntegrator
    t::Int
    str::Str
    dt::Float64
    dx::Float64
    dx_inv::Float64
    Nx::Int
    yPrev::Array{Float64, 1}
    r2::Float64
    twice1mr2::Float64
    diffMat::Tridiagonal{Float64, Array{Float64,1}}
end

StrIntegrator(str::Str, dt::Real, dx::Real) = begin
    Nx::Int = round(str.length / dx)
    yPrev::Array{Float64, 1} = zeros(Nx)
    r2::Float64 = str.c2 * dt^2. / (dx^2.)
    twice1mr2::Float64 = 2. * (1. - r2)
    twice2r::Float64 = 2. * r2
    n_2r2s::Array{Float64, 1} = fill(r2, Nx - 2)
    diag::Array{Float64, 1} = fill(twice1mr2, Nx)
    diffMat = Tridiagonal([n_2r2s..., twice2r], diag, [twice2r, n_2r2s...])
    StrIntegrator(0, str, dt, dx, 1. / dx, Nx, yPrev, r2, twice1mr2, diffMat)
end

Integrate!(si::StrIntegrator, yn, damping=false) = begin
    if si.t == 0
        si.yPrev = copy(yn)
    end
    diff = yn - si.yPrev
    square = diff .* abs.(diff)
    yn1 = (si.diffMat * yn) - si.yPrev + square * -2. * 0.5 * 1.2 * .0009 * si.str.mpu_inv
    yn1[1] = yn[1]
    yn1[end] = 0.
    si.t += 1
    si.yPrev = copy(yn)
    force = si.str.tension * (yn1[2] - yn1[1]) * si.dx_inv
    if damping == true
        yn1[1] = yn[1] + force * si.dt / 1000.
    else
        yn1[1] = yn[1]
    end
    return (yn1, force)
end

main(generate=false) = begin
    c::Real = 220.
    dt::Real = 1.0/44100.0
    dx::Real = dt * c
    length::Real = 100 * dx
    s::Str = Str(length, c, 70.)
    si::StrIntegrator = StrIntegrator(s, dt, dx)

    # yn::Array{Float64, 1} = [
    #     range(0., stop=0.005, length=Int(si.Nx * 0.15 + 1.))[1:end-1]...
    #     range(0.005, stop=0, length=Int(si.Nx * 0.85))...
    # ]
    yn::Array{Float64, 1} = [0.0004 * sin((i - 1) * pi / (si.Nx - 1)) + 0.0002 * sin(5 * (i - 1) * pi / (si.Nx - 1)) for i = 1:si.Nx]

    x::Array{Float64, 1} = range(0, stop=s.length, length=si.Nx)

    global (fig, (ax1, ax2)) = subplots(2, 1)
    ax1[:set_ylim](-0.005, 0.005)
    global line1, = ax1[:plot](x, yn)
    ax1[:set_title]("String Contour")
    ax1[:set_xlabel]("string shape (m)")
    ax1[:set_ylabel]("height offset (m)")
    global time = ax1[:text](0.02, 0.8, "", transform=ax1[:transAxes])

    fx::Array{Float64, 1} = range(0, stop=1, length=1000)
    fy::Array{Float64, 1} = fill(NaN, 1000)
    global line2, = ax2[:plot](fx, fy)
    ax2[:set_title]("Force on bridge")
    ax2[:set_xlabel]("time (s)")
    ax2[:set_ylabel]("force (N)")
    ax2[:set_ylim](-1.0, 1.0)
    ax2[:set_xlim](0.0, 1.0)

    fig[:subplots_adjust](hspace=1.0)

    init() = begin
        global line1
        global line2
        global time
        line1[:set_ydata](fill(NaN, si.Nx))
        line2[:set_ydata](fill(NaN, 1000))
        time[:set_text]("")
        return (line1, line2, time)
    end

    animate(i) = begin
        global line1
        global line2
        global time
        for j in 1:1
            (yn, f) = Integrate!(si, yn, false)
            fy[(si.t % 1000) + 1] = f
        end
        line1[:set_ydata](yn)
        time[:set_text](@sprintf("time = %.6f", si.t * dt))
        line2[:set_ydata](fy)
        return (line1, line2, time)
    end

    # anim.FuncAnimation(fig, animate, init_func=init, interval=2, blit=true, frames=100)
    # show()

    generateWAV() = begin
        out::Array{Float64,1} = zeros(882000)
        for i in 1:882000
            (yn, f) = Integrate!(si, yn, true)
            out[i] = f
        end
        wavwrite(out, "out.wav", Fs=44100)
    end

    if generate == true
        generateWAV()
    else
        anim.FuncAnimation(fig, animate, init_func=init, interval=2, blit=true, frames=100)
        show()
    end
end

main(true)
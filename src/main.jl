module Waves

using PyPlot
using PyCall
using LinearAlgebra
using Printf
using WAV
using DifferentialEquations
using RecursiveArrayTools
using Distributions
using FFTW
using DSP
using Juno

include("StringObject.jl")
include("Integrator.jl")
@pyimport matplotlib.animation as anim
pygui(true)

main(generate = false) = begin
    FFTW.set_num_threads(4)

    c::Float64 = 265.0
    # fs::Float64 = 44100.0
    length::Float64 = 0.65
    Δx::Float64 = length / 500
    Δt::Float64 = 0.5 * Δx / c
    fs::Float64 = 1.0 / Δt
    println(fs)
    Z_bridge::Float64 = 1000.0
    s::Str = Str(length, c, 700., 0.0009, Z_bridge)
    df::DragForce = DragForce(1.2, 0.5)
    si::StrIntegrator = StrIntegrator(s, df, Δt, Δx)

    h::Hammer = Hammer(0.003, 0.01, 2.5, 5e9, 6.0 / 7.0 * s.length, -0.001, 2.0)
    hi::HammerIntegrator = HammerIntegrator(0, h, Δt, 0.0)

    # yⁿ::Array{Float64,1} = [0.0004 * sin((i - 1) * pi / (si.Nₓ - 1)) + 0.0002 * sin(5 * (i - 1) * pi / (si.Nₓ - 1)) for i = 1:si.Nₓ]
    yⁿ = GeneratePluck(0.85, 0.001, si.Nₓ)
    # yⁿ::Array{Float64,1} = zeros(si.Nₓ)

    stdev::Float64 = h.width / (2 * √(2 * log(2)))
    println(stdev)
    dist = Normal(h.x, stdev)
    # dist = Uniform(h.x - 0.005, h.x + 0.005)

    x::Array{Float64,1} = range(0, stop = s.length, length = si.Nₓ)
    scale = pdf(dist, h.x)
    hammer_display = [0.01 * (pdf(dist, i) / scale - 1.0) for i in x]
    hammer = [pdf(dist, i) / scale for i in x]

    global (fig, (ax1, ax2, ax3)) = subplots(3, 1, figsize=(8, 10))
    ax1[:set_ylim](-0.005, 0.005)
    global line1, lineHammer = ax1[:plot](x, yⁿ, "k-", x, hammer_display, "-")
    ax1[:set_title]("String Contour")
    ax1[:set_xlabel]("string shape (m)")
    ax1[:set_ylabel]("height offset (m)")
    global time = ax1[:text](0.02, 0.8, "", transform = ax1[:transAxes])

    forceSize::Int64 = 8192

    Fx::Array{Float64,1} = range(0, stop = forceSize / fs, length = forceSize)
    Fy::Array{Float64,1} = fill(NaN, forceSize)
    global line2, = ax2[:plot](Fx, Fy)
    ax2[:set_title]("Force on bridge")
    ax2[:set_xlabel]("time (s)")
    ax2[:set_ylabel]("force (N)")
    ax2[:set_ylim](-25.0, 25.0)
    ax2[:set_xlim](0.0, forceSize / fs)

    fftInSize::Int64 = 16384
    fftOutSize::Int64 = 8193

    global timeDomain = fill(0.0, fftInSize)
    fftPlan = FFTW.plan_rfft(timeDomain, flags=FFTW.UNALIGNED)

    FreqX::Array{Float64,1} = range(0, stop = fs / 2.0, length = fftOutSize)
    FreqY::Array{Float64,1} = fftPlan * timeDomain
    global line3, = ax3[:loglog](FreqX, FreqY)
    ax3[:set_title]("Frequency Spectrum")
    ax3[:set_xlabel]("Frequency (hz)")
    ax3[:set_ylabel]("Magnitude")
    ax3[:set_xlim](100, fs / 2.0)
    ax3[:set_ylim](0.0, 1e4)

    fig[:subplots_adjust](hspace = 1.0)

    init() = begin
        global line1
        global line2
        global line3
        global lineHammer
        global time
        line1[:set_ydata](fill(NaN, si.Nₓ))
        line2[:set_ydata](fill(NaN, forceSize))
        line3[:set_ydata](fill(NaN, fftOutSize))
        lineHammer[:set_ydata](fill(NaN, si.Nₓ))
        time[:set_text]("")
        return (line1, line2, line3, lineHammer, time)
    end

    hammerPos = Int(round(h.x / s.length * si.Nₓ))

    animate(i) = begin
        global line1
        global line2
        global line3
        global lineHammer
        global time
        global timeDomain
        for j in 1:10
            scaledHammer = (0.001 .* hammer .+ (h.y - 0.001))
            Fh::Array{Float64,1} = Interact(h, scaledHammer, s, yⁿ)
            Fh[1:hammerPos-5] .= 0.0
            Fh[hammerPos+5:end] .= 0.0
            (yⁿ, f) = Integrate!(si, yⁿ, Fh, hammer, true)

            # Integrate!(hi, -(Fh[hammerPos]))
            Fy[(si.tₙ % forceSize) + 1] = f

            popfirst!(timeDomain)
            push!(timeDomain, f)
        end
        line1[:set_ydata](yⁿ)
        # counter += 1
        time[:set_text](@sprintf("time = %.6f", si.tₙ * Δt))
        line2[:set_ydata](Fy)
        fftOut = fftPlan * timeDomain
        line3[:set_ydata](abs.(fftOut))
        next_hammer = hammer_display .+ h.y
        lineHammer[:set_ydata](next_hammer)

        return (line1, line2, line3, lineHammer, time)
    end


    generateWAV() = begin
        outSize = Int(round(fs * 8))
        out::Array{Float64,1} = zeros(outSize)
        Juno.progress(name="calculating") do p
            for i in 1:outSize
                scaledHammer = (0.001 .* hammer .+ (h.y - 0.001))
                Fh::Array{Float64,1} = Interact(h, scaledHammer, s, yⁿ)
                Fh[1:hammerPos-1] .= 0.0
                Fh[hammerPos+1:end] .= 0.0
                (yⁿ, f) = Integrate!(si, yⁿ, Fh, hammer, true)

                Integrate!(hi, -(Fh[hammerPos]))
                out[i] = f / 20.0
                if i % 1000 == 0
                    @info "calculating" progress = i / outSize _id = p
                end
            end
            println(44100.0 / fs)
            resampled = resample(out, 44100.0 / fs)
            wavwrite(resampled, "out_round.wav", Fs = 44100)
        end
    end

    if generate == true
        generateWAV()
    else
        anim.FuncAnimation(fig, animate, init_func = init, interval = 2, blit = true, frames = 100)
        show()
    end
end

main(false)

end

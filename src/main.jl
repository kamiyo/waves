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

    ρ::Float64 = 7850.0
    # fs::Float64 = 44100.0
    length::Float64 = 0.665
    Δx::Float64 = length / 500
    Z_bridge::Float64 = 1000.0
    Ε::Float64 = 2e11
    s::Str = Str(length, ρ, 750.0, 0.001, Ε, Z_bridge)
    cl::Float64 = sqrt(s.cl²)
    Δt::Float64 = 0.05 * Δx / sqrt(s.c²)
    println(Δt / (Δx / cl))
    # Qt::Int = ceil(Δt / (Δx / cl))
    Qt::Int = 3
    println(Qt)
    Δtl::Float64 = Δt / Qt
    # Δtl::Float64 = Δt
    fs::Float64 = 1.0 / Δt
    df::DragForce = DragForce(1.2, 0.5)
    # si::StrIntegrator = StrIntegrator(s, df, Δt, Δx)
    si::StrIntegrator = StrIntegrator(s, df, Δt, Δx, Δtl, Qt)
    # si::StrIntegratorHighAcc = StrIntegratorHighAcc(s, df, Δt, Δx)

    h::Hammer = Hammer(0.008274, 0.01, 2.5, 5e9, 6.1 / 7.0 * s.length, -0.001, 2.5)
    hi::HammerIntegrator = HammerIntegrator(0, h, Δt, 0.0)

    # yⁿ::Array{Float64,1} = [0.0004 * sin((i - 1) * pi / (si.Nₓ - 1)) + 0.0002 * sin(5 * (i - 1) * pi / (si.Nₓ - 1)) for i = 1:si.Nₓ]
    yⁿ = GeneratePluck(0.85, 0.01, si.Nₓ, 0.0)
    zⁿ = zeros(si.Nₓ)
    wⁿ = zeros(si.Nₓ)
    # wⁿ[2:10] .= -Δx / 2
    # yⁿ = [sin(pi / 180) * s.length * (1 - i / (si.Nₓ-1)) for i = 0:si.Nₓ-1]
    # yⁿ::Array{Float64,1} = zeros(si.Nₓ)

    stdev::Float64 = h.width / (2 * √(2 * log(2)))
    # println(stdev)
    dist = Normal(h.x, stdev)
    left = cdf(dist, h.x - 0.005)
    right = cdf(dist, h.x + 0.005)
    massUnderFW = right - left

    # dist = Uniform(h.x - 0.005, h.x + 0.005)
    hammerPos = Int(round(h.x / s.length * si.Nₓ))

    x::Array{Float64,1} = range(0, step = Δx, length = si.Nₓ)
    scale = pdf(dist, h.x)
    hammer_x = x[hammerPos-5:hammerPos+5]
    hammer_display = [0.001 * (pdf(dist, i) / scale - 1.0) for i in hammer_x]
    hammer = [pdf(dist, i) / scale for i in x]

    m::Array{Float64,1} = [h.mass * (cdf(dist, i + Δx) - cdf(dist, i)) for i in x]

    global (fig, (ax1, ax2, ax3)) = subplots(3, 1, figsize=(10, 10))
    ax1[:set_ylim](-0.02, 0.02)
    global line1, lineHammer = ax1[:plot](x .+ wⁿ, yⁿ, "k-", hammer_x, hammer_display, "-")
    ax1[:set_title]("String Contour")
    ax1[:set_xlabel]("string shape (m)")
    ax1[:set_ylabel]("height offset (m)")
    global time = ax1[:text](0.02, 0.8, "", transform = ax1[:transAxes])

    global ax4 = ax1[:twinx]()
    global lineLongitude, = ax4[:plot](x, wⁿ, "g-")
    ax4[:set_ylabel]("Longitudinal Offset (m)")
    ax4[:set_ylim](-0.005, 0.005)

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
    ax3[:set_xlim](100, 8e3)
    ax3[:set_ylim](0.0, 1e4)

    fig[:subplots_adjust](hspace = 1.0)

    init() = begin
        global line1
        global line2
        global line3
        global lineLongitude
        global lineHammer
        global time
        line1[:set_data](fill(NaN, si.Nₓ), fill(NaN, si.Nₓ))
        line2[:set_ydata](fill(NaN, forceSize))
        line3[:set_ydata](fill(NaN, fftOutSize))
        lineLongitude[:set_data](fill(NaN, si.Nₓ), fill(NaN, si.Nₓ))
        lineHammer[:set_ydata](fill(NaN, si.Nₓ))
        time[:set_text]("")
        return (line1, line2, line3, lineLongitude, lineHammer, time)
    end

    animate(i) = begin
        global line1
        global line2
        global line3
        global lineHammer
        global lineLongitude
        global time
        global timeDomain
        for j in 1:1
            scaledHammer = (0.0001 .* hammer .+ (h.y - 0.0001))
            Fh::Array{Float64,1} = Interact(h, scaledHammer, s, yⁿ)
            Fh[1:hammerPos-5] .= 0.0
            Fh[hammerPos+5:end] .= 0.0
            (yⁿ, wⁿ, f) = Integrate!(si, yⁿ, wⁿ, Fh, hammer, true)

            # Integrate!(hi, -sum(Fh[hammerPos-10:hammerPos+10]))
            Fy[(si.tₙ % forceSize) + 1] = f

            popfirst!(timeDomain)
            push!(timeDomain, f)
        end

        δy_δx = (si.δ * yⁿ) ./ si.Δx
        θⁿ = atan.(δy_δx)
        δy = wⁿ .* sin.(θⁿ)
        δx = wⁿ .* cos.(θⁿ)

        line1[:set_data](x .+ δx, yⁿ .+ δy)
        # counter += 1
        time[:set_text](@sprintf("time = %.6f", si.tₙ * Δt))
        line2[:set_ydata](Fy)
        fftOut = fftPlan * timeDomain
        line3[:set_ydata](abs.(fftOut))
        next_hammer = hammer_display .+ h.y
        lineHammer[:set_ydata](next_hammer)
        lineLongitude[:set_data](x, wⁿ)

        return (line1, line2, line3, lineLongitude, lineHammer, time)
    end


    generateWAV() = begin
        outSize = Int(round(fs * 8))
        out::Array{Float64,1} = zeros(outSize)
        Juno.progress(name="calculating") do p
            for i in 1:outSize
                scaledHammer = (0.001 .* hammer .+ (h.y - 0.001))
                Fh::Array{Float64,1} = Interact(h, scaledHammer, s, yⁿ)
                Fh[1:hammerPos-5] .= 0.0
                Fh[hammerPos+5:end] .= 0.0
                (yⁿ, f) = Integrate!(si, yⁿ, Fh, hammer, true)

                # Integrate!(hi, -sum(Fh[hammerPos-10:hammerPos+10]))
                out[i] = f / 20.0
                if i % 1000 == 0
                    @info "calculating" progress = i / outSize _id = p
                end
            end
            println(44100.0 / fs)
            resampled = resample(out, 44100.0 / fs)
            wavwrite(resampled, "out_altered.wav", Fs = 44100)
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

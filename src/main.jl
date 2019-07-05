module Waves

using PyPlot
using PyCall
using LinearAlgebra
using Printf
using WAV
using Distributions
using FFTW
using DSP
using Juno
using Unitful

include("StringObject.jl")
include("Integrator.jl")
anim = pyimport("matplotlib.animation")
pygui(true)

main(generate = false) = begin
    FFTW.set_num_threads(4)

    density = 7850.0u"kg/m^3"
    length = 0.665u"m"
    Δx = length / 500
    impedance = 1000.0u"kg/s"
    youngs = 2e11u"Pa"
    tension=760.0u"N"
    diameter=0.001u"m"
    s::Str = Str(length, density, tension, diameter, youngs, impedance)

    c_long = sqrt(s.c_long²)
    Δt_trans = 0.2 * Δx / sqrt(s.c_trans²)

    Qₜ::Int = ceil(Δx / sqrt(s.c_trans²) / (Δx / c_long))
    Δt_long = Δt_trans / Qₜ
    fs::typeof(1.0u"Hz") = 1.0 / Δt_trans

    df::DragForce = DragForce(1.2u"kg/m^3", 0.5)
    si::StrIntegrator = StrIntegrator(s, df, Δt_trans, Δt_long, Δx, Qₜ)

    h::Hammer = Hammer(0.008274u"kg", 0.01u"m", 2.5, 5e9u"N/m^(5/2)", 6.1 / 7.0 * s.length, -0.005u"m", 4.0u"m/s")
    hi::HammerIntegrator = HammerIntegrator(0, h, Δt_trans, 0.0u"m/s^2")

    # yⁿ::Array{Float64,1} = [0.0004 * sin((i - 1) * pi / (si.Nₓ - 1)) + 0.0002 * sin(5 * (i - 1) * pi / (si.Nₓ - 1)) for i = 1:si.Nₓ]
    # yⁿ = GeneratePluck(0.85, 0.01u"m", si.Nₓ, 0.0u"m")
    zⁿ = zeros(si.Nₓ)
    # yⁿ = zeros(typeof(1.0u"m"), si.Nₓ)
    wⁿ = zeros(typeof(1.0u"m"), si.Nₓ)

    # wⁿ[2:10] .= -Δx / 2
    yⁿ = [sin(pi / 180) * s.length * (1 - i / (si.Nₓ-1)) for i = 0:si.Nₓ-1]
    # yⁿ::Array{Float64,1} = zeros(si.Nₓ)

    stdev = ustrip(uconvert(u"m", h.width)) / (2 * √(2 * log(2)))
    # println(stdev)
    unitless_hammer_x = ustrip(uconvert(u"m", h.x))
    dist = Normal(unitless_hammer_x, stdev)
    left = cdf(dist, unitless_hammer_x - 0.005)
    right = cdf(dist, unitless_hammer_x + 0.005)
    massUnderFW = right - left

    # dist = Uniform(h.x - 0.005, h.x + 0.005)
    hammerPos = Int(round(h.x / s.length * si.Nₓ))

    x = range(0.0u"m", step = Δx, length = si.Nₓ)
    scale = pdf(dist, unitless_hammer_x)
    hammer_x = x[hammerPos-5:hammerPos+5]
    hammer_display = [0.001 * (pdf(dist, ustrip(uconvert(u"m", i))) / scale - 1.0)u"m" for i in hammer_x]
    hammer = [pdf(dist, ustrip(uconvert(u"m", i)))u"m" / scale for i in x]

    using3D()
    # m = [h.mass * (cdf(dist, ustrip(uconvert(u"m", i + Δx))) - cdf(dist, ustrip(uconvert(u"m", i)))) for i in x]
    global fig = figure(figsize=(14, 10))
    global ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    global ax4 = fig.add_subplot(2, 2, 2)
    global ax2 = fig.add_subplot(2, 2, 3)
    global ax3 = fig.add_subplot(2, 2, 4)
    ax1.set_zlim3d(-0.01, 0.01)
    global line1, = ax1.plot(ustrip.(x), zⁿ, ustrip.(yⁿ), "k-", zdir="y")
    global lineHammer, = ax1.plot3D(ustrip.(hammer_x), zeros(size(hammer_x)), ustrip.(hammer_display), "-",)
    ax1.set_title("String Contour")
    ax1.set_xlabel("string shape (m)")
    ax1.set_zlabel("height offset (m)")
    global time = ax1.text2D(0.02, 0.8, "", transform = ax1.transAxes)

    global lineLongitude, = ax4.plot(ustrip.(x), ustrip.(wⁿ), "g-")
    ax4.set_ylabel("Longitudinal Offset (m)")
    ax4.set_ylim(-0.0005, 0.0005)

    forceSize::Int64 = 8192

    Fx::Array{Float64,1} = range(0, stop = forceSize / ustrip(fs), length = forceSize)
    Fy::Array{Float64,1} = fill(0, forceSize)
    global line2, = ax2.plot(Fx, Fy)
    ax2.set_title("Force on bridge")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("force (N)")
    ax2.set_ylim(-25.0, 25.0)
    ax2.set_xlim(0.0, forceSize / ustrip(fs))

    fftInSize::Int64 = 16384
    fftOutSize::Int64 = 8193

    global timeDomain = fill(0.0, fftInSize)
    fftPlan = FFTW.plan_rfft(timeDomain, flags=FFTW.UNALIGNED)

    FreqX::Array{Float64,1} = range(0, stop = ustrip(fs) / 2.0, length = fftOutSize)
    FreqY::Array{Float64,1} = fftPlan * timeDomain
    global line3, = ax3.loglog(FreqX, FreqY)
    ax3.set_title("Frequency Spectrum")
    ax3.set_xlabel("Frequency (hz)")
    ax3.set_ylabel("Magnitude")
    ax3.set_xlim(100, 8e3)
    ax3.set_ylim(1, 1e5)

    fig.subplots_adjust(hspace = 0.5, wspace=0.5)


    init() = begin
        global line1
        global line2
        global line3
        global lineLongitude
        global lineHammer
        global time
        line1.set_data(fill(NaN, si.Nₓ), fill(NaN, si.Nₓ))
        line1.set_3d_properties(fill(NaN, si.Nₓ))
        line2.set_ydata(fill(NaN, forceSize))
        line3.set_ydata(fill(NaN, fftOutSize))
        lineLongitude.set_data(fill(NaN, si.Nₓ), fill(NaN, si.Nₓ))
        lineHammer.set_ydata(fill(NaN, si.Nₓ))
        time.set_text("")
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
        for j in 1:10
            scaledHammer = (0.0001 .* hammer .+ (h.y - 0.0001u"m"))
            Fh::NewtonArray = Interact(h, scaledHammer, s, yⁿ)
            Fh[1:hammerPos-5] .= 0.0u"N"
            Fh[hammerPos+5:end] .= 0.0u"N"
            (yⁿ, wⁿ, f) = Integrate!(si, yⁿ, wⁿ, Fh, hammer, true)

            Integrate!(hi, -sum(Fh[hammerPos-10:hammerPos+10]))
            Fy[(ustrip(si.t_trans) % forceSize) + 1] = ustrip(f)

            popfirst!(timeDomain)
            push!(timeDomain, ustrip(f))
        end
        δy_δx = (si.δ * yⁿ) ./ si.Δx
        θⁿ = atan.(δy_δx)
        δy = wⁿ .* sin.(θⁿ)
        δx = wⁿ .* cos.(θⁿ)

        line1.set_data(ustrip.(x .+ δx), zⁿ)
        line1.set_3d_properties(ustrip.(yⁿ .+ δy))
        # line1.set_data(ustrip.(x), ustrip.(yⁿ))
        # counter += 1
        time.set_text(string("time = ", si.t_trans * si.Δt_trans))
        line2.set_ydata(Fy)
        fftOut = fftPlan * timeDomain
        line3.set_ydata(abs.(fftOut))
        next_hammer = hammer_display .+ h.y
        lineHammer.set_data(ustrip.(hammer_x), zeros(size(hammer_x)))
        lineHammer.set_3d_properties(ustrip.(next_hammer))
        lineLongitude.set_data(ustrip.(x), ustrip.(wⁿ))

        return (line1, line2, line3, lineLongitude, lineHammer, time)
    end


    generateWAV() = begin
        outSize = Int(round(ustrip(fs * 8)))
        out::Array{Float64,1} = zeros(outSize)
        Juno.progress(name="calculating") do p
            for i in 1:outSize
                scaledHammer = (0.0001 .* hammer .+ (h.y - 0.0001u"m"))
                Fh::NewtonArray = Interact(h, scaledHammer, s, yⁿ)
                Fh[1:hammerPos-5] .= 0.0u"N"
                Fh[hammerPos+5:end] .= 0.0u"N"
                (yⁿ, wⁿ, f) = Integrate!(si, yⁿ, wⁿ, Fh, hammer, true)

                Integrate!(hi, -sum(Fh[hammerPos-10:hammerPos+10]))

                out[i] = ustrip(f / 20.0)
                if i % 1000 == 0
                    @info "calculating" progress = i / outSize _id = p
                end
            end
            println(44100.0 / ustrip(fs))
            resampled = resample(out, 44100.0 / ustrip(fs))
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

using LinearAlgebra
using SparseArrays
using DataStructures

MeterArray = typeof(fill(1.0u"m", 1))
MeterSquaredArray = typeof(fill(1.0u"m^2", 1))
VelocitySquaredArray = typeof(fill(1.0u"m^2/s^2", 1))
NewtonArray = typeof(fill(1.0u"N", 1))

mutable struct StrIntegrator
    t_trans::Int
    t_long::Int
    str::Str
    drag::DragForce
    Δt_trans::typeof(1.0u"s")
    Δt_long::typeof(1.0u"s")
    Qₜ::Int
    Δx::typeof(1.0u"m")
    Nₓ::Int
    yⁿ⁻¹::MeterArray
    wⁿ⁻¹::MeterArray
    r_trans²::Float64
    r_long²::Float64
    r_ds::typeof(1.0u"m^-1")
    Δy::Tridiagonal{Float64,Array{Float64,1}}
    Δw::Tridiagonal{Float64,Array{Float64,1}}
    δ::Tridiagonal{Float64,Array{Float64,1}}
    δ²::Tridiagonal{Float64,Array{Float64,1}}
    δ⁴::SparseMatrixCSC{Float64,Int64}
end

StrIntegrator(str::Str, drag::DragForce, Δt_trans::typeof(1.0u"s"), Δt_long::typeof(1.0u"s"), Δx::typeof(1.0u"m"), Qₜ::Int = 0) = begin
    Nₓ::Int = round(str.length / Δx)
    Δx = str.length / Nₓ
    yⁿ⁻¹::MeterArray = zeros(typeof(1.0u"m"), Nₓ)
    wⁿ⁻¹::MeterArray = zeros(typeof(1.0u"m"), Nₓ)
    r_trans²::Float64 = str.c_trans² * Δt_trans^2. / (Δx^2.)
    r_long²::Float64 = str.c_long² * Δt_long^2. / (Δx^2.)
    r_ds::typeof(1.0u"m^-1") = ((Δt_long ^ 2) * str.c_long² / (2.0 * Δx ^ 3))
    n_2r2s::Array{Float64,1} = fill(r_trans², Nₓ - 2)
    n_2rl2s::Array{Float64,1} = fill(r_long², Nₓ - 2)
    diag::Array{Float64,1} = fill(2. * (1. - r_trans²), Nₓ)
    diagl::Array{Float64,1} = fill(2. * (1. - r_long²), Nₓ)
    Δy::Tridiagonal{Float64,Array{Float64,1}} = Tridiagonal(
        [n_2r2s..., 2. * r_trans²],
        diag,
        [2. * r_trans², n_2r2s...]
    )
    Δw::Tridiagonal{Float64,Array{Float64,1}} = Tridiagonal(
        [n_2rl2s..., 2. * r_long²],
        diagl,
        [2. * r_long², n_2rl2s...]
    )
    δ::Tridiagonal{Float64,Array{Float64,1}} = Tridiagonal(
        [fill(-0.5, Nₓ - 2)..., 0.0],
        zeros(Nₓ),
        [0.0, fill(0.5, Nₓ - 2)...]
    )
    δ²::Tridiagonal{Float64,Array{Float64,1}} = Tridiagonal(
        [fill(1.0, Nₓ - 2)..., 2.0],
        fill(-2.0, Nₓ),
        [2.0, fill(1.0, Nₓ - 2)...]
    )
    δ⁴::SparseMatrixCSC{Float64,Int64} = spdiagm(
        -3 => [zeros(Nₓ - 4)..., 1.0],
        -2 => [ones(Nₓ - 3)..., -4.0],
        -1 => [fill(-4.0, Nₓ - 2)..., 7.0],
        0 => [-4.0, 7.0, fill(6.0, Nₓ - 4)..., 7.0, -4.0],
        1 => [7.0, fill(-4.0, Nₓ - 2)...],
        2 => [-4.0, ones(Nₓ - 3)...],
        3 => [1.0, zeros(Nₓ - 4)...]
    ) .* ((Δt_trans ^ 2.0 / Δx ^ 4.0) * str.youngs * str.cross_area * (str.diameter / 4.0) ^ 2.0 / str.linear_density)
    # show(stdout, "text/plain", δ⁴)
    StrIntegrator(0, 0, str, drag, Δt_trans, Δt_long, Qₜ, str.length / Nₓ, Nₓ, yⁿ⁻¹, wⁿ⁻¹, r_trans², r_long², r_ds, Δy, Δw, δ, δ², δ⁴)
end

IntegrateW!(si::StrIntegrator, coupled::MeterArray, wⁿ::MeterArray) = begin
    wⁿ⁺¹ = (si.Δw * wⁿ) - si.wⁿ⁻¹ + coupled
    wⁿ⁺¹[1] = 0.0u"m"
    wⁿ⁺¹[end] = 0.0u"m"
    si.t_long += 1
    si.wⁿ⁻¹ = copy(wⁿ)
    return wⁿ⁺¹
end

CalcDragForce(df::DragForce, str::Str, v²::VelocitySquaredArray, Δx::typeof(1.0u"m")) = begin
    return -2.0 * v² * df.C * df.ρ_air * str.diameter * Δx
end

Integrate!(si::StrIntegrator, yⁿ::MeterArray, wⁿ::MeterArray, Fh::NewtonArray, hammer::MeterArray, damping::Bool = false) = begin
    if si.t_trans == 0
        si.yⁿ⁻¹ = copy(yⁿ)
        si.wⁿ⁻¹ = copy(wⁿ)
    end
    # Verlet integration
    str = si.str

    δs = (si.δ * yⁿ) .* (si.δ² * yⁿ)
    F_coupled = si.r_ds .* δs
    wⁿ⁺¹ = copy(wⁿ)

    for i in 1:si.Qₜ
        wⁿ⁺¹ = IntegrateW!(si, F_coupled, wⁿ⁺¹)
    end

    drag = si.drag
    δy = (yⁿ - si.yⁿ⁻¹) / si.Δt_trans
    v² = δy .* abs.(δy)
    F_bridge = uconvert(u"N", str.tension + str.force_coeff * ((yⁿ[2] - yⁿ[1]) / si.Δx) ^ 2.0) * sin(atan((yⁿ[2] - yⁿ[1]) / si.Δx))
    # yⁿ⁺¹ = (si.Δy * yⁿ - (si.δ⁴ * ustrip.(yⁿ)) * unit(1.0u"m")) - si.yⁿ⁻¹
    yⁿ⁺¹ = (si.Δy * yⁿ) - si.yⁿ⁻¹
    yⁿ⁺¹ += (si.Δt_trans ^ 2.0) / (str.linear_density * si.Δx) .* (Fh .+ CalcDragForce(drag, str, v², si.Δx))
    yⁿ⁺¹[1] = yⁿ[1] + si.Δt_trans * F_bridge / str.impedance
    yⁿ⁺¹[end] = 0.0u"m"

    si.t_trans += 1
    si.yⁿ⁻¹ = copy(yⁿ)

    return (yⁿ⁺¹, wⁿ⁺¹, F_bridge)
end

mutable struct HammerIntegrator
    t::Int64
    hammer::Hammer
    Δt::typeof(1.0u"s")
    prev_a::typeof(1.0u"m/s^2")
end

Integrate!(hi::HammerIntegrator, F::typeof(1.0u"N")) = begin
    hammer = hi.hammer
    a = F / hammer.mass
    # Leapfrog Integration
    hammer.y += hi.Δt * (hammer.v + 0.5 * hi.prev_a * hi.Δt)
    hammer.v += hi.Δt * 0.5 * (hi.prev_a + a)
    hi.t += 1
end

Interact(hammer::Hammer, hn::MeterArray, str::Str, yn::MeterArray) = begin
    # println(hammer.x / str.length)
    zn = max.((hn .- yn), 0.0u"m")
    Fn = hammer.K .* (zn .^ hammer.p)
    return Fn
end

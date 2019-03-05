using LinearAlgebra
using SparseArrays
using DataStructures

mutable struct StrIntegrator
    tₙ::Int
    tlₙ::Int
    str::Str
    drag::DragForce
    Δt::Float64
    Δtl::Float64
    Qt::Int
    Δx::Float64
    Nₓ::Int
    yⁿ⁻¹::Array{Float64,1}
    wⁿ⁻¹::Array{Float64,1}
    r²::Float64
    rl²::Float64
    Δy::SparseMatrixCSC{Float64,Int64}
    Δw::SparseMatrixCSC{Float64,Int64}
    δ::SparseMatrixCSC{Float64,Int64}
    δ²::SparseMatrixCSC{Float64,Int64}
end

StrIntegrator(str::Str, drag::DragForce, Δt::Float64, Δx::Float64, Δtl::Float64 = 0.0, Qt::Int = 0) = begin
    Nₓ::Int = round(str.length / Δx)
    yⁿ⁻¹::Array{Float64,1} = zeros(Nₓ)
    wⁿ⁻¹::Array{Float64,1} = zeros(Nₓ)
    r²::Float64 = str.c² * Δt^2. / (Δx^2.)
    rl²::Float64 = str.cl² * Δtl^2. / (Δx^2.)
    n_2r2s::Array{Float64,1} = fill(r², Nₓ - 2)
    n_2rl2s::Array{Float64,1} = fill(rl², Nₓ - 2)
    diag::Array{Float64,1} = fill(2. * (1. - r²), Nₓ)
    diagl::Array{Float64,1} = fill(2. * (1. - rl²), Nₓ)
    Δy::SparseMatrixCSC{Float64,Int64} = spdiagm(
        1 => [2. * r², n_2r2s...],
        0 => diag,
        -1 => [n_2r2s..., 2. * r²],
    )
    Δw::SparseMatrixCSC{Float64,Int64} = spdiagm(
        1 => [2. * rl², n_2rl2s...],
        0 => diagl,
        -1 => [n_2rl2s..., 2. * rl²]
    )
    δ::SparseMatrixCSC{Float64,Int64} = spdiagm(
        1 => [0.0, fill(0.5, Nₓ - 2)...],
        0 => zeros(Nₓ),
        -1 => [fill(-0.5, Nₓ - 2)..., 0.0]
    )
    δ²::SparseMatrixCSC{Float64,Int64} = spdiagm(
        1 => [2.0, fill(1.0, Nₓ - 2)...],
        0 => fill(-2.0, Nₓ),
        -1 => [fill(1.0, Nₓ - 2)..., 2.0]
    )
    StrIntegrator(0, 0, str, drag, Δt, Δtl, Qt, str.length / Nₓ, Nₓ, yⁿ⁻¹, wⁿ⁻¹, r², rl², Δy, Δw, δ, δ²)
end

const weights = [-0.1, 0.6, -1.4, 0.4]

mutable struct StrIntegratorHighAcc
    tₙ::Int
    str::Str
    drag::DragForce
    Δt::Float64
    Δx::Float64
    Nₓ::Int
    yⁿ⁻¹::Array{Float64,1}
    r²::Float64
    Δ::SparseMatrixCSC{Float64,Int64}
end

StrIntegratorHighAcc(str::Str, drag::DragForce, Δt::Float64, Δx::Float64) = begin
    Nₓ::Int = round(str.length / Δx)
    yⁿ⁻¹::Array{Float64,1} = zeros(Nₓ)
    r²::Float64 = str.c² * Δt^2. / (Δx^2.)
    diag::Array{Float64,1} = fill(2.0 - 2.5 * r², Nₓ - 2) # n-2 because first and last elements have von neumann boundary conditions
    diag1::Array{Float64,1} = fill(r² * 4.0 / 3.0, Nₓ - 4)
    diag2::Array{Float64,1} = fill(r² * -1.0 / 12.0, Nₓ - 4)
    Δ::SparseMatrixCSC{Float64,Int64} = spdiagm(
        3 => [-1.0 / 12.0 * r², zeros(Nₓ - 4)...],
        2 => [-1.0 / 8.0 * r², -1.0 / 6.0 * r², diag2...],
        1 => [13.0 / 4.0 * r², 2 * r², diag1..., 2.0 / 3.0 * r²],
        0 => [2.0 - 73.0 / 24.0 * r², diag..., 2.0 - 73.0 / 24.0 * r²],
        -1 => [2.0 / 3.0 * r², diag1..., 2 * r², 13.0 / 4.0 * r²],
        -2 => [diag2..., -1.0 / 6.0 * r², -1.0 / 8.0 * r²],
        -3 => [zeros(Nₓ - 4)..., -1.0 / 12.0 * r²]
    )
    dropzeros!(Δ)
    StrIntegratorHighAcc(0, str, drag, Δt, str.length / Nₓ, Nₓ, yⁿ⁻¹, r², Δ)
end

Integrate!(si::StrIntegrator, yⁿ::Array{Float64,1}, wⁿ::Array{Float64,1}, Fh::Array{Float64,1}, hammer::Array{Float64,1}, damping::Bool = false) = begin
    if si.tₙ == 0
        si.yⁿ⁻¹ = copy(yⁿ)
        si.wⁿ⁻¹ = copy(wⁿ)
    end
    # Verlet integration
    str = si.str

    wⁿ⁺¹::Array{Float64,1} = wⁿ
    curr_wn = copy(wⁿ)
    δs = (si.δ * yⁿ) .* (si.δ² * yⁿ)

    for i in 1:si.Qt
        # println("Start")
        # show(stdout, "text/plain", δs[420:430])
        # println()
        #
        # δs[1] = 0.0
        # δs[end] = 0.0
        if si.tlₙ % si.Qt == 0
            wⁿ⁺¹ = (si.Δw * curr_wn) - si.wⁿ⁻¹ + ((si.Δtl ^ 2) * str.cl² / (si.Δx ^ 3)) .* δs
        else
            wⁿ⁺¹ = (si.Δw * curr_wn) - si.wⁿ⁻¹
        end

        # wⁿ⁺¹ = (si.Δw * curr_wn) - si.wⁿ⁻¹ + ((si.Δtl ^ 2) * str.cl² / (si.Δx ^ 3)) .* δs
        # show(stdout, "text/plain", (si.Δw * curr_wn)[420:430])
        # println()
        # show(stdout, "text/plain", si.wⁿ⁻¹[420:430])
        # println()
        # show(stdout, "text/plain", (((si.Δtl ^ 2) * str.cl² / (si.Δx ^ 3)) .* δs)[420:430])
        # println()
        wⁿ⁺¹[1] = 0.0
        wⁿ⁺¹[end] = 0.0
        si.tlₙ += 1
        si.wⁿ⁻¹ = copy(wⁿ)
        curr_wn = copy(wⁿ⁺¹)
    end

    drag = si.drag
    δy = yⁿ - si.yⁿ⁻¹
    v² = δy .* abs.(δy)
    F_bridge = str.T * (yⁿ[2] - yⁿ[1]) / si.Δx
    # F_bridge = sin(atan((yⁿ[2] - yⁿ[1]) / si.Δx)) * (str.T + wⁿ⁺¹[2])
    # F_bridge = (yⁿ[2] - yⁿ[1]) / si.Δx * (str.T + 0.125 * str.Ε * pi * (str.σ ^ 2.0) * ((yⁿ[2] - yⁿ[1]) / si.Δx) ^ 2.0)
    yⁿ⁺¹ = (si.Δy * yⁿ) - si.yⁿ⁻¹ + ((damping == true) ? CalcDragForce(drag, str, v², si.Δx) : zeros(si.Nₓ))
    yⁿ⁺¹ += (si.Δt ^ 2.0) / (str.μ * si.Δx) .* Fh
    yⁿ⁺¹[1] = yⁿ[1] + ((damping == true) ? si.Δt * F_bridge / str.Z : 0.0)
    yⁿ⁺¹[end] = 0.0

    si.tₙ += 1
    si.yⁿ⁻¹ = copy(yⁿ)

    return (yⁿ⁺¹, wⁿ⁺¹, F_bridge)
end

Integrate!(si::StrIntegratorHighAcc, yⁿ::Array{Float64,1}, Fh::Array{Float64,1}, hammer::Array{Float64,1}, damping::Bool = false) = begin
    if si.tₙ == 0
        si.yⁿ⁻¹ = copy(yⁿ)
    end
    # Verlet integration
    str = si.str
    drag = si.drag
    Δy = yⁿ - si.yⁿ⁻¹
    v² = Δy .* abs.(Δy)
    F_bridge = str.T * (yⁿ[2] - yⁿ[1]) / si.Δx
    yⁿ⁺¹ = (si.Δ * yⁿ) - si.yⁿ⁻¹ + ((damping == true) ? CalcDragForce(drag, str, v²) : zeros(si.Nₓ))
    yⁿ⁺¹ += (si.Δt ^ 2.0) / (0.25 * str.ρ * si.Δx * pi * str.σ ^ 2.0) .* Fh
    yⁿ⁺¹[1] = yⁿ[1] + ((damping == true) ? si.Δt * F_bridge / str.Z : 0.0)
    yⁿ⁺¹[end] = 0.0
    si.tₙ += 1
    si.yⁿ⁻¹ = copy(yⁿ)
    return (yⁿ⁺¹, F_bridge)
end

mutable struct HammerIntegrator
    t::Int64
    hammer::Hammer
    Δt::Float64
    prev_a::Float64
end

Integrate!(hi::HammerIntegrator, F::Float64) = begin
    hammer = hi.hammer
    a = F / hammer.mass
    # Leapfrog Integration
    hammer.y += hi.Δt * (hammer.v + 0.5 * hi.prev_a * hi.Δt)
    hammer.v += hi.Δt * 0.5 * (hi.prev_a + a)
    hi.t += 1
end

Interact(hammer::Hammer, hn::Array{Float64,1}, str::Str, yn::Array{Float64,1}) = begin
    # println(hammer.x / str.length)
    zn = max.((hn .- yn), 0.0)
    Fn = hammer.K .* (zn .^ hammer.p)
    return Fn
end

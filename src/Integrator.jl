using LinearAlgebra

mutable struct StrIntegrator
    tₙ::Int
    str::Str
    drag::DragForce
    Δt::Float64
    Δx::Float64
    Nₓ::Int
    yⁿ⁻¹::Array{Float64,1}
    r²::Float64
    Δ::Tridiagonal{Float64,Array{Float64,1}}
end

StrIntegrator(str::Str, drag::DragForce, Δt::Float64, Δx::Float64) = begin
    Nₓ::Int = round(str.length / Δx)
    yⁿ⁻¹::Array{Float64,1} = zeros(Nₓ)
    r²::Float64 = str.c² * Δt^2. / (Δx^2.)
    n_2r2s::Array{Float64,1} = fill(r², Nₓ - 2)
    diag::Array{Float64,1} = fill(2. * (1. - r²), Nₓ)
    Δ::Tridiagonal{Float64,Array{Float64,1}} = Tridiagonal([n_2r2s..., 2. * r²], diag, [2. * r², n_2r2s...])
    StrIntegrator(0, str, drag, Δt, str.length / Nₓ, Nₓ, yⁿ⁻¹, r², Δ)
end

Integrate!(si::StrIntegrator, yⁿ::Array{Float64,1}, Fh::Array{Float64,1}, hammer::Array{Float64,1}, damping::Bool = false) = begin
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
    # @printf("hfs: %.6f", (si.Δt ^ 2.0) * Fh / (str.μ * si.Δx))
    yⁿ⁺¹ += (si.Δt ^ 2.0) / (str.μ * si.Δx) .* Fh
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

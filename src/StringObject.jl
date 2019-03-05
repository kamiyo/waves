using LinearAlgebra

struct Str
    length::Float64
    c²::Float64
    cl²::Float64
    ρ::Float64
    T::Float64
    σ::Float64
    μ::Float64
    Ε::Float64
    Z::Float64
    A::Float64
end

Str(length::Float64, ρ::Float64, T::Float64, σ::Float64, Ε::Float64, Z::Float64) = begin
    A::Float64 = pi * (σ / 2) ^ 2.0
    μ::Float64 = A * ρ
    c²::Float64 = T / μ
    cl²::Float64 = Ε * A / μ
    println(Ε * A)
    println(cl² * μ - T)
    Str(length, c², cl², ρ, T, σ, μ, Ε, Z, A)
end

struct DragForce
    ρₐᵢᵣ::Float64
    C::Float64
end

CalcDragForce(df::DragForce, str::Str, v²::Array{Float64,1}, Δx::Float64) = begin
    return -2.0 * v² * df.C * df.ρₐᵢᵣ * str.σ * Δx
end

GeneratePluck(position::Float64, height::Float64, n::Int64, start::Float64 = 0.0) = begin
    return [
        range(start, stop=height, length=Int(floor(n * position + 1.)))[1:end-1]...
        range(height, stop=0., length=Int(floor(n * (1 - position))))...
    ]
end

mutable struct Hammer
    mass::Float64
    width::Float64
    p::Float64
    K::Float64
    x::Float64
    y::Float64
    v::Float64
end

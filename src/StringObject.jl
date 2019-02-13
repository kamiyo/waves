using LinearAlgebra

struct Str
    length::Float64
    c²::Float64
    T::Float64
    σ::Float64
    μ::Float64
    Z::Float64
end

Str(length::Float64, c::Float64, T::Float64, σ::Float64, Z::Float64) = (
    c²::Float64 = c^2;
    μ::Float64 = T / c²;
    Str(length, c², T, σ, μ, Z)
)

struct DragForce
    ρₐᵢᵣ::Float64
    C::Float64
end

CalcDragForce(df::DragForce, str::Str, v²::Array{Float64,1}) = begin
    return -2.0 * v² * df.C * df.ρₐᵢᵣ * str.σ / str.μ
end

GeneratePluck(position::Float64, height::Float64, n::Int64) = begin
    return [
        range(0., stop=height, length=Int(floor(n * position + 1.)))[1:end-1]...
        range(height, stop=0, length=Int(floor(n * (1 - position))))...
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

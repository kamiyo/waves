using LinearAlgebra

struct Str
    length::typeof(1.0u"m")
    c_trans²::typeof(1.0u"m^2/s^2")
    c_long²::typeof(1.0u"m^2/s^2")
    density::typeof(1.0u"kg/m^3")
    tension::typeof(1.0u"N")
    diameter::typeof(1.0u"m")
    linear_density::typeof(1.0u"kg/m")
    youngs::typeof(1.0u"Pa")
    impedance::typeof(1.0u"kg/s")
    cross_area::typeof(1.0u"m^2")
    force_coeff::typeof(1.0u"N")
end

Str(length::typeof(1.0u"m"), density::typeof(1.0u"kg/m^3"), tension::typeof(1.0u"N"), diameter::typeof(1.0u"m"), youngs::typeof(1.0u"Pa"), impedance::typeof(1.0u"kg/s")) = begin
    cross_area = pi * (diameter / 2) ^ 2.0
    linear_density = cross_area * density
    c² = tension / linear_density
    cl² = youngs * cross_area / linear_density
    force_coeff = youngs * cross_area / 2
    Str(length, c², cl², density, tension, diameter, linear_density, youngs, impedance, cross_area, force_coeff)
end

struct DragForce
    ρ_air::typeof(1.0u"kg/m^3")
    C::Float64
end

GeneratePluck(position::Float64, height::typeof(1.0u"m"), n::Int64, start::typeof(1.0u"m") = 0.0u"m") = begin
    return [
        range(start, stop=height, length=Int(floor(n * position + 1.)))[1:end-1]...
        range(height, stop=0.0u"m", length=Int(floor(n * (1 - position))))...
    ]
end

mutable struct Hammer
    mass::typeof(1.0u"kg")
    width::typeof(1.0u"m")
    p::Float64
    K
    x::typeof(1.0u"m")
    y::typeof(1.0u"m")
    v::typeof(1.0u"m/s")
end

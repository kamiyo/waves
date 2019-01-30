#pragma once
#include <cmath>

class String
{
public:
    String(double l, double t, double c) :
        length(l),
        tension(t),
        c2(pow(c, 2.))
    {
        mpu = t / c2;
    }
    ~String() {}

private:
    double length;
    double tension;
    double mpu;
    double c2;
};


import matplotlib.pyplot as plt
import sys
import array
import struct
import matplotlib.animation as animation
import numpy as np
import itertools as itt
from IPython.display import HTML
import wave

np.__config__.show()

def convert(data, fout, chunk_size = 1024 * 1024):
    chunk_size *= 4 # samples to bytes
    waveout = wave.open(fout, 'wb')
    waveout.setparams((1, 2, 44100, 0, 'NONE', ''))
    curr_data = data
    while True:
        raw_floats = curr_data[:chunk_size]
        curr_data = curr_data[chunk_size:]
        if raw_floats.size == 0:
            return
        floats = array.array('f', raw_floats)
        samples = [int(sample * 32767) for sample in floats]
        raw_ints = struct.pack("<%dh" % len(samples), *samples)
        waveout.writeframes(raw_ints)

class String:
    def __init__(self, *, length, c=None, tension=None, mass_per_unit_length=None):
        self.length = length
        if c is not None:
            self.c_squared = c ** 2
        else:
            self.tension = tension
            self.mass_per_unit_length = mass_per_unit_length
            self.c_squared = tension / mass_per_unit_length

class StringIntegrator:
    def __init__(self, String, dt, dx):
        self.t = 0
        self.String = String
        self.dt = dt
        self.dx = dx
        self.length_x = int(String.length / dx)
        self.y_prev = np.zeros(self.length_x + 2)
        self.r_squared = String.c_squared * dt**2 / dx**2
        self.twice_1_r2 = 2 * (1 - self.r_squared)
        self.diff_mat = np.zeros((self.length_x + 2, self.length_x + 2))
        diags = np.diag_indices(self.length_x + 1)
        prev_diags = (diags[0] + 1, diags[1])
        self.diff_mat[prev_diags] = self.r_squared
        next_diags = (diags[0], diags[1] + 1)
        self.diff_mat[next_diags] = self.r_squared

    def integrate(self, current):
        y_current = np.concatenate(([current[1]], np.array(current), [current[-2]]))
        if self.t is 0:
            self.y_prev = y_current

        y_next = self.twice_1_r2 * y_current - self.y_prev + np.matmul(self.diff_mat, y_current)
        y_next[1] = 0
        y_next[-2] = current[-1]
        self.t += 1
        self.y_prev = y_current
        force = 70 * (y_next[-3] - y_next[-2]) / self.dx
        # y_next[-2] += force * dt / 1000
        return y_next[1:-1], force

c = 220
# dx = 0.01
# dt = dx/c
dt = 1.0/44100.0
dx = dt * c
print(1 / dt)
print(dx)
s = String(length=1, c=c)
si = StringIntegrator(s, dx=dx, dt=dt)

y_curr = np.concatenate((np.linspace(0, 0.002, num=int(si.length_x * 0.2 + 1))[:-1], np.linspace(0.002, 0, num=int(si.length_x * 0.8))))
# y_curr = 0.01 * np.sin(np.linspace(0, np.pi, si.length_x))

fig, (ax1, ax2) = plt.subplots(2, 1)

x = np.arange(0, s.length - dx, si.dx)

ax1.set_ylim([-0.002, 0.002])
line1, = ax1.plot(x, y_curr)
ax1.set_title('String Contour')
ax1.set_xlabel('string (m)')
ax1.set_ylabel('height offset (m)')
time_text = ax1.text(0.02, 0.8, '', transform=ax1.transAxes)

force_x = np.arange(-2000 * dt, 0, dt)
force_curr = np.zeros(2000)

xlim = np.array([-0.01, 0])
ax2.set_ylim(-0.2, 0.2)
ax2.set_xlim(xlim)
ax2.set_title('Force on RH Bridge')
ax2.set_xlabel('time')
ax2.set_ylabel('Force (N)')

line2, = ax2.plot(force_x, force_curr)
def init():
    line1.set_ydata([np.nan] * len(x))
    line2.set_xdata([np.nan] * len(force_x))
    line2.set_ydata([np.nan] * len(force_x))
    time_text.set_text('')
    return line1, line2, time_text

def animate(i):
    global y_curr, force_curr, force_x, xlim
    y0 = y_curr
    for _ in itt.repeat(None, 100):
        y_next, force = si.integrate(y0)
        y0 = y_next
        force_curr = np.concatenate((force_curr[1:], [force]))
        # dy = dt * force / 100
        # y0 += np.linspace(0, dy, len(y0))
        force_x = force_x + dt
        xlim = xlim + dt

    line1.set_ydata(y0)
    time_text.set_text('time = %.6f' % (float(si.t) * dt))

    line2.set_xdata(force_x)
    ax2.set_xlim(xlim)
    line2.set_ydata(force_curr)
    y_curr = y0
    return line1, line2, ax2, time_text

anim = animation.FuncAnimation(
    fig, animate, init_func=init, interval=2, blit=False, save_count=50
)

plt.subplots_adjust(hspace=0.5)
plt.show()

# out = np.empty(2205000)
# i = 0

# y0 = y_curr

# for _ in itt.repeat(None, 2205000):
#     y_next, force = si.integrate(y0)
#     y0 = y_next
#     out[i] = force
#     if i % 1000 == 0:
#         print(i)
#     i += 1

# convert(out, open('out.wav', 'wb'))

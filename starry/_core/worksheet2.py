from aesara_theano_fallback import aesara as theano
import aesara_theano_fallback.tensor as tt
import numpy as np
import starry

#THEANO_FLAGS = "optimizer=None"
#THEANO_FLAGS = "mode=DebugMode"
#THEANO_FLAGS = "exception_verbosity=high"

starry.config.lazy = False
starry.config.quiet = False

ydeg = 2
nt = 10
nw = 100
wav = np.linspace(1082.5, 1083.5, nw)
inc = 90.
veq = 100000.

star = starry.Primary(starry.DopplerMap(ydeg=ydeg, nt = nt, wav=wav, inc=inc, veq=veq), m=1.0, r=1.0, prot=1.0)
#star = starry.Primary(starry.Map(ydeg = ydeg, nt = nt, nw = nw, amp=1.0), m=1.0, r=1.0, prot=1.0)

planet = starry.kepler.Secondary(
    starry.Map(ydeg=ydeg, nt=nt, nw=nw, amp=5e-10),  # the surface map
    m=0,  # mass in solar masses
    r=0.2,  # radius in solar radii
    porb=2.0,  # orbital period in days
    prot=1.0,  # rotation period in days (synchronous)
    Omega=0.,  # longitude of ascending node in degrees
    ecc=0.,  # eccentricity
    w=0.,  # longitude of pericenter in degrees
    t0=0,  # time of transit in days
)

vsini = star.map.ops.enforce_bounds(veq * tt.sin(inc), 0.0, star.map.ops.vsini_max)
x = star.map.ops.get_x(vsini)
print("x")
print(x)

deg = star.map.ops.ydeg + star.map.ops.udeg
x = tt.as_tensor_variable(x)
sijk = tt.zeros((deg + 1, deg + 1, 2, tt.shape(x)[0]))

print(tt.shape(x)[0].eval())


# Initial conditions
r2 = tt.maximum(1 - x ** 2, tt.zeros_like(x))
print("r2")
print(r2.eval())
xo = xo * tt.ones_like(x)
yo = yo * tt.ones_like(x)
ro = ro * tt.ones_like(x)
#ro = tt.switch(
#    tt.gt(xo - ro, x),
#   0.,
#    tt.switch(tt.gt(xo + ro, x), ro, 0.)
#)

print("ro")
print(ro.eval())

print("yo")
print(yo.eval())

# Silly hack to prevent issues with the undefined derivative at x = 1
# This just computes the square root of r2, zeroing out values very
# close to zero.
r = tt.maximum(1 - x ** 2, tt.zeros_like(x) + 1e-100) ** 0.5
r = tt.switch(tt.gt(r, 1e-49), r, tt.zeros_like(r))

sijk = tt.set_subtensor(sijk[0, 0, 0], 2 * r)
sijk = tt.set_subtensor(sijk[0, 0, 1], 0.5 * np.pi * r2)

# Upward recursion in j
for j in range(2, deg + 1, 2):
    sijk = tt.set_subtensor(
        sijk[0, j, 0], ((j - 1.0) / (j + 1.0)) * r2 * sijk[0, j - 2, 0]
    )
    sijk = tt.set_subtensor(
        sijk[0, j, 1], ((j - 1.0) / (j + 2.0)) * r2 * sijk[0, j - 2, 1]
    )

# Upward recursion in i
for i in range(1, deg + 1):
    sijk = tt.set_subtensor(sijk[i, 0, 0], sijk[i - 1, 0, 0] * x)
    sijk = tt.set_subtensor(sijk[i, 0, 1], sijk[i - 1, 0, 1] * x)
    sijk = tt.set_subtensor(sijk[i, 1, 0], sijk[i - 1, 1, 0] * x)
    sijk = tt.set_subtensor(sijk[i, 1, 1], sijk[i - 1, 1, 1] * x)

    # Upward recursion in j
    for j in range(2, deg + 1):
        sijk = tt.set_subtensor(
            sijk[i, j, 0], sijk[i - 1, j, 0] * x
        )
        sijk = tt.set_subtensor(
            sijk[i, j, 1], sijk[i - 1, j, 1] * x
        )

print("sijk")
print(sijk.eval())

# Limits for occultation
chi = tt.maximum((ro ** 2 - (x - xo) ** 2), tt.zeros_like(x) + 1e-100) ** 0.5

print("chi")
print(chi.eval())

ul = tt.switch(
    tt.gt(xo - ro, x), 0, 
    tt.switch(tt.gt(x, xo + ro), 0,
    tt.switch(tt.gt(yo + chi, r), tt.ones_like(r), (yo + chi) / r)
    )
)

print("ul")
print(ul.shape.eval())

ll = tt.switch(
    tt.gt(xo - ro, x), 0,
    tt.switch(tt.gt(x, xo + ro), 0,
    tt.switch(tt.gt(yo - chi, -1 * r), (yo - chi) / r, -1 * tt.ones_like(r))
    )
)

print("ll")
print(ll.shape.eval())

# Boundary conditions for occultation
sijk_o = tt.zeros((deg + 1, deg + 1, 2, tt.shape(x)[0]))

print("tt.shape(x)[0]")
#print(tt.shape(x)[0].eval())

print("sijk_o initiation")
#print(sijk_o.eval())

I = tt.zeros((deg + 1, tt.shape(x)[0]))

print("I initiation")
print(I.shape.eval())


I = tt.set_subtensor(
    I[0], 0.5 * (tt.arcsin(ul) - tt.arcsin(ll) + ul * (1 - ul ** 2) ** 0.5 - ll * (1 - ll ** 2) ** 0.5)
)
I = tt.set_subtensor(
    I[1], ((1 - ll) ** (3 / 2) - (1 - ul) ** (3 / 2)) / 3
)

print("I after setting first two elements")
print(I.shape.eval())


sijk_o = tt.set_subtensor(sijk_o[0, 0, 0], (ul * r) - (ll * r))

print("sijk_o after setting 0,0,0")
print(sijk_o.eval())

sijk_o = tt.set_subtensor(sijk_o[0, 1, 0], 0.5 * (ul ** 2 - ll ** 2) * r2)

print("sijk_o after setting 0,1,0")
print(sijk_o.eval())

sijk_o = tt.set_subtensor(sijk_o[0 ,0, 1], I[0] * r2)

print("sijk_o after setting 0,0,1")
print(sijk_o.eval())

sijk_o = tt.set_subtensor(sijk_o[0, 1, 1], I[1] * r ** 3)

print("sijk_o after setting 0,1,1")
print(sijk_o.eval())

print("sijk_o after setting 1,0,0")
print(sijk_o.eval())

# Upward recursion in j
for j in range(2, deg + 1):
    sijk_o = tt.set_subtensor(
        sijk_o[0, j, 0], (1.0 / (j + 1.0)) * ((ul * r) ** (j + 1.0) - (ll * r) ** (j + 1.0))
    )
    I = tt.set_subtensor(
        I[j], 
        1./(j + 2.0) * ((j - 1.0) * I[j - 2] - ul ** (j - 1.0) * (1 - ul) ** (3./2.) + ll ** (j - 1.0) * (1 - ll) ** (3./2.))
    )
    sijk_o = tt.set_subtensor(
        sijk_o[0, j, 1], r ** (j + 2.0) * I[j]
    )

print("sijk_o after upward recursion in j")
#print(sijk_o.eval())

print("I after upward recursion in j")
#print(I.eval())


#x = tt.tile(x, (deg + 1, 1))
#print("x after tile")
#print(x.eval())

#a = sijk_o[0,:,:,:] * x
#print("a after sijk_o[0] * x")
#print(a.eval())

# Upward recursion in i
for i in range(1, deg + 1):
    sijk_o = tt.set_subtensor(sijk_o[i, 0, 0], sijk_o[i - 1, 0, 0] * x)
    sijk_o = tt.set_subtensor(sijk_o[i, 0, 1], sijk_o[i - 1, 0, 1] * x)
    sijk_o = tt.set_subtensor(sijk_o[i, 1, 0], sijk_o[i - 1, 1, 0] * x)
    sijk_o = tt.set_subtensor(sijk_o[i, 1, 1], sijk_o[i - 1, 1, 1] * x)
    for j in range(2, deg + 1):
        sijk_o = tt.set_subtensor(
            sijk_o[i, j, 0], sijk_o[i - 1, j, 0] * x
        )
        sijk_o = tt.set_subtensor(
            sijk_o[i, j, 1], sijk_o[i - 1, j, 1] * x
        )

print("sijk_o after upward recursion")
print(sijk_o.eval())

#Subtract sijk_o from sijk element by element using set_subtensor
for i in range(0, deg + 1):
    for j in range(0, deg + 1):
        sijk = tt.set_subtensor(sijk[i, j, 0], sijk[i, j, 0] - sijk_o[i, j, 0])
        sijk = tt.set_subtensor(sijk[i, j, 1], sijk[i, j, 1] - sijk_o[i, j, 1])


print("sijk_new")
print(sijk.eval())

 # Full vector
N = (deg + 1) ** 2
s = tt.zeros((N, tt.shape(x)[0]))
n = np.arange(N)
LAM = np.floor(np.sqrt(n))
DEL = 0.5 * (n - LAM ** 2)
i = np.array(np.floor(LAM - DEL), dtype=int)
j = np.array(np.floor(DEL), dtype=int)
k = np.array(np.ceil(DEL) - np.floor(DEL), dtype=int)
s = tt.set_subtensor(s[n], sijk[i, j, k])

print("s")
print(s.eval())


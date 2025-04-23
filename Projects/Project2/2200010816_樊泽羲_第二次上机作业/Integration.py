import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 4.0/(1.0 + x**2)

# Composite Midpoint Rule
def midpoint_integral(f, a, b, n):
    h = (b - a) / n
    x = a + (np.arange(n) + 0.5)*h
    return h * np.sum(f(x))

# Composite Trapezoidal Rule
def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    x = a + np.arange(n+1)*h
    y = f(x)
    return h*(0.5*y[0] + y[1:-1].sum() + 0.5*y[-1])

# Composite Simpson’s Rule
def composite_simpson(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule")
    h = (b - a) / n
    x = a + np.arange(n+1)*h
    y = f(x)
    return h/3 * (y[0] + 2*y[2:n:2].sum() + 4*y[1:n:2].sum() + y[n])

# Romberg Integration
def romberg_integration(f, a, b, max_iter):
    R = np.zeros((max_iter, max_iter))
    R[0,0] = (b - a)*(f(a) + f(b))/2
    for i in range(1, max_iter):
        h = (b - a)/2**i
        summ = sum(f(a + (2*k-1)*h) for k in range(1, 2**(i-1)+1))
        R[i,0] = R[i-1,0]/2 + h*summ
        for j in range(1, i+1):
            R[i,j] = (4**j * R[i,j-1] - R[i-1,j-1])/(4**j - 1)
    return R

# Adaptive Simpson’s Rule
def adaptive_simpson(f, a, b, eps):
    def aux(a, b, eps, fa, fm, fb):
        h = b - a
        I = (h/6)*(fa + 4*fm + fb)
        c = (a + b)/2
        fd = f((a + c)/2); fe = f((c + b)/2)
        IL = (h/12)*(fa + 4*fd + fm)
        IR = (h/12)*(fm + 4*fe + fb)
        if abs(IL + IR - I) < 15*eps:
            return IL + IR + (IL + IR - I)/15
        return (aux(a, c, eps/2, fa, fd, fm) +
                aux(c, b, eps/2, fm, fe, fb))
    fa, fb = f(a), f(b)
    fm = f((a+b)/2)
    return aux(a, b, eps, fa, fm, fb)

# Plotting and verification
n_vals = 2**np.arange(1, 27)

plt.figure(figsize=(12, 8))

# Midpoint
err_M = [abs(midpoint_integral(f,0,1,n) - np.pi) for n in n_vals]
plt.subplot(2,2,1)
plt.plot(np.log(n_vals), -np.log(err_M), '-o')
plt.title('Composite Midpoint: 2nd-order')

# Trapezoid
err_T = [abs(trapezoidal_rule(f,0,1,n) - np.pi) for n in n_vals]
plt.subplot(2,2,2)
plt.plot(np.log(n_vals), -np.log(err_T), '-o')
plt.title('Composite Trapezoid: 2nd-order')

# Simpson
err_S = [abs(composite_simpson(f,0,1,n) - np.pi) for n in n_vals]
plt.subplot(2,2,3)
plt.plot(np.log(n_vals), -np.log(err_S), '-o')
plt.title('Composite Simpson: 4th-order')

# Romberg
max_k = 15
R = romberg_integration(f,0,1,max_k)
n_R = 2**np.arange(max_k)
err_R = [abs(R[k,k] - np.pi) for k in range(max_k)]
plt.subplot(2,2,4)
ln_n = np.log(n_R)
ln_e = -np.log(err_R)
plt.scatter(ln_n, ln_e)
# Fit only the first 8 points
coef = np.polyfit(ln_n[:8], ln_e[:8], 2)
plt.plot(ln_n[:8], np.poly1d(coef)(ln_n[:8]), '--')
plt.title('Romberg: quadratic trend (first 8 points)')

plt.tight_layout()
plt.show()
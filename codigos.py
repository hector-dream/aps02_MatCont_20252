# ------------ TAREFA 1 ------------
import numpy as np

def f(x, y):
    return 3*x**2 + 3*x*y + 2*y**2 + x + 2*y

def grad(x, y):
    return np.array([6*x + 3*y + 1, 3*x + 4*y + 2], dtype=float)

def gradient_descent(alpha, x0=0.0, y0=0.0, tol=1e-5, max_iter=1_000_000):
    x, y = float(x0), float(y0)
    it = 0
    g = grad(x, y)
    while np.linalg.norm(g, 2) > tol and it < max_iter:
        x, y = (np.array([x, y]) - alpha * g)
        g = grad(x, y)
        it += 1
    return it, x, y, f(x,y), np.linalg.norm(g,2)

for a in [0.10, 0.15, 0.20, 0.30, 0.50]:
    it, x, y, fv, gn = gradient_descent(a)
    status = "OK" if np.isfinite(x) and np.isfinite(y) else "DIVERGE"
    print(f"alpha={a:<4}  iters={it:<6}  x={x:.6f}  y={y:.6f}  f={fv:.6f}  ||grad||={gn:.2e}  {status}")

# ------------ TAREFA 2 ------------
import numpy as np
import math

def g(x, y):
    S = x**2 + y**2 + 3 + 2*x**2*math.exp(-y**2)
    return math.sqrt(S) + (x - 2)**2

def gradiente_g(x, y):
    S = x**2 + y**2 + 3 + 2*x**2*math.exp(-y**2)
    R = math.sqrt(S)
    e = math.exp(-y**2)
    dgx = (x + 2*x*e) / R + 2*(x - 2)
    dgy = (y - 2*x**2*y*e) / R
    return np.array([dgx, dgy])

def gradiente_descendente_g(alpha, x0=0, y0=1, tol=1e-6, max_iter=1_000_000):
    x, y = float(x0), float(y0)
    for i in range(max_iter):
        grad = gradiente_g(x, y)
        norma = np.linalg.norm(grad)
        if norma < tol:
            break
        x -= alpha * grad[0]
        y -= alpha * grad[1]
    return x, y, g(x, y), i + 1, norma

valores_alpha = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

for alpha in valores_alpha:
    x, y, g_val, it, norma = gradiente_descendente_g(alpha)
    if np.isnan(x) or np.isinf(x) or np.isnan(y) or np.isinf(y):
        print(f"α = {alpha:.2f} → DIVERGE")
    else:
        print(f"α = {alpha:.2f} | x = {x:.6f}, y = {y:.6f}, g = {g_val:.9f}, iterações = {it}, ||grad|| = {norma:.2e}")

# ------------ TAREFA 3 ------------
import numpy as np
import math

def h(x, y):
    A = math.exp(-(x*x + y*y))
    B = math.exp(-(x-2)**2 - (y-3)**2)
    return 4*A + 3*B - x*x/7 - y*y/10 + 2

def grad_h(x, y):
    A = math.exp(-(x*x + y*y))
    B = math.exp(-(x-2)**2 - (y-3)**2)
    dhdx = -8*x*A + 3*B*(-2*x + 4) - (2*x/7)
    dhdy = -8*y*A + 3*B*(-2*y + 6) - (y/5)
    return np.array([dhdx, dhdy], float)

def gradient_ascent(alpha, x0, y0, tol=1e-6, max_iter=200000):
    x, y = float(x0), float(y0)
    g = grad_h(x, y)
    it = 0
    while np.linalg.norm(g, 2) > tol and it < max_iter:
        x += alpha * g[0]
        y += alpha * g[1]
        g = grad_h(x, y)
        it += 1
    return x, y, h(x, y), it, np.linalg.norm(g, 2)

m1 = gradient_ascent(alpha=0.10, x0=0.0, y0=0.0)
m2 = gradient_ascent(alpha=0.10, x0=2.0, y0=3.0)

print(f"Máximo 1 (0,0):  x={m1[0]:.6f}, y={m1[1]:.6f}, h={m1[2]:.6f}, iterações={m1[3]}")
print(f"Máximo 2 (2,3):  x={m2[0]:.6f}, y={m2[1]:.6f}, h={m2[2]:.6f}, iterações={m2[3]}")

# ------------ TAREFA 4 ------------
import numpy as np

def f(x, y):
    return 3*x**2 + 3*x*y + 2*y**2 + x + 2*y

def grad(x, y):
    return np.array([6*x + 3*y + 1, 3*x + 4*y + 2], float)

H = np.array([[6.0, 3.0],
              [3.0, 4.0]])

def gradiente_descendente_passo_variavel(x0=0.0, y0=0.0, tol=1e-5, max_iter=100000):
    x, y = float(x0), float(y0)
    it = 0
    g = grad(x, y)

    while np.linalg.norm(g, 2) > tol and it < max_iter:
        g = grad(x, y)
        Hg = H @ g
        alpha = float(g @ g) / float(g @ Hg)
        x -= alpha * g[0]
        y -= alpha * g[1]
        it += 1

    return x, y, f(x, y), it, np.linalg.norm(grad(x, y), 2), alpha

xv, yv, fv, itv, normv, last_alpha = gradiente_descendente_passo_variavel()
print(f"[Passo variável] x={xv:.6f}, y={yv:.6f}, f={fv:.6f}, iterações={itv}, ||grad||={normv:.2e}, último α={last_alpha:.6f}")

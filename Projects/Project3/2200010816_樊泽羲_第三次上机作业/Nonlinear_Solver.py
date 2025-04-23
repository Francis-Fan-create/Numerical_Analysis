import numpy as np
import matplotlib.pyplot as plt

def backtracking_line_search(f, x, d, fx, B_inv, rho=0.5, c=1e-4):
    """
    Backtracking line search to satisfy a Wolfe-type condition:
        ||f(x + α d)|| <= (1 - c α) ||f(x)||
    """
    α = 1.0
    norm_fx = np.linalg.norm(fx)
    # Avoid division by zero or very small norm_fx
    if norm_fx < 1e-15:
        return 0.0, fx # No step needed if already at solution

    while True:
        x_new = x + α * d
        f_new = f(x_new)
        if np.linalg.norm(f_new) <= (1 - c * α) * norm_fx:
            return α, f_new
        α *= rho
        # Prevent infinitely small steps
        if α < 1e-14:
             # Return the current best if step size becomes too small
             # This might indicate issues with the search direction or function
             # For simplicity, return the state before this loop started
             # A more robust implementation might try a different strategy
             # or signal failure.
             # Returning alpha=0 and original fx to avoid changing x.
             # Alternatively, could return last computed f_new.
             # Let's return the last computed f_new as it might be closer.
             x_last_try = x + α/rho * d # x before the last shrink
             f_last_try = f(x_last_try)
             # Check if the last try was better than the original
             if np.linalg.norm(f_last_try) < norm_fx:
                 return α/rho, f_last_try
             else: # If not better, signal no progress with alpha=0
                 return 0.0, fx


def approx_jacobian(f, x, eps=1e-8):
    """
    Finite-difference approximation of Jacobian J(x) ∈ R^{n×n}.
    """
    n = x.size
    J = np.zeros((n, n))
    fx = f(x)
    for i in range(n):
        dx = np.zeros_like(x)
        # Use relative step size for potentially better accuracy
        h = eps * (abs(x[i]) if x[i] != 0 else 1.0)
        dx[i] = h
        # Check if h is too small, causing floating point issues
        if x[i] + h == x[i]:
             h = eps # Fallback to absolute step size
             dx[i] = h
             if x[i] + h == x[i]:
                 # If still no change, Jacobian column might be zero or need smaller eps
                 # For simplicity, we proceed, but this could be flagged.
                 pass # Or raise an error/warning
        J[:, i] = (f(x + dx) - fx) / h
    return J

def broyden_solver(f, x0, max_iter=50, tol=1e-6):
    """
    Quasi-Newton solver using Broyden's rank-one update with Sherman-Morrison.

    Arguments:
        f         -- function mapping R^n to R^n.
        x0        -- initial guess (1d numpy array).
        max_iter  -- maximum number of iterations.
        tol       -- tolerance for stopping on residual norm.

    Returns:
        x          -- approximate root.
        history    -- list of residual norms at each iterate.
    """
    x = x0.copy()
    n = x.size

    # Initial Jacobian approx via finite differences, and its inverse
    try:
        B = approx_jacobian(f, x)
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        print("Error: Initial Jacobian is singular. Cannot start Broyden's method.")
        return x, [np.linalg.norm(f(x))] # Return initial state

    history = []

    # Initial residual
    fx = f(x)
    norm_fx = np.linalg.norm(fx)
    history.append(norm_fx)

    if norm_fx < tol:
        print("Initial guess is already within tolerance.")
        return x, history

    for k in range(max_iter):
        # search direction: d = - B_inv * f(x)
        d = -B_inv.dot(fx)

        # line search to find α and f(x+αd)
        α, f_new = backtracking_line_search(f, x, d, fx, B_inv)

        # If line search fails to find a step (alpha=0), stop iterating.
        if α == 0.0:
            print(f"Warning: Line search failed to find a step at iteration {k+1}. Stopping.")
            break

        # update iterate
        s = α * d
        x_next = x + s

        # compute new residual (already computed in line search)
        fx_next = f_new

        # check convergence
        norm_fx_next = np.linalg.norm(fx_next)
        history.append(norm_fx_next)
        if norm_fx_next < tol:
            print(f"Converged in {k+1} iterations.")
            x = x_next # Update x to the final solution
            break

        # function increment
        y = fx_next - fx

        # Sherman–Morrison update for inverse:
        # B_{k+1}^{-1} = B_k^{-1} + (s - B_k^{-1} y) s^T B_k^{-1} / (s^T B_k^{-1} y)
        # Let u = s - B_inv @ y
        # Let v = s
        # Denominator = v^T @ B_inv @ y = s^T @ B_inv @ y
        # Numerator = u @ v^T @ B_inv = (s - B_inv @ y) @ s^T @ B_inv
        # This seems overly complex. Let's use the standard SM formula for B_inv update directly.
        # B_{k+1} = B_k + (y - B_k s) s^T / (s^T s)  <- This is Broyden's "good" update for B
        # The inverse update using SM on B_k update:
        # B_{k+1}^{-1} = B_k^{-1} + (s - B_k^{-1} y) / (s^T B_k^{-1} y) * s^T B_k^{-1}
        Binv_y = B_inv.dot(y)
        s_T_Binv_y = s.T.dot(Binv_y)

        if abs(s_T_Binv_y) < 1e-12:
            # Denominator is too small, skip update to avoid instability
            # Recompute Jacobian occasionally might be a better strategy here
            print(f"Warning: Denominator in Broyden update is near zero at iteration {k+1}. Skipping update.")
            B_inv_next = B_inv # Keep the old inverse
        else:
            u = s - Binv_y
            vT_Binv = s.T.dot(B_inv)
            B_inv_next = B_inv + np.outer(u, vT_Binv) / s_T_Binv_y

        # prepare for next iteration
        x, fx = x_next, fx_next
        B_inv = B_inv_next # Only update B_inv

        if k == max_iter - 1:
            print("Maximum iterations reached without convergence.")

    return x, history

# ================================================
# System 1: Extended Powell Singular Function
# ================================================
def extended_powell_singular_system(x):
    """
    System of n nonlinear equations (n must be multiple of 4):
      f[4i]   = x[4i]   + 10*x[4i+1]
      f[4i+1] = sqrt(5)*(x[4i+2] - x[4i+3])
      f[4i+2] = (x[4i+1] - 2*x[4i+2])**2
      f[4i+3] = sqrt(10)*(x[4i] - x[4i+3])**2
    Uses 0-based indexing corresponding to 1-based in problem description.
    f_{4i-3} -> f[4*i]
    f_{4i-2} -> f[4*i+1]
    f_{4i-1} -> f[4*i+2]
    f_{4i}   -> f[4*i+3]
    """
    n = x.size
    if n % 4 != 0:
        raise ValueError("n must be a multiple of 4 for Extended Powell Singular Function")
    f = np.zeros_like(x)
    for i in range(n // 4):
        idx0 = 4 * i
        idx1 = 4 * i + 1
        idx2 = 4 * i + 2
        idx3 = 4 * i + 3
        f[idx0] = x[idx0] + 10 * x[idx1]
        f[idx1] = np.sqrt(5) * (x[idx2] - x[idx3])
        f[idx2] = (x[idx1] - 2 * x[idx2])**2
        f[idx3] = np.sqrt(10) * (x[idx0] - x[idx3])**2
    return f

print("--- Solving Extended Powell Singular Function ---")
# choose problem size and initial guess
n_powell = 8  # must be multiple of 4
x0_powell = np.tile([3.0, -1.0, 0.0, 1.0], n_powell // 4)

# solve
solution_powell, residuals_powell = broyden_solver(extended_powell_singular_system, x0_powell, max_iter=100, tol=1e-8)
print("Approximate solution (Powell):", solution_powell)
print(f"Final residual norm (Powell): {residuals_powell[-1]:.2e}")


# visualize convergence
plt.figure(figsize=(6,4))
plt.semilogy(residuals_powell, marker='o', linestyle='-')
plt.xlabel("Iteration")
plt.ylabel(r"Residual Norm $\|f(x)\|_2$")
plt.title("Broyden Convergence: Extended Powell Singular Function")
plt.grid(True, which="both", ls="--")
plt.tight_layout()
# plt.show() # Show plots individually or all at the end

# ================================================
# System 2: Trigonometric Function
# ================================================
def trigonometric_system(x):
    """
    System of n nonlinear equations:
      f_i(x) = n - sum_{j=1}^{n} cos(x_j) + i*(1 - cos(x_i)) - sin(x_i)
    Uses 0-based indexing for x and f, but the coefficient 'i' follows the
    1-based definition from the problem description (i.e., uses i+1).
    """
    n = x.size
    f = np.zeros_like(x)
    sum_cos_x = np.sum(np.cos(x))
    for i in range(n):
        # Using (i+1) for the coefficient as per the 1-based formula f_i
        f[i] = n - sum_cos_x + (i + 1) * (1 - np.cos(x[i])) - np.sin(x[i])
    return f

print("\n--- Solving Trigonometric Function ---")
# choose problem size and initial guess
n_trig = 4 # Example size
x0_trig = np.full(n_trig, 1.0 / n_trig)

# solve
solution_trig, residuals_trig = broyden_solver(trigonometric_system, x0_trig, max_iter=100, tol=1e-8)
print("Approximate solution (Trig):", solution_trig)
print(f"Final residual norm (Trig): {residuals_trig[-1]:.2e}")


# visualize convergence
plt.figure(figsize=(6,4))
plt.semilogy(residuals_trig, marker='s', linestyle=':')
plt.xlabel("Iteration")
plt.ylabel(r"Residual Norm $\|f(x)\|_2$")
plt.title(f"Broyden Convergence: Trigonometric Function (n={n_trig})")
plt.grid(True, which="both", ls="--")
plt.tight_layout()
# plt.show()

# ================================================
# System 3: Wood Function
# ================================================
def wood_system(x):
    """
    System of 4 nonlinear equations (Wood Function).
    f1(x)=(x1-1)^2
    f2(x)=100(x1-x2)^2
    f3(x)=90(x3-x4)^2+(x3-1)^2  <- Assuming x3, x4 indices (0-based: x[2], x[3])
    f4(x)=10.1((1-x2)^2+(1-x4)^2)+19.8(1-x2)(1-x4) <- Assuming x2, x4 indices (0-based: x[1], x[3])
    """
    n = x.size
    if n != 4:
        raise ValueError("Wood function is defined for n=4")
    f = np.zeros(4)
    f[0] = (x[0] - 1)**2
    f[1] = 100 * (x[0] - x[1])**2
    f[2] = 90 * (x[2] - x[3])**2 + (x[2] - 1)**2 # Using 0-based indices x[2], x[3]
    f[3] = 10.1 * ((1 - x[1])**2 + (1 - x[3])**2) + 19.8 * (1 - x[1]) * (1 - x[3]) # Using 0-based indices x[1], x[3]
    return f

print("\n--- Solving Wood Function ---")
# Initial guess
x0_wood = np.array([-3.0, -1.0, -3.0, -1.0])

# solve
solution_wood, residuals_wood = broyden_solver(wood_system, x0_wood, max_iter=200, tol=1e-8) # Increased max_iter for Wood
print("Approximate solution (Wood):", solution_wood)
print(f"Final residual norm (Wood): {residuals_wood[-1]:.2e}")


# visualize convergence
plt.figure(figsize=(6,4))
plt.semilogy(residuals_wood, marker='^', linestyle='--')
plt.xlabel("Iteration")
plt.ylabel(r"Residual Norm $\|f(x)\|_2$")
plt.title("Broyden Convergence: Wood Function (n=4)")
plt.grid(True, which="both", ls="--")
plt.tight_layout()

# Show all plots
plt.show()
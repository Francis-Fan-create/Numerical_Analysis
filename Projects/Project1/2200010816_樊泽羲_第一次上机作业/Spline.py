import numpy as np
import matplotlib.pyplot as plt

def TriSolve(a, b, c, d):
    """
    Solves a tridiagonal system A*x = d using the Thomas algorithm.
    The tridiagonal matrix A has main diagonal a, lower diagonal b and upper diagonal c.
    
    Parameters:
        a : 1D array of length n (main diagonal)
        b : 1D array of length n-1 (sub-diagonal)
        c : 1D array of length n-1 (super-diagonal)
        d : 1D array of length n (right-hand side)
        
    Returns:
        x : 1D array solution of length n
    """
    n = len(d)
    a_prime = np.zeros(n)
    c_prime = np.zeros(n-1)
    d_prime = np.zeros(n)
    
    # Initialization
    a_prime[0] = a[0]
    c_prime[0] = c[0] / a_prime[0]
    d_prime[0] = d[0]
    
    # Forward elimination
    for k in range(1, n):
        a_prime[k] = a[k] - b[k-1] * c_prime[k-1]
        if k < n - 1:
            c_prime[k] = c[k] / a_prime[k]
        d_prime[k] = d[k] - c_prime[k-1] * d_prime[k-1]
    
    # Back substitution
    x = np.zeros(n)
    x[-1] = d_prime[-1] / a_prime[-1]
    for k in range(n-2, -1, -1):
        x[k] = (d_prime[k] - b[k] * x[k+1]) / a_prime[k]
    
    return x

def ObjectFunction(x):
    """
    Computes the target function f(x) = exp(sin(x)) + cos(4*x).
    
    Parameters:
        x : array-like, input variable
        
    Returns:
        y : array-like, function values
    """
    return np.exp(np.sin(x)) + np.cos(4*x)

def InvTri(a, b, c):
    """
    Computes the inverse of a tridiagonal matrix A given by its diagonals.
    The matrix A is defined by:
        - a: main diagonal (length n)
        - b: sub-diagonal (length n-1)
        - c: super-diagonal (length n-1)
    
    The inverse is computed column-by-column by solving A*x = e_i.
    
    Parameters:
        a : 1D array (main diagonal)
        b : 1D array (sub-diagonal)
        c : 1D array (super-diagonal)
    
    Returns:
        A_inv : 2D array (n x n), the inverse of the matrix A
    """
    n = len(a)
    A_inv = np.zeros((n, n))
    # Solve A*x = e_i for each unit vector e_i
    for i in range(n):
        d = np.zeros(n)
        d[i] = 1  # e_i
        x = TriSolve(a, b, c, d)
        A_inv[:, i] = x
    return A_inv

def DerValue(A, B, C, n, Y):
    """
    Computes the vector of second derivatives (M) at the spline nodes.
    This function uses the inverse of a tridiagonal matrix along with a 
    correction procedure to obtain the coefficients.
    
    Parameters:
        A : 1D array of length n (main diagonal entries, constant 2's)
        B : 1D array of length n (B values, here constant 0.5's)
        C : 1D array of length n (C values, here constant 0.5's)
        n : integer, number of subintervals
        Y : 1D array of length n (right-hand side, computed from finite differences)
    
    Returns:
        Answer : 1D array, the computed second derivatives at the nodes (length n)
    """
    # Note: In MATLAB indexing, B(2:n) becomes B[1:] in Python,
    #       and C(1:n-1) becomes C[:-1].
    A0 = InvTri(A, B[1:], C[:-1])
    
    # Build temp1 as an n x n zero matrix, then set its last row, first column.
    temp1 = np.zeros((n, n))
    temp1[-1, 0] = B[0]  # B(1) in MATLAB corresponds to B[0]
    
    # Compute A1 using a correction term
    A1 = (np.eye(n) - (A0 @ temp1) / (1 + B[0] * A0[0, -1])) @ A0
    
    # Build temp2 as an n x n zero matrix, then set its first row, last column.
    temp2 = np.zeros((n, n))
    temp2[0, -1] = C[-1]  # C(n) in MATLAB is C[-1] in Python
    
    # Compute A2 using another correction term
    A2 = (np.eye(n) - (A1 @ temp2) / (1 + C[-1] * A1[-1, 0])) @ A1
    
    # The final answer multiplies Y (treated as a row vector) on the left by A2.
    Answer = Y @ A2
    return Answer

def compute_error(n):
    """
    Computes the maximum spline interpolation error for a given number of subintervals n.
    
    Parameters:
        n : integer, number of subintervals (the number of nodes is n+1)
    
    Returns:
        error_max : float, the maximum error (in the infinity norm) over all subintervals.
    """
    h = 2 * np.pi / n  # step size
    # Create n+1 equally spaced nodes over [0, 2*pi]
    x = np.linspace(0, 2*np.pi, n+1)
    Y_values = ObjectFunction(x)  # function values at the nodes

    # Compute finite differences for the first derivative (periodic)
    temp = (Y_values[1:] - Y_values[:-1]) / h
    temp = np.concatenate((temp, [temp[0]]))  # enforce periodicity
    
    # Right-hand side for the spline system based on differences of slopes
    b_rhs = 3 * (temp[1:] - temp[:-1]) / h  # length n

    # Construct the constant tridiagonal matrix coefficients
    A_diag = np.full(n, 2.0)    # main diagonal (all 2's)
    B_diag = np.full(n, 0.5)    # lower diagonal (all 0.5's)
    C_diag = np.full(n, 0.5)    # upper diagonal (all 0.5's)
    
    # Compute the second derivatives (M) at the interior nodes
    M = DerValue(A_diag, B_diag, C_diag, n, b_rhs)
    # For periodicity, prepend M[-1] to obtain M of length n+1
    M = np.concatenate(([M[-1]], M))
    
    # Compute the cubic spline coefficients for each interval
    coeff = np.zeros((n, 4))
    error_max = 0
    for i in range(n):
        xi = x[i]
        xip1 = x[i+1]
        Mi = M[i]
        Mi1 = M[i+1]
        # Contribution from second derivative at node i
        term1 = Mi * np.array([-1, 3*xip1, -3*xip1**2, xip1**3]) / (6*h)
        # Contribution from second derivative at node i+1
        term2 = Mi1 * np.array([1, -3*xi, 3*xi**2, -xi**3]) / (6*h)
        # Contribution from function values and first derivative approximations
        term3 = np.zeros(4)
        vec = (Y_values[i] / h - Mi * h / 6) * np.array([-1, xip1]) + (Y_values[i+1] / h - Mi1 * h / 6) * np.array([1, -xi])
        term3[2:] = vec
        coeff[i, :] = term1 + term2 + term3
        
        # Evaluate the spline on a fine grid within the current subinterval
        xx = np.linspace(xi, xip1, 1000)
        yy_spline = np.polyval(coeff[i, :], xx)
        yy_true = ObjectFunction(xx)
        error_interval = np.max(np.abs(yy_spline - yy_true))
        error_max = max(error_max, error_interval)
    return error_max

def main():
    # Define the list of subinterval counts to test
    n_values = [5, 10, 20, 30, 50, 100, 300, 500]
    errors = []
    
    print(f'{"n":>5s} {"e_h":>15s}')
    print("-"*22)
    for n in n_values:
        e_h = compute_error(n)
        errors.append(e_h)
        print(f'{n:5d} {e_h:15.6e}')
    
    # Convert lists to numpy arrays for regression and plotting
    n_arr = np.array(n_values)
    errors_arr = np.array(errors)
    
    # For convergence plot, we work with log scale:
    log_n = np.log(n_arr)
    log_error = -np.log(errors_arr)  # using negative log(error)
    
    # Fit a line to the data: log_error = m * log_n + b
    coeff_fit = np.polyfit(log_n, log_error, 1)
    m_fit, b_fit = coeff_fit
    fitted_line = m_fit * log_n + b_fit
    
    # Expected line has slope 4 (i.e., -log(error) = 4 log(n) + constant)
    # Choose the intercept so that the expected line passes through the average point.
    b_expected = np.mean(log_error - 4 * log_n)
    expected_line = 4 * log_n + b_expected
    
    # Compute difference metrics:
    slope_difference = abs(m_fit - 4)
    rmse_diff = np.sqrt(np.mean((fitted_line - expected_line)**2))
    
    # Plot the convergence data
    plt.figure(figsize=(8, 6))
    plt.plot(log_n, log_error, 'ko', markersize=8, label='Data')
    plt.plot(log_n, fitted_line, 'b-', linewidth=2, label=f'Fitted line (slope={m_fit:.3f})')
    plt.plot(log_n, expected_line, 'r--', linewidth=2, label='Expected line (slope=4)')
    plt.xlabel('log(n)')
    plt.ylabel('-log(e_h)')
    plt.title('Convergence Plot')
    plt.legend()
    plt.grid(True)
    plt.annotate(f"Slope difference: {slope_difference:.3e}\nRMSE difference: {rmse_diff:.3e}",
                 xy=(0.05, 0.75), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"))
    plt.show()

if __name__ == '__main__':
    main()

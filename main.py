import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# =========================
# Put your data here
# =========================

x_values = np.array([1, 2, 3, 4, 5], dtype=float)
y_values = np.array([2.1, 4.0, 6.2, 7.9, 10.1], dtype=float)

x_uncertainty = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)
y_uncertainty = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=float)


# =========================
# Function to fit
# =========================

def fit_function(x, a, b):
    """
    Linear function:
        f(x) = ax + b
    """
    return a * x + b


def numerical_derivative(function, x, params, step=1e-6):
    """
    Estimates df/dx numerically so x uncertainty can be included.
    """
    return (function(x + step, *params) - function(x - step, *params)) / (2 * step)


def chi_squared(params, x, y, x_err, y_err):
    """
    Calculates chi-squared using both x and y uncertainties.

    sigma_effective^2 = sigma_y^2 + (df/dx * sigma_x)^2
    """
    y_fit = fit_function(x, *params)
    dydx = numerical_derivative(fit_function, x, params)

    sigma_effective = np.sqrt(y_err**2 + (dydx * x_err)**2)

    return np.sum(((y - y_fit) / sigma_effective) ** 2)


def numerical_hessian(function, params, step=1e-4):
    """
    Numerically estimates the Hessian matrix of chi-squared.
    This is used to estimate parameter uncertainties.
    """
    params = np.array(params, dtype=float)
    n = len(params)
    hessian = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            params_pp = params.copy()
            params_pm = params.copy()
            params_mp = params.copy()
            params_mm = params.copy()

            params_pp[i] += step
            params_pp[j] += step

            params_pm[i] += step
            params_pm[j] -= step

            params_mp[i] -= step
            params_mp[j] += step

            params_mm[i] -= step
            params_mm[j] -= step

            hessian[i, j] = (
                function(params_pp)
                - function(params_pm)
                - function(params_mp)
                + function(params_mm)
            ) / (4 * step**2)

    return hessian


# =========================
# Fit data
# =========================

initial_guess = [1, 0]

result = minimize(
    chi_squared,
    initial_guess,
    args=(x_values, y_values, x_uncertainty, y_uncertainty)
)

best_fit_params = result.x
minimum_chi_squared = result.fun

degrees_of_freedom = len(x_values) - len(best_fit_params)
chi_squared_per_dof = minimum_chi_squared / degrees_of_freedom


# =========================ff
# Parameter uncertainties
# =========================

chi_squared_for_hessian = lambda params: chi_squared(
    params,
    x_values,
    y_values,
    x_uncertainty,
    y_uncertainty
)

hessian = numerical_hessian(chi_squared_for_hessian, best_fit_params)

covariance_matrix = 2 * np.linalg.inv(hessian)
parameter_uncertainties = np.sqrt(np.diag(covariance_matrix))


# =========================
# Print results
# =========================

print("Best-fit parameters:")
print(f"a = {best_fit_params[0]} ± {parameter_uncertainties[0]}")
print(f"b = {best_fit_params[1]} ± {parameter_uncertainties[1]}")

print()
print(f"Minimum chi^2 = {minimum_chi_squared}")
print(f"Degrees of freedom = {degrees_of_freedom}")
print(f"Minimum chi^2 / dof = {chi_squared_per_dof}")


# =========================
# Plot data and fit
# =========================

x_smooth = np.linspace(
    min(x_values) - 0.5,
    max(x_values) + 0.5,
    500
)

y_smooth = fit_function(x_smooth, *best_fit_params)

plt.errorbar(
    x_values,
    y_values,
    xerr=x_uncertainty,
    yerr=y_uncertainty,
    fmt="o",
    capsize=4,
    label=None
)

plt.plot(
    x_smooth,
    y_smooth,
    label=r"$f(x) = ax + b$"
)

results_text = (
    r"$f(x) = ax + b$" "\n"
    rf"$a = {best_fit_params[0]:.4g} \pm {parameter_uncertainties[0]:.2g}$" "\n"
    rf"$b = {best_fit_params[1]:.4g} \pm {parameter_uncertainties[1]:.2g}$" "\n"
    rf"$\chi^2/\mathrm{{dof}} = {chi_squared_per_dof:.4g}$"
)

plt.text(
    0.05,
    0.95,
    results_text,
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox={
        "boxstyle": "round",
        "facecolor": "white",
        "alpha": 0.85
    }
)

plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Data with Best-Fit Function")
plt.grid(True)

plt.show()
"""
Menu Cost Model (Sheshinski-Weiss type)

    V(p) = max{ V^n(p), V^a }
    V^n(p) = pi(p) + beta * V(p / pi_inf)
    V^a    = max_{p'} { pi(p') - kappa + beta * V(p' / pi_inf) }
    pi(p)  = (p^(1-epsilon) - A * p^(-epsilon)) * Y

Solved by value function iteration on a discretized price grid,
using cubic spline interpolation to evaluate V(p/pi_inf) off-grid.

This is a direct port of menu_cost_model.jl to Python (numpy + scipy +
matplotlib), preserving the same structure, variable names, and outputs.
"""

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# ---------------- Parameters ----------------
epsilon = 6.0     # elasticity of substitution (ss markup = epsilon/(epsilon-1))
A       = 0.85    # real marginal cost (mc), constant, A < 1
Y       = 1.0     # aggregate demand, constant
beta    = 0.99    # discount factor
kappa   = 0.02    # menu cost, real units
pi_inf  = 1.005   # gross steady-state inflation rate

# Static optimal relative price (flexible-price benchmark)
pstar = epsilon / (epsilon - 1) * A


def profit(p):
    """Period profit function pi(p) = (p^(1-eps) - A p^(-eps)) Y."""
    return (p ** (1 - epsilon) - A * p ** (-epsilon)) * Y


# ---------------- Grid ----------------
pmin, pmax = 0.70, 1.30
N = 4000
pgrid = np.linspace(pmin, pmax, N)
prof = profit(pgrid)


# ---------------- Value Function Iteration ----------------
def solve_menu_cost(tol=1e-10, maxit=10_000, verbose=True):
    """
    Solve the menu cost Bellman equation by value function iteration.

    Returns
    -------
    V, Vn, Va, p_reset : converged value function, inaction value,
                          action value, and optimal reset price.
    """
    V = prof / (1 - beta)          # initial guess: perpetual inaction
    p_next = pgrid / pi_inf

    Va = None
    p_reset = None

    for it in range(1, maxit + 1):
        # cubic spline interpolant of V, evaluated at p/pi_inf for every grid point
        spline = CubicSpline(pgrid, V, extrapolate=True)
        cont = spline(p_next)          # V(p/pi_inf) for every grid point p

        Vn = prof + beta * cont

        # V^a: choose p' on the grid maximizing profit(p') - kappa + beta V(p'/pi_inf)
        # Note: this objective has the same functional form as Vn evaluated at p'=pgrid,
        # so we can reuse Vn directly instead of a separate inner maximization loop.
        obj = prof + beta * cont - kappa
        idx_a = np.argmax(obj)
        Va = obj[idx_a]
        p_reset = pgrid[idx_a]

        Vnew = np.maximum(Vn, Va)
        diff = np.max(np.abs(Vnew - V))
        V = Vnew

        if diff < tol:
            if verbose:
                print(f"Converged at iteration {it}, diff = {diff:.3e}")
            break
    else:
        print(f"Warning: did not converge within {maxit} iterations, diff = {diff:.3e}")

    # Recompute Vn one more time at the converged V for plotting / inaction region
    spline = CubicSpline(pgrid, V, extrapolate=True)
    cont = spline(p_next)
    Vn = prof + beta * cont

    return V, Vn, Va, p_reset


V, Vn, Va, p_reset = solve_menu_cost()

# ---------------- Inaction region ----------------
inaction_idx = np.where(Vn >= Va - 1e-8)[0]
p_lo = pgrid[inaction_idx[0]]
p_hi = pgrid[inaction_idx[-1]]

print(f"Static optimal price p*         = {pstar:.4f}")
print(f"Reset price after adjustment p' = {p_reset:.4f}")
print(f"Inaction region                 = [{p_lo:.4f}, {p_hi:.4f}]")
print(f"Value of adjustment V^a         = {Va:.4f}")

########################################################################
# Plot: replica of the lecture-note figure
#   - Vn(p): hump-shaped curve (solid)
#   - Va   : horizontal dashed line
#   - inaction region shaded/marked
########################################################################

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(pgrid, Vn, label=r"$V^n(p)$", linewidth=2.5, color="steelblue")
ax.axhline(Va, label=r"$V^a$", linestyle="--", linewidth=2, color="darkorange")

# Shade the inaction region
ax.axvspan(p_lo, p_hi, alpha=0.12, color="green", label="inaction region")

# Mark the boundaries and the reset price
ax.axvline(p_lo, color="green", linestyle=":", linewidth=1.5)
ax.axvline(p_hi, color="green", linestyle=":", linewidth=1.5)
ax.scatter([p_reset], [Va], color="red", s=50, zorder=5, label="reset price p'")
ax.axvline(pstar, color="gray", linestyle="-.", linewidth=1.2, label="static optimum p*")

ax.set_xlim(pmin, pmax)
ax.set_xlabel("p")
ax.set_ylabel("Value")
ax.set_title("Menu Cost Model: Value Functions")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("menu_cost_value_functions.png", dpi=150)
print("Figure saved to menu_cost_value_functions.png")

########################################################################
# Optional: simulate one firm's relative price path to illustrate
# lumpy adjustment under trend inflation
########################################################################


def simulate_price_path(T, p0=None):
    """Simulate a single firm's relative price path under trend inflation,
    resetting to p_reset whenever the price drifts below p_lo."""
    if p0 is None:
        p0 = p_reset
    path = np.zeros(T)
    p = p0
    for t in range(T):
        if p < p_lo:
            p = p_reset
        path[t] = p
        p = p / pi_inf   # erosion of relative price under trend inflation
    return path


T = 40
path = simulate_price_path(T)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.step(range(1, T + 1), path, where="post", linewidth=2, color="purple")
ax2.axhline(p_lo, color="green", linestyle=":", linewidth=1)
ax2.axhline(p_hi, color="green", linestyle=":", linewidth=1)
ax2.set_xlabel("t")
ax2.set_ylabel("relative price p_t(f)")
ax2.set_title("Simulated Lumpy Price Adjustment")
ax2.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig("menu_cost_price_path.png", dpi=150)
print("Figure saved to menu_cost_price_path.png")

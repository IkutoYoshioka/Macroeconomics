########################################################################
# Menu Cost Model (Sheshinski-Weiss type)
#
#   V(p) = max{ V^n(p), V^a }
#   V^n(p) = π(p) + β V(p/π_inf)
#   V^a    = max_{p'} { π(p') − κ + β V(p'/π_inf) }
#   π(p)   = (p^(1-ε) − A p^(-ε)) Y
#
# Solved by value function iteration on a discretized price grid,
# using cubic spline interpolation to evaluate V(p/π_inf) off-grid.
########################################################################

using Interpolations
using Plots

# ---------------- Parameters ----------------
const ϵ      = 6.0     # elasticity of substitution (ss markup = ϵ/(ϵ-1))
const A      = 0.85    # real marginal cost (mc), constant, A < 1
const Y      = 1.0     # aggregate demand, constant
const β      = 0.99    # discount factor
const κ      = 0.02    # menu cost, real units
const πinf   = 1.005   # gross steady-state inflation rate

# Static optimal relative price (flexible-price benchmark)
const pstar = ϵ / (ϵ - 1) * A

# Period profit function
profit(p) = (p^(1 - ϵ) - A * p^(-ϵ)) * Y

# ---------------- Grid ----------------
const pmin, pmax = 0.70, 1.30
const N = 4000
const prange = range(pmin, pmax, length = N)   # keep as AbstractRange for interpolation
const pgrid = collect(prange)
const prof  = profit.(pgrid)

# ---------------- Value Function Iteration ----------------
function solve_menu_cost(; tol = 1e-10, maxit = 10_000, verbose = true)
    V = prof ./ (1 - β)              # initial guess: perpetual inaction
    Vn = similar(V)
    p_next = pgrid ./ πinf

    local Va, p_reset
    for it in 1:maxit
        itp = cubic_spline_interpolation(prange, V; extrapolation_bc = Line())
        cont = itp.(p_next)          # V(p/πinf) for every grid point p

        Vn .= prof .+ β .* cont

        # V^a: choose p' on the grid maximizing profit(p') - κ + β V(p'/πinf)
        obj = prof .+ β .* cont .- κ  # same functional form evaluated at p' = pgrid
        idx_a = argmax(obj)
        Va = obj[idx_a]
        p_reset = pgrid[idx_a]

        Vnew = max.(Vn, Va)
        diff = maximum(abs.(Vnew .- V))
        V .= Vnew

        if diff < tol
            verbose && println("Converged at iteration $it, diff = $(diff)")
            break
        end
        if it == maxit
            @warn "Did not converge within maxit iterations, diff = $diff"
        end
    end

    # Recompute V^n one more time at the converged V for plotting/inaction region
    itp = cubic_spline_interpolation(prange, V; extrapolation_bc = Line())
    cont = itp.(p_next)
    Vn .= prof .+ β .* cont

    return V, Vn, Va, p_reset
end

V, Vn, Va, p_reset = solve_menu_cost()

# ---------------- Inaction region ----------------
inaction_idx = findall(Vn .>= Va - 1e-8)
p_lo = pgrid[first(inaction_idx)]
p_hi = pgrid[last(inaction_idx)]

println("Static optimal price p*        = ", round(pstar, digits = 4))
println("Reset price after adjustment p'= ", round(p_reset, digits = 4))
println("Inaction region                = [", round(p_lo, digits = 4), ", ", round(p_hi, digits = 4), "]")
println("Value of adjustment V^a        = ", round(Va, digits = 4))

########################################################################
# Plot: replica of the lecture-note figure
#   - V^n(p): hump-shaped curve (solid)
#   - V^a   : horizontal dashed line
#   - inaction region shaded/marked
########################################################################

plot(pgrid, Vn,
     label = "Vⁿ(p)",
     linewidth = 2.5,
     color = :steelblue,
     xlabel = "p",
     ylabel = "Value",
     title  = "Menu Cost Model: Value Functions",
     legend = :bottomright,
     xlims  = (pmin, pmax),
     grid   = true)

hline!([Va], label = "Vᵃ", linestyle = :dash, linewidth = 2, color = :darkorange)

# Shade the inaction region
vspan!([p_lo, p_hi], alpha = 0.12, color = :green, label = "inaction region")

# Mark the boundaries and the reset price
vline!([p_lo, p_hi], color = :green, linestyle = :dot, linewidth = 1.5, label = false)
scatter!([p_reset], [Va], color = :red, markersize = 5, label = "reset price p'")
vline!([pstar], color = :gray, linestyle = :dashdot, linewidth = 1.2, label = "static optimum p*")

savefig("menu_cost_value_functions.png")
println("Figure saved to menu_cost_value_functions.png")

########################################################################
# Optional: simulate one firm's relative price path to illustrate
# lumpy adjustment under trend inflation
########################################################################

function simulate_price_path(T; p0 = p_reset)
    path = zeros(T)
    p = p0
    for t in 1:T
        # if p has drifted below the inaction region lower bound, reset
        if p < p_lo
            p = p_reset
        end
        path[t] = p
        p = p / πinf   # erosion of relative price under trend inflation
    end
    return path
end

T = 40
path = simulate_price_path(T)

plot(1:T, path,
     seriestype = :steppost,
     linewidth = 2,
     color = :purple,
     xlabel = "t",
     ylabel = "relative price p_t(f)",
     title  = "Simulated Lumpy Price Adjustment",
     legend = false,
     grid   = true)
hline!([p_lo, p_hi], color = :green, linestyle = :dot, linewidth = 1)

savefig("menu_cost_price_path.png")
println("Figure saved to menu_cost_price_path.png")
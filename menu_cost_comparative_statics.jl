########################################################################
# Menu Cost Model (Sheshinski-Weiss type): Comparative Statics
#
#   V(p) = max{ V^n(p), V^a }
#   V^n(p) = π(p) + β V(p/π_inf)
#   V^a    = max_{p'} { π(p') − κ + β V(p'/π_inf) }
#   π(p)   = (p^(1-ε) − A p^(-ε)) Y
#
# All structural parameters (ε, A, Y, β, κ, πinf) are passed as
# arguments to solve_menu_cost, so the model can be re-solved cheaply
# for many parameter values without restarting Julia.
########################################################################

using Interpolations
using Plots

# ---------------- Grid (shared across all parameterizations) ----------------
const pmin, pmax = 0.60, 1.40
const N = 4000
const prange = range(pmin, pmax, length = N)
const pgrid  = collect(prange)

# ---------------- Model solved as a function of parameters ----------------
"""
    solve_menu_cost(; ϵ, A, Y, β, κ, πinf, tol, maxit, verbose)

Solve the menu cost Bellman equation by value function iteration and
return a NamedTuple with V, Vn, Va, p_reset, p_lo, p_hi, pstar, plus
the parameters used (for labeling plots).
"""
function solve_menu_cost(; ϵ = 6.0, A = 0.85, Y = 1.0, β = 0.99,
                            κ = 0.02, πinf = 1.005,
                            tol = 1e-10, maxit = 10_000, verbose = false)

    profit(p) = (p^(1 - ϵ) - A * p^(-ϵ)) * Y
    prof  = profit.(pgrid)
    pstar = ϵ / (ϵ - 1) * A
    p_next = pgrid ./ πinf

    V  = prof ./ (1 - β)     # initial guess
    Vn = similar(V)
    local Va, p_reset

    for it in 1:maxit
        itp  = cubic_spline_interpolation(prange, V; extrapolation_bc = Line())
        cont = itp.(p_next)

        Vn .= prof .+ β .* cont

        obj = prof .+ β .* cont .- κ
        idx_a = argmax(obj)
        Va = obj[idx_a]
        p_reset = pgrid[idx_a]

        Vnew = max.(Vn, Va)
        diff = maximum(abs.(Vnew .- V))
        V .= Vnew

        if diff < tol
            verbose && println("Converged at iteration $it, diff = $diff")
            break
        end
        it == maxit && @warn "Did not converge, diff = $diff"
    end

    # recompute Vn at converged V for inaction-region bookkeeping
    itp  = cubic_spline_interpolation(prange, V; extrapolation_bc = Line())
    cont = itp.(p_next)
    Vn .= prof .+ β .* cont

    inaction_idx = findall(Vn .>= Va - 1e-8)
    p_lo = pgrid[first(inaction_idx)]
    p_hi = pgrid[last(inaction_idx)]

    return (V = V, Vn = Vn, Va = Va, p_reset = p_reset,
            p_lo = p_lo, p_hi = p_hi, pstar = pstar,
            ϵ = ϵ, A = A, β = β, κ = κ, πinf = πinf)
end

########################################################################
# Helper: print a one-line summary of a solved case
########################################################################
function summarize(res; label = "")
    println(rpad(label, 22),
            "  p* = ", round(res.pstar, digits = 4),
            "  p' = ", round(res.p_reset, digits = 4),
            "  inaction = [", round(res.p_lo, digits = 4), ", ",
                                round(res.p_hi, digits = 4), "]",
            "  width = ", round(res.p_hi - res.p_lo, digits = 4))
end

########################################################################
# Helper: overlay Vn(p) curves for several solved cases on one plot
########################################################################
function plot_comparison(results, labels; title = "Comparative Statics",
                          xlims_ = (pmin, pmax))
    plt = plot(xlabel = "p", ylabel = "Value", title = title,
               legend = :bottomright, xlims = xlims_, grid = true)
    palette_colors = palette(:tab10)
    for (i, (res, lab)) in enumerate(zip(results, labels))
        plot!(plt, pgrid, res.Vn, label = lab, linewidth = 2.2,
              color = palette_colors[mod1(i, 10)])
        vline!(plt, [res.p_lo, res.p_hi],
               color = palette_colors[mod1(i, 10)], linestyle = :dot,
               linewidth = 1, label = false, alpha = 0.6)
    end
    return plt
end

########################################################################
# 1) Baseline case
########################################################################
baseline = solve_menu_cost(verbose = true)
summarize(baseline; label = "baseline")

########################################################################
# 2) Comparative statics over the menu cost κ
#    Prediction: larger κ -> wider inaction region (more price stickiness)
########################################################################
κ_values = [0.005, 0.02, 0.05, 0.10]
κ_results = [solve_menu_cost(κ = k) for k in κ_values]
println("\n--- Varying κ (menu cost) ---")
for (res, k) in zip(κ_results, κ_values)
    summarize(res; label = "κ = $k")
end

plt_kappa = plot_comparison(κ_results, ["κ = $k" for k in κ_values];
                             title = "Effect of Menu Cost κ on Vⁿ(p)",
                             xlims_ = (0.8, 1.2))
savefig(plt_kappa, "compstat_kappa.png")

########################################################################
# 3) Comparative statics over trend inflation π_inf
#    Result (verified numerically): HIGHER inflation WIDENS the inaction
#    region and shifts the reset price p' upward. Intuition: under
#    positive trend inflation the firm anticipates future erosion of
#    its relative price, so it optimally resets to a HIGHER price than
#    the static optimum, which in turn widens the tolerable upper band.
#    (This is the opposite of the naive guess that faster erosion alone
#    would narrow the inaction region -- the reset-price response
#    dominates.)
########################################################################
πinf_values = [1.000, 1.005, 1.02, 1.05]
πinf_results = [solve_menu_cost(πinf = p) for p in πinf_values]
println("\n--- Varying π_inf (steady-state gross inflation) ---")
for (res, p) in zip(πinf_results, πinf_values)
    summarize(res; label = "πinf = $p")
end

plt_pi = plot_comparison(πinf_results, ["πinf = $p" for p in πinf_values];
                          title = "Effect of Trend Inflation on Vⁿ(p)",
                          xlims_ = (0.8, 1.2))
savefig(plt_pi, "compstat_piinf.png")

########################################################################
# 4) Comparative statics over the discount factor β
########################################################################
β_values = [0.90, 0.95, 0.99, 0.999]
β_results = [solve_menu_cost(β = b) for b in β_values]
println("\n--- Varying β (discount factor) ---")
for (res, b) in zip(β_results, β_values)
    summarize(res; label = "β = $b")
end

plt_beta = plot_comparison(β_results, ["β = $b" for b in β_values];
                            title = "Effect of Discount Factor β on Vⁿ(p)",
                            xlims_ = (0.8, 1.2))
savefig(plt_beta, "compstat_beta.png")

########################################################################
# 5) Comparative statics over the elasticity of substitution ε
#    Prediction: higher ε -> lower markup -> pstar shifts down
########################################################################
ϵ_values = [3.0, 6.0, 10.0, 20.0]
ϵ_results = [solve_menu_cost(ϵ = e) for e in ϵ_values]
println("\n--- Varying ε (elasticity of substitution) ---")
for (res, e) in zip(ϵ_results, ϵ_values)
    summarize(res; label = "ε = $e")
end

plt_eps = plot_comparison(ϵ_results, ["ε = $e" for e in ϵ_values];
                           title = "Effect of Elasticity ε on Vⁿ(p)",
                           xlims_ = (0.6, 1.4))
savefig(plt_eps, "compstat_epsilon.png")

########################################################################
# 6) Summary bar chart: inaction-region width across κ and π_inf grids
#    (a small 2D comparative-statics heatmap-style summary)
########################################################################
κ_grid    = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
πinf_grid = [1.000, 1.005, 1.01, 1.02, 1.05]

width_matrix = zeros(length(κ_grid), length(πinf_grid))
for (i, k) in enumerate(κ_grid), (j, p) in enumerate(πinf_grid)
    res = solve_menu_cost(κ = k, πinf = p)
    width_matrix[i, j] = res.p_hi - res.p_lo
end

heatmap(πinf_grid, κ_grid, width_matrix,
        xlabel = "π_inf", ylabel = "κ",
        title  = "Inaction-Region Width across (κ, π_inf)",
        color  = :viridis)

savefig("compstat_heatmap_width.png")
println("\nFigure saved to compstat_heatmap_width.png")

println("\nAll comparative statics figures saved:")
println("  compstat_kappa.png")
println("  compstat_piinf.png")
println("  compstat_beta.png")
println("  compstat_epsilon.png")
println("  compstat_heatmap_width.png")

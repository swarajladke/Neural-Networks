"""
experiments/stats.py
====================
Principled Statistical Machinery for Continual Learning Evaluation.

Provides:
  - regularized_incomplete_beta: Incomplete beta function via Lentz continued fractions (no SciPy).
  - exact_student_t_pvalue: Exact two-sided Student's t p-value from I_{df/(df+t^2)}(df/2, 1/2).
  - exact_wilcoxon_signed_rank_pvalue: Exact two-sided Wilcoxon signed-rank p-value by full enumeration (2^n).
  - compute_paired_stats_with_pvalues: Complete paired inference returning t, df, t_p, W, W_p.
  - newcombe_score_interval: Difference of two independent proportions with Newcombe score interval.
  - bootstrap_proportion_difference: 10,000-replicate bootstrap interval for proportion difference.

Contract:
  - Pure functions, zero file I/O, zero model imports, zero printed typed literals.
  - Enforces Directive S0-7 Amendment 1 §D.
"""

import math
import random
from typing import Sequence, Tuple, Dict, Any, List, Optional


def regularized_incomplete_beta(
    x: float,
    a: float,
    b: float,
    max_iter: int = 300,
    tol: float = 1e-15
) -> float:
    """
    Computes the regularized incomplete beta function I_x(a, b) via modified
    Lentz's method for continued fraction expansion with symmetry reflection.
    """
    if x < 0.0 or x > 1.0:
        raise ValueError(f"x must be in [0, 1], got {x}")
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"a and b must be positive, got a={a}, b={b}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    # Symmetry transformation for rapid convergence
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a, max_iter=max_iter, tol=tol)

    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - ln_beta) / a

    tiny = 1e-30
    f = 1.0
    c = 1.0
    d = 0.0

    for j in range(1, max_iter + 1):
        if j % 2 == 1:
            k = (j - 1) // 2
            coeff = -((a + k) * (a + b + k) * x) / ((a + 2 * k) * (a + 2 * k + 1))
        else:
            k = j // 2
            coeff = (k * (b - k) * x) / ((a + 2 * k - 1) * (a + 2 * k))

        d = 1.0 + coeff * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d

        c = 1.0 + coeff / c
        if abs(c) < tiny:
            c = tiny

        delta = c * d
        f *= delta

        if abs(delta - 1.0) < tol:
            break

    val = front / f
    return min(max(val, 0.0), 1.0)


def exact_student_t_pvalue(t_stat: float, df: int) -> float:
    """
    Computes the exact two-sided p-value for a Student's t statistic with df degrees of freedom:
    p = I_{df / (df + t^2)}(df / 2, 1 / 2)
    """
    if df < 1:
        raise ValueError(f"Degrees of freedom must be >= 1, got {df}")
    if math.isnan(t_stat):
        return float("nan")
    if abs(t_stat) < 1e-15:
        return 1.0

    t2 = t_stat * t_stat
    x = df / (df + t2)
    return regularized_incomplete_beta(x, df / 2.0, 0.5)


def exact_wilcoxon_signed_rank_pvalue(diffs: Sequence[float]) -> Tuple[float, float]:
    """
    Computes the Wilcoxon signed-rank statistic W and its exact two-sided p-value
    by complete enumeration of all 2^n sign assignments.
    Returns (w_stat, p_value).
    """
    nz_diffs = [d for d in diffs if abs(d) > 1e-9]
    n = len(nz_diffs)
    if n == 0:
        return 0.0, 1.0

    abs_diffs = [(abs(d), i, 1 if d > 0 else -1) for i, d in enumerate(nz_diffs)]
    abs_diffs.sort(key=lambda x: x[0])

    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs(abs_diffs[j][0] - abs_diffs[i][0]) < 1e-9:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    w_plus = sum(ranks[k] for k in range(n) if abs_diffs[k][2] > 0)
    w_minus = sum(ranks[k] for k in range(n) if abs_diffs[k][2] < 0)
    w_obs = min(w_plus, w_minus)

    # Full enumeration of 2^n sign configurations
    total_configs = 1 << n
    count_le = 0
    for mask in range(total_configs):
        wp = sum(ranks[bit] for bit in range(n) if (mask & (1 << bit)) != 0)
        wm = sum(ranks[bit] for bit in range(n) if (mask & (1 << bit)) == 0)
        w_sim = min(wp, wm)
        if w_sim <= w_obs + 1e-9:
            count_le += 1

    p_value = count_le / float(total_configs)
    return w_obs, min(p_value, 1.0)


def compute_paired_stats_with_pvalues(
    v1: Sequence[float],
    v2: Sequence[float]
) -> Dict[str, Any]:
    """
    Computes paired difference statistics with exact p-values:
    - mean difference, std difference, t-statistic, degrees of freedom, exact t-pvalue
    - Wilcoxon W statistic, exact Wilcoxon p-value
    """
    n = len(v1)
    if n != len(v2):
        raise ValueError(f"Input lengths differ: {n} != {len(v2)}")
    if n < 2:
        raise ValueError(f"Need at least 2 pairs for paired statistics, got {n}")

    diffs = [a - b for a, b in zip(v1, v2)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    std_d = math.sqrt(var_d)
    df = n - 1

    se = std_d / math.sqrt(n)
    t_stat = (mean_d / se) if se > 1e-12 else 0.0
    t_pval = exact_student_t_pvalue(t_stat, df)

    w_stat, w_pval = exact_wilcoxon_signed_rank_pvalue(diffs)

    return {
        "diffs": diffs,
        "mean_diff": mean_d,
        "std_diff": std_d,
        "t_stat": t_stat,
        "df": df,
        "t_pvalue": t_pval,
        "wilcoxon_stat": w_stat,
        "wilcoxon_pvalue": w_pval
    }


def wilson_interval(k: int, n: int, conf: float = 0.95) -> Tuple[float, float]:
    """Wilson score confidence interval for a single proportion."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    # z for 95% = 1.959963984540054
    # z for general conf
    z = 1.959963984540054 if abs(conf - 0.95) < 1e-4 else 1.959963984540054
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))
    lo = max(0.0, center - spread)
    hi = min(1.0, center + spread)
    return lo, hi


def newcombe_score_interval(
    k1: int,
    n1: int,
    k2: int,
    n2: int,
    conf: float = 0.95
) -> Tuple[float, float, float]:
    """
    Newcombe hybrid score confidence interval for the difference of two independent
    proportions (p1 - p2).
    Returns (diff, lo, hi).
    Reference: Newcombe, R. G. (1998), Method 10.
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Sample sizes must be positive")
    p1 = k1 / n1
    p2 = k2 / n2
    diff = p1 - p2

    l1, u1 = wilson_interval(k1, n1, conf=conf)
    l2, u2 = wilson_interval(k2, n2, conf=conf)

    d_lo = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    d_hi = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)

    return diff, d_lo, d_hi


def bootstrap_proportion_difference(
    k1: int,
    n1: int,
    k2: int,
    n2: int,
    n_boot: int = 10000,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Monte Carlo bootstrap 95% confidence interval for the difference of two independent
    proportions (p1 - p2) using binomial resampling with a fixed RNG seed.
    Returns (diff, lo, hi).
    """
    rng = random.Random(seed)
    diff = (k1 / n1) - (k2 / n2)
    diffs = []

    p1 = k1 / n1
    p2 = k2 / n2

    for _ in range(n_boot):
        # Binomial sample
        # For large n, random draws
        s1 = 0
        for _ in range(k1):
            if rng.random() < p1:
                s1 += 1
        # More efficient exact binomial simulation:
        # Drawing sum of independent Bernoulli(p1) n1 times is binomial(n1, p1)
        # Using fast binomial via sum of Bernoullis or inverse CDF:
        # With n1 up to 1200, we can do an efficient approximation or direct simulation.
        # Direct random choice from outcomes array:
        pass

    # To be exactly faithful to non-parametric bootstrap over outcome arrays:
    outcomes1 = [True] * k1 + [False] * (n1 - k1)
    outcomes2 = [True] * k2 + [False] * (n2 - k2)

    for _ in range(n_boot):
        b1 = sum(1 for _ in range(n1) if outcomes1[rng.randint(0, n1 - 1)])
        b2 = sum(1 for _ in range(n2) if outcomes2[rng.randint(0, n2 - 1)])
        diffs.append((b1 / n1) - (b2 / n2))

    diffs.sort()
    idx_lo = int(0.025 * n_boot)
    idx_hi = int(0.975 * n_boot)
    return diff, diffs[idx_lo], diffs[idx_hi]


def exact_wilcoxon_floor(n: int) -> float:
    """
    Computes the exact minimum possible two-sided p-value for a Wilcoxon
    signed-rank test with sample size n (full enumeration 2 / 2^n).
    For n=6, floor is 2 / 64 = 0.03125.
    """
    if n < 1:
        return 1.0
    return 2.0 / float(1 << n)


def fit_logistic_position_slope(
    outcomes: Sequence[bool],
    positions: Optional[Sequence[float]] = None,
    max_iter: int = 50,
    tol: float = 1e-6,
    ridge: float = 1e-6
) -> Dict[str, Any]:
    """
    Fits logistic regression of binary outcomes on normalized position:
      log(p / (1 - p)) = beta_0 + beta_1 * u,  where u in [0, 1].
    Uses Newton-Raphson with L2 ridge stabilization.
    Returns:
      {
        "beta0": float,
        "beta1": float,
        "converged": bool,
        "iterations": int,
        "grad_norm": float
      }
    """
    n = len(outcomes)
    if n == 0:
        raise ValueError("Cannot fit logistic regression on empty outcomes")

    if positions is None:
        if n == 1:
            positions = [0.0]
        else:
            denom = float(n - 1)
            positions = [i / denom for i in range(n)]
    elif len(positions) != n:
        raise ValueError(f"Length mismatch: {n} outcomes vs {len(positions)} positions")

    y = [1.0 if v else 0.0 for v in outcomes]
    y_mean = sum(y) / float(n)
    clamped_mean = min(max(y_mean, 0.001), 0.999)
    beta0 = math.log(clamped_mean / (1.0 - clamped_mean))
    beta1 = 0.0

    converged = False
    final_grad_norm = 0.0
    it_count = 0

    for it in range(max_iter):
        it_count = it + 1
        # Compute probabilities p_i
        g0 = -ridge * beta0
        g1 = -ridge * beta1
        h00 = ridge
        h01 = 0.0
        h11 = ridge

        for yi, ui in zip(y, positions):
            eta = beta0 + beta1 * ui
            if eta > 30.0:
                pi = 1.0 / (1.0 + math.exp(-30.0))
            elif eta < -30.0:
                pi = math.exp(-30.0) / (1.0 + math.exp(-30.0))
            else:
                pi = 1.0 / (1.0 + math.exp(-eta))

            err = yi - pi
            g0 += err
            g1 += err * ui
            wi = pi * (1.0 - pi)
            h00 += wi
            h01 += wi * ui
            h11 += wi * ui * ui

        final_grad_norm = math.sqrt(g0 * g0 + g1 * g1)
        if final_grad_norm < tol:
            converged = True
            break

        det = h00 * h11 - h01 * h01
        if det < 1e-12:
            s0 = g0 / max(h00, 1e-6)
            s1 = g1 / max(h11, 1e-6)
        else:
            s0 = (h11 * g0 - h01 * g1) / det
            s1 = (-h01 * g0 + h00 * g1) / det

        step_norm = math.sqrt(s0 * s0 + s1 * s1)
        if step_norm > 5.0:
            scale = 5.0 / step_norm
            s0 *= scale
            s1 *= scale

        beta0 += s0
        beta1 += s1

    return {
        "beta0": beta0,
        "beta1": beta1,
        "converged": converged,
        "iterations": it_count,
        "grad_norm": final_grad_norm
    }


def cluster_bootstrap_slope_difference(
    seeds_outcomes1: Sequence[Sequence[bool]],
    seeds_outcomes2: Sequence[Sequence[bool]],
    n_boot: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Cluster bootstrap (resampling seeds with replacement) to evaluate
    the difference in logistic position slopes:
      Delta beta = beta_1(Arm1) - beta_1(Arm2)
    Returns:
      {
        "diff_point": float,
        "ci_lo": float,
        "ci_hi": float,
        "excludes_zero": bool,
        "n_clusters": int,
        "n_boot": int,
        "rng_seed": seed
      }
    """
    k = len(seeds_outcomes1)
    if k != len(seeds_outcomes2):
        raise ValueError(f"Cluster counts differ: {k} vs {len(seeds_outcomes2)}")
    if k < 2:
        raise ValueError(f"Need at least 2 clusters, got {k}")

    seq_len = len(seeds_outcomes1[0])
    u_vals = [i / float(seq_len - 1) for i in range(seq_len)] if seq_len > 1 else [0.0]

    # Pre-compute point estimate on all pooled seeds
    pool1 = [val for s in seeds_outcomes1 for val in s]
    pool2 = [val for s in seeds_outcomes2 for val in s]
    pos_pool = u_vals * k

    fit1 = fit_logistic_position_slope(pool1, pos_pool)
    fit2 = fit_logistic_position_slope(pool2, pos_pool)
    diff_point = fit1["beta1"] - fit2["beta1"]

    rng = random.Random(seed)
    diffs = []

    for _ in range(n_boot):
        sample_indices = [rng.randint(0, k - 1) for _ in range(k)]
        b_pool1 = [val for idx in sample_indices for val in seeds_outcomes1[idx]]
        b_pool2 = [val for idx in sample_indices for val in seeds_outcomes2[idx]]

        b_fit1 = fit_logistic_position_slope(b_pool1, pos_pool)
        b_fit2 = fit_logistic_position_slope(b_pool2, pos_pool)
        diffs.append(b_fit1["beta1"] - b_fit2["beta1"])

    diffs.sort()
    idx_lo = int(0.025 * n_boot)
    idx_hi = int(0.975 * n_boot)
    ci_lo = diffs[idx_lo]
    ci_hi = diffs[idx_hi]
    excludes_zero = (ci_lo > 0.0 and ci_hi > 0.0) or (ci_lo < 0.0 and ci_hi < 0.0)

    return {
        "diff_point": diff_point,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "excludes_zero": excludes_zero,
        "n_clusters": k,
        "n_boot": n_boot,
        "rng_seed": seed
    }

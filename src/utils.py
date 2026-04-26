import numpy as np


def compute_metrics(log: dict, V_corr=None, soc_corr=None):
    """
    Compute EKF and optional PINN performance metrics.

    Parameters
    ----------
    log      : dict   output of machine2_ekf.run_cosim()
    V_corr   : np.ndarray | None   PINN-corrected voltage [V]
    soc_corr : np.ndarray | None   PINN-corrected SOC [0-1]

    Returns
    -------
    dict of scalar metrics
    """
    noise_std = float(np.std(log["V_meas"] - log["V_true"]) + 1e-8)

    v_rmse_ekf = float(np.sqrt(np.mean((log["V_true"] - log["V_est"]) ** 2)) * 1000)
    s_rmse_ekf = float(np.sqrt(np.mean((log["soc_true"] - log["soc_est"]) ** 2)) * 100)
    s_max_ekf  = float(np.max(np.abs(log["soc_true"] - log["soc_est"])) * 100)

    inn_rms        = float(np.sqrt(np.mean(log["innov"] ** 2)) * 1000)
    inn_mean       = float(np.mean(log["innov"]) * 1000)
    inn_noise_ratio = inn_rms / (noise_std * 1000 + 1e-8)

    tr_P_init  = float(log["P_tr"][0])
    tr_P_final = float(log["P_tr"][-1])
    convergence = tr_P_final / (tr_P_init + 1e-12) * 100.0

    nis_mean = float(np.mean(log["NIS"]))
    nis_std  = float(np.std(log["NIS"]))

    m = dict(
        v_rmse_ekf      = v_rmse_ekf,
        s_rmse_ekf      = s_rmse_ekf,
        s_max_ekf       = s_max_ekf,
        inn_rms         = inn_rms,
        inn_mean        = inn_mean,
        inn_noise_ratio = inn_noise_ratio,
        tr_P_init       = tr_P_init,
        tr_P_final      = tr_P_final,
        convergence_pct = convergence,
        nis_mean        = nis_mean,
        nis_std         = nis_std,
        noise_std_mv    = noise_std * 1000,
    )

    if V_corr is not None and soc_corr is not None:
        v_rmse_pinn = float(
            np.sqrt(np.mean((log["V_true"] - V_corr) ** 2)) * 1000)
        s_rmse_pinn = float(
            np.sqrt(np.mean((log["soc_true"] - soc_corr) ** 2)) * 100)
        improv_v = (v_rmse_ekf - v_rmse_pinn) / (v_rmse_ekf + 1e-8) * 100
        improv_s = (s_rmse_ekf - s_rmse_pinn) / (s_rmse_ekf + 1e-8) * 100
        inn_pinn = float(
            np.sqrt(np.mean((log["V_true"] - V_corr) ** 2))) / noise_std
        m.update(dict(
            v_rmse_pinn     = v_rmse_pinn,
            s_rmse_pinn     = s_rmse_pinn,
            improv_v        = improv_v,
            improv_s        = improv_s,
            inn_noise_pinn  = inn_pinn,
        ))

    return m


def cycle_table(log: dict):
    """
    Build cycle-by-cycle uncertainty analytics table.

    Returns
    -------
    list[dict]  one entry per detected cycle
    """
    t      = log["t"]
    soc    = log["soc_true"]
    P_tr   = log["P_tr"]
    P_soc  = log["P_soc"]

    dt     = float(np.median(np.diff(t)))
    rows   = []
    cycle  = 1
    i      = 0
    N      = len(t)

    baseline_peak = None

    while i < N:
        # detect start of discharge (SOC near top)
        if soc[i] < 0.95:
            i += 1
            continue

        seg_start = i
        # advance until SOC drops below 0.10 or end
        while i < N and soc[i] > 0.05:
            i += 1
        if i >= N:
            break

        seg = slice(seg_start, i)
        if (i - seg_start) < 5:
            i += 1
            continue

        soc_start = float(soc[seg_start])
        soc_min   = float(np.min(soc[seg]))
        soc_end   = float(soc[i - 1])
        dur_min   = (i - seg_start) * dt / 60.0
        peak_P    = float(np.max(P_tr[seg]))
        peak_soc  = float(np.max(np.sqrt(P_soc[seg]))) * 100.0

        if baseline_peak is None:
            baseline_peak = peak_P
            delta_pct     = 0.0
            status        = "Baseline"
        else:
            delta_pct = (peak_P - baseline_peak) / (baseline_peak + 1e-12) * 100.0
            if   abs(delta_pct) <= 10.0: status = "Stable"
            elif abs(delta_pct) <= 25.0: status = "Warning"
            else:                        status = "Diverging"

        rows.append(dict(
            cycle       = cycle,
            soc_start   = round(soc_start * 100, 1),
            soc_min     = round(soc_min   * 100, 1),
            soc_end     = round(soc_end   * 100, 1),
            dur_min     = round(dur_min,          1),
            peak_P      = peak_P,
            delta_pct   = round(delta_pct,        1),
            peak_soc_pct= round(peak_soc,         4),
            status      = status,
        ))
        cycle += 1

    return rows


def ekf_assessment(metrics: dict):
    """
    Generate engineering assessment flags from metrics.

    Returns
    -------
    list[dict]  each entry: {level, message}
    level: "error" | "warning" | "ok"
    """
    flags = []
    inr = metrics.get("inn_noise_ratio", 999)
    con = metrics.get("convergence_pct", 100)
    nis = metrics.get("nis_mean", 999)

    if inr > 3.0:
        flags.append(dict(
            level   = "error",
            message = (f"EKF DIVERGING — Innovation/Noise = {inr:.2f}× "
                       f"(limit 3×). Reduce sensor noise or increase R.")))
    elif inr > 1.5:
        flags.append(dict(
            level   = "warning",
            message = (f"EKF STRESSED — Innovation/Noise = {inr:.2f}× "
                       f"(limit 1.5×). Consider increasing R or cycles.")))
    else:
        flags.append(dict(
            level   = "ok",
            message = (f"EKF CONSISTENT — Innovation/Noise = {inr:.2f}× ✓")))

    if con > 20.0:
        flags.append(dict(
            level   = "warning",
            message = (f"Partial Convergence — tr(P) = {con:.1f}% of initial. "
                       f"More cycles or better P₀ would help.")))
    else:
        flags.append(dict(
            level   = "ok",
            message = (f"Converged — tr(P) = {con:.1f}% of initial ✓")))

    if 0.5 < nis < 2.0:
        flags.append(dict(
            level   = "ok",
            message  = (f"NIS consistent — mean NIS = {nis:.3f} "
                        f"(expected ≈ 1.0) ✓")))
    else:
        flags.append(dict(
            level   = "warning",
            message = (f"NIS inconsistent — mean NIS = {nis:.3f} "
                       f"(expected ≈ 1.0). Q/R may need tuning.")))

    inn_mean = metrics.get("inn_mean", 999)
    if abs(inn_mean) > 5.0:
        flags.append(dict(
            level   = "warning",
            message = (f"Innovation bias — mean = {inn_mean:.2f} mV "
                       f"(expected ≈ 0). Check OCV table accuracy.")))
    else:
        flags.append(dict(
            level   = "ok",
            message = (f"Innovation unbiased — mean = {inn_mean:.2f} mV ✓")))

    return flags


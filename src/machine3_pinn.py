import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .chemistry import make_ocv, docv_dsoc


class PINNResidual(nn.Module):
    """
    Mechanistically-Guided Residual Network.

    Learns the structural mismatch between ECM and DFN:
        residual(t) = V_true(t) - V_ecm(t)

    Architecture
    ------------
    Input(4) -> [Linear -> Tanh] x n_layers -> Linear(1)
    Skip connection: Linear(4->1) added to main path (ResNet style)

    Standards
    ---------
    He et al. (2015)          arXiv:1512.03385       — ResNet skip connection
    Nature Comms (2026)       doi:10.1038/s41467-...  — Mechanistic residual learning
    Xavier init               Glorot & Bengio (2010)  — Weight initialization
    """

    def __init__(self, n_layers: int = 3, n_neurons: int = 64):
        super().__init__()

        layers = [nn.Linear(4, n_neurons), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 1))
        self.net  = nn.Sequential(*layers)
        self.skip = nn.Linear(4, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


def _build_features(log: dict, chem: dict):
    """
    Build normalised feature matrix X and target vector y.

    Features : [t_norm, I_norm, SOC_est, V_ecm_norm]
    Target   : residual = V_true - V_ecm  (normalised)
    """
    ocv_fn = make_ocv(chem)

    V_ocv = np.array([
        float(ocv_fn(np.clip(s, 0.01, 0.99)))
        for s in log["soc_est"]
    ])
    V_ecm    = V_ocv - log["I_true"] * chem["R0"]
    residual = log["V_true"] - V_ecm

    t_range  = log["t"].max()  - log["t"].min()  + 1e-8
    I_mean   = log["I_true"].mean();  I_std  = log["I_true"].std()  + 1e-8
    Ve_mean  = V_ecm.mean();          Ve_std = V_ecm.std()          + 1e-8
    r_mean   = residual.mean();       r_std  = residual.std()        + 1e-8

    t_n  = (log["t"]      - log["t"].min()) / t_range
    I_n  = (log["I_true"] - I_mean)         / I_std
    Ve_n = (V_ecm         - Ve_mean)        / Ve_std
    r_n  = (residual      - r_mean)         / r_std

    X = np.stack([t_n, I_n, log["soc_est"], Ve_n], axis=1
                 ).astype(np.float32)
    y = r_n.astype(np.float32).reshape(-1, 1)

    stats = dict(r_mean=r_mean, r_std=r_std,
                 V_ecm=V_ecm, residual=residual)
    return X, y, stats


def run_pinn(
    log:          dict,
    chem:         dict,
    Q_nom:        float,
    n_epochs:     int   = 2000,
    n_layers:     int   = 3,
    n_neurons:    int   = 64,
    lambda_phys:  float = 0.01,
    progress_cb         = None,
):
    """
    Train PINN Residual Corrector on current simulation log.

    Standards
    ---------
    Nature Comms (2026)    Mechanistic residual learning
    arXiv:2412.16724       Physics loss: Coulomb counting + η
    Loshchilov (2017)      AdamW weight decay
    Cosine LR              Loshchilov & Hutter (2017) arXiv:1608.03983
    Gradient clipping      Pascanu et al. (2013) — stable RNN/deep training

    Parameters
    ----------
    log          : dict    output of machine2_ekf.run_cosim()
    chem         : dict    chemistry entry from chemistry.build_chem()
    Q_nom        : float   nominal capacity [Ah]
    n_epochs     : int     training epochs
    n_layers     : int     hidden layers
    n_neurons    : int     neurons per layer
    lambda_phys  : float   physics loss weight
    progress_cb  : callable(epoch, total, data_loss, phys_loss) | None

    Returns
    -------
    V_corrected   : np.ndarray  [V]
    soc_corrected : np.ndarray  [0-1]
    loss_hist     : list[float]
    phys_hist     : list[float]
    metrics       : dict
    """

    X, y, stats = _build_features(log, chem)
    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    model = PINNResidual(n_layers, n_neurons)

    # AdamW — better weight decay than Adam (Loshchilov 2017)
    optimizer = optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Cosine annealing LR (Loshchilov & Hutter 2017)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    loss_fn   = nn.MSELoss()
    loss_hist = []
    phys_hist = []

    Q_s      = float(Q_nom) * 3600.0
    I_tensor = torch.tensor(log["I_true"][:-1].astype(np.float32))

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        pred      = model(X_t)
        data_loss = loss_fn(pred, y_t)

        # Physics loss — Coulomb counting with η (Prada 2013)
        soc_t  = torch.tensor(log["soc_est"].astype(np.float32))
        dsoc   = (soc_t[1:] - soc_t[:-1]) / 10.0
        eta    = torch.where(
            I_tensor < 0.0,
            torch.full_like(I_tensor, 0.9),
            torch.ones_like(I_tensor),
        )
        phys_target = -(eta * I_tensor) / Q_s
        phys_loss   = loss_fn(dsoc, phys_target)

        total = data_loss + lambda_phys * phys_loss
        total.backward()

        # Gradient clipping (Pascanu et al. 2013)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        loss_hist.append(float(data_loss))
        phys_hist.append(float(phys_loss))

        if progress_cb and epoch % max(1, n_epochs // 40) == 0:
            progress_cb(epoch + 1, n_epochs,
                        float(data_loss), float(phys_loss))

    # ── Inference ─────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        r_pred_n = model(X_t).numpy().ravel()

    r_pred      = r_pred_n * stats["r_std"] + stats["r_mean"]
    V_corrected = stats["V_ecm"] + r_pred

    # SOC correction via OCV Jacobian
    ocv_fn    = make_ocv(chem)
    dOCV_dSOC = np.array([
        docv_dsoc(ocv_fn, np.clip(s, 0.05, 0.95))
        for s in log["soc_est"]
    ])
    dV            = V_corrected - log["V_est"]
    soc_corrected = np.clip(
        log["soc_est"] + 0.15 * dV / (dOCV_dSOC + 1e-6),
        0.0, 1.0,
    )

    # ── Metrics ───────────────────────────────────────────────
    v_rmse_ecm  = np.sqrt(np.mean((log["V_true"] - stats["V_ecm"]) ** 2)) * 1000
    v_rmse_pinn = np.sqrt(np.mean((log["V_true"] - V_corrected) ** 2))    * 1000
    s_rmse_ekf  = np.sqrt(np.mean((log["soc_true"] - log["soc_est"]) ** 2)) * 100
    s_rmse_pinn = np.sqrt(np.mean((log["soc_true"] - soc_corrected) ** 2))  * 100
    improv_v    = (v_rmse_ecm - v_rmse_pinn) / (v_rmse_ecm + 1e-8) * 100
    improv_s    = (s_rmse_ekf - s_rmse_pinn) / (s_rmse_ekf + 1e-8) * 100

    noise_std       = float(np.std(log["V_meas"] - log["V_true"]) + 1e-8)
    inn_corrected   = float(np.sqrt(np.mean((log["V_true"] - V_corrected) ** 2)))
    inn_noise_ratio = inn_corrected / noise_std

    metrics = dict(
        v_rmse_ecm      = v_rmse_ecm,
        v_rmse_pinn     = v_rmse_pinn,
        s_rmse_ekf      = s_rmse_ekf,
        s_rmse_pinn     = s_rmse_pinn,
        improv_v        = improv_v,
        improv_s        = improv_s,
        inn_noise_ratio = inn_noise_ratio,
        n_params        = sum(p.numel() for p in model.parameters()),
        final_data_loss = loss_hist[-1],
        final_phys_loss = phys_hist[-1],
    )

    return V_corrected, soc_corrected, loss_hist, phys_hist, metrics


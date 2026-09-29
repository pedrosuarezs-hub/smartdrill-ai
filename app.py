from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial import cKDTree
from scipy.stats import norm

st.set_page_config(page_title="SmartDrill AI", page_icon="⛏️", layout="wide")

WEIGHTS = {"global": 0.20, "entropy": 0.20, "grade": 0.60}
STOP_U = 0.018
STOP_G = 0.020
STOP_M = 0.030
STOP_PERSIST = 3
STOP_MIN_ITER = 4

st.markdown(
    """
    <style>
      .block-container{padding-top:1.1rem;padding-bottom:3rem;max-width:1450px}
      .sd-title{font-size:2.15rem;font-weight:850;letter-spacing:-.03em;margin-bottom:.1rem}
      .sd-sub{color:#8f9db2;margin-bottom:1.1rem}
      [data-testid="stMetricValue"]{font-size:1.7rem}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="sd-title">SmartDrill AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sd-sub">Geostatistical active learning for adaptive copper resource-definition drilling</div>',
    unsafe_allow_html=True,
)


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().upper() for c in out.columns]
    return out


def validate(collar, assay, survey=None):
    collar = normalize_cols(collar)
    assay = normalize_cols(assay)
    survey = normalize_cols(survey) if survey is not None else None
    issues, warnings = [], []
    req_c = ["BHID", "XCOLLAR", "YCOLLAR", "ZCOLLAR"]
    req_a = ["BHID", "FROM_M", "TO_M", "CU_PCT"]
    req_s = ["BHID", "AT_M", "AZIMUTH", "DIP"]

    mc = [c for c in req_c if c not in collar.columns]
    ma = [c for c in req_a if c not in assay.columns]
    if mc:
        issues.append(f"COLLAR missing: {mc}")
    if ma:
        issues.append(f"ASSAY missing: {ma}")

    if survey is not None:
        ms = [c for c in req_s if c not in survey.columns]
        if ms:
            issues.append(f"SURVEY missing: {ms}")
    else:
        warnings.append("No SURVEY supplied; holes will be treated as vertical.")

    if not issues:
        from_m = pd.to_numeric(assay["FROM_M"], errors="coerce")
        to_m = pd.to_numeric(assay["TO_M"], errors="coerce")
        if (to_m <= from_m).any():
            issues.append("ASSAY has TO_M <= FROM_M.")

        unknown = sorted(set(assay["BHID"].astype(str)) - set(collar["BHID"].astype(str)))
        if unknown:
            issues.append(f"ASSAY BHIDs absent from COLLAR: {unknown[:10]}")

    return issues, warnings


def direction(az_deg, dip_deg):
    az = math.radians(float(az_deg))
    dip = math.radians(float(dip_deg))
    horizontal = math.cos(dip)
    return np.array(
        [horizontal * math.sin(az), horizontal * math.cos(az), math.sin(dip)],
        dtype=float,
    )


def hole_position(md, collar_xyz, survey_hole):
    md = float(max(md, 0.0))
    if survey_hole is None or survey_hole.empty:
        return collar_xyz + np.array([0.0, 0.0, -md])

    s = survey_hole.sort_values("AT_M").copy()
    if float(s.iloc[0]["AT_M"]) > 0:
        first = s.iloc[0].copy()
        first["AT_M"] = 0.0
        s = pd.concat([pd.DataFrame([first]), s], ignore_index=True)

    rows = s.to_dict("records")
    pos = collar_xyz.astype(float).copy()

    for i, row in enumerate(rows):
        start = float(row["AT_M"])
        if start >= md:
            break

        next_row = rows[i + 1] if i + 1 < len(rows) else row
        end = min(float(next_row["AT_M"]), md) if i + 1 < len(rows) else md
        if end <= start:
            continue

        vector = direction(row["AZIMUTH"], row["DIP"]) + direction(
            next_row["AZIMUTH"], next_row["DIP"]
        )
        n = np.linalg.norm(vector)
        vector = direction(row["AZIMUTH"], row["DIP"]) if n == 0 else vector / n
        pos += (end - start) * vector

        if end >= md:
            break

    return pos


def sample_points(collar, assay, survey=None):
    collar = normalize_cols(collar)
    assay = normalize_cols(assay)
    survey = normalize_cols(survey) if survey is not None else None

    assay = assay.copy()
    for c in ["FROM_M", "TO_M", "CU_PCT"]:
        assay[c] = pd.to_numeric(assay[c], errors="coerce")
    assay = assay.dropna(subset=["FROM_M", "TO_M", "CU_PCT"])
    assay["MID_M"] = (assay["FROM_M"] + assay["TO_M"]) / 2.0

    collar_map = {str(r.BHID): r for r in collar.itertuples(index=False)}
    survey_groups = {}

    if survey is not None and not survey.empty:
        for c in ["AT_M", "AZIMUTH", "DIP"]:
            survey[c] = pd.to_numeric(survey[c], errors="coerce")
        survey = survey.dropna(subset=["AT_M", "AZIMUTH", "DIP"])
        survey_groups = {
            str(k): g for k, g in survey.groupby(survey["BHID"].astype(str))
        }

    rows = []
    for r in assay.itertuples(index=False):
        bhid = str(r.BHID)
        c = collar_map[bhid]
        xyz0 = np.array(
            [float(c.XCOLLAR), float(c.YCOLLAR), float(c.ZCOLLAR)], dtype=float
        )
        p = hole_position(float(r.MID_M), xyz0, survey_groups.get(bhid))
        rows.append(
            {
                "BHID": bhid,
                "MID_M": float(r.MID_M),
                "X": float(p[0]),
                "Y": float(p[1]),
                "Z": float(p[2]),
                "CU_PCT": float(r.CU_PCT),
            }
        )

    return pd.DataFrame(rows)


@dataclass
class GeoCfg:
    rx: float
    ry: float
    rz: float
    nugget: float
    sill: float
    threshold: float
    length: float


def infer_cfg(samples, threshold, length):
    ex = max(float(samples.X.max() - samples.X.min()), 1.0)
    ey = max(float(samples.Y.max() - samples.Y.min()), 1.0)
    ez = max(float(samples.Z.max() - samples.Z.min()), 1.0)
    sill = max(float(samples.CU_PCT.var(ddof=1)), 1e-5)
    return GeoCfg(
        max(0.55 * ex, 50),
        max(0.55 * ey, 50),
        max(0.55 * ez, 30),
        max(0.05 * sill, 1e-6),
        sill,
        threshold,
        length,
    )


class OKModel:
    def __init__(self, xyz, values, cfg):
        self.xyz = np.asarray(xyz, float)
        self.values = np.asarray(values, float)
        self.cfg = cfg
        self.scale = np.array([cfg.rx, cfg.ry, cfg.rz], float)
        self.psill = max(cfg.sill - cfg.nugget, 1e-8)
        self.scaled = self.xyz / self.scale
        self.tree = cKDTree(self.scaled)

    def cov(self, r):
        r = np.asarray(r, float)
        return self.psill * np.where(r < 1, 1 - 1.5 * r + 0.5 * r**3, 0)

    def predict(self, query, k=10, need_pred=True):
        query = np.asarray(query, float)
        pred = np.empty(len(query)) if need_pred else None
        var = np.empty(len(query))

        for start in range(0, len(query), 900):
            q = query[start : start + 900]
            d, idx = self.tree.query(q / self.scale, k=min(k, len(self.xyz)))
            if idx.ndim == 1:
                idx = idx[:, None]
                d = d[:, None]

            kk = idx.shape[1]
            neigh = self.scaled[idx]
            pair_dist = np.linalg.norm(
                neigh[:, :, None, :] - neigh[:, None, :, :], axis=-1
            )

            C = self.cov(pair_dist)
            ii = np.arange(kk)
            C[:, ii, ii] += self.cfg.nugget + 1e-10
            c = self.cov(d)

            B = len(q)
            A = np.zeros((B, kk + 1, kk + 1))
            A[:, :kk, :kk] = C
            A[:, :kk, kk] = 1
            A[:, kk, :kk] = 1

            b = np.zeros((B, kk + 1))
            b[:, :kk] = c
            b[:, kk] = 1

            try:
                sol = np.linalg.solve(A, b[..., None])[..., 0]
            except np.linalg.LinAlgError:
                sol = np.stack(
                    [np.linalg.lstsq(A[i], b[i], rcond=None)[0] for i in range(B)]
                )

            w = sol[:, :kk]
            lam = sol[:, kk]

            if need_pred:
                pred[start : start + B] = (w * self.values[idx]).sum(axis=1)

            v = self.psill + self.cfg.nugget - (w * c).sum(axis=1) - lam
            var[start : start + B] = np.maximum(v, 0)

        return pred, var


def monitor_grid(samples, nx=11, ny=9, nz=6):
    x = np.linspace(samples.X.min(), samples.X.max(), nx)
    y = np.linspace(samples.Y.min(), samples.Y.max(), ny)
    z = np.linspace(samples.Z.min(), samples.Z.max(), nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def generate_candidates(collar, min_dist, length):
    collar = normalize_cols(collar)
    xs = np.linspace(collar.XCOLLAR.min(), collar.XCOLLAR.max(), 8)
    ys = np.linspace(collar.YCOLLAR.min(), collar.YCOLLAR.max(), 7)
    existing = collar[["XCOLLAR", "YCOLLAR"]].to_numpy(float)

    rows = []
    idx = 1
    z = float(collar.ZCOLLAR.median())

    for y in ys:
        for x in xs:
            d = np.sqrt(((existing - np.array([x, y])) ** 2).sum(axis=1)).min()
            if d < min_dist:
                continue
            rows.append(
                {
                    "CANDIDATE_ID": f"C{idx:03d}",
                    "XCOLLAR": float(x),
                    "YCOLLAR": float(y),
                    "ZCOLLAR": z,
                    "AZIMUTH": 0.0,
                    "DIP": -90.0,
                    "LENGTH_M": float(length),
                }
            )
            idx += 1

    return pd.DataFrame(rows)


def candidate_xyz(row, step=15):
    depths = np.arange(step / 2, float(row.LENGTH_M), step)
    return np.column_stack(
        [
            np.full_like(depths, float(row.XCOLLAR)),
            np.full_like(depths, float(row.YCOLLAR)),
            np.full_like(depths, float(row.ZCOLLAR)) - depths,
        ]
    )


def rank_candidates(samples, candidate_df, cfg, topn=12):
    xyz = samples[["X", "Y", "Z"]].to_numpy(float)
    vals = samples.CU_PCT.to_numpy(float)

    model = OKModel(xyz, vals, cfg)
    monitor = monitor_grid(samples)
    pred0, var0 = model.predict(monitor)

    sd0 = np.sqrt(np.maximum(var0, 1e-12))
    p0 = 1 - norm.cdf((cfg.threshold - pred0) / sd0)
    H0 = -(p0 * np.log(p0 + 1e-12) + (1 - p0) * np.log(1 - p0 + 1e-12))
    gw0 = sd0 * (0.5 + np.clip(pred0 / cfg.threshold, 0, 2))

    rows = []
    for r in candidate_df.itertuples(index=False):
        cxyz = candidate_xyz(r)
        augmented = np.vstack([xyz, cxyz])

        geom = OKModel(augmented, np.zeros(len(augmented)), cfg)
        _, vnew = geom.predict(monitor, need_pred=False)

        sdnew = np.sqrt(np.maximum(vnew, 1e-12))
        pnew = 1 - norm.cdf((cfg.threshold - pred0) / sdnew)
        Hnew = -(
            pnew * np.log(pnew + 1e-12)
            + (1 - pnew) * np.log(1 - pnew + 1e-12)
        )
        gwnew = sdnew * (0.5 + np.clip(pred0 / cfg.threshold, 0, 2))

        rows.append(
            {
                **r._asdict(),
                "GLOBAL_GAIN": float(np.maximum(var0 - vnew, 0).mean()),
                "ENTROPY_GAIN": float(np.maximum(H0 - Hnew, 0).mean()),
                "GRADE_GAIN": float(np.maximum(gw0 - gwnew, 0).mean()),
            }
        )

    df = pd.DataFrame(rows)
    for c in ["GLOBAL_GAIN", "ENTROPY_GAIN", "GRADE_GAIN"]:
        lo, hi = df[c].min(), df[c].max()
        df[c + "_N"] = (df[c] - lo) / (hi - lo + 1e-12)

    df["INFORMATION_SCORE"] = (
        WEIGHTS["global"] * df.GLOBAL_GAIN_N
        + WEIGHTS["entropy"] * df.ENTROPY_GAIN_N
        + WEIGHTS["grade"] * df.GRADE_GAIN_N
    )

    df = df.sort_values(
        ["INFORMATION_SCORE", "GLOBAL_GAIN"], ascending=False
    ).reset_index(drop=True)
    df["RANK"] = np.arange(1, len(df) + 1)

    state = {
        "mean_var": float(var0.mean()),
        "mean_grade": float(pred0.mean()),
        "metal_proxy": float(np.where(pred0 >= cfg.threshold, pred0, 0).sum()),
    }
    return df.head(topn), state


def slice_model(samples, cfg, zvalue):
    x = np.linspace(samples.X.min(), samples.X.max(), 50)
    y = np.linspace(samples.Y.min(), samples.Y.max(), 42)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = np.full_like(X, zvalue)
    q = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    model = OKModel(
        samples[["X", "Y", "Z"]].to_numpy(float),
        samples.CU_PCT.to_numpy(float),
        cfg,
    )
    pred, var = model.predict(q)
    return x, y, pred.reshape(len(y), len(x)), var.reshape(len(y), len(x))


def stop_eval(history):
    if history is None or history.empty:
        return "CONTINUE", 0, "Insufficient history.", pd.DataFrame()

    h = normalize_cols(history).sort_values("ITERATION").copy()
    req = ["ITERATION", "MEAN_VAR", "MEAN_GRADE", "METAL_PROXY"]
    if any(c not in h.columns for c in req):
        return "CONTINUE", 0, f"HISTORY needs {req}", h

    for c in req:
        h[c] = pd.to_numeric(h[c], errors="coerce")

    h["UNCERTAINTY_REDUCTION_FRAC"] = np.nan
    h["MEAN_GRADE_CHANGE_FRAC"] = np.nan
    h["METAL_PROXY_CHANGE_FRAC"] = np.nan

    for i in range(1, len(h)):
        p = h.iloc[i - 1]
        c = h.iloc[i]
        h.iloc[i, h.columns.get_loc("UNCERTAINTY_REDUCTION_FRAC")] = (
            (p.MEAN_VAR - c.MEAN_VAR) / max(abs(p.MEAN_VAR), 1e-12)
        )
        h.iloc[i, h.columns.get_loc("MEAN_GRADE_CHANGE_FRAC")] = (
            abs(c.MEAN_GRADE - p.MEAN_GRADE) / max(abs(p.MEAN_GRADE), 1e-12)
        )
        h.iloc[i, h.columns.get_loc("METAL_PROXY_CHANGE_FRAC")] = (
            abs(c.METAL_PROXY - p.METAL_PROXY) / max(abs(p.METAL_PROXY), 1e-12)
        )

    run = 0
    for r in h.itertuples(index=False):
        stable = (
            r.ITERATION >= STOP_MIN_ITER
            and pd.notna(r.UNCERTAINTY_REDUCTION_FRAC)
            and r.UNCERTAINTY_REDUCTION_FRAC <= STOP_U
            and r.MEAN_GRADE_CHANGE_FRAC <= STOP_G
            and r.METAL_PROXY_CHANGE_FRAC <= STOP_M
        )
        run = run + 1 if stable else 0

    decision = "STOP" if run >= STOP_PERSIST else "CONTINUE"
    reason = (
        "Stability criteria persisted for the required consecutive updates."
        if decision == "STOP"
        else "At least one stability criterion has not persisted long enough."
    )
    return decision, run, reason, h


def demo_data():
    rng = np.random.default_rng(20260929)
    collar_rows, assay_rows, survey_rows = [], [], []
    hole = 1

    for y in np.linspace(150, 750, 4):
        for x in np.linspace(180, 1020, 5):
            bh = f"DH{hole:03d}"
            collar_rows.append(
                {
                    "BHID": bh,
                    "XCOLLAR": x,
                    "YCOLLAR": y,
                    "ZCOLLAR": 1000.0,
                }
            )
            survey_rows.append(
                {"BHID": bh, "AT_M": 0.0, "AZIMUTH": 0.0, "DIP": -90.0}
            )

            for fr in np.arange(0, 360, 30):
                mid = fr + 15
                dx = (x - 600) / 330
                dy = (y - 450) / 240
                dz = (mid - 210) / 160
                trend = np.exp(-0.5 * (dx * dx + dy * dy + dz * dz))
                cu = max(0.01, 0.04 + 0.78 * trend + rng.normal(0, 0.025))
                assay_rows.append(
                    {
                        "BHID": bh,
                        "FROM_M": float(fr),
                        "TO_M": float(fr + 30),
                        "CU_PCT": float(cu),
                    }
                )
            hole += 1

    history = pd.DataFrame(
        [
            {"ITERATION": 0, "MEAN_VAR": 0.01320, "MEAN_GRADE": 0.301, "METAL_PROXY": 142.0},
            {"ITERATION": 1, "MEAN_VAR": 0.01250, "MEAN_GRADE": 0.298, "METAL_PROXY": 139.0},
            {"ITERATION": 2, "MEAN_VAR": 0.01195, "MEAN_GRADE": 0.295, "METAL_PROXY": 136.5},
            {"ITERATION": 3, "MEAN_VAR": 0.01155, "MEAN_GRADE": 0.293, "METAL_PROXY": 134.8},
            {"ITERATION": 4, "MEAN_VAR": 0.01135, "MEAN_GRADE": 0.292, "METAL_PROXY": 133.8},
            {"ITERATION": 5, "MEAN_VAR": 0.01117, "MEAN_GRADE": 0.291, "METAL_PROXY": 132.8},
            {"ITERATION": 6, "MEAN_VAR": 0.01099, "MEAN_GRADE": 0.290, "METAL_PROXY": 131.9},
        ]
    )

    return (
        pd.DataFrame(collar_rows),
        pd.DataFrame(assay_rows),
        pd.DataFrame(survey_rows),
        history,
    )


with st.sidebar:
    st.header("SmartDrill project")
    mode = st.radio("Data mode", ["Demo dataset", "Upload project data"])
    threshold = st.number_input(
        "Study threshold (% Cu)", min_value=0.01, value=0.20, step=0.01
    )
    length = st.number_input(
        "Candidate hole length (m)", min_value=100, value=450, step=25
    )
    min_distance = st.number_input(
        "Minimum collar spacing (m)", min_value=10, value=70, step=10
    )

if mode == "Demo dataset":
    collar, assay, survey, history = demo_data()
else:
    st.subheader("Upload drilling tables")
    a, b, c, d = st.columns(4)
    with a:
        fc = st.file_uploader("COLLAR.csv", type=["csv"])
    with b:
        fs = st.file_uploader("SURVEY.csv (optional)", type=["csv"])
    with c:
        fa = st.file_uploader("ASSAY.csv", type=["csv"])
    with d:
        fh = st.file_uploader("HISTORY.csv (optional)", type=["csv"])

    if fc is None or fa is None:
        st.info("Upload at least COLLAR.csv and ASSAY.csv.")
        st.stop()

    collar = pd.read_csv(fc)
    assay = pd.read_csv(fa)
    survey = pd.read_csv(fs) if fs else None
    history = pd.read_csv(fh) if fh else pd.DataFrame()

issues, warnings = validate(collar, assay, survey)
for warning in warnings:
    st.warning(warning)

if issues:
    st.error("Input validation failed.")
    for issue in issues:
        st.write("•", issue)
    st.stop()

samples = sample_points(collar, assay, survey)
if len(samples) < 20:
    st.error("At least 20 valid assay midpoint samples are recommended.")
    st.stop()

base = infer_cfg(samples, float(threshold), float(length))

with st.sidebar:
    st.divider()
    st.subheader("Covariance model")
    rx = st.number_input("Range X (m)", min_value=10.0, value=float(round(base.rx, 1)))
    ry = st.number_input("Range Y (m)", min_value=10.0, value=float(round(base.ry, 1)))
    rz = st.number_input("Range Z (m)", min_value=10.0, value=float(round(base.rz, 1)))
    nug = st.number_input(
        "Nugget", min_value=0.0, value=float(base.nugget), format="%.6f"
    )
    sill = st.number_input(
        "Total sill", min_value=0.000001, value=float(base.sill), format="%.6f"
    )

cfg = GeoCfg(rx, ry, rz, nug, sill, float(threshold), float(length))

tabs = st.tabs(
    [
        "Overview",
        "Geostatistical model",
        "Next drillhole",
        "Adaptive stopping",
        "Data QA",
        "Methodology",
    ]
)

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Drillholes", f"{normalize_cols(collar).BHID.nunique():,}")
    c2.metric("Assay samples", f"{len(samples):,}")
    c3.metric("Mean Cu", f"{samples.CU_PCT.mean():.3f}%")
    c4.metric("Study threshold", f"{threshold:.2f}% Cu")

    fig = px.scatter_3d(
        samples,
        x="X",
        y="Y",
        z="Z",
        color="CU_PCT",
        hover_name="BHID",
        title="3D assay midpoint distribution",
        labels={"CU_PCT": "Cu (%)"},
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=45, b=0))
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    z = st.slider(
        "Diagnostic slice elevation",
        float(samples.Z.min()),
        float(samples.Z.max()),
        float(samples.Z.median()),
    )
    with st.spinner("Running ordinary kriging..."):
        x, y, estimate, variance = slice_model(samples, cfg, z)

    left, right = st.columns(2)
    with left:
        fig = go.Figure(
            go.Heatmap(x=x, y=y, z=estimate, colorbar=dict(title="Cu %"))
        )
        fig.update_layout(
            title=f"Estimated Cu at Z={z:.1f} m",
            height=510,
            xaxis_title="Easting",
            yaxis_title="Northing",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = go.Figure(
            go.Heatmap(x=x, y=y, z=variance, colorbar=dict(title="Variance"))
        )
        fig.update_layout(
            title=f"Kriging uncertainty at Z={z:.1f} m",
            height=510,
            xaxis_title="Easting",
            yaxis_title="Northing",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "v2 uses an anisotropic spherical covariance model. Formal automatic variogram fitting is a planned v3 milestone."
    )

with tabs[2]:
    candidate_df = generate_candidates(collar, float(min_distance), float(length))

    if candidate_df.empty:
        st.error("No candidates satisfy the spacing constraint.")
    else:
        with st.spinner("Scoring candidate holes..."):
            ranking, state = rank_candidates(
                samples, candidate_df, cfg, min(12, len(candidate_df))
            )

        a, b, c = st.columns(3)
        a.metric("Mean kriging variance", f"{state['mean_var']:.5f}")
        b.metric("Estimated mean grade", f"{state['mean_grade']:.3f}% Cu")
        c.metric("Metal proxy", f"{state['metal_proxy']:.2f}")

        top = ranking.iloc[0]
        st.success(
            f"Recommended next hole: {top.CANDIDATE_ID} | "
            f"E {top.XCOLLAR:.1f} | N {top.YCOLLAR:.1f} | "
            f"Dip {top.DIP:.0f}° | Length {top.LENGTH_M:.0f} m | "
            f"Score {top.INFORMATION_SCORE:.3f}"
        )

        fig = px.scatter(
            ranking,
            x="XCOLLAR",
            y="YCOLLAR",
            size="INFORMATION_SCORE",
            color="INFORMATION_SCORE",
            hover_name="CANDIDATE_ID",
            title="Top candidate collars by information score",
        )
        current_collars = normalize_cols(collar)
        fig.add_scatter(
            x=current_collars.XCOLLAR,
            y=current_collars.YCOLLAR,
            mode="markers",
            marker=dict(size=8, symbol="x"),
            name="Existing collars",
        )
        fig.update_layout(height=540)
        st.plotly_chart(fig, use_container_width=True)

        cols = [
            "RANK",
            "CANDIDATE_ID",
            "XCOLLAR",
            "YCOLLAR",
            "DIP",
            "LENGTH_M",
            "GLOBAL_GAIN",
            "ENTROPY_GAIN",
            "GRADE_GAIN",
            "INFORMATION_SCORE",
        ]
        st.dataframe(ranking[cols], use_container_width=True, hide_index=True)
        st.download_button(
            "Download ranked candidates CSV",
            ranking[cols].to_csv(index=False).encode(),
            "smartdrill_ranked_candidates.csv",
            "text/csv",
        )

with tabs[3]:
    decision, run, reason, evaluated = stop_eval(history)

    if decision == "STOP":
        st.success("STOP CAMPAIGN")
    else:
        st.warning("CONTINUE DRILLING")

    st.write(reason)

    if not evaluated.empty and "UNCERTAINTY_REDUCTION_FRAC" in evaluated.columns:
        latest = evaluated.iloc[-1]
        a, b, c, d = st.columns(4)
        a.metric("Persistence", f"{run}/{STOP_PERSIST}")
        b.metric(
            "Uncertainty reduction",
            "—"
            if pd.isna(latest.UNCERTAINTY_REDUCTION_FRAC)
            else f"{100 * latest.UNCERTAINTY_REDUCTION_FRAC:.2f}%",
        )
        c.metric(
            "Mean-grade change",
            "—"
            if pd.isna(latest.MEAN_GRADE_CHANGE_FRAC)
            else f"{100 * latest.MEAN_GRADE_CHANGE_FRAC:.2f}%",
        )
        d.metric(
            "Metal-proxy change",
            "—"
            if pd.isna(latest.METAL_PROXY_CHANGE_FRAC)
            else f"{100 * latest.METAL_PROXY_CHANGE_FRAC:.2f}%",
        )

        st.line_chart(evaluated.set_index("ITERATION")[["MEAN_VAR"]])
        st.dataframe(evaluated, use_container_width=True, hide_index=True)
    else:
        st.info(
            "Upload HISTORY.csv with ITERATION, MEAN_VAR, MEAN_GRADE, METAL_PROXY to evaluate autonomous stopping."
        )

with tabs[4]:
    st.subheader("Input quality summary")
    st.write("✅ Required schemas passed")
    st.write(f"Valid assay midpoints: {len(samples):,}")
    st.write(
        f"Cu range: {samples.CU_PCT.min():.3f}–{samples.CU_PCT.max():.3f}%"
    )
    st.write(
        f"X: {samples.X.min():.1f}–{samples.X.max():.1f} | "
        f"Y: {samples.Y.min():.1f}–{samples.Y.max():.1f} | "
        f"Z: {samples.Z.min():.1f}–{samples.Z.max():.1f}"
    )
    st.dataframe(samples.head(500), use_container_width=True, hide_index=True)

with tabs[5]:
    st.markdown(
        """
        ### Research logic implemented in v2

        1. Validate COLLAR, SURVEY and ASSAY.
        2. Convert assay intervals to 3D midpoint coordinates.
        3. Run local ordinary kriging and calculate spatial uncertainty.
        4. Generate vertical candidate drillholes.
        5. Rank candidates with the frozen Phase 7 acquisition weights:
           - 0.20 global variance reduction,
           - 0.20 threshold entropy reduction,
           - 0.60 grade-weighted uncertainty reduction.
        6. Recommend the next information-rich hole.
        7. Evaluate the autonomous stopping rule from campaign history:
           uncertainty reduction <= 1.8%, mean-grade change <= 2.0%,
           metal-proxy change <= 3.0%, persisting for 3 consecutive updates.

        ### Current limitations

        - Research prototype, not a compliant mineral-resource reporting package.
        - Candidate generator is currently vertical-hole only.
        - Prototype desurvey approximation; validated minimum-curvature desurvey is a v3 milestone.
        - Variogram fitting is not yet automatic.
        - No geological domain or wireframe constraints or density model yet.
        - 0.20% Cu is a study threshold, not an economic cutoff grade.
        """
    )

st.caption("SmartDrill AI v2 — functional research prototype for Minexcellence 2026.")

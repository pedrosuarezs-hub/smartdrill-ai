from __future__ import annotations

import io
import math
import zipfile

import numpy as np
import pandas as pd


EXPECTED_FILES = {
    "Collar.csv",
    "Survey.csv",
    "COMP_A.csv",
    "COMP_B.csv",
    "COMP_C.csv",
    "COMP_D.csv",
    "COMP_E.csv",
}


def direction_vector(azimuth_deg: float, dip_deg: float) -> np.ndarray:
    az = math.radians(float(azimuth_deg))
    dip = math.radians(float(dip_deg))
    horizontal = math.cos(dip)
    return np.array(
        [
            horizontal * math.sin(az),
            horizontal * math.cos(az),
            math.sin(dip),
        ],
        dtype=float,
    )


def minimum_curvature_increment(length_m: float, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    dogleg = math.acos(dot)
    if dogleg < 1e-10:
        rf = 1.0
    else:
        rf = (2.0 / dogleg) * math.tan(dogleg / 2.0)
    return (float(length_m) / 2.0) * (v1 + v2) * rf


def parse_lunahuasi_zip(uploaded_file) -> dict:
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        missing = sorted(EXPECTED_FILES - names)
        if missing:
            raise ValueError(f"ZIP is missing expected files: {missing}")

        tables = {
            name: pd.read_csv(zf.open(name))
            for name in EXPECTED_FILES
        }

    collar = tables["Collar.csv"].copy()
    survey = tables["Survey.csv"].copy()

    # The downloaded public package stores dip as a positive magnitude.
    # SmartDrill uses negative-downward dip for Cartesian trajectory calculations.
    collar = collar.rename(
        columns={
            "HoleID": "BHID",
            "Depth (m)": "LENGTH_M",
            "EASTING": "XCOLLAR",
            "NORTHING": "YCOLLAR",
            "ELEVATION": "ZCOLLAR",
            "Dip": "DIP_RAW",
            "Azimuth": "AZIMUTH",
            "Start Date": "START_DATE",
            "Season": "SEASON",
        }
    )
    collar["DIP"] = -pd.to_numeric(collar["DIP_RAW"], errors="coerce").abs()

    survey = survey.rename(
        columns={
            "HoleID": "BHID",
            "DEPTH": "AT_M",
            "Dip": "DIP_RAW",
            "Azimuth": "AZIMUTH",
        }
    )
    survey["DIP"] = -pd.to_numeric(survey["DIP_RAW"], errors="coerce").abs()

    comps = []
    for name in ["COMP_A.csv", "COMP_B.csv", "COMP_C.csv", "COMP_D.csv", "COMP_E.csv"]:
        d = tables[name].copy()
        d = d.loc[:, ~d.columns.astype(str).str.startswith("Unnamed")]
        d["SOURCE_FILE"] = name
        comps.append(d)

    composites = pd.concat(comps, ignore_index=True)

    return {
        "collar": collar,
        "survey": survey,
        "composites": composites,
        "raw_tables": tables,
    }


def _survey_groups(survey: pd.DataFrame):
    out = {}
    for bhid, g in survey.groupby("BHID"):
        gg = g.sort_values("AT_M").copy()
        gg["AT_M"] = pd.to_numeric(gg["AT_M"], errors="coerce")
        gg["AZIMUTH"] = pd.to_numeric(gg["AZIMUTH"], errors="coerce")
        gg["DIP"] = pd.to_numeric(gg["DIP"], errors="coerce")
        gg = gg.dropna(subset=["AT_M", "AZIMUTH", "DIP"])
        out[str(bhid)] = gg
    return out


def position_at_md(
    bhid: str,
    md: float,
    collar_map: dict,
    survey_groups: dict,
):
    c = collar_map[str(bhid)]
    pos = np.array([float(c.XCOLLAR), float(c.YCOLLAR), float(c.ZCOLLAR)], dtype=float)
    md = float(md)

    sh = survey_groups.get(str(bhid))
    if sh is None or sh.empty:
        v = direction_vector(c.AZIMUTH, c.DIP)
        return pos + md * v, "COLLAR_ORIENTATION_FALLBACK"

    s = sh.copy().sort_values("AT_M")
    if float(s.iloc[0]["AT_M"]) > 0:
        first = pd.DataFrame(
            [
                {
                    "BHID": bhid,
                    "AT_M": 0.0,
                    "AZIMUTH": c.AZIMUTH,
                    "DIP": c.DIP,
                }
            ]
        )
        s = pd.concat([first, s], ignore_index=True).sort_values("AT_M")

    rows = s.to_dict("records")
    current_md = 0.0

    for i in range(len(rows) - 1):
        r1 = rows[i]
        r2 = rows[i + 1]
        seg_start = float(r1["AT_M"])
        seg_end = float(r2["AT_M"])

        if md <= seg_start:
            break

        use_end = min(md, seg_end)
        seg_len = use_end - seg_start
        if seg_len <= 0:
            continue

        v1 = direction_vector(r1["AZIMUTH"], r1["DIP"])

        if use_end < seg_end:
            frac = seg_len / max(seg_end - seg_start, 1e-12)
            v2_full = direction_vector(r2["AZIMUTH"], r2["DIP"])
            v2 = (1 - frac) * v1 + frac * v2_full
            v2 = v2 / np.linalg.norm(v2)
        else:
            v2 = direction_vector(r2["AZIMUTH"], r2["DIP"])

        pos += minimum_curvature_increment(seg_len, v1, v2)
        current_md = use_end

        if md <= seg_end:
            return pos, "DOWNHOLE_SURVEY"

    last = rows[-1]
    last_depth = float(last["AT_M"])
    if md > current_md:
        current_md = max(current_md, last_depth)
        v = direction_vector(last["AZIMUTH"], last["DIP"])
        pos += max(md - current_md, 0.0) * v
        return pos, "SURVEY_PLUS_LAST_ORIENTATION_EXTRAPOLATION"

    return pos, "DOWNHOLE_SURVEY"


def build_drill_traces(collar: pd.DataFrame, survey: pd.DataFrame, step_m: float = 25.0) -> pd.DataFrame:
    collar_map = {str(r.BHID): r for r in collar.itertuples(index=False)}
    groups = _survey_groups(survey)
    rows = []

    for bhid, c in collar_map.items():
        length = float(c.LENGTH_M)
        depths = list(np.arange(0.0, length, step_m))
        if not depths or depths[-1] < length:
            depths.append(length)

        for md in depths:
            xyz, method = position_at_md(bhid, md, collar_map, groups)
            rows.append(
                {
                    "BHID": bhid,
                    "MD_M": float(md),
                    "X": float(xyz[0]),
                    "Y": float(xyz[1]),
                    "Z": float(xyz[2]),
                    "POSITION_METHOD": method,
                    "SEASON": c.SEASON,
                }
            )

    return pd.DataFrame(rows)


def build_composite_midpoints(
    collar: pd.DataFrame,
    survey: pd.DataFrame,
    composites: pd.DataFrame,
) -> pd.DataFrame:
    collar_map = {str(r.BHID): r for r in collar.itertuples(index=False)}
    groups = _survey_groups(survey)
    rows = []

    for _, r in composites.iterrows():
        bhid = str(r["HoleID"])
        mid = 0.5 * (float(r["From (m)"]) + float(r["To (m)"]))
        xyz, method = position_at_md(bhid, mid, collar_map, groups)

        rows.append(
            {
                "BHID": bhid,
                "NESTED_LEVEL": str(r["Nested Level"]),
                "FROM_M": float(r["From (m)"]),
                "TO_M": float(r["To (m)"]),
                "MID_M": mid,
                "WIDTH_M": float(r["Width (m)"]),
                "EST_TRUE_WIDTH_M": float(r["Est True Width (m)"]),
                "CU_PCT": float(r["Cu %"]),
                "AU_GPT": float(r["Au g/t"]),
                "AG_GPT": float(r["Ag g/t"]),
                "CUEQ_PCT": float(r["CuEq %"]),
                "PRESS_RELEASE_DATE": str(r["Press Release Date"]),
                "X": float(xyz[0]),
                "Y": float(xyz[1]),
                "Z": float(xyz[2]),
                "POSITION_METHOD": method,
                "SOURCE_FILE": str(r["SOURCE_FILE"]),
            }
        )

    return pd.DataFrame(rows)


def audit_lunahuasi(collar: pd.DataFrame, survey: pd.DataFrame, composites: pd.DataFrame) -> dict:
    missing_survey = sorted(set(collar["BHID"]) - set(survey["BHID"]))
    missing_composites = sorted(set(collar["BHID"]) - set(composites["HoleID"]))

    return {
        "holes": int(collar["BHID"].nunique()),
        "total_drilled_m": float(collar["LENGTH_M"].sum()),
        "mean_hole_length_m": float(collar["LENGTH_M"].mean()),
        "survey_rows": int(len(survey)),
        "survey_holes": int(survey["BHID"].nunique()),
        "holes_without_survey": missing_survey,
        "composite_intervals": int(len(composites)),
        "composite_holes": int(composites["HoleID"].nunique()),
        "holes_without_composites": missing_composites,
    }

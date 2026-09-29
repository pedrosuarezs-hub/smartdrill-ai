from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from realdata import (
    audit_lunahuasi,
    build_composite_midpoints,
    build_drill_traces,
    parse_lunahuasi_zip,
)

st.set_page_config(
    page_title="SmartDrill AI | Lunahuasi",
    page_icon="🌐",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container{
        padding-top:1.1rem;
        padding-bottom:3rem;
        max-width:1500px;
      }
      .hero{
        padding:1.25rem 1.45rem;
        border:1px solid rgba(98,167,255,.25);
        border-radius:18px;
        background:
          radial-gradient(circle at 85% 15%, rgba(75,209,160,.16), transparent 30%),
          linear-gradient(135deg, rgba(20,34,57,.98), rgba(10,18,32,.98));
        margin-bottom:1rem;
      }
      .hero h1{
        margin:0;
        font-size:2.2rem;
        letter-spacing:-.035em;
      }
      .hero p{
        margin:.35rem 0 0;
        color:#9aa9bd;
      }
      .tag{
        display:inline-block;
        padding:.28rem .58rem;
        border-radius:999px;
        border:1px solid rgba(75,209,160,.35);
        background:rgba(75,209,160,.08);
        color:#8ce8c3;
        font-size:.78rem;
        font-weight:700;
        margin-right:.35rem;
        margin-top:.55rem;
      }
      [data-testid="stMetric"]{
        background:linear-gradient(180deg, rgba(23,36,59,.95), rgba(13,23,39,.95));
        border:1px solid rgba(88,115,153,.22);
        padding:.75rem 1rem;
        border-radius:14px;
      }
      [data-testid="stMetricValue"]{font-size:1.65rem}
      .qa{
        padding:.85rem 1rem;
        border-radius:12px;
        background:rgba(246,198,103,.08);
        border:1px solid rgba(246,198,103,.24);
      }
      .ok{
        padding:.85rem 1rem;
        border-radius:12px;
        background:rgba(75,209,160,.08);
        border:1px solid rgba(75,209,160,.22);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>SmartDrill AI · Lunahuasi Real Data Explorer</h1>
      <p>3D visualization and technical QA of the public Lunahuasi drilling package.</p>
      <span class="tag">REAL COLLARS</span>
      <span class="tag">DOWNHOLE SURVEYS</span>
      <span class="tag">REPORTED COMPOSITES</span>
      <span class="tag">PHASE 9B</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Lunahuasi project")
uploaded = st.sidebar.file_uploader(
    "Upload the official Lunahuasi ZIP",
    type=["zip"],
    help="Use Lunahuasi-Drill-Data-May-21-2026-rev.zip or a compatible package.",
)

st.sidebar.divider()
st.sidebar.caption(
    "This page is intentionally separated from the synthetic resource-model module because "
    "the public package contains reported nested composite intervals, not continuous raw assays."
)

if uploaded is None:
    st.info(
        "Upload the Lunahuasi ZIP in the sidebar. SmartDrill will parse Collar.csv, Survey.csv, "
        "and COMP_A–E automatically."
    )
    st.markdown(
        """
        ### What this real-data module will show

        - actual collar locations and drill lengths;
        - deviated downhole traces reconstructed from survey stations;
        - COMP_A–E reported mineralized intervals in 3D;
        - Cu, Au, Ag and CuEq filters;
        - interval-width and nested-level filters;
        - drill-season filters;
        - survey-coverage QA;
        - downloadable cleaned SmartDrill tables.

        **Important:** this module does not treat the reported composites as a complete assay
        database. That prevents double-counting nested intervals or interpreting unreported
        intervals as zero grade.
        """
    )
    st.stop()

try:
    data = parse_lunahuasi_zip(uploaded)
except Exception as exc:
    st.error(f"Could not parse the uploaded ZIP: {exc}")
    st.stop()

collar = data["collar"]
survey = data["survey"]
composites = data["composites"]

with st.spinner("Reconstructing 3D drill traces and composite midpoint coordinates..."):
    traces = build_drill_traces(collar, survey, step_m=25.0)
    midpoints = build_composite_midpoints(collar, survey, composites)
    audit = audit_lunahuasi(collar, survey, composites)

season_map = collar.set_index("BHID")["SEASON"].to_dict()
midpoints["SEASON"] = midpoints["BHID"].map(season_map)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Drillholes", f"{audit['holes']}")
k2.metric("Total drilling", f"{audit['total_drilled_m']:,.0f} m")
k3.metric("Surveyed holes", f"{audit['survey_holes']}/{audit['holes']}")
k4.metric("Reported intervals", f"{audit['composite_intervals']:,}")
k5.metric("Composite holes", f"{audit['composite_holes']}/{audit['holes']}")

st.markdown(
    """
    <div class="qa">
      <b>Data interpretation:</b> COMP_A–E are nested reported composite intercepts.
      They are excellent for a real drilling/intercept case study, but they are not a
      continuous raw assay database and should not be stacked as independent samples
      for a resource model.
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "3D Intercept Explorer",
        "Drill Plan",
        "Data QA",
        "Downloads",
        "AI Transfer Status",
    ]
)

with tabs[0]:
    f1, f2, f3, f4 = st.columns([1.1, 1.1, 1.1, 1.2])

    with f1:
        levels = sorted(midpoints["NESTED_LEVEL"].dropna().unique())
        selected_levels = st.multiselect(
            "Composite levels",
            levels,
            default=["COMP_A"],
            help="Start with COMP_A because it is the broadest reported composite level.",
        )

    with f2:
        grade_field = st.selectbox(
            "Colour by",
            ["CU_PCT", "CUEQ_PCT", "AU_GPT", "AG_GPT"],
            format_func=lambda x: {
                "CU_PCT": "Cu (%)",
                "CUEQ_PCT": "CuEq (%)",
                "AU_GPT": "Au (g/t)",
                "AG_GPT": "Ag (g/t)",
            }[x],
        )

    with f3:
        min_width = st.number_input(
            "Minimum reported width (m)",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )

    with f4:
        seasons = sorted(midpoints["SEASON"].dropna().astype(str).unique())
        selected_seasons = st.multiselect(
            "Drilling seasons",
            seasons,
            default=seasons,
        )

    filtered = midpoints[
        midpoints["NESTED_LEVEL"].isin(selected_levels)
        & (midpoints["WIDTH_M"] >= float(min_width))
        & midpoints["SEASON"].astype(str).isin(selected_seasons)
    ].copy()

    st.caption(
        f"Displaying {len(filtered):,} reported intervals from "
        f"{filtered['BHID'].nunique() if len(filtered) else 0} holes."
    )

    fig = go.Figure()

    for bhid, g in traces.groupby("BHID"):
        fig.add_trace(
            go.Scatter3d(
                x=g["X"],
                y=g["Y"],
                z=g["Z"],
                mode="lines",
                line=dict(color="rgba(145,160,180,0.28)", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if len(filtered):
        hover = (
            "Hole %{customdata[0]}<br>"
            "Level %{customdata[1]}<br>"
            "From %{customdata[2]:.1f} m<br>"
            "To %{customdata[3]:.1f} m<br>"
            "Width %{customdata[4]:.1f} m<br>"
            "Cu %{customdata[5]:.2f}%<br>"
            "Au %{customdata[6]:.2f} g/t<br>"
            "Ag %{customdata[7]:.1f} g/t<br>"
            "CuEq %{customdata[8]:.2f}%<extra></extra>"
        )

        marker_sizes = 4 + 8 * (
            filtered["WIDTH_M"] / max(filtered["WIDTH_M"].quantile(.95), 1.0)
        ).clip(0, 1)

        fig.add_trace(
            go.Scatter3d(
                x=filtered["X"],
                y=filtered["Y"],
                z=filtered["Z"],
                mode="markers",
                name="Reported composites",
                marker=dict(
                    size=marker_sizes,
                    color=filtered[grade_field],
                    colorscale="Turbo",
                    opacity=.88,
                    colorbar=dict(title=grade_field),
                    line=dict(width=.3, color="rgba(255,255,255,.45)"),
                ),
                customdata=filtered[
                    [
                        "BHID",
                        "NESTED_LEVEL",
                        "FROM_M",
                        "TO_M",
                        "WIDTH_M",
                        "CU_PCT",
                        "AU_GPT",
                        "AG_GPT",
                        "CUEQ_PCT",
                    ]
                ].to_numpy(),
                hovertemplate=hover,
            )
        )

    fig.update_layout(
        height=760,
        margin=dict(l=0, r=0, t=35, b=0),
        title="Lunahuasi 3D drilling and reported mineralized intercepts",
        scene=dict(
            xaxis_title="Easting",
            yaxis_title="Northing",
            zaxis_title="Elevation",
            bgcolor="rgba(0,0,0,0)",
            aspectmode="data",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

    if len(filtered):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Median Cu", f"{filtered['CU_PCT'].median():.2f}%")
        c2.metric("Maximum Cu", f"{filtered['CU_PCT'].max():.2f}%")
        c3.metric("Median width", f"{filtered['WIDTH_M'].median():.1f} m")
        c4.metric("Maximum CuEq", f"{filtered['CUEQ_PCT'].max():.2f}%")

with tabs[1]:
    plan = go.Figure()

    for bhid, g in traces.groupby("BHID"):
        plan.add_trace(
            go.Scatter(
                x=g["X"],
                y=g["Y"],
                mode="lines",
                line=dict(color="rgba(130,150,175,.28)", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    plan.add_trace(
        go.Scatter(
            x=collar["XCOLLAR"],
            y=collar["YCOLLAR"],
            mode="markers+text",
            text=collar["BHID"],
            textposition="top center",
            marker=dict(size=7, color=collar["LENGTH_M"], colorscale="Blues"),
            customdata=collar[["LENGTH_M", "SEASON", "AZIMUTH", "DIP"]].to_numpy(),
            hovertemplate=(
                "Hole %{text}<br>"
                "Length %{customdata[0]:.1f} m<br>"
                "Season %{customdata[1]}<br>"
                "Azimuth %{customdata[2]:.1f}°<br>"
                "Dip %{customdata[3]:.1f}°<extra></extra>"
            ),
            name="Collars",
        )
    )

    plan.update_layout(
        title="Collar plan and horizontal drill-trace projections",
        height=680,
        xaxis_title="Easting",
        yaxis_title="Northing",
        yaxis_scaleanchor="x",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(plan, use_container_width=True)

with tabs[2]:
    st.subheader("Technical data QA")

    qa1, qa2, qa3 = st.columns(3)
    qa1.metric("Holes without survey rows", len(audit["holes_without_survey"]))
    qa2.metric("Holes without reported composites", len(audit["holes_without_composites"]))
    qa3.metric(
        "Mean hole length",
        f"{audit['mean_hole_length_m']:.1f} m",
    )

    if audit["holes_without_survey"]:
        st.warning(
            "No Survey.csv rows for: "
            + ", ".join(audit["holes_without_survey"])
            + ". Their 3D traces use collar orientation as a fallback."
        )

    if audit["holes_without_composites"]:
        st.info(
            "No reported composite intervals for: "
            + ", ".join(audit["holes_without_composites"])
        )

    methods = midpoints["POSITION_METHOD"].value_counts().rename_axis("Method").reset_index(name="Intervals")
    st.markdown("#### Composite midpoint positioning method")
    st.dataframe(methods, use_container_width=True, hide_index=True)

    st.markdown("#### Nested-level summary")
    level_summary = (
        midpoints.groupby("NESTED_LEVEL")
        .agg(
            Intervals=("BHID", "size"),
            Holes=("BHID", "nunique"),
            Width_m=("WIDTH_M", "sum"),
            Cu_mean=("CU_PCT", "mean"),
            Cu_median=("CU_PCT", "median"),
            Cu_max=("CU_PCT", "max"),
        )
        .reset_index()
    )
    st.dataframe(level_summary, use_container_width=True, hide_index=True)

    st.markdown(
        """
        **Interpretation rule**

        - COMP_A should be treated as the broadest reported intercept level.
        - COMP_B–E are progressively nested/selective sub-intervals.
        - Do not merge all levels into a kriging dataset: this would double-count overlapping
          intervals and bias grades upward.
        - Missing reported intervals are not zero-grade assays.
        """
    )

with tabs[3]:
    st.subheader("Export cleaned SmartDrill tables")

    collar_export = collar[
        [
            "BHID",
            "XCOLLAR",
            "YCOLLAR",
            "ZCOLLAR",
            "LENGTH_M",
            "AZIMUTH",
            "DIP",
            "SEASON",
            "START_DATE",
        ]
    ].copy()

    survey_export = survey[["BHID", "AT_M", "AZIMUTH", "DIP"]].copy()

    st.download_button(
        "Download cleaned Collar CSV",
        collar_export.to_csv(index=False).encode("utf-8"),
        file_name="Lunahuasi_Collar_SmartDrill.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download cleaned Survey CSV",
        survey_export.to_csv(index=False).encode("utf-8"),
        file_name="Lunahuasi_Survey_SmartDrill.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download 3D composite midpoint CSV",
        midpoints.to_csv(index=False).encode("utf-8"),
        file_name="Lunahuasi_Composite_Midpoints_3D.csv",
        mime="text/csv",
    )

    st.caption(
        "Dip is exported with the SmartDrill convention: negative values indicate downward drilling."
    )

with tabs[4]:
    st.subheader("Real-data active-learning transfer status")

    st.markdown(
        """
        <div class="ok">
          <b>Completed:</b> real collar geometry, survey trajectories, reported-composite
          positioning, filters, data QA and 3D visualization are now integrated into SmartDrill.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Why the full AI resource-model validation is not enabled yet

        The public ZIP contains reported **composited intercepts**, not a continuous raw assay
        table along every drilled metre. Running the synthetic kriging workflow directly on all
        COMP_A–E records would be methodologically incorrect because the levels are nested.

        ### Safe next experiment

        We can perform an **intercept-level retrospective study** using COMP_A only, explicitly
        labelled as a proof of concept. That study can ask whether SmartDrill would prioritize
        spatial areas associated with subsequently reported mineralized intercepts.

        It must not be described as a mineral-resource estimate or as full resource-definition
        validation.

        ### What would unlock the stronger validation

        A public or permissioned dataset containing continuous or regularly composited raw assays,
        collar coordinates and downhole surveys across the full drillholes.
        """
    )

st.caption(
    "SmartDrill AI · Phase 9B real-data explorer. Public Lunahuasi composites are used with explicit data limitations."
)

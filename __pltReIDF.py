# _pltReIDF.py
# ------------------------------------------------------------
# EnergyPlus time-series helpers with FLOOR-LEVEL support
# ------------------------------------------------------------
from __future__ import annotations

import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
# --- Consistent variable naming + palette helpers ---

def build_global_heatbalance_colormap(long_df: pd.DataFrame) -> dict[str, tuple]:
    """
    Build a consistent red/blue/purple color mapping for all variables across all plots.
    Each variable's color family (R/B/P) depends on its overall sign pattern in the dataset.
    """
    from matplotlib import cm
    import numpy as np

    if "Variable" not in long_df.columns or "Value" not in long_df.columns:
        raise ValueError("long_df must contain 'Variable' and 'Value' columns.")

    # 1) Determine sign behavior for each variable across all data
    summary = (long_df.groupby("Variable")["Value"]
               .agg(["min", "max", "mean"])
               .reset_index())
    summary["sign_type"] = np.select(
        [
            (summary["min"] < 0) & (summary["max"] > 0),
            (summary["mean"] >= 0),
        ],
        ["bidir", "gain"],
        default="loss"
    )

    # 2) Assign colors deterministically
    gains  = summary.query("sign_type == 'gain'")["Variable"].sort_values().tolist()
    losses = summary.query("sign_type == 'loss'")["Variable"].sort_values().tolist()
    bidir  = summary.query("sign_type == 'bidir'")["Variable"].sort_values().tolist()

    reds    = cm.get_cmap("Reds",    len(gains) + 2)
    blues   = cm.get_cmap("Blues",   len(losses) + 2)
    purples = cm.get_cmap("Purples", len(bidir)  + 2)

    cmap = {}
    for i, v in enumerate(gains):   cmap[v]  = reds(i + 1)
    for i, v in enumerate(losses):  cmap[v]  = blues(i + 1)
    for i, v in enumerate(bidir):   cmap[v]  = purples(i + 1)

    return cmap


# ---------------------- Parsing & Loading ----------------------

def parse_eplus_datetime(s: str, default_year: int = 2021):
    s = str(s).strip()
    try:
        return pd.to_datetime(f"{default_year}/{s}", format="%Y/%m/%d  %H:%M:%S")
    except Exception:
        return pd.to_datetime(s, errors="coerce")


def detect_zone_type(name: str) -> str:
    u = str(name).upper()
    keywords = [
        "MEETING","OPENOFFICE","OFFICE","ATRIUM","LAB","STORE",
        "CIRCULATION","STAIR","CAFE","EQUIPMENT","CORRIDOR","LOBBY",
        "CLASSROOM","KITCHEN","TOILET","RESTROOM","SERVER","VRF","ROOM"
    ]
    for k in keywords:
        if k in u:
            return k
    return "OTHER"


# --------- Floor & Area metadata builders (from your two CSVs) ---------

def _norm_zone_key(s: str) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip().rstrip(",").upper()


def _parse_floors_list(x):
    if pd.isna(x) or str(x).strip() == "":
        return []
    try:
        return ast.literal_eval(str(x))
    except Exception:
        return []


def build_zone_meta_from_csvs(
    summary_csv: str = "ref/zones_by_floor_summary.csv",
    areas_csv: str   = "ref/zone_areas.csv",
    mode: str = "primary"  # "primary" (100% to floor_primary) | "split" (even share across floors_found)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      zone_meta: columns ['Zone','Floor','Area_m2'] (Zone repeated per floor in split mode; 'Zone' is UPPER key)
      floor_areas: ['Floor','Floor Area [m2]']
    Expects:
      summary_csv: columns 'zone', 'floor_primary', optional 'floors_found_str'
      areas_csv:   columns 'ZONE' (UPPER), 'Area [m2]'
    """
    summary = pd.read_csv(summary_csv)
    areas   = pd.read_csv(areas_csv)

    summary["zone_key"] = summary["zone"].apply(_norm_zone_key)
    areas["ZONE_KEY"]   = areas["ZONE"].apply(_norm_zone_key)

    if "floors_found_str" in summary.columns:
        summary["floors_found"] = summary["floors_found_str"].apply(_parse_floors_list)
    else:
        summary["floors_found"] = [[] for _ in range(len(summary))]

    zones = summary.merge(
        areas[["ZONE_KEY", "Area [m2]"]],
        left_on="zone_key",
        right_on="ZONE_KEY",
        how="left"
    )

    rows = []
    for _, r in zones.iterrows():
        zone_key = r.get("zone_key", "")
        area = r.get("Area [m2]")
        if not zone_key or pd.isna(area):
            continue

        if mode.lower() == "split":
            floors = r.get("floors_found", [])
            if isinstance(floors, float) and np.isnan(floors):
                floors = []
            if not floors:
                fp = r.get("floor_primary")
                if pd.isna(fp):
                    continue
                floors = [int(fp)]
            share = float(area) / max(len(floors), 1)
            for f in floors:
                rows.append({"Zone": zone_key, "Floor": int(f), "Area_m2": share})
        else:  # "primary"
            fp = r.get("floor_primary")
            if pd.isna(fp):
                continue
            rows.append({"Zone": zone_key, "Floor": int(fp), "Area_m2": float(area)})

    zone_meta = pd.DataFrame(rows)
    if zone_meta.empty:
        zone_meta = pd.DataFrame(columns=["Zone","Floor","Area_m2"])

    floor_areas = (
        zone_meta.groupby("Floor", as_index=False)["Area_m2"]
                 .sum()
                 .rename(columns={"Area_m2":"Floor Area [m2]"})
                 .sort_values("Floor")
    )
    return zone_meta, floor_areas


def attach_floor_area(long_df: pd.DataFrame, zone_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns: Floor (int) and Area_m2 to long_df by Zone key.
    long_df['Zone'] may include mixed case; we normalize for join.
    """
    df = long_df.copy()
    df["Zone_key"] = df["Zone"].str.upper().str.strip().str.rstrip(",")
    zm = zone_meta.copy()
    zm["Zone_key"] = zm["Zone"].str.upper().str.strip().str.rstrip(",")
    df = df.merge(zm[["Zone_key","Floor","Area_m2"]], on="Zone_key", how="left")
    df.drop(columns=["Zone_key"], inplace=True)
    return df


# ---------------------- Core Long Loader ----------------------

def load_long(
    input_csv: str,
    assume_year: int = 2021,
    areas_df: pd.DataFrame | None = None,
    zone_meta_df: pd.DataFrame | None = None  # NEW optional: to attach Floor + Area_m2
) -> pd.DataFrame:
    """
    Return long-format df with columns:
    Date/Time | Zone | Variable | Value | Year | Month | ZoneType | (Area_m2?) | (Floor?)
    If zone_meta_df is provided (Zone, Floor, Area_m2), those columns are merged in.
    If areas_df is provided (Zone, Area_m2), Area_m2 is merged (and can be overridden by zone_meta_df).
    """
    df = pd.read_csv(input_csv)
    if "Date/Time" not in df.columns:
        raise ValueError("Expected 'Date/Time' column not found.")
    df["Date/Time"] = df["Date/Time"].apply(lambda s: parse_eplus_datetime(s, default_year=assume_year))
    df = df.dropna(subset=["Date/Time"])

    zonevar_cols = [c for c in df.columns if ":" in str(c)]
    df_zonevars = df[["Date/Time"] + zonevar_cols].copy()

    long_df = df_zonevars.melt(id_vars=["Date/Time"], var_name="ZoneVariable", value_name="Value")
    zv = long_df["ZoneVariable"].astype(str)
    long_df["Zone"] = zv.apply(lambda s: s.split(":", 1)[0] if ":" in s else s)
    long_df["Variable"] = zv.apply(lambda s: s.split(":", 1)[1] if ":" in s else "UNKNOWN")
    long_df = long_df.drop(columns=["ZoneVariable"])

    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce").fillna(0.0)
    long_df["Year"] = long_df["Date/Time"].dt.year
    long_df["Month"] = long_df["Date/Time"].dt.to_period("M")
    long_df["ZoneType"] = long_df["Zone"].apply(detect_zone_type)

    # Merge areas from areas_df (if provided)
    if areas_df is not None and {"Zone","Area_m2"}.issubset(areas_df.columns):
        long_df = long_df.merge(
            areas_df[["Zone","Area_m2"]].assign(Zone=areas_df["Zone"].astype(str)),
            on="Zone", how="left"
        )

    # Merge Floor + Area_m2 from zone_meta_df (takes precedence for Area_m2 if present)
    if zone_meta_df is not None and "Zone" in zone_meta_df.columns:
        # zone_meta may be repeated per floor; we keep many-to-one at this stage
        # (aggregations will handle 'Floor' dimension).
        zmeta = zone_meta_df.copy()
        zmeta["Zone"] = zmeta["Zone"].astype(str)
        # If multiple rows per Zone, we won't collapse here; merge will create duplicates by design.
        long_df = long_df.merge(zmeta[["Zone","Floor","Area_m2"]], on="Zone", how="left")

    return long_df


# ---------------------- Aggregations ----------------------

def _convert_units(series_wh: pd.Series, units: str) -> pd.Series:
    return series_wh / 1000.0 if str(units).lower() == "kwh" else series_wh


def compute_aggs(long_df: pd.DataFrame) -> dict:
    """
    Precompute all standard aggregations (in Wh); now also includes floor-level if 'Floor' present.
    """
    out = {}
    out["monthly_zone_variable_sum"]     = long_df.groupby(["Month","Zone","Variable"], as_index=False)["Value"].sum()
    out["yearly_zone_variable_sum"]      = long_df.groupby(["Year","Zone","Variable"], as_index=False)["Value"].sum()
    out["monthly_zonetype_variable_sum"] = long_df.groupby(["Month","ZoneType","Variable"], as_index=False)["Value"].sum()
    out["yearly_zonetype_variable_sum"]  = long_df.groupby(["Year","ZoneType","Variable"], as_index=False)["Value"].sum()
    out["monthly_building_variable_sum"] = long_df.groupby(["Month","Variable"], as_index=False)["Value"].sum()
    out["yearly_building_variable_sum"]  = long_df.groupby(["Year","Variable"], as_index=False)["Value"].sum()

    if "Floor" in long_df.columns:
        # keep rows with concrete floor numbers
        lf = long_df.dropna(subset=["Floor"]).copy()
        out["monthly_floor_variable_sum"] = lf.groupby(["Month","Floor","Variable"], as_index=False)["Value"].sum()
        out["yearly_floor_variable_sum"]  = lf.groupby(["Year","Floor","Variable"], as_index=False)["Value"].sum()
    else:
        out["monthly_floor_variable_sum"] = pd.DataFrame(columns=["Month","Floor","Variable","Value"])
        out["yearly_floor_variable_sum"]  = pd.DataFrame(columns=["Year","Floor","Variable","Value"])

    return out


def get_df(
    aggs: dict,
    scale: str = "monthly",           # "monthly" | "yearly"
    scope: str = "building",          # "building" | "zonetype" | "zone" | "floor"
    zone: str | None = None,
    zonetype: str | None = None,
    floor: int | None = None,         # NEW
    variables: list[str] | None = None,
    units: str = "kWh",               # "Wh" | "kWh"
    per_m2: bool = False,
    total_area: float | None = None,  # used for building-level per m²
    long_df_for_area: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Fetch an aggregated dataframe at the requested scale/scope, with optional filtering,
    units conversion, and per-m² normalization. (Now supports scope="floor")
    Returns a tidy dataframe keyed by Month/Year and Variable with Value (+ Value_per_m2 if requested).
    """
    scale = scale.lower()
    scope = scope.lower()

    key_map = {
        ("monthly","building"): "monthly_building_variable_sum",
        ("monthly","zonetype"): "monthly_zonetype_variable_sum",
        ("monthly","zone"):     "monthly_zone_variable_sum",
        ("monthly","floor"):    "monthly_floor_variable_sum",
        ("yearly","building"):  "yearly_building_variable_sum",
        ("yearly","zonetype"):  "yearly_zonetype_variable_sum",
        ("yearly","zone"):      "yearly_zone_variable_sum",
        ("yearly","floor"):     "yearly_floor_variable_sum",
    }
    base_key = key_map.get((scale, scope))
    if base_key is None:
        raise ValueError("Invalid combination of scale/scope.")

    df = aggs[base_key].copy()

    # Apply scope filters
    if scope == "zone" and zone is not None:
        df = df[df["Zone"] == zone]
    if scope == "zonetype" and zonetype is not None:
        df = df[df["ZoneType"] == zonetype]
    if scope == "floor" and floor is not None:
        df = df[df["Floor"] == floor]

    # Filter variables if provided
    if variables:
        df = df[df["Variable"].isin(variables)]

    # Units conversion (from Wh)
    df["Value"] = _convert_units(df["Value"], units)

    # Per m² normalization
    if per_m2:
        if scope == "building":
            if total_area is None and long_df_for_area is not None and "Area_m2" in long_df_for_area.columns:
                total_area = long_df_for_area.dropna(subset=["Area_m2"])[["Zone","Area_m2"]].drop_duplicates()["Area_m2"].sum()
            df["Value_per_m2"] = df["Value"] / (total_area if (total_area and total_area > 0) else np.nan)

        elif scope == "zonetype":
            if long_df_for_area is None or "Area_m2" not in long_df_for_area.columns:
                df["Value_per_m2"] = np.nan
            else:
                areas = long_df_for_area.dropna(subset=["Area_m2"])[["ZoneType","Zone","Area_m2"]].drop_duplicates()
                type_area = areas.groupby("ZoneType", as_index=False)["Area_m2"].sum()
                df = df.merge(type_area, on="ZoneType", how="left")
                df["Value_per_m2"] = df["Value"] / df["Area_m2"]

        elif scope == "zone":
            if long_df_for_area is None or "Area_m2" not in long_df_for_area.columns:
                df["Value_per_m2"] = np.nan
            else:
                areas = long_df_for_area[["Zone","Area_m2"]].drop_duplicates()
                df = df.merge(areas, on="Zone", how="left")
                df["Value_per_m2"] = df["Value"] / df["Area_m2"]

        elif scope == "floor":
            if long_df_for_area is None or not {"Area_m2","Floor"}.issubset(long_df_for_area.columns):
                df["Value_per_m2"] = np.nan
            else:
                # Sum unique areas by floor (avoid double counting Zone x Floor)
                floor_areas = (long_df_for_area.dropna(subset=["Floor","Area_m2"])
                                [["Zone","Floor","Area_m2"]]
                                .drop_duplicates()
                                .groupby("Floor", as_index=False)["Area_m2"].sum()
                                .rename(columns={"Area_m2":"Floor_Area_m2"}))
                df = df.merge(floor_areas, on="Floor", how="left")
                df["Value_per_m2"] = df["Value"] / df["Floor_Area_m2"]

    return df


# ---------------------- Legend coverage helper (optional) ----------------------

def check_legend_variables(
    df: pd.DataFrame,
    scale: str = "monthly",
    per_m2: bool = False
) -> dict:
    """
    Returns which variables *should* appear in the legend based on the data.
    """
    scale = scale.lower()
    time_key = "Month" if scale == "monthly" else "Year"
    value_col = "Value_per_m2" if (per_m2 and "Value_per_m2" in df.columns) else "Value"

    tmp = df.copy()
    if time_key == "Month":
        tmp[time_key] = pd.to_datetime(tmp[time_key].astype(str) + "-01", errors="coerce")

    piv = tmp.pivot_table(index=time_key, columns="Variable", values=value_col, aggfunc="sum").fillna(0)
    nonzero = piv.loc[:, (piv != 0).any(axis=0)]
    expected_vars = list(nonzero.columns)

    return {
        "n_expected_in_legend": len(expected_vars),
        "expected_variables": expected_vars
    }

# --- helper: snap to nearest-up multiple of a step (default 2.5) ---
def _snap_up(value: float, step: float = 2.5) -> float:
    if value is None or not np.isfinite(value):
        return step
    if value <= 0:
        return step
    return np.ceil(value / step) * step


def plot_heat_balance(
    df,
    scale="monthly",
    scope="building",
    units="kWh",
    per_m2=False,
    title_suffix="",
    ylim_abs=None,
    global_color_map: dict[str, tuple] | None = None  # NEW: consistent colors across plots
):
    """
    Heat-balance stacked bar chart with fixed color "scenarios":
      - Reds   = gains (positive-only)
      - Blues  = losses (negative-only)
      - Purple = bidirectional (both + and -)
    Net Total line is removed.
    Y axis is symmetric and snapped to the nearest UP multiple of 2.5 that covers stacks.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch

    # --- Styling ---
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        'figure.titlesize': 20,
    })

    scale = scale.lower()
    time_key = "Month" if scale == "monthly" else "Year"
    value_col = "Value_per_m2" if (per_m2 and "Value_per_m2" in df.columns) else "Value"

    df = df.copy()
    if time_key == "Month":
        df[time_key] = pd.to_datetime(df[time_key].astype(str) + "-01")
        bar_width = 25
    else:
        df[time_key] = pd.to_datetime(df[time_key].astype(str) + "-01-01")
        bar_width = 120

    # --- Build tidy pivot ---
    pivot_df = df.pivot_table(index=time_key, columns="Variable", values=value_col, aggfunc="sum").fillna(0)
    pivot_df = pivot_df.sort_index()

    # --- Clean variable names for nicer legend labels ---
    def clean_name(v):
        v = v.replace("Zone Air Heat Balance ", "")
        v = v.replace(" Rate [W](Hourly)", "")
        return v.strip()

    rename_map = {v: clean_name(v) for v in pivot_df.columns}
    pivot_df.rename(columns=rename_map, inplace=True)

    # --- Classify variables (for stacking and color "families") ---
    # classify variables by sign on the CLEANED column names
    var_state = {}
    for v in pivot_df.columns:
        vals = pivot_df[v]
        has_pos = (vals > 0).any()
        has_neg = (vals < 0).any()
        if has_pos and has_neg:
            var_state[v] = "bidir"
        elif vals.mean() >= 0:
            var_state[v] = "gain"
        else:
            var_state[v] = "loss"

    gains  = sorted([v for v, t in var_state.items() if t == "gain"])
    losses = sorted([v for v, t in var_state.items() if t == "loss"])
    bidir  = sorted([v for v, t in var_state.items() if t == "bidir"])

    # ---- Color selection
    if global_color_map is not None:
        # 1) ensure global map uses cleaned keys
        #    (if your global builder returns raw names, map them through clean_name first)
        # global_color_map = { clean_name(k): c for k, c in global_color_map.items() }

        color_map = dict(global_color_map)  # copy
        # 2) fill any missing variables deterministically, preserving R/B/P families
        missing_g = [v for v in gains  if v not in color_map]
        missing_l = [v for v in losses if v not in color_map]
        missing_b = [v for v in bidir  if v not in color_map]

        if missing_g or missing_l or missing_b:
            from matplotlib import cm
            if missing_g:
                reds = cm.get_cmap("Reds", len(missing_g) + 2)
                for i, v in enumerate(missing_g):
                    color_map[v] = reds(i + 1)
            if missing_l:
                blues = cm.get_cmap("Blues", len(missing_l) + 2)
                for i, v in enumerate(missing_l):
                    color_map[v] = blues(i + 1)
            if missing_b:
                purples = cm.get_cmap("Purples", len(missing_b) + 2)
                for i, v in enumerate(missing_b):
                    color_map[v] = purples(i + 1)
    else:
        # local fallback: consistent families within THIS plot
        from matplotlib import cm
        reds    = cm.get_cmap("Reds",    len(gains)  + 2)
        blues   = cm.get_cmap("Blues",   len(losses) + 2)
        purples = cm.get_cmap("Purples", len(bidir)  + 2)

        color_map = {}
        for i, v in enumerate(gains):   color_map[v] = reds(i + 1)
        for i, v in enumerate(losses):  color_map[v] = blues(i + 1)
        for i, v in enumerate(bidir):   color_map[v] = purples(i + 1)


    # --- Positive / Negative parts for stacking ---
    pos_df = pivot_df.clip(lower=0)
    neg_df = pivot_df.clip(upper=0)

    fig, ax = plt.subplots(figsize=(14, 8))

    # --- Draw order: gains → bidir (top); bidir → losses (bottom) ---
    bottom = np.zeros(len(pos_df))
    for v in sorted(gains):
        if np.any(pos_df[v] != 0):
            ax.bar(pos_df.index, pos_df[v], width=bar_width, bottom=bottom,
                   color=color_map[v], label=v)
            bottom += pos_df[v]
    for v in sorted(bidir):
        if np.any(pos_df[v] != 0):
            ax.bar(pos_df.index, pos_df[v], width=bar_width, bottom=bottom,
                   color=color_map[v], label=v)
            bottom += pos_df[v]

    bottom = np.zeros(len(neg_df))
    for v in sorted(bidir):
        if np.any(neg_df[v] != 0):
            ax.bar(neg_df.index, neg_df[v], width=bar_width, bottom=bottom,
                   color=color_map[v], label=v)
            bottom += neg_df[v]
    for v in sorted(losses):
        if np.any(neg_df[v] != 0):
            ax.bar(neg_df.index, neg_df[v], width=bar_width, bottom=bottom,
                   color=color_map[v], label=v)
            bottom += neg_df[v]

    # --- Axes styling ---
    ax.axhline(0, color="black", linewidth=1.5)
    ttl_units = f"{units}/m²" if per_m2 else units
    title_core = f"Heat Balance — {scale.capitalize()} ({str(scope).capitalize()})"
    ax.set_title(f"{title_core} [{ttl_units}] {title_suffix}".strip())
    ax.set_xlabel(scale.capitalize())
    ax.set_ylabel(f"Total Energy [{ttl_units}]")
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- Symmetric Y with snapping to 2.5 multiples ---
    # Compute required coverage from actual stack totals (top of positive stack, bottom of negative stack)
    pos_stack_top = (pos_df.sum(axis=1).max() if not pos_df.empty else 0.0) or 0.0
    neg_stack_bot = (neg_df.sum(axis=1).min() if not neg_df.empty else 0.0) or 0.0
    max_abs_needed = max(abs(pos_stack_top), abs(neg_stack_bot))

    # If user provided ylim_abs, still snap it UP to the nearest 2.5 multiple;
    # else compute from data and snap.
    target = ylim_abs if (ylim_abs is not None) else max_abs_needed
    snapped = _snap_up(float(target), step=2.5)
    if snapped == 0:
        snapped = 2.5
    ax.set_ylim(-snapped, snapped)

    # --- X-axis ticks ---
    if scale == "yearly":
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # --- Legend (variables only; NO net total) ---
    vars_list = list(pivot_df.columns)
    var_handles = [Patch(facecolor=color_map[v], label=v) for v in vars_list]
    fig.legend(
        handles=var_handles,
        labels=[h.get_label() for h in var_handles],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        ncol=1
    )

    plt.tight_layout(rect=[0, 0, 0.78, 1])
    plt.show()

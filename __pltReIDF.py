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



def _snap_up(value: float, step: float = 2.5) -> float:
    import numpy as np
    if value is None or not np.isfinite(value) or value <= 0:
        return step
    return np.ceil(value / step) * step

def clean_var_name(v: str) -> str:
    v = str(v)
    v = v.replace("Zone Air Heat Balance ", "")
    v = v.replace(" Rate [W](Hourly)", "")
    return v.strip()

def plot_heat_balance(
    df,
    scale="monthly",
    scope="building",
    units="kWh",
    per_m2=False,
    title_suffix="",
    ylim_abs=None,
    fixed_color_map: dict[str, tuple | str] | None = None,   # REQUIRED for fixed colors
    strict_colors: bool = True,   # True: raise if a variable color is missing; False: fallback gray
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch

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
    vcol = "Value_per_m2" if (per_m2 and "Value_per_m2" in df.columns) else "Value"

    df = df.copy()
    if time_key == "Month":
        df[time_key] = pd.to_datetime(df[time_key].astype(str) + "-01")
        bar_width = 25
    else:
        df[time_key] = pd.to_datetime(df[time_key].astype(str) + "-01-01")
        bar_width = 120

    # pivot and CLEAN names to match your dict keys
    piv = (df.pivot_table(index=time_key, columns="Variable", values=vcol, aggfunc="sum")
             .fillna(0)
             .sort_index())
    piv.columns = [clean_var_name(c) for c in piv.columns]

    # color lookup: ONLY from fixed_color_map
    if fixed_color_map is None:
        raise ValueError("fixed_color_map is required when strict colors are requested.")
    colors = dict(fixed_color_map)
    missing = [c for c in piv.columns if c not in colors]
    if missing and strict_colors:
        raise KeyError(f"Missing colors for variables: {missing}")
    fallback = "#bdbdbd"  # neutral gray
    color_of = lambda name: colors.get(name, fallback)

    # sign-based stacking (geometry only; no color inference)
    pos = piv.clip(lower=0)
    neg = piv.clip(upper=0)

    fig, ax = plt.subplots(figsize=(14, 8))

    # order legend/stack deterministically by column name
    cols = sorted(piv.columns)

    # positive stack
    bottom = np.zeros(len(pos))
    for v in cols:
        vals = pos[v].values
        if np.any(vals != 0):
            ax.bar(pos.index, vals, width=bar_width, bottom=bottom, color=color_of(v), label=v)
            bottom += vals

    # negative stack
    bottom = np.zeros(len(neg))
    for v in cols:
        vals = neg[v].values
        if np.any(vals != 0):
            ax.bar(neg.index, vals, width=bar_width, bottom=bottom, color=color_of(v), label=v)
            bottom += vals

    # symmetric y-axis snapped to 2.5 multiples
    pos_top = pos.sum(axis=1).max() if not pos.empty else 0.0
    neg_bot = neg.sum(axis=1).min() if not neg.empty else 0.0
    needed = max(abs(pos_top), abs(neg_bot))
    target = float(ylim_abs) if (ylim_abs is not None) else needed
    y = _snap_up(target, 2.5)
    if y == 0: y = 2.5
    ax.set_ylim(-y, y)
    ax.axhline(0, color="black", lw=1.5)

    # ticks/labels
    if scale == "yearly":
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ttl_units = f"{units}/m²" if per_m2 else units
    ax.set_title(f"Heat Balance — {scale.capitalize()} ({str(scope).capitalize()}) [{ttl_units}] {title_suffix}".strip())
    ax.set_xlabel(scale.capitalize()); ax.set_ylabel(f"Total Energy [{ttl_units}]")
    ax.grid(True, linestyle="--", alpha=0.5)

    # legend (variables only; order = sorted cols)
    handles = [Patch(facecolor=color_of(v), label=v) for v in cols]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    plt.tight_layout(rect=[0, 0, 0.78, 1])
    plt.show()

def _to_month_period(x):
    import pandas as pd
    if isinstance(x, pd.Period):
        return x.asfreq("M")
    return pd.Period(pd.to_datetime(str(x)).strftime("%Y-%m-01"), freq="M")

def plot_month_compare_across_building_and_floors(
    aggs,
    long_df,
    floors,
    months=None,                  # accepts None, list of Period/Timestamp/str
    units="kWh",
    per_m2=True,
    fixed_color_map=None,         # or global_color_map; keys = CLEANED names
    strict_colors=True,
    title="Heat Balance — Building & Floors (Monthly)",
    floor_labeler=None            # e.g., lambda f: f"{int(f)}F" if f>=0 else f"B{abs(int(f))}F"
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if fixed_color_map is None:
        raise ValueError("Provide fixed_color_map/global_color_map with cleaned variable names.")

    # --- normalize Month to Period('M') everywhere ---
    if "Month" in long_df.columns and not isinstance(long_df["Month"].dtype, pd.PeriodDtype):
        long_df = long_df.copy()
        long_df["Month"] = long_df["Month"].apply(_to_month_period)

    # derive months from data if not supplied
    if months is None:
        months = list(pd.Index(long_df["Month"].dropna().unique()).sort_values())
    else:
        months = [_to_month_period(m) for m in months]

    # default floor labeler
    if floor_labeler is None:
        def floor_labeler(f):
            f = int(f)
            return f"{f}F" if f >= 0 else f"B{abs(f)}F"

    # collect building + floors
    frames = []
    b = get_df(aggs, scale="monthly", scope="building", units=units, per_m2=per_m2, long_df_for_area=long_df)
    if "Month" in b and not isinstance(b["Month"].dtype, pd.PeriodDtype):
        b["Month"] = b["Month"].apply(_to_month_period)
    b["Entity"] = "Building"
    frames.append(b)

    for f in floors:
        df_f = get_df(aggs, scale="monthly", scope="floor", floor=f, units=units, per_m2=per_m2, long_df_for_area=long_df)
        if df_f.empty:
            continue
        if "Month" in df_f and not isinstance(df_f["Month"].dtype, pd.PeriodDtype):
            df_f["Month"] = df_f["Month"].apply(_to_month_period)
        df_f["Entity"] = floor_labeler(f)
        frames.append(df_f)

    if not frames:
        print("No data to plot.")
        return

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df[all_df["Month"].isin(months)]

    # choose value column
    vcol = "Value_per_m2" if per_m2 and "Value_per_m2" in all_df.columns else "Value"

    # pivot to compute common symmetric y
    piv_all = (all_df.pivot_table(index=["Month","Entity"], columns="Variable", values=vcol, aggfunc="sum")
                      .fillna(0.0))
    # clean variable names to match your color dict keys
    piv_all.columns = [clean_var_name(c) for c in piv_all.columns]

    pos_top = piv_all.clip(lower=0).sum(axis=1).max()
    neg_bot = piv_all.clip(upper=0).sum(axis=1).min()
    from math import ceil
    step = 2.5
    ymax = step if (pos_top==neg_bot==0) else ceil(max(abs(pos_top), abs(neg_bot))/step)*step

    # layout
    ncol = min(4, max(1, len(months)))
    nrow = int(np.ceil(len(months)/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6*ncol, 4.8*nrow), squeeze=False)
    ttl_units = f"{units}/m²" if per_m2 else units
    fig.suptitle(f"{title} [{ttl_units}]", fontsize=18, y=0.98)

    # plot each month
    for i, m in enumerate(months):
        ax = axes[i//ncol][i % ncol]
        if m not in piv_all.index.get_level_values(0):
            ax.set_visible(False); continue

        sub = piv_all.xs(m, level=0)  # DataFrame indexed by Entity
        # Building first, then floors in alphanumeric order
        sub = sub.reindex(["Building"] + sorted([e for e in sub.index if e!="Building"]))

        cols = sorted(sub.columns)  # deterministic legend order

        # color lookup
        missing = [c for c in cols if c not in fixed_color_map]
        if missing and strict_colors:
            raise KeyError(f"Missing colors for variables: {missing}")
        color_of = lambda name: fixed_color_map.get(name, "#bdbdbd")

        # stacks
        pos = sub.clip(lower=0)
        neg = sub.clip(upper=0)
        x = np.arange(len(sub.index)); width = 0.62

        bottom = np.zeros(len(sub))
        for v in cols:
            vals = pos[v].values
            if np.any(vals != 0): ax.bar(x, vals, width, bottom=bottom, color=color_of(v))
            bottom += vals
        bottom = np.zeros(len(sub))
        for v in cols:
            vals = neg[v].values
            if np.any(vals != 0): ax.bar(x, vals, width, bottom=bottom, color=color_of(v))
            bottom += vals

        ax.axhline(0, color="black", lw=1)
        ax.set_ylim(-ymax, ymax)
        ax.set_xticks(x, sub.index, rotation=0)
        ax.set_title(str(m)); ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylabel(ttl_units)

    # one legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=fixed_color_map.get(c, "#bdbdbd"), label=c) for c in sorted(piv_all.columns)]
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(0.99, 0.5), frameon=False)
    plt.tight_layout(rect=[0,0,0.92,0.95])
    plt.show()

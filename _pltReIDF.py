import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch                  # NEW: for proxy legend handles
from matplotlib.lines import Line2D                   # NEW: for net line in legend


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


def load_long(input_csv: str, assume_year: int = 2021, areas_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Return long-format df with columns:
    Date/Time | Zone | Variable | Value | Year | Month | ZoneType | Area_m2 (if supplied)
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

    if areas_df is not None and {"Zone","Area_m2"}.issubset(areas_df.columns):
        long_df = long_df.merge(areas_df[["Zone","Area_m2"]], on="Zone", how="left")

    return long_df


# ---------------------- Aggregations ----------------------

def _convert_units(series_wh: pd.Series, units: str) -> pd.Series:
    return series_wh / 1000.0 if str(units).lower() == "kwh" else series_wh


def compute_aggs(long_df: pd.DataFrame) -> dict:
    """Precompute all standard aggregations (in Wh); convert later when fetching."""
    out = {}
    out["monthly_zone_variable_sum"]     = long_df.groupby(["Month","Zone","Variable"], as_index=False)["Value"].sum()
    out["yearly_zone_variable_sum"]      = long_df.groupby(["Year","Zone","Variable"], as_index=False)["Value"].sum()
    out["monthly_zonetype_variable_sum"] = long_df.groupby(["Month","ZoneType","Variable"], as_index=False)["Value"].sum()
    out["yearly_zonetype_variable_sum"]  = long_df.groupby(["Year","ZoneType","Variable"], as_index=False)["Value"].sum()
    out["monthly_building_variable_sum"] = long_df.groupby(["Month","Variable"], as_index=False)["Value"].sum()
    out["yearly_building_variable_sum"]  = long_df.groupby(["Year","Variable"], as_index=False)["Value"].sum()
    return out


def get_df(
    aggs: dict,
    scale: str = "monthly",           # "monthly" | "yearly"
    scope: str = "building",          # "building" | "zonetype" | "zone"
    zone: str | None = None,
    zonetype: str | None = None,
    variables: list[str] | None = None,
    units: str = "kWh",               # "Wh" | "kWh"
    per_m2: bool = False,
    total_area: float | None = None,  # used for building-level per m²
    long_df_for_area: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Fetch an aggregated dataframe at the requested scale/scope, with optional filtering,
    units conversion, and per-m² normalization.
    Returns a tidy dataframe keyed by Month/Year and Variable with Value (and Value_per_m2 if requested).
    """
    scale = scale.lower()
    scope = scope.lower()

    # Select the base aggregated table in Wh
    key_map = {
        ("monthly","building"): "monthly_building_variable_sum",
        ("monthly","zonetype"): "monthly_zonetype_variable_sum",
        ("monthly","zone"):     "monthly_zone_variable_sum",
        ("yearly","building"):  "yearly_building_variable_sum",
        ("yearly","zonetype"):  "yearly_zonetype_variable_sum",
        ("yearly","zone"):      "yearly_zone_variable_sum",
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

    # Filter variables if provided
    if variables:
        df = df[df["Variable"].isin(variables)]

    # Units conversion (from Wh to desired)
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

    return df


# ---------------------- Legend coverage helper (NEW) ----------------------

def check_legend_variables(
    df: pd.DataFrame,
    scale: str = "monthly",
    per_m2: bool = False
) -> dict:
    """
    Returns which variables *should* appear in the legend based on the data.
    Useful to confirm coverage when many components exist.
    """
    scale = scale.lower()
    time_key = "Month" if scale == "monthly" else "Year"
    value_col = "Value_per_m2" if (per_m2 and "Value_per_m2" in df.columns) else "Value"

    tmp = df.copy()
    if time_key == "Month":
        # ensure a sortable datetime index if needed
        tmp[time_key] = pd.to_datetime(tmp[time_key].astype(str) + "-01", errors="coerce")

    piv = tmp.pivot_table(index=time_key, columns="Variable", values=value_col, aggfunc="sum").fillna(0)
    nonzero = piv.loc[:, (piv != 0).any(axis=0)]
    expected_vars = list(nonzero.columns)

    return {
        "n_expected_in_legend": len(expected_vars),
        "expected_variables": expected_vars
    }
def plot_heat_balance(
    df,
    scale="monthly",           # "monthly" | "yearly"
    scope="building",
    units="kWh",
    per_m2=False,
    title_suffix="",
    show_net=True,
    ylim_abs=None,             # NEW: symmetric y-limit (e.g. 10 -> (-10, 10))
    show_tick_each_bar=True
):
    """
    Heat-balance stacked bar chart with:
    - Reds = gains (positive-only)
    - Blues = losses (negative-only)
    - Purples = bidirectional (both + and -)
    - Legend outside plot area
    - Symmetrical y-limits if `ylim_abs` is given (e.g. ylim_abs=10 -> (-10,10))
    - Yearly x-axis: YYYY, Monthly x-axis: tick per bar
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm, patches, lines
    import matplotlib.dates as mdates

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

    pivot_df = df.pivot_table(index=time_key, columns="Variable", values=value_col, aggfunc="sum").fillna(0)
    pivot_df = pivot_df.sort_index()
    net = pivot_df.sum(axis=1)

    # --- Classify variables ---
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

    gains = [v for v, t in var_state.items() if t == "gain"]
    losses = [v for v, t in var_state.items() if t == "loss"]
    bidir  = [v for v, t in var_state.items() if t == "bidir"]

    # --- Color maps ---
    reds    = cm.get_cmap("Reds",    len(gains) + 2)
    blues   = cm.get_cmap("Blues",   len(losses) + 2)
    purples = cm.get_cmap("Purples", len(bidir)  + 2)

    color_map = {}
    for i, v in enumerate(gains):  color_map[v] = reds(i + 1)
    for i, v in enumerate(losses): color_map[v] = blues(i + 1)
    for i, v in enumerate(bidir):  color_map[v] = purples(i + 1)

    # --- Clean variable names ---
    def clean_name(v):
        v = v.replace("Zone Air Heat Balance ", "")
        v = v.replace(" Rate [W](Hourly)", "")
        return v.strip()

    rename_map = {v: clean_name(v) for v in pivot_df.columns}
    pivot_df.rename(columns=rename_map, inplace=True)
    color_map = {rename_map[k]: v for k, v in color_map.items()}

    pos_df = pivot_df.clip(lower=0)
    neg_df = pivot_df.clip(upper=0)

    fig, ax = plt.subplots(figsize=(14, 8))

    # --- Draw order: gains → bidir (top); bidir → losses (bottom) ---
    bottom = np.zeros(len(pos_df))
    for v in [rename_map[x] for x in gains]:
        if np.any(pos_df[v] != 0):
            ax.bar(pos_df.index, pos_df[v], width=bar_width, bottom=bottom,
                   color=color_map[v], label=v)
            bottom += pos_df[v]
    for v in [rename_map[x] for x in bidir]:
        if np.any(pos_df[v] != 0):
            ax.bar(pos_df.index, pos_df[v], width=bar_width, bottom=bottom,
                   color=color_map[v], label=v)
            bottom += pos_df[v]

    bottom = np.zeros(len(neg_df))
    for v in [rename_map[x] for x in bidir]:
        if np.any(neg_df[v] != 0):
            ax.bar(neg_df.index, neg_df[v], width=bar_width, bottom=bottom,
                   color=color_map[v], label=v)
            bottom += neg_df[v]
    for v in [rename_map[x] for x in losses]:
        if np.any(neg_df[v] != 0):
            ax.bar(neg_df.index, neg_df[v], width=bar_width, bottom=bottom,
                   color=color_map[v], label=v)
            bottom += neg_df[v]

    # --- Net line ---
    net_handle = None
    if show_net:
        (net_line,) = ax.plot(pivot_df.index, net, color='black', linewidth=3)
        from matplotlib.lines import Line2D
        net_handle = Line2D([0], [0], color='black', lw=3, label="Net Total")

    # --- Axes styling ---
    ax.axhline(0, color="black", linewidth=1.5)
    ttl_units = f"{units}/m²" if per_m2 else units
    title_core = f"Heat Balance — {scale.capitalize()} ({scope.capitalize()})"
    ax.set_title(f"{title_core} [{ttl_units}] {title_suffix}".strip())
    ax.set_xlabel(scale.capitalize())
    ax.set_ylabel(f"Total Energy [{ttl_units}]")
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- Symmetrical y-limit ---
    if ylim_abs is not None:
        ax.set_ylim(-abs(ylim_abs), abs(ylim_abs))

    # --- X-axis ticks ---
    if scale == "yearly":
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # --- Legend outside ---
    from matplotlib.patches import Patch
    vars_list = list(pivot_df.columns)
    var_handles = [Patch(facecolor=color_map[v], label=v) for v in vars_list]
    if net_handle:
        var_handles.append(net_handle)

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

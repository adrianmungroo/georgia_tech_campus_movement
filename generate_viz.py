"""
generate_viz.py
Reads edge_time_series.parquet + network_files/walk_edges_clean.geojson
and renders a fully self-contained index.html visualization.

Usage:
    python3 generate_viz.py
"""

import base64
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ paths
HERE        = Path(__file__).parent
PARQUET         = HERE / "edge_time_series.parquet"
PARQUET_DENSITY = HERE / "edge_time_series_density.parquet"
GEOJSON     = HERE / "network_files" / "walk_edges_clean.geojson"
TEMPLATE    = HERE / "template.html"
OUTPUT      = HERE / "index.html"

COORD_PREC  = 5   # decimal places for coordinates (~1 m precision)
TIMEZONE    = "US/Eastern"


# ------------------------------------------------------------------ loaders
def load_data():
    print("  loading parquet…")
    ts = pd.read_parquet(PARQUET)
    print(f"    shape: {ts.shape}  cols: {ts.columns[:3].tolist()}…")

    print("  loading GeoJSON…")
    edges = gpd.read_file(GEOJSON)
    print(f"    {len(edges)} edges loaded")

    if PARQUET_DENSITY.is_file():
        ts_d = pd.read_parquet(PARQUET_DENSITY).reindex(index=ts.index, columns=ts.columns, fill_value=0)
        print(f"    density parquet shape: {ts_d.shape}")
    else:
        print("    (no edge_time_series_density.parquet — deriving from counts ÷ length_ft)")
        oid_len = {str(int(r["OBJECTID"])): float(r["length_ft"]) for _, r in edges.iterrows()}
        L = ts.columns.map(lambda c: oid_len.get(c, 1.0)).astype(float)
        L = np.maximum(L, 1e-6)
        ts_d = ts.div(L, axis=1).astype(np.float32)

    return ts, ts_d, edges


# ------------------------------------------------------------------ geometry
def build_geo_json(edges, oids_sorted):
    """
    Returns a compact JSON array index-aligned to oids_sorted.
    Each element: {"oid": int, "c": [[lon, lat], ...]}
    """
    oid_to_coords = {}
    for _, row in edges.iterrows():
        oid = str(int(row["OBJECTID"]))
        coords = [
            [round(x, COORD_PREC), round(y, COORD_PREC)]
            for x, y in row.geometry.coords
        ]
        oid_to_coords[oid] = coords

    result = []
    for oid in oids_sorted:
        result.append({
            "oid": int(oid),
            "c":   oid_to_coords.get(oid, [])
        })

    return json.dumps(result, separators=(",", ":"))


# ------------------------------------------------------------------ matrix
def build_ts_b64(ts, oids_sorted):
    """
    Reorders columns to match oids_sorted, casts to Int16, base64-encodes.
    Returns (b64_string, global_max).
    """
    ts_sorted = ts[oids_sorted]
    global_max = int(ts_sorted.values.max())

    arr = ts_sorted.values.astype(np.int16)  # (n_steps, N_EDGES)
    b64 = base64.b64encode(arr.tobytes()).decode("ascii")
    return b64, global_max


def build_ts_float_b64(ts, oids_sorted):
    """Float32 row-major matrix for density values."""
    ts_sorted = ts[oids_sorted]
    gmax = float(ts_sorted.values.max())
    arr = ts_sorted.values.astype(np.float32)
    b64 = base64.b64encode(arr.tobytes()).decode("ascii")
    return b64, gmax


def build_edge_lengths_b64(edges, oids_sorted):
    oid_len = {str(int(r["OBJECTID"])): float(r["length_ft"]) for _, r in edges.iterrows()}
    arr = np.array([oid_len.get(oid, 1.0) for oid in oids_sorted], dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode("ascii")


# ------------------------------------------------------------------ labels
def build_labels(ts):
    """
    Converts UTC index to US/Eastern, returns JSON array of formatted strings.
    Example: ["Sun Apr 13, 8:00 PM", ...]
    """
    ts_edt = ts.index.tz_convert(TIMEZONE)
    labels = [dt.strftime("%-a %-b %-d, %-I:%M %p") for dt in ts_edt]
    return json.dumps(labels, separators=(",", ":"))


# ------------------------------------------------------------------ colormaps (dark→bright: hot, viridis, inferno, plasma)
CMAP_DEFS = {
    "hot":     "hot",
    "viridis": "viridis",
    "inferno": "inferno",
    "plasma":  "plasma",
    "reds":    "Reds",
    "ylord":   "YlOrRd",
}


def build_all_cmaps():
    """
    Returns JSON object mapping cmap key → 20-stop [[r,g,b], ...] table.
    Default active cmap: hot (dark→bright).
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps
    stops  = 20
    result = {}
    for key, mpl_name in CMAP_DEFS.items():
        cmap  = colormaps[mpl_name]
        table = []
        for i in range(stops):
            t = i / (stops - 1)
            r, g, b, _ = cmap(t)
            table.append([round(r * 255), round(g * 255), round(b * 255)])
        result[key] = table
    return json.dumps(result, separators=(",", ":"))


# ------------------------------------------------------------------ campus totals
def build_campus_totals(ts):
    """
    Sum all edges per timestep → JSON array of ints.
    """
    totals = ts.values.sum(axis=1).astype(int).tolist()
    return json.dumps(totals, separators=(",", ":"))


def build_campus_totals_density(ts, edges):
    """Total traversals / total network length (ft) per timestep — comparable campus intensity."""
    total_len = float(edges["length_ft"].sum())
    if total_len <= 0:
        total_len = 1.0
    totals = (ts.values.sum(axis=1) / total_len).astype(float).tolist()
    return json.dumps(totals, separators=(",", ":"))


# ------------------------------------------------------------------ render
def render(template_str, replacements):
    html = template_str
    for token, value in replacements.items():
        html = html.replace(token, value)
    return html


# ------------------------------------------------------------------ main
def main():
    print("=== generate_viz.py ===")

    ts, ts_density, edges = load_data()

    # Sorted OBJECTID strings — defines column order throughout
    oids_sorted = sorted(ts.columns, key=lambda x: int(x))
    n_edges = len(oids_sorted)
    n_steps = len(ts)

    print("  building geometry JSON…")
    geo_json = build_geo_json(edges, oids_sorted)

    print("  encoding time series (Int16 + base64)…")
    ts_b64, global_max = build_ts_b64(ts, oids_sorted)

    print("  encoding density matrix (Float32 + base64)…")
    ts_d_b64, global_max_d = build_ts_float_b64(ts_density, oids_sorted)

    print("  encoding edge lengths (Float32)…")
    lens_b64 = build_edge_lengths_b64(edges, oids_sorted)

    print("  building timestamp labels…")
    ts_labels = build_labels(ts)

    print("  building colormaps…")
    cmaps_json = build_all_cmaps()

    print("  building campus totals…")
    campus_totals = build_campus_totals(ts)
    campus_totals_d = build_campus_totals_density(ts, edges)

    print("  computing percentiles…")
    all_vals = ts.values.flatten()
    p1, p99 = int(np.percentile(all_vals, 1)), int(np.percentile(all_vals, 99))
    all_d = ts_density.values.flatten()
    p1d, p99d = float(np.percentile(all_d, 1)), float(np.percentile(all_d, 99))

    print("  reading template…")
    template_str = TEMPLATE.read_text(encoding="utf-8")

    replacements = {
        "__GEO_JSON__":              geo_json,
        "__TS_B64__":                ts_b64,
        "__TS_B64_DENSITY__":        ts_d_b64,
        "__EDGE_LENGTHS_B64__":       lens_b64,
        "__TS_LABELS__":             ts_labels,
        "__CMAPS__":                 cmaps_json,
        "__CAMPUS_TOTALS__":         campus_totals,
        "__CAMPUS_TOTALS_DENSITY__": campus_totals_d,
        "__N_EDGES__":               str(n_edges),
        "__N_STEPS__":               str(n_steps),
        "__N_STEPS_MINUS1__":        str(n_steps - 1),
        "__GLOBAL_MAX__":            str(global_max),
        "__P1__":                    str(p1),
        "__P99__":                   str(p99),
        "__GLOBAL_MAX_D__":          repr(global_max_d),
        "__P1_D__":                  repr(p1d),
        "__P99_D__":                 repr(p99d),
    }

    print("  rendering HTML…")
    html = render(template_str, replacements)

    OUTPUT.write_text(html, encoding="utf-8")
    size_kb = len(html.encode("utf-8")) / 1024
    print(f"\nWrote: {OUTPUT}  ({size_kb:.0f} KB)")
    print("\nOpen index.html in a browser to view the visualization.")


if __name__ == "__main__":
    main()

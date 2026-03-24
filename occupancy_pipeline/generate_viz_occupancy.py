"""
Reads edge_occupancy_series.parquet + parent's walk_edges_clean.geojson, writes index_occupancy.html.

Uses ../template.html in-memory with patches (Int32 matrix + occupancy copy). Does not modify
template.html, generate_viz.py, or index.html.

Usage (from this folder):
    python3 simulate_network_occupancy.py   # if parquet missing
    python3 generate_viz_occupancy.py
"""

import base64
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE     = Path(__file__).resolve().parent
BASE     = HERE.parent
PARQUET  = HERE / "edge_occupancy_series.parquet"
GEOJSON  = BASE / "network_files" / "walk_edges_clean.geojson"
TEMPLATE = BASE / "template.html"
OUTPUT   = HERE / "index_occupancy.html"

COORD_PREC  = 5
TIMEZONE    = "US/Eastern"

TEMPLATE_PATCHES = [
    (
        "<title>GT Campus Pedestrian Flow</title>",
        "<title>GT Campus Pedestrian Occupancy</title>",
    ),
    (
        "// DECODE MATRIX  (Int16, row-major, 288 × 1493)",
        "// DECODE MATRIX  (Int32, row-major, steps × edges) — occupancy viz",
    ),
    ("return new Int16Array(buf);", "return new Int32Array(buf);"),
    (
        "<h1>GT Campus<br>Pedestrian Flow</h1>",
        "<h1>GT Campus<br>Pedestrian Occupancy</h1>",
    ),
    (
        '<div class="subtitle">1,493 edges &middot; 288 time steps<br>10-minute bins &middot; Apr 13–15 2025</div>',
        '<div class="subtitle">1,493 edges &middot; 288 time steps<br>Mid-bin snapshot occupancy &middot; Apr 13–15 2025</div>',
    ),
    (
        '<span class="stat-label">Edge traversals</span>',
        '<span class="stat-label">People on edges</span>',
    ),
    (
        'title="Average traversals per active edge (flow concentration)"',
        'title="Average people on edge among active edges"',
    ),
    ('<div class="legend-title">Traversals / 10 min</div>',
     '<div class="legend-title">People on edge (snapshot)</div>'),
    (
        '<span class="ts-stat-label">48 hr traversals</span>',
        '<span class="ts-stat-label">48 hr sum (snapshots)</span>',
    ),
    (
        "`${count} traversal${count !== 1 ? 's' : ''}`",
        "`${count} person${count !== 1 ? 's' : ''}`",
    ),
]


def load_data():
    print("  loading parquet…")
    ts = pd.read_parquet(PARQUET)
    print(f"    shape: {ts.shape}")
    print("  loading GeoJSON…")
    edges = gpd.read_file(GEOJSON)
    print(f"    {len(edges)} edges loaded")
    return ts, edges


def build_geo_json(edges, oids_sorted):
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


def build_ts_b64(ts, oids_sorted):
    ts_sorted  = ts[oids_sorted]
    global_max = int(ts_sorted.values.max())
    arr        = ts_sorted.values.astype(np.int32)
    raw_bytes  = arr.tobytes()
    b64        = base64.b64encode(raw_bytes).decode("ascii")
    return b64, global_max


def build_labels(ts):
    ts_edt = ts.index.tz_convert(TIMEZONE)
    labels = [dt.strftime("%-a %-b %-d, %-I:%M %p") for dt in ts_edt]
    return json.dumps(labels, separators=(",", ":"))


def build_all_cmaps():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps
    stops  = 20
    result = {}
    for key, mpl_name in {
        "hot": "hot", "viridis": "viridis", "inferno": "inferno",
        "plasma": "plasma", "reds": "Reds", "ylord": "YlOrRd",
    }.items():
        cmap  = colormaps[mpl_name]
        table = []
        for i in range(stops):
            t = i / (stops - 1)
            r, g, b, _ = cmap(t)
            table.append([round(r * 255), round(g * 255), round(b * 255)])
        result[key] = table
    return json.dumps(result, separators=(",", ":"))


def build_campus_totals(ts):
    totals = ts.values.sum(axis=1).astype(int).tolist()
    return json.dumps(totals, separators=(",", ":"))


def apply_occupancy_patches(html: str) -> str:
    for old, new in TEMPLATE_PATCHES:
        if old not in html:
            raise ValueError(f"template.html missing expected fragment; update patches: {old[:60]}…")
        html = html.replace(old, new, 1)
    return html


def main():
    print("=== generate_viz_occupancy.py ===")
    ts, edges = load_data()
    oids_sorted = sorted(ts.columns, key=lambda x: int(x))
    n_edges     = len(oids_sorted)
    n_steps     = len(ts)

    geo_json    = build_geo_json(edges, oids_sorted)
    ts_b64, global_max = build_ts_b64(ts, oids_sorted)
    ts_labels   = build_labels(ts)
    cmaps_json  = build_all_cmaps()
    campus_totals = build_campus_totals(ts)

    all_vals = ts.values.flatten()
    p1, p99  = int(np.percentile(all_vals, 1)), int(np.percentile(all_vals, 99))

    template_str = apply_occupancy_patches(TEMPLATE.read_text(encoding="utf-8"))

    for token, value in {
        "__GEO_JSON__":        geo_json,
        "__TS_B64__":          ts_b64,
        "__TS_LABELS__":       ts_labels,
        "__CMAPS__":           cmaps_json,
        "__CAMPUS_TOTALS__":   campus_totals,
        "__N_EDGES__":         str(n_edges),
        "__N_STEPS__":         str(n_steps),
        "__N_STEPS_MINUS1__":  str(n_steps - 1),
        "__GLOBAL_MAX__":      str(global_max),
        "__P1__":              str(p1),
        "__P99__":             str(p99),
    }.items():
        template_str = template_str.replace(token, value)

    OUTPUT.write_text(template_str, encoding="utf-8")
    kb = len(template_str.encode("utf-8")) / 1024
    print(f"\nWrote: {OUTPUT}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()

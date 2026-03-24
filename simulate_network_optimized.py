"""
Simulate pedestrian traversals → edge_time_series.parquet + edge_time_series_density.parquet (10-min bins).
Density = traversals per foot of edge length (per bin).

Run from any cwd:
    python3 simulate_network_optimized.py
"""
from __future__ import annotations

import geopandas as gpd
import math
import pickle
import warnings
from itertools import islice
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TIME_BIN   = 10   # minutes
WALK_SPEED = 4.6  # ft/s (~3.1 mph) — edge timing in simulation
MAX_WALK_SPEED = 5.0  # ft/s (~3.4 mph) — walkers only; skip faster implied speeds
BIN_NS     = int(TIME_BIN * 60 * 1e9)
MAX_PATH_OPTIONS = 2
TARGET_BEST_PATH_PROB = 0.90
RNG_SEED = 42


def tobler_fps(slope):
    """Tobler's hiking function: walking speed in ft/s as a function of slope (rise/run)."""
    mph = 6 * math.exp(-3.5 * abs(slope + 0.05))
    return mph * 5280 / 3600


def calibrate_theta(costs, target_prob=TARGET_BEST_PATH_PROB, max_theta=200.0):
    costs = np.array(costs, dtype=float)
    deltas = costs - costs.min()
    if len(deltas) <= 1 or np.allclose(deltas, 0.0):
        return 0.0

    def p_best(theta):
        w = np.exp(-theta * deltas)
        return float(w[0] / w.sum())

    if p_best(max_theta) < target_prob:
        return max_theta

    lo, hi = 0.0, max_theta
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if p_best(mid) < target_prob:
            lo = mid
        else:
            hi = mid
    return hi


def main() -> None:
    BASE = Path(__file__).resolve().parent
    out_parquet = BASE / "edge_time_series.parquet"
    out_density_parquet = BASE / "edge_time_series_density.parquet"

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(f"simulate_network_optimized — BASE={BASE}")

    events_path = BASE / "simon/outputs/events/movement_events.parquet"
    if not events_path.is_file():
        raise FileNotFoundError(
            f"Missing movement events:\n  {events_path}\n"
            "Build or copy movement_events.parquet into simon/outputs/events/ first."
        )

    movement_events = pd.read_parquet(events_path)
    log(f"  loaded {len(movement_events):,} movement events")

    pkl_path = BASE / "network_files/bldg_nodes_dict.pkl"
    with open(pkl_path, "rb") as f:
        bldg_nodes_dict = pickle.load(f)
    log(f"  loaded bldg_nodes_dict ({len(bldg_nodes_dict)} buildings)")

    edges = gpd.read_file(BASE / "network_files/walk_edges_clean.geojson")
    log(f"  loaded {len(edges)} edges")

    rng = np.random.default_rng(RNG_SEED)

    G = nx.Graph()
    for _, r in edges.iterrows():
        cost = float(r["total_length"]) / tobler_fps(r["slope"])
        G.add_edge(r["from_id"], r["to_id"], weight=cost)

    edge_lookup: dict[tuple[int, int], tuple[int, int]] = {}
    for _, r in edges.iterrows():
        val = (int(r["length_ft"] / WALK_SPEED * 1e9), int(r["OBJECTID"]))
        edge_lookup[(r["from_id"], r["to_id"])] = val
        edge_lookup[(r["to_id"], r["from_id"])] = val

    def compute_path(id_a, id_b):
        A = bldg_nodes_dict[id_a]
        B = bldg_nodes_dict[id_b]
        lengths, paths = nx.multi_source_dijkstra(G, sources=A, weight="weight")
        reachable = {b: lengths[b] for b in B if b in lengths}
        if not reachable:
            return None
        return paths[min(reachable, key=reachable.get)]

    def path_to_edge_data(path):
        if path is None or len(path) < 2:
            return None
        data = []
        for i in range(len(path) - 2, -1, -1):
            k = (path[i], path[i + 1])
            if k in edge_lookup:
                data.append(edge_lookup[k])
        return data if data else None

    def path_cost(path):
        return sum(G[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1))

    def build_route_choice(path):
        if path is None or len(path) < 2:
            return None

        source, target = path[0], path[-1]
        try:
            candidates = list(
                islice(nx.shortest_simple_paths(G, source, target, weight="weight"), MAX_PATH_OPTIONS)
            )
        except Exception:
            candidates = [path]

        if not candidates:
            return None

        edge_options = []
        costs = []
        for p in candidates:
            edge_data = path_to_edge_data(p)
            if edge_data:
                edge_options.append(edge_data)
                costs.append(path_cost(p))

        if not edge_options:
            return None

        if len(edge_options) == 1:
            return edge_options, np.array([1.0]), 0.0

        theta = calibrate_theta(costs, TARGET_BEST_PATH_PROB)
        deltas = np.array(costs, dtype=float) - min(costs)
        probs = np.exp(-theta * deltas)
        probs /= probs.sum()
        cum_probs = np.cumsum(probs)
        cum_probs[-1] = 1.0
        return edge_options, cum_probs, theta

    log("Pre-computing unique OD paths...")
    unique_od = movement_events[["l1", "l2"]].drop_duplicates()
    path_cache = {}
    key_errors = 0
    for i, (_, row) in enumerate(unique_od.iterrows()):
        key = (row["l1"], row["l2"])
        try:
            path_cache[key] = compute_path(row["l1"], row["l2"])
        except KeyError:
            key_errors += 1
            path_cache[key] = None
        except Exception:
            path_cache[key] = None
        if i % 1000 == 0:
            log(f"  {i}/{len(unique_od)} OD pairs computed")

    n_paths_ok = sum(1 for v in path_cache.values() if v is not None)
    n_degenerate = sum(1 for v in path_cache.values() if v is not None and len(v) < 2)
    log(f"Done. {len(path_cache)} OD keys | {n_paths_ok} with a route | {key_errors} KeyError (missing bldg in dict)")
    if n_paths_ok == 0:
        raise RuntimeError(
            "No OD pairs got a shortest path. Check walk_edges_clean.geojson connectivity and "
            "that bldg_nodes_dict.pkl keys match movement_events l1/l2 building codes."
        )
    if n_degenerate > n_paths_ok * 0.5:
        raise RuntimeError(
            f"{n_degenerate}/{n_paths_ok} shortest paths are a single node (O and D share that node). "
            "Usually bldg_nodes_dict.pkl was built with buffer(10) in EPSG:4326 (10° ≈ 1100 km), so every "
            "entrance node matched every building. Fix: run `python3 rebuild_bldg_nodes_dict.py` from this "
            "folder (or re-run associate_nodes_with_bldgs.ipynb with EPSG:2240 buffer)."
        )

    log("Building logit route-choice cache...")
    route_choice_cache = {}
    theta_values = []
    multi_route_count = 0
    for key, path in path_cache.items():
        choice = build_route_choice(path)
        route_choice_cache[key] = choice
        if choice and len(choice[0]) > 1:
            multi_route_count += 1
            theta_values.append(choice[2])
    log("Done.")
    if theta_values:
        log(
            f"Multi-route ODs: {multi_route_count}/{len(route_choice_cache)} | "
            f"mean theta={np.mean(theta_values):.4f}"
        )
    else:
        log("No OD pairs had multiple route options under current settings.")

    min_time   = movement_events["t1"].min()
    max_time   = movement_events["t2"].max()
    time_index = pd.date_range(start=min_time, end=max_time, freq=f"{TIME_BIN}min")

    log("Simulating pedestrian movement...")
    t_ns_list: list[int] = []
    oid_list: list[int] = []
    skipped_fast = 0
    skipped_no_route = 0
    for num, event in enumerate(movement_events.itertuples(index=False)):
        choice = route_choice_cache.get((event.l1, event.l2))
        if not choice:
            skipped_no_route += 1
            continue
        edge_options, cum_probs, _ = choice
        route_idx = int(np.searchsorted(cum_probs, rng.random(), side="right"))
        edge_data = edge_options[route_idx]
        dt_sec = float(event.dt_sec)
        len_path_ft = sum(w * WALK_SPEED / 1e9 for w, _ in edge_data)
        if dt_sec <= 0 or len_path_ft / dt_sec > MAX_WALK_SPEED:
            skipped_fast += 1
            continue
        t_ns = event.t2.value
        for walk_ns, oid in edge_data:
            t_ns -= walk_ns
            t_ns_list.append(t_ns)
            oid_list.append(oid)
        if num % 50000 == 0:
            log(f"  {num}/{len(movement_events)} events processed")

    log(
        f"Skipped {skipped_no_route:,} events (no route) | "
        f"{skipped_fast:,} non-walking (implied speed > {MAX_WALK_SPEED} ft/s)."
    )

    if not t_ns_list:
        raise RuntimeError(
            "Recorded zero edge traversals — edge_time_series would be all zeros.\n"
            "Common causes: (1) l1/l2 in movement_events not in bldg_nodes_dict.pkl — "
            "re-run associate_nodes_with_bldgs.ipynb; (2) network disconnected in ArcGIS — "
            "check walk_edges_clean.geojson; (3) every event filtered as too fast — "
            "check dt_sec vs path length or raise MAX_WALK_SPEED."
        )

    log("Building edge_time_series matrix...")
    t_ns_arr = np.array(t_ns_list, dtype="int64")
    oid_arr  = np.array(oid_list,  dtype="int64")
    bin_ns = t_ns_arr - (t_ns_arr % BIN_NS)
    tz        = min_time.tz
    time_bins = pd.to_datetime(bin_ns, utc=True)
    time_bins = time_bins.tz_convert(tz) if tz else time_bins.tz_localize(None)

    records_df = pd.DataFrame({"time_bin": time_bins, "OBJECTID": oid_arr})
    counts     = records_df.groupby(["time_bin", "OBJECTID"]).size().unstack(fill_value=0)
    counts.columns = counts.columns.astype(str)

    edge_time_series = counts.reindex(
        index=time_index,
        columns=edges["OBJECTID"].astype(str).unique(),
        fill_value=0,
    )

    length_by_oid = {
        str(int(r["OBJECTID"])): float(r["length_ft"]) for _, r in edges.iterrows()
    }
    L = edge_time_series.columns.map(lambda c: length_by_oid.get(c, 1.0)).astype(float)
    L = np.maximum(L, 1e-6)
    edge_time_series_density = edge_time_series.div(L, axis=1).astype(np.float32)

    edge_time_series.to_parquet(out_parquet)
    edge_time_series_density.to_parquet(out_density_parquet)
    log(
        f"Done. Wrote {out_parquet}  shape={edge_time_series.shape}  "
        f"sum={int(edge_time_series.values.sum())}\n"
        f"      {out_density_parquet} (traversals/ft per bin)"
    )


if __name__ == "__main__":
    main()

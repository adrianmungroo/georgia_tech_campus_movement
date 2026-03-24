"""
Standalone occupancy simulator (does not modify edge_time_series pipeline).

For each movement event and each edge along the chosen route, builds the time
interval [t_entry, t_exit) (nanoseconds, UTC). For each 10-minute frame, counts
how many people are on that edge at the **midpoint** of the bin (so short trips
still register within the labeled window).

Writes: ./edge_occupancy_series.parquet (same index/column layout as edge_time_series).

Usage (from this folder):
    python3 simulate_network_occupancy.py
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
import networkx as nx
import numpy as np
import pickle
import math
from itertools import islice
import warnings

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
BASE = HERE.parent

movement_events = pd.read_parquet(BASE / "simon/outputs/events/movement_events.parquet")
with open(BASE / "network_files/bldg_nodes_dict.pkl", "rb") as f:
    bldg_nodes_dict = pickle.load(f)
edges = gpd.read_file(BASE / "network_files/walk_edges_clean.geojson")

TIME_BIN   = 10   # minutes
WALK_SPEED = 4.6  # ft/s (~3.1 mph) — edge timing in simulation
MAX_WALK_SPEED = 5.0  # ft/s (~3.4 mph) — walkers only; skip faster implied speeds
BIN_NS     = int(TIME_BIN * 60 * 1e9)
MAX_PATH_OPTIONS = 2
TARGET_BEST_PATH_PROB = 0.90
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)


def tobler_fps(slope):
    mph = 6 * math.exp(-3.5 * abs(slope + 0.05))
    return mph * 5280 / 3600


G = nx.Graph()
for _, r in edges.iterrows():
    cost = float(r["total_length"]) / tobler_fps(r["slope"])
    G.add_edge(r["from_id"], r["to_id"], weight=cost)

edge_lookup = {}
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


print("Pre-computing unique OD paths...")
unique_od = movement_events[["l1", "l2"]].drop_duplicates()
path_cache = {}
for i, (_, row) in enumerate(unique_od.iterrows()):
    key = (row["l1"], row["l2"])
    try:
        path_cache[key] = compute_path(row["l1"], row["l2"])
    except Exception:
        path_cache[key] = None
    if i % 1000 == 0:
        print(f"  {i}/{len(unique_od)} OD pairs computed")
n_ok = sum(1 for v in path_cache.values() if v is not None)
n_degen = sum(1 for v in path_cache.values() if v is not None and len(v) < 2)
print(f"Done. {len(path_cache)} paths cached.\n")
if n_ok and n_degen > n_ok * 0.5:
    raise RuntimeError(
        f"{n_degen}/{n_ok} OD paths are a single node — bldg_nodes_dict.pkl is likely wrong. "
        "Run `python3 rebuild_bldg_nodes_dict.py` from the parent student_movement_model folder."
    )

print("Building logit route-choice cache...")
route_choice_cache = {}
theta_values = []
multi_route_count = 0
for key, path in path_cache.items():
    choice = build_route_choice(path)
    route_choice_cache[key] = choice
    if choice and len(choice[0]) > 1:
        multi_route_count += 1
        theta_values.append(choice[2])
print("Done.\n")
if theta_values:
    print(
        f"Multi-route ODs: {multi_route_count}/{len(route_choice_cache)} | "
        f"mean theta={np.mean(theta_values):.4f}"
    )
else:
    print("No OD pairs had multiple route options under current settings.")
print()

min_time   = movement_events["t1"].min()
max_time   = movement_events["t2"].max()
time_index = pd.date_range(start=min_time, end=max_time, freq=f"{TIME_BIN}min")
half_bin = pd.Timedelta(minutes=TIME_BIN / 2)
snap_index = time_index + half_bin
snap_ns = snap_index.asi8.astype(np.int64)

edge_cols = edges["OBJECTID"].astype(str).unique()
oid_to_col = {int(oid): j for j, oid in enumerate(edge_cols)}
n_times = len(time_index)
n_edges = len(edge_cols)
occ = np.zeros((n_times, n_edges), dtype=np.int32)

print("Simulating edge occupancy (mid-bin snapshots)...")
skipped_fast = 0
for num, event in enumerate(movement_events.itertuples(index=False)):
    choice = route_choice_cache.get((event.l1, event.l2))
    if not choice:
        continue
    edge_options, cum_probs, _ = choice
    route_idx = int(np.searchsorted(cum_probs, rng.random(), side="right"))
    edge_data = edge_options[route_idx]
    dt_sec = float(event.dt_sec)
    len_path_ft = sum(w * WALK_SPEED / 1e9 for w, _ in edge_data)
    if dt_sec <= 0 or len_path_ft / dt_sec > MAX_WALK_SPEED:
        skipped_fast += 1
        continue
    t_ns = int(event.t2.value)
    for walk_ns, oid in edge_data:
        t_ns -= walk_ns
        t_in = t_ns
        t_out = t_in + walk_ns
        j = oid_to_col.get(oid)
        if j is None:
            continue
        i0 = np.searchsorted(snap_ns, t_in, side="left")
        i1 = np.searchsorted(snap_ns, t_out, side="left")
        if i1 > i0:
            occ[i0:i1, j] += 1
    if num % 50000 == 0:
        print(f"  {num}/{len(movement_events)} events processed")

print(f"Skipped {skipped_fast} non-walking events (implied speed > {MAX_WALK_SPEED} ft/s).")
edge_occupancy_series = pd.DataFrame(
    occ,
    index=time_index,
    columns=edge_cols.astype(str),
)
out_parquet = HERE / "edge_occupancy_series.parquet"
edge_occupancy_series.to_parquet(out_parquet)
print(f"Done. Saved {out_parquet}  shape={edge_occupancy_series.shape}")

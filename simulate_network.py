import geopandas as gpd
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import random

movement_events = pd.read_parquet("/home/shared/cake/sub-projects/student_movement_model/simon/outputs/events/movement_events.parquet")

# load the bldg_nodes_dict which associates building ids with network nodes
with open('./network_files/bldg_nodes_dict.pkl', 'rb') as f:
    bldg_nodes_dict = pickle.load(f)

# load the network of nodes and edges
edges = gpd.read_file("/home/shared/cake/sub-projects/student_movement_model/network_files/walk_edges_clean.geojson")
nodes = gpd.read_file("/home/shared/cake/sub-projects/student_movement_model/network_files/walk_nodes_clean.geojson")

TIME_BIN = 10 # minutes
WALK_SPEED = 4.6  # ft/s (~3.1 mph) — used for back-calculation timing
MAX_WALK_SPEED = 5.0  # ft/s (~3.4 mph) — skip events faster than a brisk walk
MIN_DT = pd.Timedelta(minutes=1)
MAX_DT = pd.Timedelta(minutes=10)

min_time = movement_events["t1"].min()
max_time = movement_events["t2"].max()
time_index = pd.date_range(start=min_time, end=max_time, freq=f"{TIME_BIN}min")

# Build the empty time–edge matrix
edge_time_series = pd.DataFrame(
    0,  # Initialize with zeros
    index=time_index,
    columns=edges["OBJECTID"].unique(),
    dtype=float
)

weight_field = "length_ft"
G = nx.Graph()
for i, r in edges.iterrows():
    G.add_edge(r["from_id"], r["to_id"], weight=float(r[weight_field]), geom=r.geometry)

def calculate_shortest_bldg_path(id_a, id_b, G):
    """
    Calculate the shortest path between two building ids in the georgia tech network. 
    """
    A = bldg_nodes_dict[id_a]
    B = bldg_nodes_dict[id_b]
    lengths, paths = nx.multi_source_dijkstra(G, sources=A, weight="weight")
    reachable_targets = {b: lengths[b] for b in B if b in lengths}
    if reachable_targets:
        # Find the closest target
        closest_b, dist = min(reachable_targets.items(), key=lambda x: x[1])
    else:
        print("No reachable node from A to B.")
    return closest_b, dist, paths[closest_b]

def get_edges_from_path(path, edges):
    gdf = gpd.GeoDataFrame(columns=edges.columns) 

    for i in range(len(path) - 1):
        node_a = path[i]
        node_b = path[i + 1]

        # Find the matching edge (handle either direction if needed)
        edge = edges.loc[
            ((edges["from_id"] == node_a) & (edges["to_id"] == node_b)) |
            ((edges["from_id"] == node_b) & (edges["to_id"] == node_a))
        ]

        # Append the found edge to the GeoDataFrame
        gdf = pd.concat([gdf, edge], ignore_index=True)

        # reverse the gdf rows so we have the final edges first
        gdf_reversed = gdf.iloc[::-1] 
    return gdf_reversed

def back_calculate_time_of_entry(edge_list, time_end, walk_speed = WALK_SPEED):
    rolling_time = time_end
    for i, r in edge_list.iterrows():
        edge_length = r["length_ft"]
        walk_time = edge_length/walk_speed
        edge_list.loc[i, "time_of_entry"] = rolling_time - pd.Timedelta(seconds=walk_time)
        rolling_time = edge_list.loc[i, "time_of_entry"]
    return edge_list

# Process all movement events
import warnings
warnings.filterwarnings("ignore")

movement_events_test = movement_events.copy()
num = 0
skipped_fast = 0
for idx, event in movement_events_test.iterrows():
    dt_sec = event["dt_sec"]
    bldg_start = event["l1"]
    bldg_end = event["l2"]
    time_end = event["t2"]
    
    try:
        # Calculate shortest path
        _, dist, path = calculate_shortest_bldg_path(bldg_start, bldg_end, G)
        
        # Get edges from path
        edge_list = get_edges_from_path(path, edges)
        
        len_path = edge_list["length_ft"].sum()
        if dt_sec <= 0 or len_path / dt_sec > MAX_WALK_SPEED:
            skipped_fast += 1
            continue  # walkers only — implied speed too high (vehicle, bike, bad Wi‑Fi, etc.)

        # Back calculate entry times
        edge_list = back_calculate_time_of_entry(edge_list, time_end)
        edge_list["time_bin"] = edge_list["time_of_entry"].dt.floor(f"{TIME_BIN}min")
        
        # Update edge_time_series
        for i, r in edge_list.iterrows():
            edge_time_series.loc[r["time_bin"], r["OBJECTID"]] += 1
            
    except Exception as e:
        # Skip events that can't be processed
        continue
        
    # Progress indicator
    if num % 100 == 0:
        print(f"Processed {num}/{len(movement_events_test)} events")
    num += 1

print(f"Skipped {skipped_fast} non-walking events (implied speed > {MAX_WALK_SPEED} ft/s).")
edge_time_series.to_parquet("./edge_time_series.parquet")
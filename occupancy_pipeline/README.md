# Edge occupancy pipeline

Self-contained add-on for the student movement model. It does **not** change the original flow pipeline (`simulate_network_optimized.py` → `edge_time_series.parquet` → `generate_viz.py` → `index.html`).

## Problem it solves

The original time series counts **traversals binned by edge entry time** (10-minute bins). One person can contribute to **many edges in the same bin**, so summing over edges is hard to interpret as “how many people are moving.”

This pipeline builds **occupancy**: at each animation frame, the value is **how many simulated people are on that edge at one instant** (see “Time semantics” below). At any instant, each person is on **at most one** edge.

## Contents of this folder

| File | Role |
|------|------|
| `simulate_network_occupancy.py` | Simulation: reads parent `simon/outputs/events/movement_events.parquet`, network GeoJSON + `bldg_nodes_dict.pkl`; writes `edge_occupancy_series.parquet` here. |
| `generate_viz_occupancy.py` | Reads `edge_occupancy_series.parquet` + parent `network_files/walk_edges_clean.geojson`; reads parent `template.html` **without modifying it**; patches strings + Int32 decode in memory; writes `index_occupancy.html` here. |
| `edge_occupancy_series.parquet` | Output matrix (optional to commit; regenerate with the simulator). Same shape convention as `edge_time_series.parquet`: rows = time steps, columns = edge `OBJECTID`. |
| `index_occupancy.html` | Standalone browser viz (optional to commit; regenerate with the generator). |

Inputs live in the parent directory (`../simon/…`, `../network_files/…`, `../template.html`).

## Time semantics

- For each movement event, the route and **back-calculated** edge intervals \([t_\text{entry}, t_\text{exit})\) match the same logic as the optimized traversal simulator (arrival at building at `t2`, walk times from `length_ft` and `WALK_SPEED`).
- Each 10-minute **row** in the matrix still aligns with the same `pd.date_range` as the original viz, but the count is evaluated at the **midpoint** of that bin (`bin_start + 5 minutes`). That way short trips that fall inside a labeled window still show up, while each frame remains a single instant per edge.

## Routing / RNG

`simulate_network_occupancy.py` mirrors `simulate_network_optimized.py`: Tobler-weighted graph, up to two shortest-path alternatives, logit probabilities with ~90% on the best path, `RNG_SEED = 42`. Running both simulators should therefore use the **same** stochastic route draws for the same events (comparable traversal vs occupancy).

## Usage

From **this** directory:

```bash
cd occupancy_pipeline
python3 simulate_network_occupancy.py   # several minutes at full event count
python3 generate_viz_occupancy.py
```

Open `index_occupancy.html` in a browser (file:// is fine).

## Technical notes

- The occupancy matrix can exceed 16-bit counts on busy edges, so the embedded viz uses **Int32** in the base64 payload; `generate_viz_occupancy.py` applies that change only inside the generated HTML, not in `template.html`.
- If `template.html` copy changes, `TEMPLATE_PATCHES` in `generate_viz_occupancy.py` may need updating if exact substrings no longer match.

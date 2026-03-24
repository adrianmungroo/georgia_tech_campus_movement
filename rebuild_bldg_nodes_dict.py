#!/usr/bin/env python3
"""
Rebuild network_files/bldg_nodes_dict.pkl after associate_nodes_with_bldgs.ipynb logic.

Uses EPSG:2240 + 10 ft buffer (same as fixed notebook). Run from student_movement_model/:

    python3 rebuild_bldg_nodes_dict.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import geopandas as gpd

BASE = Path(__file__).resolve().parent

# Same list as associate_nodes_with_bldgs.ipynb
WIFI_BUILDING_LIST = [
    "003", "002", "006", "007", "010", "015", "011", "014",
    "013", "012", "064", "065W", "065E", "060A", "059", "066", "067", "067A", "071", "073A", "073", "073B", "081", "075", "076", "077", "074",
    "085", "086", "084", "090", "083C", "095", "092", "093", "094",
    "091", "101", "100", "103", "098", "104", "108", "106", "105",
    "107", "115", "111", "116", "109", "110", "123", "124", "119",
    "117", "017", "022", "016", "016A", "020", "130", "126", "130W",
    "131", "130E", "129", "132S", "134", "135", "132N", "133", "138",
    "142", "137", "136", "147", "146", "145", "144", "149", "153",
    "155", "151", "152", "161", "160", "156", "158", "165", "168",
    "166", "167", "164", "171", "170", "172", "175", "173", "186",
    "318", "176", "180A", "177", "181", "180F", "187", "118", "180G",
    "180C", "184", "180E", "180B", "180D", "182", "189", "191N",
    "191W", "191E", "191S", "191", "025", "023A", "026", "024", "196",
    "195", "198", "200", "199", "201", "207", "209", "203", "204",
    "211", "210", "213", "217", "216", "215", "720", "850", "876",
    "785", "865", "031", "030", "033", "033A", "032", "033B", "035",
    "040", "039", "036", "038", "045", "047", "050", "041", "051",
    "051D", "051B", "051C", "051F", "055", "058", "057", "052", "056",
]

PROJ = "EPSG:2240"
BUFFER_FT = 10.0


def main() -> None:
    campus = gpd.read_file("/home/shared/cake/cake_db/vector/campus_buildings_categories_fixed.geojson")
    campus = campus[campus["ShowFeatur"] == "Yes"]

    missing = [c for c in WIFI_BUILDING_LIST if c not in campus["BLDG_CODE"].values]
    if missing:
        raise SystemExit(f"Missing BLDG_CODE in campus file: {missing[:20]}…")

    found_bldgs = campus[campus["BLDG_CODE"].isin(WIFI_BUILDING_LIST)].copy()

    nodes = gpd.read_file(BASE / "network_files/walk_nodes_clean.geojson")
    found_bldgs = found_bldgs.to_crs(nodes.crs)

    b = found_bldgs.to_crs(PROJ).copy()
    b["geometry"] = b.geometry.buffer(BUFFER_FT)
    nodes_proj = nodes.to_crs(PROJ)

    near_pts = gpd.sjoin(
        nodes_proj[nodes_proj["kind"] == 1],
        b,
        predicate="within",
        how="inner",
    )
    bldg_nodes_dict = near_pts.groupby("BLDG_CODE")["node_id"].apply(list).to_dict()

    err = [c for c in WIFI_BUILDING_LIST if c not in bldg_nodes_dict]
    if err:
        raise SystemExit(f"Buildings with no entrance nodes after join: {err}")

    out = BASE / "network_files/bldg_nodes_dict.pkl"
    with open(out, "wb") as f:
        pickle.dump(bldg_nodes_dict, f)

    # Sanity: lists should be small (not ~all nodes on campus)
    sizes = sorted(len(v) for v in bldg_nodes_dict.values())
    print(f"Wrote {out}")
    print(f"  buildings: {len(bldg_nodes_dict)}  |  node list length min/med/max: {sizes[0]} / {sizes[len(sizes)//2]} / {sizes[-1]}")
    if sizes[-1] > 80:
        print("  WARNING: some buildings have very many nodes — check buffer/CRS if unexpected.")


if __name__ == "__main__":
    main()

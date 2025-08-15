import streamlit as st
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from rectpack import newPacker
import math
import json

st.set_page_config(page_title="Box Add-On Fit Planner", page_icon="📦", layout="wide")

# -------------------------
# Data models
# -------------------------
@dataclass
class Item:
    name: str
    L: float  # inches
    W: float  # inches
    H: float  # inches (stacking height)
    rotatable_xy: bool = True

@dataclass
class Box:
    L: float
    W: float
    H: float

# -------------------------
# Defaults (editable in UI)
# -------------------------
def default_catalog() -> Dict[str, Item]:
    return {
        "16 oz jar": Item("16 oz jar", 3.39, 3.39, 4.52, True),
        "24 oz jar": Item("24 oz jar", 3.38, 3.38, 6.17, True),
        "35 oz bento": Item("35 oz bento", 8.04, 5.39, 2.85, True),
        "truffle box": Item("truffle box", 8.50, 4.00, 2.00, True),
        "matcha satchet": Item("matcha satchet", 5.00, 3.00, 0.10, True),
        "4 oz bottle": Item("4 oz bottle", 2.09, 2.09, 4.81, True),
        # New item
        "Ice pack": Item("Ice pack", 11.00, 8.00, 1.75, True),
    }

# Angela's current counts (preloaded)
DEFAULT_COUNTS = {
    "16 oz jar": 3,
    "24 oz jar": 0,
    "35 oz bento": 6,
    "truffle box": 1,
    "matcha satchet": 1,
    "4 oz bottle": 5,
    "Ice pack": 0,  # New
}

BOX_PRESETS = {
    "Big 19.875×16.875×11.875": Box(19.875, 16.875, 11.875),
    "Small 16×10×13": Box(16.0, 10.0, 13.0),
    "Custom": None,
}

# -------------------------
# Packing helpers (2D per layer using rectpack)
# -------------------------
def pack_rectangles(rects: List[Tuple[float, float]], bin_size: Tuple[float, float]) -> Optional[List[Tuple[float, float, float, float]]]:
    if not rects:
        return []
    packer = newPacker(rotation=True)
    binW, binH = bin_size  # rectpack uses (width, height)
    packer.add_bin(binW, binH)
    for i, (w, h) in enumerate(rects):
        packer.add_rect(w, h, rid=i)
    packer.pack()
    all_rects = packer.rect_list()
    if len(all_rects) < len(rects):
        return None
    placed = sorted([(x, y, w, h) for (_, x, y, w, h, rid) in all_rects], key=lambda t: (t[1], t[0]))
    return placed

class Layer:
    def __init__(self, box_L: float, box_W: float):
        self.box_L = box_L
        self.box_W = box_W
        self.items: List[Tuple[str, float, float, float]] = []
        self.height: float = 0.0
        self.placement: Optional[List[Tuple[float, float, float, float]]] = None

    def try_add(self, item: Item) -> bool:
        candidate_rects = [(it[1], it[2]) for it in self.items] + [(item.L, item.W)]
        placed = pack_rectangles(candidate_rects, (self.box_L, self.box_W))
        if placed is None:
            return False
        self.items.append((item.name, item.L, item.W, item.H))
        self.placement = placed
        self.height = max(self.height, item.H)
        return True

    def summary(self) -> Dict:
        area_used = sum(w*h for (_,_,w,h) in [(0,0,it[1],it[2]) for it in self.items])
        return {
            "items": [name for (name, _, _, _) in self.items],
            "height": self.height,
            "used_area": area_used,
            "free_area": self.box_L*self.box_W - area_used,
            "count": len(self.items),
        }

def try_pack_all(items_expanded: List[Item], box: Box) -> Tuple[bool, List[Layer], float]:
    layers: List[Layer] = []
    for it in sorted(items_expanded, key=lambda x: (-x.H, max(x.L, x.W))):
        placed = False
        for layer in sorted(layers, key=lambda L: L.summary()["free_area"], reverse=True):
            if layer.try_add(it):
                placed = True
                break
        if not placed:
            new_layer = Layer(box.L, box.W)
            if not new_layer.try_add(it):
                return False, layers, math.inf
            layers.append(new_layer)
    total_height = sum(L.height for L in layers)
    return (total_height <= box.H + 1e-6), layers, total_height

def expand_items(catalog: Dict[str, Item], counts: Dict[str, int]) -> List[Item]:
    out: List[Item] = []
    for name, n in counts.items():
        if n <= 0:
            continue
        item = catalog[name]
        out.extend([item] * int(n))
    return out

def greedy_addon_search(base_counts: Dict[str, int], candidates: List[str], max_add_per: Dict[str, int], catalog: Dict[str, Item], box: Box, trials:int=50, seed:int=42):
    import random
    rng = random.Random(seed)
    results = []
    orders = [candidates[:]]
    for _ in range(trials-1):
        order = candidates[:]
        rng.shuffle(order)
        orders.append(order)
    for order in orders:
        counts = dict(base_counts)
        for name in order:
            cap = max_add_per.get(name, 0)
            for _ in range(cap):
                test_counts = dict(counts)
                test_counts[name] = test_counts.get(name,0) + 1
                items = expand_items(catalog, test_counts)
                ok, layers, h = try_pack_all(items, box)
                if ok:
                    counts = test_counts
                else:
                    break
        extras = {k: counts.get(k,0) - base_counts.get(k,0) for k in set(list(counts.keys())+list(base_counts.keys()))}
        total_added = sum(max(0, v) for v in extras.values())
        items = expand_items(catalog, counts)
        ok, layers, h = try_pack_all(items, box)
        if ok:
            results.append((extras, total_added, layers, h, counts))
    results.sort(key=lambda r: (-r[1], r[3]))
    unique = []
    seen = set()
    for r in results:
        key = json.dumps({k:int(v) for k,v in r[0].items()}, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(r)
        if len(unique) >= 15:
            break
    return unique

def layout_table(layers: List[Layer]) -> List[Dict]:
    rows = []
    for i, L in enumerate(layers, start=1):
        counts = {}
        for name, *_ in L.items:
            counts[name] = counts.get(name, 0) + 1
        rows.append({
            "Layer": i,
            "Height (in)": round(L.height, 3),
            "Items": ", ".join(f"{k}×{v}" for k,v in sorted(counts.items())),
            "Free area (sq in)": round(L.summary()["free_area"], 2)
        })
    return rows

# -------------------------
# UI
# -------------------------
st.title("📦 Box Add-On Fit Planner")
st.caption("Preloaded with Angela's items and updated box presets.")

# --- Box presets ---
st.sidebar.header("Box Preset")
preset = st.sidebar.selectbox("Choose box", options=list(BOX_PRESETS.keys()), index=0)

if "box_dims" not in st.session_state:
    st.session_state.box_dims = (BOX_PRESETS["Big 19.875×16.875×11.875"].L, BOX_PRESETS["Big 19.875×16.875×11.875"].W, BOX_PRESETS["Big 19.875×16.875×11.875"].H)

if preset != "Custom":
    b = BOX_PRESETS[preset]
    st.session_state.box_dims = (b.L, b.W, b.H)

st.sidebar.header("Box Dimensions (in)")
L = st.sidebar.number_input("Length", value=float(st.session_state.box_dims[0]), min_value=0.1, step=0.1, format="%.2f", key="L_input")
W = st.sidebar.number_input("Width",  value=float(st.session_state.box_dims[1]), min_value=0.1, step=0.1, format="%.2f", key="W_input")
H = st.sidebar.number_input("Height", value=float(st.session_state.box_dims[2]), min_value=0.1, step=0.1, format="%.2f", key="H_input")
box = Box(L, W, H)

# --- Catalog editor ---
st.sidebar.header("Catalog (edit if needed)")
cat = default_catalog()
for name in list(cat.keys()):
    with st.sidebar.expander(name, expanded=False):
        col1, col2, col3 = st.columns(3)
        cat[name].L = col1.number_input(f"{name} — L", value=cat[name].L, step=0.01, format="%.2f", key=f"L_{name}")
        cat[name].W = col2.number_input(f"{name} — W", value=cat[name].W, step=0.01, format="%.2f", key=f"W_{name}")
        cat[name].H = col3.number_input(f"{name} — H", value=cat[name].H, step=0.01, format="%.2f", key=f"H_{name}")
        cat[name].rotatable_xy = st.checkbox("Allow L×W rotation", value=True, key=f"rot_{name}")

# --- What it comes with ---
st.subheader("1) What it comes with")
st.write("Enter the **base items** included in the order.")

counts = {}
cols = st.columns(3)
names = list(default_catalog().keys())
for idx, name in enumerate(names):
    default_val = DEFAULT_COUNTS.get(name, 0)
    counts[name] = cols[idx % 3].number_input(f"{name}", min_value=0, value=int(default_val), step=1, key=f"qty_{name}")

if st.button("Check 'comes with' fit ✅", type="primary"):
    items = expand_items(cat, counts)
    ok, layers, height_used = try_pack_all(items, box)
    if ok:
        st.success(f"Fits! Estimated stacked height: **{height_used:.2f} in** of **{box.H:.2f} in**. Layers: {len(layers)}.")
    else:
        if math.isinf(height_used):
            st.error(f"Does not fit in the {box.L}×{box.W} footprint.")
        else:
            st.error(f"Does not fit. Estimated stacked height needed: **{height_used:.2f} in** > box height **{box.H:.2f} in**.")
    if layers:
        st.markdown("**Layer breakdown (base items)**")
        st.dataframe(layout_table(layers), use_container_width=True)

st.divider()

# --- What can add on ---
st.subheader("2) What can add on")
st.write("Pick add-on types to try and set per-type caps.")

candidate_types = st.multiselect("Add-on candidates", options=names, default=["16 oz jar", "4 oz bottle", "matcha satchet", "truffle box", "35 oz bento", "Ice pack"])
caps = {}
cap_cols = st.columns(5)
for i, name in enumerate(candidate_types):
    default_cap = 4 if name != "matcha satchet" else 20
    caps[name] = cap_cols[i % 5].number_input(f"Max extra {name}", min_value=0, value=default_cap, step=1, key=f"cap_{name}")

trials = st.slider("Search permutations", min_value=5, max_value=200, value=50, step=5)

if st.button("Find add-on combos 🎯"):
    base_items = expand_items(cat, counts)
    base_ok, base_layers, base_h = try_pack_all(base_items, box)
    if not base_ok:
        st.warning("Your base ('comes with') does not fit yet. Reduce items, then try add-ons.")
    res = greedy_addon_search(counts, candidate_types, caps, cat, box, trials=trials, seed=123)
    if not res:
        st.info("No add-on combos found.")
    else:
        st.success(f"Found {len(res)} add-on combo(s).")
        table_rows = []
        for i, (extras, total_added, layers, h, final_counts) in enumerate(res, start=1):
            table_rows.append({
                "Option": i,
                "Total added": total_added,
                "Height used (in)": round(h, 2),
                "Layers": len(layers),
                "Add-ons": ", ".join([f"{k}×{v}" for k,v in extras.items() if v>0]) if any(v>0 for v in extras.values()) else "—"
            })
        st.dataframe(table_rows, use_container_width=True)
        opt_idx = st.number_input("Option to preview", min_value=1, max_value=len(res), value=1, step=1)
        extras, total_added, layers, h, final_counts = res[opt_idx-1]
        st.markdown(f"**Selected option {opt_idx}** → height used **{h:.2f} in** / layers **{len(layers)}**.")
        st.dataframe(layout_table(layers), use_container_width=True)

st.divider()
st.caption("Estimator only — real-world packing may require protective materials.")

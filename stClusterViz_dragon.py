# fixing messed up broomstick

## choose noise value -done

# use different labels in chart -  2nd change is to  make it possible to choose different labels on the icicle/treemap charts. Currently it uses the columns used to build it. We should also add the option to show them in hover. For that we need the analogy of my label_converter I think.
# finally done - it's possible to choose the 2nd set of columns as names but GUI is messy and need to remove debugging stuff later

# To fix first:

# 1 -  add preview!!!!
# 2 - fix UI: make it clear what the seletors do, remove debugging stuff

# TM (and icicle) filtering!
# bubble charts with enrichment
# timeline


# other metrics - slope, CAGR etc

# otehr charts, enrichment etc


import io
#from platform import node
import re
from collections import defaultdict

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT_LABEL = "WHOLE CLUSTERABLE CORPUS"

# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="Cluster Viz", layout="wide")
st.title("📊 Cluster Viz")


# ============================================================
# Cached loader
# ============================================================
@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes), dtype="string", low_memory=False)
    return pd.read_excel(io.BytesIO(file_bytes), dtype="string", sheet_name=0, engine="openpyxl")


# ============================================================
# Helpers: Scout-style eps column detection
# Pattern: <base>_eps<eps>  e.g. 5D_nn12_mins3_MSC48_eps0.13
# ============================================================
EPS_COL_RX = re.compile(r"^(?P<base>.+)_eps(?P<eps>\d+(?:\.\d+)?)$", flags=re.IGNORECASE)


def infer_eps_groups(df: pd.DataFrame) -> dict[str, list[tuple[float, str]]]:
    """
    Returns:
      { base_string: [(eps, column_name), ...] } sorted eps descending for each base.
    Only columns matching '<base>_eps<eps>' are included.
    """
    groups = defaultdict(list)

    for col in df.columns:
        m = EPS_COL_RX.match(str(col))
        if not m:
            continue
        base = m.group("base")
        eps = float(m.group("eps"))
        groups[base].append((eps, str(col)))

    out = {}
    for base, pairs in groups.items():
        pairs.sort(key=lambda x: x[0], reverse=True)  # coarse -> fine
        out[base] = pairs

    return out


def choose_eps_group(df_wide: pd.DataFrame, key_prefix: str = "epsgrp"):
    """
    Streamlit UI: pick which hierarchical clustering set (base) to visualize.
    Returns: (base, eps_cols, eps_values) or (None, None, None)
    """
    st.subheader("Hierarchical clustering columns")

    st.info(
        "📌 **Auto-detection convention (Scout style)**\n\n"
        "stClusterViz auto-detects hierarchical clustering columns named like:\n"
        "`<constant_part>_eps<eps>`\n\n"
        "Example:\n"
        "`5D_nn12_mins3_MSC48_eps0.13`\n\n"
        "If multiple clusterings exist in the file, select which set to visualize."
    )

    groups = infer_eps_groups(df_wide)

    if not groups:
        st.error(
            "No columns matching the pattern `<base>_eps<eps>` were found.\n\n"
            "If your file uses a different naming convention, we can add manual selection "
            "or an interactive renaming helper later."
        )
        return None, None, None

    bases = sorted(groups.keys())
    #base = st.selectbox(
    #    "Select clustering run (constant part before `_eps...`):",
    #    options=bases,
    #)

    base = st.selectbox(
        "Select clustering run (constant part before `_eps...`):",
        options=bases,
        key=f"{key_prefix}__base_select",
    )

    eps_pairs = groups[base]
    if len(eps_pairs) < 2:
        st.warning(f"Selected group '{base}' has < 2 eps columns; need at least 2 for parent/child edges.")
        return None, None, None

    st.success(f"Detected {len(eps_pairs)} eps levels for '{base}'.")
    st.dataframe(
        pd.DataFrame(eps_pairs, columns=["eps", "column"]),
        use_container_width=True,
        hide_index=True,
    )

    eps_values = [e for e, _ in eps_pairs]
    eps_cols = [c for _, c in eps_pairs]
    return base, eps_cols, eps_values


# ============================================================
# Option A builder: parent/child edge list
# ============================================================
def build_df_for_plot_edges_from_cols(
    df_wide: pd.DataFrame,
    eps_cols: list[str],
    eps_values: list[float] | None = None,
    id_col: str | None = None,
    drop_noise: bool = True,
    noise_values=(-1, "-1", None, "None", "nan", "NaN", "NA", "N/A", ""),
    min_edge_value: int = 1,
) -> pd.DataFrame:
    """
    Build parent/child edge list from explicit eps columns (already ordered coarse->fine).
    Output columns:
      parent_eps, child_eps, parent_cluster, child_cluster, value
    """
    if eps_cols is None or len(eps_cols) < 2:
        raise ValueError("Need at least two eps columns.")

    if eps_values is None:
        # fallback positional ordering only
        eps_values = list(range(len(eps_cols), 0, -1))

    if len(eps_values) != len(eps_cols):
        raise ValueError("eps_values length must match eps_cols length.")

    keep_cols = eps_cols.copy()
    if id_col and id_col in df_wide.columns:
        keep_cols = [id_col] + keep_cols

    df = df_wide[keep_cols].copy()

    for c in eps_cols:
        df[c] = df[c].where(~df[c].isin(noise_values), other=pd.NA)

    edges = []
    eps_pairs = list(zip(eps_values, eps_cols))

    for (parent_eps, parent_col), (child_eps, child_col) in zip(eps_pairs[:-1], eps_pairs[1:]):
        tmp = df[[parent_col, child_col]].copy()

        if drop_noise:
            tmp = tmp.dropna(subset=[parent_col, child_col])

        g = (
            tmp.groupby([parent_col, child_col])
               .size()
               .reset_index(name="value")
        )

        g["parent_eps"] = float(parent_eps)
        g["child_eps"] = float(child_eps)
        g = g.rename(columns={parent_col: "parent_cluster", child_col: "child_cluster"})

        if min_edge_value and min_edge_value > 1:
            g = g[g["value"] >= min_edge_value]

        edges.append(g)

    df_edges = pd.concat(edges, ignore_index=True)
    df_edges["parent_cluster"] = df_edges["parent_cluster"].astype(str)
    df_edges["child_cluster"] = df_edges["child_cluster"].astype(str)

    return df_edges.sort_values(["parent_eps", "child_eps", "value"], ascending=[False, False, False])


# ============================================================
# Sankey helpers
# ============================================================
def edges_to_sankey_inputs(df_edges: pd.DataFrame, label_mode: str = "eps + cluster"):
    """
    Convert edge list into nodes + links for Plotly Sankey.

    label_mode:
      - "eps + cluster"  => "eps 0.13 | 7"
      - "cluster only"   => "7"  (note: clusters across eps levels become distinct nodes anyway)
    """
    parent_nodes = df_edges[["parent_eps", "parent_cluster"]].rename(
        columns={"parent_eps": "eps", "parent_cluster": "cluster"}
    )
    child_nodes = df_edges[["child_eps", "child_cluster"]].rename(
        columns={"child_eps": "eps", "child_cluster": "cluster"}
    )
    df_nodes = pd.concat([parent_nodes, child_nodes], ignore_index=True).drop_duplicates()

    # stable ordering: coarse eps first
    df_nodes = df_nodes.sort_values(["eps", "cluster"], ascending=[False, True]).reset_index(drop=True)
    df_nodes["node_id"] = range(len(df_nodes))

    if label_mode == "cluster only":
        df_nodes["label"] = df_nodes["cluster"].astype(str)
    else:
        df_nodes["label"] = df_nodes.apply(lambda r: f"eps {r['eps']} | {r['cluster']}", axis=1)

    node_index = {(float(r.eps), str(r.cluster)): int(r.node_id) for r in df_nodes.itertuples(index=False)}

    df_links = df_edges.copy()
    df_links["source"] = df_links.apply(lambda r: node_index[(float(r.parent_eps), str(r.parent_cluster))], axis=1)
    df_links["target"] = df_links.apply(lambda r: node_index[(float(r.child_eps), str(r.child_cluster))], axis=1)

    return df_nodes, df_links


def _truncate(s: str, n: int = 35) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n-1] + "…"

def plot_sankey(df_nodes: pd.DataFrame, df_links: pd.DataFrame, title: str, max_label_chars: int = 35):
    short_labels = [_truncate(x, max_label_chars) for x in df_nodes["label"].tolist()]
    full_labels = df_nodes["label"].tolist()

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=12,
                    thickness=14,
                    line=dict(color="rgba(0,0,0,0.15)", width=0.5),
                    label=short_labels,
                    customdata=full_labels,
                    hovertemplate="%{customdata}<extra></extra>",
                ),
                link=dict(
                    source=df_links["source"].tolist(),
                    target=df_links["target"].tolist(),
                    value=df_links["value"].astype(int).tolist(),
                    hovertemplate="Flow: %{value}<extra></extra>",
                ),
            )
        ]
    )
    fig.update_layout(title=title, height=750)
    return fig

def build_df_for_plot_flow(
    df_wide: pd.DataFrame,
    eps_cols: list[str],
    eps_values: list[float],
    root_label: str = "WHOLE CLUSTERABLE CORPUS",
    noise_values=("-1", "-1.0", "", "nan", "NaN", "None", None),
) -> pd.DataFrame:
    """
    Notebook-style df_for_plot builder.

    Key idea:
    - Build a leaf->root path table (one row per LEAF cluster) by taking, for each leaf,
      the most common ancestor label at each eps level.
    - Then create parent->child edges between adjacent eps levels.
    - child size = TOTAL size of that child node at that eps (doc count),
      which is valid because the leaf-path construction enforces nesting globally.
    - Noise is excluded as a node (never shown as a cluster).
    """
   

    df = df_wide[eps_cols].copy()

    ##### addition to Cauldron:
    ### this will drop all points which are always noise across all eps levels before we build anything
    ### rather than letting the noise points disappear level by level only
    # Drop ONLY documents that are noise at ALL selected eps levels
    
    ## (hardcode noise label = -1 for now)
    #noise_set = {"-1", "-1.0"}

    noise_set = set(str(x) for x in noise_values if x is not None)
    
    tmp = df.astype("string")
    all_noise_mask = tmp.isin(list(noise_set)).all(axis=1)
    df = df.loc[~all_noise_mask].copy()
    st.write("always-noise docs dropped:", int(all_noise_mask.sum()))
    ########################## end of addition to Cauldron############


    # normalize to string; noise -> NA
    for c in eps_cols:
        #df[c] = df[c].astype("string").replace(list(noise_values), pd.NA)
        df[c] = df[c].astype("string").replace(list(noise_set), pd.NA)
        

    # Make unique node labels per eps (suffix), like your notebook CSV does (epsX_suffix)
    node_cols = []
    for eps, c in zip(eps_values, eps_cols):
        nc = f"__node_eps{eps}"
        df[nc] = df[c].where(df[c].notna(), pd.NA)
        df[nc] = df[nc].astype("string").map(lambda x: f"{x}__eps{eps}" if pd.notna(x) else pd.NA)
        node_cols.append(nc)

    # --- 1) Leaf-path table: one row per LEAF (finest eps) cluster ---
    leaf_col = node_cols[-1]
    path_df = df[node_cols].copy()

    # Drop docs that are noise at LEAF (leaf noise is not a cluster)
    path_df = path_df.dropna(subset=[leaf_col])

    # For each leaf cluster, pick the most common label at each higher eps
    def _mode(series: pd.Series):
        vc = series.value_counts(dropna=True)
        return vc.index[0] if len(vc) else pd.NA

    leaf_paths = (
        path_df.groupby(leaf_col, dropna=False)
               .agg(_mode)
               .reset_index(drop=False)
    )
    # leaf_paths has columns: leaf_col + all node_cols (mode per leaf)

    # --- 2) Build parent->child edges from adjacent eps columns ---
    rows = []

    # ROOT -> top-level (coarsest) nodes
    top_col = node_cols[0]
    top_sizes = df[top_col].dropna().value_counts()
    for child, cnt in top_sizes.items():
        rows.append({"parent": root_label, "child": child, "child size": int(cnt)})

    # adjacent levels: parent = col[i], child = col[i+1]
    for pcol, ccol in zip(node_cols[:-1], node_cols[1:]):
        # Use unique pairs from leaf_paths (this enforces global nesting)
        pairs = leaf_paths[[pcol, ccol]].dropna().drop_duplicates()

        # TOTAL size of each child node at this eps = count of docs assigned to it
        child_totals = df[ccol].dropna().value_counts()

        for _, r in pairs.iterrows():
            parent = r[pcol]
            child = r[ccol]
            # child might be rare/absent in df after NA removal; guard
            if child in child_totals.index:
                rows.append({"parent": parent, "child": child, "child size": int(child_totals[child])})

    df_for_plot = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["parent", "child"])
        .reset_index(drop=True)
    )
    return df_for_plot



# ============================================================
# Sidebar inputs
# ============================================================
st.sidebar.header("Inputs")

uploaded_file = st.sidebar.file_uploader(
    "Upload data (CSV or XLSX)",
    type=["csv", "xlsx"],
)

if uploaded_file is None:
    st.info("Upload a file to begin.")
    st.stop()

try:
    file_bytes = uploaded_file.getvalue()
    df = load_data(file_bytes, uploaded_file.name)
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# Basic cleanup (optional)
df = df.replace(
    to_replace=["", " ", "  ", "None", "NONE", "none", "NA", "N/A", "n/a"],
    value=pd.NA
)
df = df.dropna(how="all")

all_columns = df.columns.tolist()

st.sidebar.divider()
st.sidebar.subheader("Document key")
doc_key_col = st.sidebar.selectbox(
    "Unique identifier column",
    options=[""] + all_columns,
    index=0,
    help="Select a column that uniquely identifies a document (optional).",
)

st.sidebar.divider()
st.sidebar.subheader("2D coordinates")

if len(all_columns) == 0:
    st.error("No columns found in the uploaded file.")
    st.stop()

x_col = st.sidebar.selectbox("2D coord: X", options=all_columns, index=0)
y_col = st.sidebar.selectbox("2D coord: Y", options=all_columns, index=min(1, len(all_columns) - 1))

st.sidebar.divider()
st.sidebar.subheader("Series & hover")

series_col = st.sidebar.selectbox(
    "Column to distinguish series (color)",
    options=[""] + all_columns,
    index=0,
    help="Choose a categorical column to color points by (optional).",
)

hover_candidates = [c for c in all_columns if c not in {x_col, y_col}]
hover_cols = st.sidebar.multiselect(
    "Columns to include in hover",
    options=hover_candidates,
    default=[],
    help="Tip: selecting many columns (or long text columns) can slow down the plot.",
)


# ============================================================
# Tabs
# ============================================================
tabs = st.tabs([
    "📈 Scatter",
    "🧬 Hierarchy (Sankey + edges)",
    "🧊 Icicle (soon)",
    "🧭 Placeholder",
    "⬇ Exports",
])


# ============================================================
# Tab 0: Scatter
# ============================================================
with tabs[0]:
    st.subheader("2D scatterplot")

    plot_df = df.copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")

    n_before = len(plot_df)
    plot_df = plot_df.dropna(subset=[x_col, y_col])
    n_after = len(plot_df)

    if n_after == 0:
        st.error("No valid numeric X/Y points after coercion. Please select correct X/Y columns.")
    else:
        if n_after < n_before:
            st.info(f"Dropped {n_before - n_after} rows with missing/non-numeric X/Y.")

        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color=(series_col if series_col else None),
            hover_data=(hover_cols if hover_cols else None),
            opacity=0.75,
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)


def build_label_maps(
    df_wide: pd.DataFrame,
    eps_cols_id: list[str],
    eps_cols_disp: list[str],
    noise_values=("-1", "-1.0"),
):
    """
    For each eps level, create dict: numeric_label -> descriptive_label
    Uses mode (most common) descriptive label per numeric label.
    If ID and display columns are the same, we return an empty dict for that level
    (meaning: use IDs as display; no mapping needed).
    """
    maps = []
    noise_set = set(str(x) for x in noise_values if x is not None)

    for col_id, col_disp in zip(eps_cols_id, eps_cols_disp):

        # ✅ Critical fix: if user selected the same column set for display, don't group
        if col_id == col_disp:
            maps.append({})  # empty map = "use ID as display"
            continue

        #id_ser = df_wide[col_id].astype("string")
        id_ser = (df_wide[col_id]
          .astype("string")
          .str.strip()
          .str.replace(r"\.0$", "", regex=True))
        
        disp_ser = df_wide[col_disp].astype("string")

        tmp = pd.DataFrame({"id": id_ser, "disp": disp_ser}).dropna()

        mode_map = (
            tmp.groupby("id")["disp"]
               .agg(lambda s: s.value_counts().index[0] if len(s.value_counts()) else pd.NA)
               .to_dict()
        )

        # optional: force noise display text if present
        for nv in noise_set:
            if nv in mode_map:
                mode_map[nv] = "unclustered"

        maps.append(mode_map)

    return maps


# ============================================================
# Tab 1: Hierarchy (Sankey + edges)
# ============================================================
with tabs[1]:
    st.subheader("Hierarchy visualization (Sankey) + edge list")

    # Choose a usable id column (optional)
    id_col = None
    if doc_key_col and doc_key_col in df.columns:
        id_col = doc_key_col
    elif "RowID" in df.columns:
        id_col = "RowID"

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        drop_noise = st.checkbox("Drop noise (-1/empty)", value=True)
    with c2:
        min_edge_value = st.number_input("Minimum edge size", min_value=1, value=1, step=1)
    with c3:
        label_mode = st.selectbox("Node labels", ["eps + cluster", "cluster only"], index=0)
    with c4:
        st.caption(f"Doc ID column: `{id_col}`" if id_col else "Doc ID column: (not set)")

    #base, eps_cols, eps_values = choose_eps_group(df)
    base, eps_cols, eps_values = choose_eps_group(df, key_prefix="sankey")


    if eps_cols:
        try:

            ####
            # ---- Sankey readability controls (Edge filtering) ----
            c1, c2, c3 = st.columns([1, 1, 1])

            with c1:
                # Make default sensible for large corpora
                min_edge_value = st.number_input(
                    "Minimum edge size",
                    min_value=1,
                    value=25,      # <- default bumped up
                    step=1,
                    help="Hide tiny transitions. Increase for readability on large corpora."
                )

            with c2:
                top_n_edges = st.number_input(
                    "Keep top N edges (0 = all)",
                    min_value=0,
                    value=600,     # <- default cap; tweak as you like
                    step=50,
                    help="Keeps only the strongest edges globally (after min-edge filtering)."
                )

            with c3:
                keep_all_levels = st.checkbox(
                    "Keep at least 1 outgoing edge per parent node",
                    value=True,
                    help="Prevents isolated parent nodes when filtering is aggressive."
                )

            ####
            df_edges = build_df_for_plot_edges_from_cols(
                df_wide=df,
                eps_cols=eps_cols,
                eps_values=eps_values,
                id_col=id_col,
                drop_noise=drop_noise,
                min_edge_value=int(min_edge_value),
            )

            st.success(f"Edges built: {df_edges.shape[0]} rows")

            # ---- Apply edge filters ----
            df_edges = df_edges.copy()

            # 1) hard threshold
            df_edges = df_edges[df_edges["value"] >= int(min_edge_value)]

            ### 2) optional top-N strongest edges globally
            ##if int(top_n_edges) > 0 and len(df_edges) > int(top_n_edges):
            ##    df_edges = df_edges.sort_values("value", ascending=False).head(int(top_n_edges))

            ######################
            ###filter per step instead of global top-N###
            ### This retains structure across levels, instead of letting one dense region dominate the entire Sankey###
            
            top_k_per_parent = st.number_input("Keep top K per parent node (0 = off)", min_value=0, value=8, step=1)

            if int(top_k_per_parent) > 0:
                df_edges = (
                    df_edges.sort_values("value", ascending=False)
                            .groupby(["parent_eps", "child_eps", "parent_cluster"], as_index=False)
                            .head(int(top_k_per_parent))
                )



            ######################

            # 3) optional safety net: keep at least one outgoing edge per (parent_eps, parent_cluster)
            #    (useful when filtering is aggressive so the Sankey doesn't "break apart" too much)
            if keep_all_levels and not df_edges.empty:
                idx_max = (
                    df_edges.sort_values("value", ascending=False)
                        .groupby(["parent_eps", "parent_cluster"], as_index=False)
                        .head(1)
                        .index
                )
                df_edges = pd.concat([df_edges, df_edges.loc[idx_max]], ignore_index=True).drop_duplicates()

            # Optional: tell user what happened
            st.caption(
                f"After filtering: {len(df_edges):,} edges "
                f"(min_edge_value={int(min_edge_value)}, top_n_edges={int(top_n_edges)})"
            )


            # Sankey
            df_nodes, df_links = edges_to_sankey_inputs(df_edges, label_mode=label_mode)
            #fig = plot_sankey(df_nodes, df_links, title=f"{base} — hierarchy Sankey")

            max_label_chars = st.slider("Max label characters", 10, 80, 35, 5)
            fig = plot_sankey(df_nodes, df_links, title=f"{base} — hierarchy Sankey", max_label_chars=max_label_chars)
            st.plotly_chart(fig, use_container_width=True)

            # Edge table (collapsible)
            with st.expander("Show edge list table"):
                st.dataframe(df_edges, use_container_width=True, height=420)

            # Downloads
            colA, colB = st.columns(2)
            with colA:
                st.download_button(
                    "Download edges (CSV)",
                    data=df_edges.to_csv(index=False).encode("utf-8"),
                    file_name=f"edges_{base}.csv",
                    mime="text/csv",
                )
            with colB:
                st.download_button(
                    "Download nodes+links (CSV zip-like workaround)",
                    data=(
                        "NODES\n" + df_nodes.to_csv(index=False) + "\n\nLINKS\n" + df_links.to_csv(index=False)
                    ).encode("utf-8"),
                    file_name=f"sankey_nodes_links_{base}.txt",
                    mime="text/plain",
                    help="Simple combined export; we can switch to a real ZIP later if you want.",
                )

        except Exception as e:
            st.error(f"Failed to build Sankey: {e}")


# ============================================================
# Tab 2: Icicle (placeholder for next)
# ============================================================
with tabs[2]:
    st.subheader("Icicle + Treemap (IDs from labels, text from display columns)")
    st.caption("Hierarchy is built from numeric label columns. Display text can come from keywords / summaries. Noise is handled from label columns only.")

    # --- Choose ID set (numeric labels) ---
    base_id, eps_cols_id, eps_values_id = choose_eps_group(df, key_prefix="icicle_id")
    if not eps_cols_id:
        st.stop()

    # --- Choose display set (descriptive labels) ---
    base_disp, eps_cols_disp, eps_values_disp = choose_eps_group(df, key_prefix="icicle_disp")
    if not eps_cols_disp:
        st.stop()

    # --- Noise label(s) ---
    noise_text = st.text_input(
        "Noise label(s) (comma-separated)",
        value="-1",
        help="Used for numeric label columns. Example: -1  or  -1,-1.0"
    )
    noise_values_user = [x.strip() for x in noise_text.split(",") if x.strip()]

    # --- Build df_for_plot ONLY from numeric ID columns ---
    df_for_plot = build_df_for_plot_flow(
        df_wide=df,
        eps_cols=eps_cols_id,
        eps_values=eps_values_id,
        root_label=ROOT_LABEL,
        noise_values=noise_values_user,
    )

    # --- Add explicit ROOT node row for Plotly total mode (keeps it stable) ---
    if not ((df_for_plot["child"] == ROOT_LABEL) & (df_for_plot["parent"] == "")).any():
        root_total = int(df_for_plot.loc[df_for_plot["parent"] == ROOT_LABEL, "child size"].sum())
        df_for_plot = pd.concat(
            [pd.DataFrame([{"parent": "", "child": ROOT_LABEL, "child size": root_total}]), df_for_plot],
            ignore_index=True
        )

    st.caption(f"df_for_plot rows: {len(df_for_plot):,}")

    # --- Build mapping: numeric ID -> display label per eps ---
    label_maps = build_label_maps(
        df, eps_cols_id, eps_cols_disp, noise_values=noise_values_user
    )

    # Build dict keyed by eps token string (avoids float index issues)
    label_maps_by_eps = {str(eps): mp for eps, mp in zip(eps_values_id, label_maps)}

    st.write("TEST map eps0.25 key '6':", label_maps_by_eps.get("0.25", {}).get("6"))
    st.write("TEST map eps0.25 key '6.0':", label_maps_by_eps.get("0.25", {}).get("6.0"))
    
    st.write("Example mapping keys:", list(label_maps_by_eps.keys())[:3])
    st.write("Example map size at eps0.25:", len(label_maps_by_eps.get("0.25", {})))

    rx = re.compile(r"__eps([0-9]*\\.?[0-9]+)$")

    #rx = re.compile(r"__eps([0-9]*\\.?[0-9]+)$")



    def to_display(node: str) -> str:
        node = "" if pd.isna(node) else str(node).strip()

        # Keep ROOT as-is
        if node in {"", ROOT_LABEL}:
            return node

        # Robust split: expect "...__eps<eps>"
        if "__eps" not in node:
            return node

        numeric, eps_key = node.rsplit("__eps", 1)   # numeric="6", eps_key="0.25"
        numeric = numeric.strip()
        eps_key = eps_key.strip()
        suffix = "__eps" + eps_key

        mp = label_maps_by_eps.get(eps_key, {})
        if not mp:
            return node

        disp = mp.get(numeric, numeric)
        return f"{disp}{suffix}"



    sample = df_for_plot.loc[df_for_plot["child"].str.contains(r"^6__eps0\.25$", na=False), "child"].iloc[0]
    st.write("DEBUG sample repr / mapped:", repr(sample), "=>", to_display(sample))

    # Create display columns used by Plotly
    df_for_plot["child_name"] = df_for_plot["child"].apply(to_display)
    st.write(df_for_plot.loc[df_for_plot["child"]=="6__eps0.25", ["child", "child_name"]].head(1))

    st.write("Example:", df_for_plot.loc[df_for_plot["child"].str.contains("eps0.25", na=False), ["child", "child_name"]].head(3))
    st.write("repr(child) sample:", repr(df_for_plot.loc[df_for_plot["child"].str.contains("eps0.25", na=False), "child"].iloc[0]))


    df_for_plot["parent_name"] = df_for_plot["parent"].apply(to_display)

    tabA, tabB = st.tabs(["🧊 Icicle", "🟩 Treemap"])

    with tabA:
        fig_i = px.icicle(
            df_for_plot,
            names="child_name",
            parents="parent_name",
            values="child size",
            title=f"{base_id} (display: {base_disp}) — Icicle",
            hover_data={"child": True, "parent": True, "child size": True},
        )
        fig_i.update_traces(branchvalues="total")
        fig_i.update_layout(height=850)
        st.plotly_chart(fig_i, use_container_width=True)

    with tabB:
        fig_t = px.treemap(
            df_for_plot,
            names="child_name",
            parents="parent_name",
            values="child size",
            title=f"{base_id} (display: {base_disp}) — Treemap",
            hover_data={"child": True, "parent": True, "child size": True},
        )
        fig_t.update_traces(branchvalues="total")
        fig_t.update_layout(height=850)
        st.plotly_chart(fig_t, use_container_width=True)

    with st.expander("Show df_for_plot"):  #("Show df_for_plot (head)")
        #st.dataframe(df_for_plot.head(50), use_container_width=True)
        st.dataframe(df_for_plot, use_container_width=True)

# ============================================================
# Tab 3: Placeholder
# ============================================================
with tabs[3]:
    st.subheader("Placeholder")


# ============================================================
# Tab 4: Exports
# ============================================================
with tabs[4]:
    st.subheader("Exports")
    st.caption("Export features will be added later.")
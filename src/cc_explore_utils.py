
import warnings
from typing import Dict

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import Engine
except Exception:
    create_engine = None
    Engine = None

# -----------------------------
# Data loading & cleaning
# -----------------------------

LEVEL_NAME_MAP = {1: "task", 2: "step", 3: "phase"}

def connect(db_url: str):
    if create_engine is None:
        raise ImportError("SQLAlchemy is required. Please install it (e.g., pip install sqlalchemy).")
    return create_engine(db_url, future=True)

def table_exists(engine, table_name: str) -> bool:
    with engine.connect() as conn:
        try:
            result = conn.execute(text(f"SELECT 1 FROM {table_name} LIMIT 1"))
            _ = result.fetchone()
            return True
        except Exception:
            return False

def read_table(engine, table_name: str) -> pd.DataFrame:
    if not table_exists(engine, table_name):
        warnings.warn(f"Table '{table_name}' not found; returning empty DataFrame.")
        return pd.DataFrame()
    return pd.read_sql(f"SELECT * FROM {table_name}", con=engine)

def load_all(engine) -> Dict[str, pd.DataFrame]:
    return {
        "videos": read_table(engine, "videos"),
        "captions": read_table(engine, "captions"),
        "levels": read_table(engine, "levels"),
        "transcriptions": read_table(engine, "transcriptions"),
    }

def coerce_caption_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure key columns are the right dtype; non-destructive (returns a copy)."""
    if df.empty:
        return df.copy()
    out = df.copy()
    # time columns to float
    for col in ["start_time", "end_time", "duration"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    # level to int
    if "level_id" in out.columns:
        out["level_id"] = pd.to_numeric(out["level_id"], errors="coerce").astype("Int64")
    # booleans
    for col in ["is_surgical_content", "is_descriptive"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].fillna(0).astype(int)
    return out

def add_level_names(df: pd.DataFrame, level_col: str = "level_id", name_col: str = "level_name") -> pd.DataFrame:
    if df.empty or level_col not in df.columns:
        return df.copy()
    out = df.copy()
    out[name_col] = out[level_col].map(LEVEL_NAME_MAP).fillna(out[level_col].astype(str))
    return out

# -----------------------------
# Filters
# -----------------------------

def filter_captions_surgical_descriptive(captions: pd.DataFrame) -> pd.DataFrame:
    if captions.empty:
        return captions.copy()
    caps = coerce_caption_dtypes(captions)
    return caps[(caps["is_surgical_content"] == 1) & (caps["is_descriptive"] == 1)].copy()

def narrated_videos(videos: pd.DataFrame) -> pd.DataFrame:
    """Videos with total_caption_count > 0."""
    if videos.empty or "total_caption_count" not in videos.columns:
        return videos.copy()
    out = videos.copy()
    out["total_caption_count"] = pd.to_numeric(out["total_caption_count"], errors="coerce").fillna(0).astype(int)
    return out[out["total_caption_count"] > 0].copy()

def videos_with_filtered_captions(videos: pd.DataFrame, filtered_caps: pd.DataFrame) -> pd.DataFrame:
    if videos.empty:
        return videos.copy()
    if filtered_caps.empty:
        return videos.iloc[0:0].copy()
    vid_ids = set(filtered_caps["video_id"].dropna().unique().tolist())
    return videos[videos["video_id"].isin(vid_ids)].copy()

# -----------------------------
# Caption analyses & plots (Plotly)
# -----------------------------

def plot_caption_count_histograms(captions: pd.DataFrame):
    """Interactive bar chart with counts for: unfiltered, surgical, descriptive, surgical & descriptive."""
    if captions.empty:
        print("No captions to plot.")
        return None
    caps = coerce_caption_dtypes(captions)
    counts = {
        "unfiltered": len(caps),
        "surgical": int((caps["is_surgical_content"] == 1).sum()),
        "descriptive": int((caps["is_descriptive"] == 1).sum()),
        "surgical & descriptive": int(((caps["is_surgical_content"] == 1) & (caps["is_descriptive"] == 1)).sum())
    }
    df_counts = pd.DataFrame({"category": list(counts.keys()), "count": list(counts.values())})
    fig = px.bar(df_counts, x="category", y="count", title="Caption Counts (All vs. Surgical/Descriptive)")
    
    return fig

def plot_caption_counts_by_level(filtered_captions: pd.DataFrame):
    """Interactive bar of filtered caption count per level."""
    if filtered_captions.empty:
        print("No filtered captions to plot.")
        return None
    df = add_level_names(coerce_caption_dtypes(filtered_captions))
    counts = df["level_name"].value_counts().rename_axis("level").reset_index(name="count")
    counts = counts.sort_values("level")
    fig = px.bar(counts, x="level", y="count", title="Filtered Caption Count per Level")
    
    return fig

def plot_duration_violin_by_level(filtered_captions: pd.DataFrame):
    """Interactive violin: duration per level (x=levels, y=duration seconds)."""
    if filtered_captions.empty:
        print("No filtered captions to plot.")
        return None
    df = add_level_names(coerce_caption_dtypes(filtered_captions))
    df = df.dropna(subset=["duration", "level_name"])
    if df.empty:
        print("Insufficient duration data for violin plot.")
        return None
    fig = px.violin(df, x="level_name", y="duration", box=True, points="all",
                    title="Filtered Caption Duration by Level (seconds)",
                    category_orders={"level_name": ["task", "step", "phase"]})
    fig.update_layout(xaxis_title="Level", yaxis_title="Duration (seconds)")
    fig.update_yaxes(range=[-50, 1000], autorange=False)
    
    return fig

def plot_avg_caption_count_per_video_by_level(filtered_captions: pd.DataFrame):
    """
    Interactive violin for the per-video caption count by level (filtered captions only).
    """
    if filtered_captions.empty:
        print("No filtered captions to plot.")
        return None
    df = add_level_names(coerce_caption_dtypes(filtered_captions))
    if "video_id" not in df.columns:
        print("Missing 'video_id' in captions.")
        return None

    rows = []
    for lvl in ["task", "step", "phase"]:
        sub = df[df["level_name"] == lvl]
        if sub.empty:
            continue
        counts = sub.groupby("video_id")["caption_id"].count().reset_index(name="count")
        counts["level_name"] = lvl
        rows.append(counts)
    if not rows:
        print("No data to plot per-video caption counts.")
        return None

    counts_df = pd.concat(rows, ignore_index=True)
    fig = px.violin(counts_df, x="level_name", y="count", box=True, points="all",
                    title="Per-Video Caption Count by Level (Filtered)",
                    category_orders={"level_name": ["task", "step", "phase"]})
    fig.update_layout(xaxis_title="Level", yaxis_title="Caption count per video")
    
    return fig

# -----------------------------
# Video metadata analyses & plots (Plotly)
# -----------------------------

def plot_video_overview_bars(videos: pd.DataFrame, filtered_captions: pd.DataFrame):
    """
    Interactive bars: total videos, silent videos (total_caption_count=0), narrated videos (>0),
    and videos that contain surgical & descriptive captions.
    """
    if videos.empty:
        print("No videos to plot.")
        return None
    vids = videos.copy()
    if "total_caption_count" in vids.columns:
        vids["total_caption_count"] = pd.to_numeric(vids["total_caption_count"], errors="coerce").fillna(0).astype(int)
    else:
        vids["total_caption_count"] = 0

    total = len(vids)
    silent = int((vids["total_caption_count"] == 0).sum())
    narrated = int((vids["total_caption_count"] > 0).sum())

    if not filtered_captions.empty:
        vid_ids_filtered = set(filtered_captions["video_id"].dropna().unique().tolist())
        filtered_videos_count = int(vids["video_id"].isin(vid_ids_filtered).sum())
    else:
        filtered_videos_count = 0

    df_counts = pd.DataFrame({
        "category": ["total", "silent videos", "narrated videos", "filtered videos"],
        "count": [total, silent, narrated, filtered_videos_count]
    })
    fig = px.bar(df_counts, x="category", y="count", title="Video Overview")
    
    return fig

def plot_categorical_pie(videos: pd.DataFrame, column: str, min_count: int = 20):
    """
    Interactive pie chart for a categorical column in videos.
    """
    if videos.empty or column not in videos.columns:
        print(f"No data for pie: {column}")
        return None
    vc = videos[column].fillna("Unknown").astype(str).value_counts()
    vc = vc[vc >= min_count]
    if vc.empty:
        print(f"No categories >= {min_count} for {column}.")
        return None
    df = vc.rename_axis(column).reset_index(name="count")
    fig = px.pie(
        df,
        names=column,
        values="count",
        title=f"{column} distribution",
        hole=0.4
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        sort=False  # keep your current order; set True if you want descending by count
    )
    fig.update_layout(
        showlegend=False,
    )
    return fig

def plot_fps_hist(videos: pd.DataFrame, bins: int = 16):
    """Interactive histogram for FPS (e.g., ranging roughly 15-30)."""
    if videos.empty or "fps" not in videos.columns:
        print("No FPS data to plot.")
        return None
    ser = pd.to_numeric(videos["fps"], errors="coerce").dropna()
    if ser.empty:
        print("No valid FPS values.")
        return None
    df = pd.DataFrame({"fps": ser.values})
    fig = px.histogram(df, x="fps", nbins=bins, title="FPS Distribution")
    fig.update_layout(xaxis_title="Frames per second", yaxis_title="Count")
    
    return fig

def plot_video_duration_violin(videos: pd.DataFrame):
    """Interactive violin for video duration in seconds (preferred)."""
    if videos.empty or "duration" not in videos.columns:
        print("No duration data to plot.")
        return None
    ser = pd.to_numeric(videos["duration"], errors="coerce").dropna()
    if ser.empty:
        print("No valid durations.")
        return None
    df = pd.DataFrame({"duration_seconds": ser.values, "all_videos": "all"})
    fig = px.violin(df, x="all_videos", y="duration_seconds", box=True, points="all",
                    title="Video Duration Distribution (seconds)")
    fig.update_layout(xaxis_title="", yaxis_title="Duration (seconds)", showlegend=False)
    
    return fig

def plot_video_duration_hist(videos: pd.DataFrame, bins: int = 30):
    """Deprecated name: now shows an interactive violin instead of a histogram."""
    return plot_video_duration_violin(videos)

# -----------------------------
# Sampling tables
# -----------------------------

def sample_captions(df: pd.DataFrame, n: int = 10, seed: int = 42) -> pd.DataFrame:
    if df.empty:
        return df
    take = min(n, len(df))
    return df.sample(take, random_state=seed)

def sample_filtered_by_level(filtered_captions: pd.DataFrame, level_id: int, n: int = 10, seed: int = 42) -> pd.DataFrame:
    df = coerce_caption_dtypes(filtered_captions)
    sub = df[df["level_id"] == level_id]
    return sample_captions(sub, n=n, seed=seed)

# -----------------------------
# Integrity / sanity helpers
# -----------------------------

def summary_info(videos: pd.DataFrame, captions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(("videos_total", len(videos)))
    if "total_caption_count" in videos.columns:
        tcc = pd.to_numeric(videos["total_caption_count"], errors="coerce").fillna(0)
        rows.append(("videos_silent", int((tcc == 0).sum())))
        rows.append(("videos_narrated", int((tcc > 0).sum())))
    else:
        rows.append(("videos_silent", None))
        rows.append(("videos_narrated", None))

    rows.append(("captions_total", len(captions)))
    if not captions.empty:
        caps = coerce_caption_dtypes(captions)
        rows.append(("captions_surgical", int((caps["is_surgical_content"] == 1).sum())))
        rows.append(("captions_descriptive", int((caps["is_descriptive"] == 1).sum())))
        rows.append(("captions_surgical_and_descriptive", int(((caps["is_surgical_content"] == 1) & (caps["is_descriptive"] == 1)).sum())))
    else:
        rows.extend([("captions_surgical", None), ("captions_descriptive", None), ("captions_surgical_and_descriptive", None)])

    return pd.DataFrame(rows, columns=["metric", "value"])

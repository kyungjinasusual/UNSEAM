#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick pipeline to align pause & event-boundary markers from multimodal cues.

Inputs
------
- An mp4 movie file (with audio).
- A subtitle / script file (SRT strongly recommended; plain text fallback).
- (Optional) A PySceneDetect-generated pause CSV for visual "pause" segments.

Outputs
-------
- <base>.audio_silences.csv            # audio-based "silence" segments
- <base>.semantic_lowchange.csv        # low semantic change segments from subtitles
- <base>.events_llm.csv                # (optional) LLM-derived event segments/boundaries
- <base>.markers_merged.csv            # aligned markers (video_pause/audio_pause/semantic_pause/event_boundary)
- <base>.markers_timeline.png          # a simple timeline visualization

Minimal dependencies
--------------------
- numpy, pandas, matplotlib, scikit-learn, librosa (requires ffmpeg for mp4 audio decode)
- Optional: openai (for LLM event segmentation). Set OPENAI_API_KEY env var.

Compatibility
-------------
This script can ingest the pause CSV produced by your existing PySceneDetect script
(extract_pauses_with_pyscenedetect.py) so you can cross-check results.

Usage (examples)
----------------
python quick_pauses_events_pipeline.py movie.mp4 --subs movie.srt --pyscenedetect-pauses movie.pauses.csv \
  --min-audio-sec 0.6 --audio-thr-percentile 20 \
  --semantic-window-sec 5.0 --semantic-thr-percentile 35 \
  --llm-events --llm-model gpt-4o-mini

Notes
-----
- If you do not have an SRT, you can still run audio-only. If your "script" is plain text without timestamps,
  the LLM-based segmentation can still work, but semantic low-change segments require time-aligned SRT.
"""

import argparse
import os
import re
import json
import math
import subprocess
import tempfile
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# Librosa for audio energy
import librosa

# TF-IDF + cosine similarity for semantic change
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Plotting
import matplotlib.pyplot as plt


# ------------------------- Utilities -------------------------

def hhmmss_to_seconds(tc: str) -> float:
    """
    Convert timecode 'HH:MM:SS.mmm' or SRT 'HH:MM:SS,mmm' to seconds (float).
    """
    tc = tc.strip()
    tc = tc.replace(',', '.')
    h, m, s = tc.split(':')
    return 3600 * int(h) + 60 * int(m) + float(s)


def seconds_to_hhmmss(sec: float) -> str:
    ms = int(round((sec - int(sec)) * 1000))
    s_int = int(sec)
    h = s_int // 3600
    m = (s_int % 3600) // 60
    s = s_int % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def ensure_audio_wav(video_path: str, sr: int = 16000) -> str:
    """
    Extract mono PCM WAV with ffmpeg. Returns path to temp wav.
    """
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", str(sr), "-vn",
        "-acodec", "pcm_s16le", tmp_wav
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(
            "Failed to extract audio with ffmpeg. Please ensure ffmpeg is installed and in PATH."
        ) from e
    return tmp_wav


# ------------------------- (1) Audio silence detection -------------------------

def detect_audio_silences(
    video_path: str,
    frame_len_sec: float = 0.032,
    hop_len_sec: float = 0.016,
    thr_db: Optional[float] = None,
    thr_percentile: Optional[float] = 20.0,
    min_silence_sec: float = 0.5
) -> pd.DataFrame:
    """
    Energy-based silence detection.

    If thr_db is None, use a percentile threshold over short-time dB energy (lower percentile => more silence).
    """
    wav_path = ensure_audio_wav(video_path, sr=16000)
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    os.unlink(wav_path)

    frame_len = int(round(frame_len_sec * sr))
    hop_len = int(round(hop_len_sec * sr))
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len, center=True)[0]
    # Convert to dB; add epsilon to avoid log of zero
    db = 20.0 * np.log10(np.maximum(rms, 1e-10))

    if thr_db is None:
        if thr_percentile is None:
            thr_percentile = 20.0
        thr_db = np.percentile(db, thr_percentile)

    # Boolean mask for silence frames
    is_sil = db <= thr_db

    # Frame times (center of frame)
    times = librosa.frames_to_time(np.arange(len(db)), sr=sr, hop_length=hop_len, n_fft=frame_len)

    # Group consecutive True runs into segments
    segments = []
    start = None
    for i, flag in enumerate(is_sil):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            end = i - 1
            segments.append((start, end))
            start = None
    if start is not None:
        segments.append((start, len(is_sil) - 1))

    # Filter by minimum duration and format
    rows = []
    for start_idx, end_idx in segments:
        start_t = float(times[start_idx])
        end_t = float(times[end_idx]) + hop_len_sec
        if (end_t - start_t) >= min_silence_sec:
            rows.append({
                "start_sec": start_t,
                "end_sec": end_t,
                "start_tc": seconds_to_hhmmss(start_t),
                "end_tc": seconds_to_hhmmss(end_t),
                "duration_sec": end_t - start_t,
                "method": "audio_silence",
                "thr_db": thr_db,
                "thr_percentile": thr_percentile
            })
    return pd.DataFrame(rows)


# ------------------------- (2) Subtitle-based semantic change -------------------------

_SRT_BLOCK_RE = re.compile(
    r"(?P<idx>\d+)\s*\n(?P<start>\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*\n(?P<text>.*?)(?=\n\s*\n|\Z)",
    re.DOTALL
)

def parse_srt(path: str) -> List[dict]:
    """
    Minimal SRT parser (no advanced features). Returns list of dicts with start/end/text.
    """
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    blocks = []
    for m in _SRT_BLOCK_RE.finditer(text):
        start = hhmmss_to_seconds(m.group("start"))
        end = hhmmss_to_seconds(m.group("end"))
        txt = re.sub(r"\s+", " ", m.group("text")).strip()
        blocks.append({"start_sec": start, "end_sec": end, "text": txt})
    if not blocks:
        # Fallback: treat whole file as one block at time 0..inf (not great, but avoids crash)
        blocks = [{"start_sec": 0.0, "end_sec": 0.0, "text": text.strip()}]
    return blocks


def semantic_change_from_srt(
    srt_path: str,
    window_sec: float = 5.0,
    thr_percentile: float = 35.0,
    min_lowchange_sec: float = 1.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute a coarse semantic change rate from subtitle blocks using TF-IDF + cosine distance.
    Returns:
        - df_scores: time series of (mid_sec, delta_semantic)
        - df_segments: low-change segments (semantic "pause-like") above min duration
    """
    blocks = parse_srt(srt_path)
    if len(blocks) < 2:
        # Cannot compute deltas
        cols = ["mid_sec", "delta_semantic"]
        return pd.DataFrame([], columns=cols), pd.DataFrame([], columns=["start_sec","end_sec","start_tc","end_tc","duration_sec","method"])

    texts = [b["text"] for b in blocks]
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform(texts)
    # Similarity between adjacent blocks; delta = 1 - sim
    sims = []
    mids = []
    for i in range(len(blocks) - 1):
        s = cosine_similarity(X[i], X[i+1])[0, 0]
        d = 1.0 - float(s)
        mid = 0.5 * (blocks[i]["end_sec"] + blocks[i+1]["start_sec"])
        sims.append(d)
        mids.append(mid)
    df_scores = pd.DataFrame({"mid_sec": mids, "delta_semantic": sims})

    # Smooth over window_sec by simple moving average
    if len(df_scores) > 1:
        step = np.median(np.diff(df_scores["mid_sec"])) if len(df_scores) > 2 else window_sec
        k = max(1, int(round(window_sec / max(step, 1e-6))))
        df_scores["delta_smooth"] = df_scores["delta_semantic"].rolling(window=k, min_periods=1, center=True).mean()
    else:
        df_scores["delta_smooth"] = df_scores["delta_semantic"]

    # Low-change threshold by percentile (lower values mean slower semantic change)
    thr = np.percentile(df_scores["delta_smooth"], thr_percentile)

    # Build segments where delta_smooth <= thr
    is_low = df_scores["delta_smooth"] <= thr
    segments = []
    start = None
    for i, flag in enumerate(is_low):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            end = i - 1
            segments.append((start, end))
            start = None
    if start is not None:
        segments.append((start, len(is_low) - 1))

    rows = []
    for s_idx, e_idx in segments:
        start_t = float(df_scores["mid_sec"].iloc[s_idx] - 0.5 * window_sec)
        end_t = float(df_scores["mid_sec"].iloc[e_idx] + 0.5 * window_sec)
        if end_t <= start_t:
            continue
        if (end_t - start_t) >= min_lowchange_sec:
            rows.append({
                "start_sec": max(0.0, start_t),
                "end_sec": end_t,
                "start_tc": seconds_to_hhmmss(max(0.0, start_t)),
                "end_tc": seconds_to_hhmmss(end_t),
                "duration_sec": end_t - start_t,
                "method": "semantic_lowchange",
                "thr_percentile": thr_percentile,
                "window_sec": window_sec
            })

    return df_scores, pd.DataFrame(rows)


# ------------------------- (3) Optional LLM-based event segmentation -------------------------

def llm_event_segmentation_from_subs(
    srt_path: str,
    model_name: str = "gpt-4o-mini",
    max_chars: int = 12000
) -> pd.DataFrame:
    """
    Calls an OpenAI-compatible chat model (if available) to segment events with timestamps.
    Returns a DataFrame of segments [start_sec, end_sec, label].

    Requires:
      - pip install openai
      - export OPENAI_API_KEY=...
    """
    try:
        import os
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is None:
            raise RuntimeError("OPENAI_API_KEY not set.")
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print("[WARN] LLM segmentation unavailable:", e)
        return pd.DataFrame(columns=["start_sec","end_sec","start_tc","end_tc","label","method"])

    blocks = parse_srt(srt_path)
    # Prepare a compressed, time-tagged transcript
    lines = []
    total = 0
    for b in blocks:
        seg = f"[{seconds_to_hhmmss(b['start_sec'])} --> {seconds_to_hhmmss(b['end_sec'])}] {b['text']}"
        if total + len(seg) > max_chars:
            break
        lines.append(seg)
        total += len(seg)
    prompt = (
        "You are given a time-stamped transcript of a movie segment (SRT-like). "
        "Propose a compact list of major narrative events with start/end timestamps in seconds. "
        "Return ONLY JSON list of objects with keys: start_sec, end_sec, and label.\n\n"
        + "\n".join(lines)
    )

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content
        # Attempt to extract JSON
        # Find first JSON array
        m = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
        if not m:
            raise ValueError("No JSON array found in model output.")
        data = json.loads(m.group(0))
        rows = []
        for ev in data:
            try:
                s = float(ev.get("start_sec", 0.0))
                e = float(ev.get("end_sec", s))
                lab = str(ev.get("label", "event"))
                rows.append({
                    "start_sec": s,
                    "end_sec": e,
                    "start_tc": seconds_to_hhmmss(s),
                    "end_tc": seconds_to_hhmmss(e),
                    "label": lab,
                    "method": "event_llm"
                })
            except Exception:
                continue
        return pd.DataFrame(rows)
    except Exception as e:
        print("[WARN] LLM segmentation failed:", e)
        return pd.DataFrame(columns=["start_sec","end_sec","start_tc","end_tc","label","method"])


# ------------------------- Merge & visualize -------------------------

def load_pyscenedetect_pauses(path: str) -> pd.DataFrame:
    """
    Load pauses from extract_pauses_with_pyscenedetect.py output:
      columns: start_tc,end_tc,start_sec,end_sec,duration_sec,start_frame,end_frame,...
    """
    df = pd.read_csv(path)
    # normalize expected columns
    need = ["start_sec","end_sec","start_tc","end_tc","duration_sec"]
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column in PySceneDetect pause CSV: {c}")
    df = df[need].copy()
    df["method"] = "video_pause_pyscenedetect"
    return df


def merge_markers(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    dfs = [d for d in dfs if d is not None and len(d) > 0]
    if not dfs:
        return pd.DataFrame(columns=["start_sec","end_sec","start_tc","end_tc","duration_sec","label","method"])
    # Ensure columns exist
    out = []
    for d in dfs:
        d = d.copy()
        if "duration_sec" not in d.columns:
            d["duration_sec"] = (d["end_sec"] - d["start_sec"]).clip(lower=0.0)
        if "label" not in d.columns:
            d["label"] = ""
        out.append(d[["start_sec","end_sec","start_tc","end_tc","duration_sec","label","method"]])
    merged = pd.concat(out, axis=0, ignore_index=True).sort_values("start_sec")
    return merged


def visualize_timeline(merged: pd.DataFrame, base: str, total_dur: Optional[float] = None):
    """
    Simple timeline plot with category lanes.
    """
    if merged.empty:
        print("[INFO] Nothing to visualize.")
        return
    # Assign y lanes by method
    methods = list(merged["method"].unique())
    y_map = {m: i for i, m in enumerate(methods)}
    plt.figure(figsize=(12, 2 + 0.6 * len(methods)))

    # plot segments as horizontal bars (boundaries as short segments if end==start)
    for _, r in merged.iterrows():
        y = y_map[r["method"]]
        x0 = r["start_sec"]
        x1 = r["end_sec"]
        if x1 <= x0 + 1e-3:
            # event boundary: draw a vertical line
            plt.vlines([x0], y - 0.3, y + 0.3)
        else:
            plt.hlines(y, x0, x1, linewidth=6)

    plt.yticks(list(y_map.values()), list(y_map.keys()))
    plt.xlabel("Time (sec)")
    if total_dur is not None and total_dur > 0:
        plt.xlim(0, total_dur)
    plt.tight_layout()
    out_png = f"{base}.markers_timeline.png"
    plt.savefig(out_png, dpi=160)
    print(f"[OK] Saved timeline plot: {out_png}")


def probe_video_duration_sec(video_path: str) -> Optional[float]:
    """
    Use ffprobe to read duration.
    """
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(res.stdout.strip())
    except Exception:
        return None


# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=str, help="Path to input video (e.g., movie.mp4)")
    ap.add_argument("--subs", type=str, default=None, help="Path to subtitle/script file (SRT preferred)")
    ap.add_argument("--pyscenedetect-pauses", type=str, default=None, help="Path to PySceneDetect pauses CSV (optional)")

    # Audio silence params
    ap.add_argument("--min-audio-sec", type=float, default=0.5, help="Minimum silence duration (sec)")
    ap.add_argument("--audio-thr-db", type=float, default=None, help="Absolute dB threshold (if set, overrides percentile)")
    ap.add_argument("--audio-thr-percentile", type=float, default=20.0, help="Percentile for dB threshold (lower => more silence)")
    ap.add_argument("--audio-frame-sec", type=float, default=0.032, help="RMS frame length (sec)")
    ap.add_argument("--audio-hop-sec", type=float, default=0.016, help="RMS hop length (sec)")

    # Semantic change params
    ap.add_argument("--semantic-window-sec", type=float, default=5.0, help="Smoothing window (sec) over delta_semantic")
    ap.add_argument("--semantic-thr-percentile", type=float, default=35.0, help="Percentile for low-change threshold")
    ap.add_argument("--semantic-min-sec", type=float, default=1.0, help="Minimum low-change segment duration (sec)")

    # LLM options
    ap.add_argument("--llm-events", action="store_true", help="Enable LLM-based event segmentation")
    ap.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="OpenAI model name")

    args = ap.parse_args()
    video_path = args.video
    base = os.path.splitext(os.path.basename(video_path))[0]

    total_dur = probe_video_duration_sec(video_path)

    # (1) Audio silences
    print("[*] Detecting audio silences...")
    df_audio = detect_audio_silences(
        video_path=video_path,
        frame_len_sec=args.audio_frame_sec,
        hop_len_sec=args.audio_hop_sec,
        thr_db=args.audio_thr_db,
        thr_percentile=args.audio_thr_percentile,
        min_silence_sec=args.min_audio_sec
    )
    audio_csv = f"{base}.audio_silences.csv"
    df_audio.to_csv(audio_csv, index=False)
    print(f"[OK] {len(df_audio)} audio silence segments -> {audio_csv}")

    # (2) Semantic change & low-change segments
    df_sem_scores = pd.DataFrame()
    df_sem_low = pd.DataFrame()
    if args.subs and os.path.exists(args.subs):
        print("[*] Computing semantic change from subtitles...")
        df_sem_scores, df_sem_low = semantic_change_from_srt(
            args.subs,
            window_sec=args.semantic_window_sec,
            thr_percentile=args.semantic_thr_percentile,
            min_lowchange_sec=args.semantic_min_sec
        )
        sem_csv = f"{base}.semantic_lowchange.csv"
        df_sem_low.to_csv(sem_csv, index=False)
        print(f"[OK] {len(df_sem_low)} semantic low-change segments -> {sem_csv}")
        # Also save scores for inspection
        df_sem_scores.to_csv(f"{base}.semantic_change_scores.csv", index=False)
    else:
        print("[INFO] --subs not provided or file missing; skipping semantic analysis.")

    # (3) LLM event segmentation (optional)
    df_llm = pd.DataFrame()
    if args.llm_events and args.subs and os.path.exists(args.subs):
        print("[*] Running LLM-based event segmentation (requires OPENAI_API_KEY)...")
        df_llm = llm_event_segmentation_from_subs(args.subs, model_name=args.llm_model)
        ev_csv = f"{base}.events_llm.csv"
        if len(df_llm) > 0:
            df_llm.to_csv(ev_csv, index=False)
            print(f"[OK] {len(df_llm)} LLM events -> {ev_csv}")
        else:
            print("[WARN] No LLM events produced.")
    elif args.llm_events:
        print("[WARN] --llm-events specified but --subs missing; skipping LLM segmentation.")

    # (4) Load PySceneDetect pauses (optional) to align
    df_video_pauses = pd.DataFrame()
    if args.pyscenedetect_pauses and os.path.exists(args.pyscenedetect_pauses):
        print("[*] Loading PySceneDetect pauses...")
        df_video_pauses = load_pyscenedetect_pauses(args.pyscenedetect_pauses)
        print(f"[OK] Loaded {len(df_video_pauses)} video pauses (PySceneDetect).")

    # Prepare merged markers
    dfs_to_merge = [df_video_pauses, df_audio]
    if len(df_sem_low) > 0:
        dfs_to_merge.append(df_sem_low)
    if len(df_llm) > 0:
        dfs_to_merge.append(df_llm)

    merged = merge_markers(dfs_to_merge)
    merged_csv = f"{base}.markers_merged.csv"
    merged.to_csv(merged_csv, index=False)
    print(f"[OK] Merged markers -> {merged_csv} (n={len(merged)})")

    # (5) Visualize
    print("[*] Visualizing timeline...")
    visualize_timeline(merged, base=base, total_dur=total_dur)
    print("[DONE]")

if __name__ == "__main__":
    main()

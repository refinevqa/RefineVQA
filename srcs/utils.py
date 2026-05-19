import os
import re
import socket
import numpy as np
import av


ANS_ORDER = ["A", "B", "C", "D", "E"]


def get_container_hostname():
    try:
        return socket.gethostname()
    except Exception:
        return "None"


def find_file_by_video_id(root_folder, video_id):
    for dirpath, _, filenames in os.walk(root_folder):
        if video_id in filenames:
            return os.path.join(dirpath, video_id)
    return None


def _read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index, end_index = indices[0], indices[-1]
    idx_set = set(indices.tolist() if hasattr(indices, "tolist") else list(indices))
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in idx_set:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def _resolve_video_path(video_root, vid):
    candidates = [str(vid) + ".mp4", str(vid) + ".mkv", str(vid) + ".webm", str(vid)]
    for name in candidates:
        p = find_file_by_video_id(video_root, name)
        if p is not None:
            return p
    raise FileNotFoundError(f"Video with ID {vid} not found in {video_root}")


def load_video_uniform(video_root, vid, num_frames=32):
    """Uniformly sample `num_frames` frames from a video — used for the initial description."""
    video_path = _resolve_video_path(video_root, vid)
    container = av.open(video_path)
    total = container.streams.video[0].frames
    if total <= 0:
        total = sum(1 for _ in container.decode(video=0))
        container = av.open(video_path)
    indices = np.linspace(0, max(total - 1, 0), num_frames).astype(int)
    return _read_video_pyav(container, indices)


def load_video_at_indices(video_root, vid, indices):
    """Load frames at the explicit `indices` — used for the zoom-in description."""
    video_path = _resolve_video_path(video_root, vid)
    container = av.open(video_path)
    indices = np.array(sorted(set(int(i) for i in indices))).astype(int)
    return _read_video_pyav(container, indices)


def extract_candidate_frames(video_root, vid, candidate_count=128):
    """Densely sample candidate frames (used by the frame selector)."""
    video_path = _resolve_video_path(video_root, vid)
    container = av.open(video_path)
    total = container.streams.video[0].frames
    if total <= 0:
        total = sum(1 for _ in container.decode(video=0))
        container = av.open(video_path)
    if total <= candidate_count:
        indices = np.arange(0, total).astype(int)
    else:
        indices = np.linspace(0, total - 1, candidate_count).astype(int)
    frames = _read_video_pyav(container, indices)
    return frames, indices


def extract_questions(text):
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*\d\.\)\s]+", "", line)
        if "?" in line:
            out.append(line)
    return out


def format_options(options):
    parts = []
    for i, opt in enumerate(options):
        parts.append(f"{ANS_ORDER[i]}.{opt}")
    return " ".join(parts)


def parse_answer_letter(text, num_options=5):
    if not text:
        return None
    m = re.search(r"\b([A-E])\b", text.strip())
    if m and ANS_ORDER.index(m.group(1)) < num_options:
        return m.group(1)
    return None


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

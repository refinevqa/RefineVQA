"""RefineVQA — iterative refinement of video descriptions via LLM feedback.

Entry point. Every model and pipeline knob is exposed via CLI flags; the
script supports the three iterative-refinement variants from the paper
(AIoR, GIoR, BPoR).
"""

import argparse
import json
import os
from tqdm import tqdm

from srcs.models import LLMWrapper, VideoLMWrapper, VisionCaptioner
from srcs.frame_selector import build_frame_selector
from srcs.refine_vqa import RefineVQA
from srcs.utils import ANS_ORDER, get_container_hostname, to_serializable


def parse_args():
    p = argparse.ArgumentParser(description="RefineVQA inference")

    # ---- I/O ---------------------------------------------------------------
    p.add_argument("--input_file", type=str, required=True,
                   help="JSONL file with rows containing {video, question, options, answer}.")
    p.add_argument("--video_path", type=str, required=True,
                   help="Root directory containing the video files.")
    p.add_argument("--results_path", type=str, default="results",
                   help="Output directory root.")
    p.add_argument("--run_name", type=str, default=None,
                   help="Optional subfolder name; defaults to a tag built from the variant + iters.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N samples (debug).")
    p.add_argument("--resume", action="store_true",
                   help="Skip videos that already have a result JSON.")

    # ---- variant -----------------------------------------------------------
    p.add_argument("--variant", type=str, default="AIoR",
                   choices=["AIoR", "GIoR", "BPoR", "aior", "gior", "bpor"],
                   help="Iterative refinement strategy.")
    p.add_argument("--max_iterations", type=int, default=3,
                   help="AIoR: maximum refinement iterations (early-stop allowed).")
    p.add_argument("--num_iterations", type=int, default=3,
                   help="GIoR: fixed number of refinement iterations.")
    p.add_argument("--num_feedback_questions", type=int, default=3,
                   help="BPoR: number of feedback questions generated at once.")

    # ---- models ------------------------------------------------------------
    p.add_argument("--vlm_model", type=str,
                   default="llava-hf/llava-onevision-qwen2-7b-ov-hf",
                   help="HuggingFace identifier for the VideoLM.")
    p.add_argument("--llm_model", type=str,
                   default="microsoft/Phi-3.5-mini-instruct",
                   help="HuggingFace identifier for the LLM.")
    p.add_argument("--caption_model", type=str,
                   default="llava-hf/llava-v1.5-7b",
                   help="HuggingFace identifier for the captioner (caption-based selector only).")
    p.add_argument("--download_dir", type=str, default=None,
                   help="HF cache directory passed to vllm.")
    p.add_argument("--vlm_tp", type=int, default=1, help="VideoLM tensor parallel size.")
    p.add_argument("--llm_tp", type=int, default=1, help="LLM tensor parallel size.")
    p.add_argument("--vlm_max_model_len", type=int, default=8192)
    p.add_argument("--llm_max_model_len", type=int, default=8192)
    p.add_argument("--vlm_gpu_util", type=float, default=0.5)
    p.add_argument("--llm_gpu_util", type=float, default=0.4)

    # ---- frame sampling ----------------------------------------------------
    p.add_argument("--frame_selector", type=str, default="image",
                   choices=["image", "caption"],
                   help="Frame selector for the zoom-in stage.")
    p.add_argument("--num_frames", type=int, default=32,
                   help="Frames given to the VideoLM in each call.")
    p.add_argument("--candidate_frames", type=int, default=128,
                   help="Dense candidate pool inspected by the frame selector.")

    # ---- environment -------------------------------------------------------
    p.add_argument("--gpu_num", type=str, default="0",
                   help="CUDA_VISIBLE_DEVICES value.")

    return p.parse_args()


def load_inputs(path):
    """Read a JSONL where each row has at least: video, question, options, answer.

    Supports the NExT-QA dump format used by the original Iter-Feedback-VQA
    repo (``doc.video``, ``doc.question``, ``doc.a0..a4``, ``doc.answer``),
    or a flat schema ``{video, question, options, answer}``.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "doc" in r:
                doc = r["doc"]
                video = doc["video"]
                question = doc["question"]
                options = [doc[k] for k in ("a0", "a1", "a2", "a3", "a4") if k in doc]
                answer = doc.get("answer")
                doc_id = r.get("doc_id")
            else:
                video = r["video"]
                question = r["question"]
                options = r.get("options") or r.get("choices") or []
                answer = r.get("answer")
                doc_id = r.get("doc_id") or r.get("qid")
            rows.append({
                "doc_id": doc_id,
                "video": video,
                "question": question,
                "options": options,
                "answer": answer,
            })
    return rows


def _short(name):
    return name.split("/")[-1]


def _run_tag(args):
    variant = args.variant.upper()
    if variant == "AIoR".upper():
        iters_tag = f"maxIter{args.max_iterations}"
    elif variant == "GIoR".upper():
        iters_tag = f"iter{args.num_iterations}"
    else:
        iters_tag = f"nQ{args.num_feedback_questions}"
    return (
        f"{variant}_{iters_tag}_fs-{args.frame_selector}_"
        f"vlm-{_short(args.vlm_model)}_llm-{_short(args.llm_model)}"
    )


def main():
    args = parse_args()
    args.variant = args.variant.upper()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    container = get_container_hostname()
    run_name = args.run_name or _run_tag(args)
    save_folder = os.path.join(args.results_path, container, run_name)
    os.makedirs(save_folder, exist_ok=True)
    print(f"Saving results to: {save_folder}")

    print(f"Loading VideoLM: {args.vlm_model}")
    videolm = VideoLMWrapper(
        args.vlm_model,
        tensor_parallel_size=args.vlm_tp,
        max_model_len=args.vlm_max_model_len,
        download_dir=args.download_dir,
        gpu_memory_utilization=args.vlm_gpu_util,
    )
    print(f"Loading LLM: {args.llm_model}")
    llm = LLMWrapper(
        args.llm_model,
        tensor_parallel_size=args.llm_tp,
        max_model_len=args.llm_max_model_len,
        download_dir=args.download_dir,
        gpu_memory_utilization=args.llm_gpu_util,
    )

    captioner = None
    if args.frame_selector == "caption":
        print(f"Loading captioner: {args.caption_model}")
        captioner = VisionCaptioner(
            args.caption_model,
            download_dir=args.download_dir,
        )
    selector = build_frame_selector(args.frame_selector, captioner=captioner)

    pipeline = RefineVQA(videolm=videolm, llm=llm, frame_selector=selector, args=args)

    rows = load_inputs(args.input_file)
    if args.limit:
        rows = rows[: args.limit]

    by_video = {}
    for r in rows:
        by_video.setdefault(r["video"], []).append(r)

    for vid, items in tqdm(list(by_video.items()), desc=run_name):
        out_path = os.path.join(save_folder, f"{vid}.json")
        if args.resume and os.path.exists(out_path):
            continue
        results = []
        for item in items:
            try:
                trace = pipeline.run(
                    video_root=args.video_path,
                    vid=vid,
                    question=item["question"],
                    options=item["options"],
                )
            except Exception as exc:
                trace = {"error": str(exc)}
            target_letter = None
            if item["answer"] is not None:
                try:
                    target_letter = ANS_ORDER[int(item["answer"])]
                except (ValueError, TypeError, IndexError):
                    target_letter = str(item["answer"])
            results.append({
                "doc_id": item["doc_id"],
                "video": vid,
                "question": item["question"],
                "options": item["options"],
                "answer": item["answer"],
                "target_letter": target_letter,
                "trace": trace,
                "predicted_letter": (trace.get("answer") or {}).get("letter") if isinstance(trace, dict) else None,
            })
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=to_serializable)

    print("Done.")


if __name__ == "__main__":
    main()

"""RefineVQA pipeline with AIoR / GIoR / BPoR variants.

Reproduces Section 3 of the paper:

  Initial description -> (Feedback -> Zoom-in -> Refinement) x K -> Answer
"""

import re

from .prompts import (
    ANSWER_PROMPT,
    FEEDBACK_BATCH_PROMPT,
    FEEDBACK_PROMPT,
    REFINE_BATCH_PROMPT,
    REFINE_PROMPT,
    vlm_initial_description_prompt,
    vlm_zoomin_description_prompt,
)
from .utils import (
    extract_candidate_frames,
    format_options,
    load_video_at_indices,
    load_video_uniform,
    parse_answer_letter,
)


def _parse_feedback(text):
    """Parse the LLM feedback output into (enough, reasoning, feedback_question)."""
    enough = False
    reasoning = ""
    feedback_question = ""

    m = re.search(r"Decision\s*:\s*(Enough|Not\s*Enough)", text, re.IGNORECASE)
    if m:
        enough = m.group(1).strip().lower() == "enough"
    else:
        # If the model failed to emit a Decision line, treat ambiguity as
        # "not enough" so iteration continues — but only if a question exists.
        enough = "?" not in text

    r = re.search(r"Reasoning\s*:\s*(.+)", text)
    if r:
        reasoning = r.group(1).split("\n")[0].strip()

    q = re.search(r"Feedback Question\s*:\s*(.+)", text)
    if q:
        feedback_question = q.group(1).split("\n")[0].strip()
    else:
        # Fall back to the first line ending in '?'
        for line in text.split("\n"):
            line = line.strip()
            if line.endswith("?"):
                feedback_question = line.lstrip("-*0123456789. )").strip()
                break
    return enough, reasoning, feedback_question


def _parse_batch_questions(text, n):
    questions = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*\d\.\)\s]+", "", line)
        if "?" in line and line not in questions:
            questions.append(line)
        if len(questions) >= n:
            break
    return questions


class RefineVQA:
    def __init__(self, videolm, llm, frame_selector, args):
        self.videolm = videolm
        self.llm = llm
        self.frame_selector = frame_selector
        self.args = args

    # --- atomic steps -------------------------------------------------------

    def initial_description(self, video_root, vid):
        video = load_video_uniform(video_root, vid, num_frames=self.args.num_frames)
        prompt = vlm_initial_description_prompt()
        return self.videolm.generate(prompt, video)

    def feedback_step(self, description, question, options):
        opts = format_options(options)
        prompt = FEEDBACK_PROMPT % (description, question, opts)
        raw = self.llm.generate(prompt)
        enough, reasoning, fq = _parse_feedback(raw)
        return {"raw": raw, "enough": enough, "reasoning": reasoning, "feedback_question": fq}

    def batch_feedback_step(self, description, question, options, n):
        opts = format_options(options)
        prompt = FEEDBACK_BATCH_PROMPT % (n, description, question, opts)
        raw = self.llm.generate(prompt)
        questions = _parse_batch_questions(raw, n)
        return {"raw": raw, "questions": questions}

    def zoomin_description(self, video_root, vid, feedback_question, candidate_frames, candidate_indices):
        sel_kwargs = {}
        if self.args.frame_selector == "caption":
            sel_kwargs["video_id"] = vid
        selected = self.frame_selector.select(
            candidate_frames,
            candidate_indices,
            feedback_question,
            top_k=self.args.num_frames,
            **sel_kwargs,
        )
        video = load_video_at_indices(video_root, vid, selected)
        prompt = vlm_zoomin_description_prompt(feedback_question)
        text = self.videolm.generate(prompt, video)
        return text, selected

    def refine_step(self, initial, zoom_in):
        prompt = REFINE_PROMPT % (initial, zoom_in)
        return self.llm.generate(prompt)

    def refine_batch_step(self, initial, zoom_ins):
        zoom_block = "\n".join(
            f"Zoom-in description {i+1}: {z}" for i, z in enumerate(zoom_ins)
        )
        prompt = REFINE_BATCH_PROMPT % (initial, zoom_block)
        return self.llm.generate(prompt)

    def answer_step(self, description, question, options):
        opts = format_options(options)
        prompt = ANSWER_PROMPT % (description, question, opts)
        raw = self.llm.generate(prompt)
        letter = parse_answer_letter(raw, num_options=len(options))
        return {"raw": raw, "letter": letter}

    # --- variants -----------------------------------------------------------

    def run(self, video_root, vid, question, options):
        variant = self.args.variant.lower()
        if variant == "aior":
            return self._run_aior(video_root, vid, question, options)
        if variant == "gior":
            return self._run_gior(video_root, vid, question, options)
        if variant == "bpor":
            return self._run_bpor(video_root, vid, question, options)
        raise ValueError(f"Unknown variant: {self.args.variant}")

    def _prep_candidates(self, video_root, vid):
        return extract_candidate_frames(
            video_root, vid, candidate_count=self.args.candidate_frames
        )

    def _run_aior(self, video_root, vid, question, options):
        trace = {"variant": "AIoR", "iterations": []}
        description = self.initial_description(video_root, vid)
        trace["initial_description"] = description

        candidate_frames, candidate_indices = self._prep_candidates(video_root, vid)
        for it in range(self.args.max_iterations):
            fb = self.feedback_step(description, question, options)
            iter_log = {"feedback": fb}
            if fb["enough"] or not fb["feedback_question"]:
                trace["iterations"].append(iter_log)
                break

            zoom_in, sel = self.zoomin_description(
                video_root, vid, fb["feedback_question"], candidate_frames, candidate_indices
            )
            description = self.refine_step(description, zoom_in)
            iter_log["zoom_in"] = zoom_in
            iter_log["selected_frames"] = sel
            iter_log["refined_description"] = description
            trace["iterations"].append(iter_log)

        ans = self.answer_step(description, question, options)
        trace["final_description"] = description
        trace["answer"] = ans
        return trace

    def _run_gior(self, video_root, vid, question, options):
        trace = {"variant": "GIoR", "iterations": []}
        description = self.initial_description(video_root, vid)
        trace["initial_description"] = description

        candidate_frames, candidate_indices = self._prep_candidates(video_root, vid)
        k = self.args.num_iterations
        for it in range(k):
            fb = self.feedback_step(description, question, options)
            # GIoR forces refinement regardless of the Enough decision.
            if not fb["feedback_question"]:
                # Skip iteration if the LLM gave us nothing to act on.
                trace["iterations"].append({"feedback": fb})
                continue

            zoom_in, sel = self.zoomin_description(
                video_root, vid, fb["feedback_question"], candidate_frames, candidate_indices
            )
            description = self.refine_step(description, zoom_in)
            trace["iterations"].append({
                "feedback": fb,
                "zoom_in": zoom_in,
                "selected_frames": sel,
                "refined_description": description,
            })

        ans = self.answer_step(description, question, options)
        trace["final_description"] = description
        trace["answer"] = ans
        return trace

    def _run_bpor(self, video_root, vid, question, options):
        trace = {"variant": "BPoR"}
        description = self.initial_description(video_root, vid)
        trace["initial_description"] = description

        candidate_frames, candidate_indices = self._prep_candidates(video_root, vid)
        n = self.args.num_feedback_questions
        batch = self.batch_feedback_step(description, question, options, n)
        zoom_ins, selected_all = [], []
        for fq in batch["questions"][:n]:
            zoom_in, sel = self.zoomin_description(
                video_root, vid, fq, candidate_frames, candidate_indices
            )
            zoom_ins.append(zoom_in)
            selected_all.append(sel)

        if zoom_ins:
            description = self.refine_batch_step(description, zoom_ins)

        ans = self.answer_step(description, question, options)
        trace["feedback_questions"] = batch["questions"]
        trace["zoom_ins"] = zoom_ins
        trace["selected_frames"] = selected_all
        trace["final_description"] = description
        trace["answer"] = ans
        return trace

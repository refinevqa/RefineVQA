"""Prompts used throughout the RefineVQA pipeline.

The pipeline produces five distinct kinds of generations:

  1. Initial description (VideoLM, no question access).
  2. Feedback decision + feedback question (LLM).
  3. Zoom-in description grounded on a feedback question (VideoLM).
  4. Refinement that merges initial and zoom-in descriptions (LLM).
  5. Final answer over the refined description (LLM).
"""

# ---------------------------------------------------------------------------
# VLM (VideoLM) prompts
# ---------------------------------------------------------------------------

def vlm_initial_description_prompt():
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<video>\n"
        "Please provide a detailed description of the video, focusing on the "
        "main subjects, their actions, and the temporal flow of events from "
        "beginning to end."
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def vlm_zoomin_description_prompt(feedback_question):
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<video>\n"
        f"{feedback_question}\n"
        "Answer the question above based only on what is visible in the video, "
        "describing the relevant motion and visual details."
        "<|im_end|>\n<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

FEEDBACK_PROMPT = """You are a strict reasoning assistant. Read a video description and a multiple-choice question, then judge whether the description contains enough visual information to answer the question.

Rules:
- If the description clearly supports a single correct option, output exactly: Decision: Enough
- Otherwise, identify what visual evidence is missing and write one follow-up question that, if answered by inspecting the video, would resolve the ambiguity.
- The follow-up question must be answerable from the video alone.

Format your reply exactly as one of:
Decision: Enough
Reasoning: <one sentence>

OR

Decision: Not Enough
Reasoning: <one sentence describing what is missing>
Feedback Question: <one concise follow-up question, ending with '?'>

Example 1:
Video description: A child sits in a high chair and laughs while a woman waves a green cloth in front of them.
Question: why does the woman touch the baby at the start of the video?
Options: A.put on bib B.adjusting her hair C.prevent baby from falling D.hugging her E.to feed her
Decision: Not Enough
Reasoning: The description does not specify what the woman does with her hand at the very start of the video.
Feedback Question: What does the woman do with her hand toward the baby at the very start of the video?

Example 2:
Video description: A man swings a golf club, finishes the swing, then turns toward the camera and gives a thumbs up before walking off.
Question: what did the lady do while turning back?
Options: A.walk away B.thumbs up C.put down her club D.applying cream on face E.caressing for the dog
Decision: Enough
Reasoning: The description states the person gives a thumbs up after the swing.

Now process the following:
Video description: %s
Question: %s
Options: %s
"""


FEEDBACK_BATCH_PROMPT = """You are a strict reasoning assistant. Given a video description and a multiple-choice question, list %d distinct follow-up questions that probe different missing visual details needed to answer the question.

Each question must:
- Be answerable from the video.
- End with a question mark.
- Target a different piece of missing evidence.

Format:
1. <question 1>
2. <question 2>
...

Video description: %s
Question: %s
Options: %s

Questions:
"""


REFINE_PROMPT = """You will be given two descriptions of the same video. Please combine and refine them into a single, coherent, time-ordered description that preserves the global temporal flow of the first description while incorporating the additional visual details from the second.

Rules:
- Keep the temporal order of events from the initial description.
- Insert the new details from the zoom-in description at the moments they occur.
- Do not introduce information that is not present in either description.
- Output only the refined description, no headings.

Initial description: %s
Zoom-in description: %s

Refined description:"""


REFINE_BATCH_PROMPT = """You will be given an initial video description and several zoom-in descriptions answering different follow-up questions about the same video. Please merge them into a single, coherent, time-ordered description that preserves the temporal flow of the initial description while integrating the relevant details from each zoom-in description.

Rules:
- Keep the temporal order of events from the initial description.
- Integrate the additional details at the moments they occur.
- Do not introduce information not present in the descriptions.
- Output only the refined description.

Initial description: %s
%s
Refined description:"""


ANSWER_PROMPT = """You are a helpful assistant that answers questions about the video description.
You are given a Context (video description) and you choose exactly one of the provided options.
Output a short reasoning sentence and then the answer in the exact format "Answer:<LETTER>.<text>".

Video description: A child sits on a blanket next to a large dog that is lying still on its back; the child gently pats the dog and the dog occasionally moves a paw.
Question: what does the dog do after lying still for a while in the middle?
Options: A.lie down B.shake its body C.move to right side D.look behind E.put hand on floor
Reasoning: The dog is mostly still, so the most likely subtle action is putting its paw on the floor.
Answer:E.put hand on floor

Video description: %s
Question: %s
Options: %s
Reasoning:"""

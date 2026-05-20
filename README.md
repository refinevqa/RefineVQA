# RefineVQA

Reference reproduction of *ReFineVQA: Iterative Refinement of Video Description via Feedback Generation for Video Question Answering* (WACV 2026). Implements the full pipeline — initial description, feedback generation, zoom-in description, refinement, answer — and all three refinement variants: **AIoR**, **GIoR**, **BPoR**.

## Input format

`--input_file` is a JSONL file. Each row must describe one (video, question, options) sample. Two schemas are accepted:

Flat:

```json
{"doc_id": 0, "video": "<video-id>", "question": "...", "options": ["...", "...", "...", "...", "..."], "answer": 2}
```

Or the NExT-QA dump used by the original Iter-Feedback-VQA repo (rows containing a `doc` field with `video`, `question`, `a0`..`a4`, `answer`).

`--video_path` is the root directory containing the corresponding video files (`<video>.mp4`). Files are looked up recursively.

## Usage

`--vlm_model` / `--llm_model` / `--caption_model` accept any HuggingFace identifier that vllm can load — swap in any VideoLM, LLM, or captioner.

Inference (AIoR — early-stopping iteration, the default in the paper):

```bash
python main.py \
    --input_file <path>.jsonl \
    --video_path <video-root-dir> \
    --variant AIoR \
    --max_iterations {N} \
    --vlm_model {videolm} \
    --llm_model {llm} \
    --frame_selector image \
    --gpu_num 0
```

GIoR — fixed number of refinement iterations:

```bash
python main.py \
    --input_file <path>.jsonl --video_path <video-root-dir> \
    --variant GIoR --num_iterations {N} \
    --vlm_model {videolm} \
    --llm_model {llm}
```

BPoR — generate multiple feedback questions in one shot, merge in a single refinement:

```bash
python main.py \
    --input_file <path>.jsonl --video_path <video-root-dir> \
    --variant BPoR --num_feedback_questions {N} \
    --vlm_model {videolm} \
    --llm_model {llm}
```

Defaults for every flag are visible via `python main.py --help`.


## Citations

```
@InProceedings{Shin_2026_WACV,
    author    = {Shin, Jeongwan and Hur, Chan and Cho, Seongmin and Choi, Jaeho and Park, Hyeyoung},
    title     = {ReFineVQA: Iterative Refinement of Video Description via Feedback Generation for Video Question Answering},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {March},
    year      = {2026},
    pages     = {7647-7657}
}
```
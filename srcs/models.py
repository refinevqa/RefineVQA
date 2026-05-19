"""Lightweight wrappers around vllm for the VideoLM and the LLM.

The wrappers are intentionally model-agnostic — any HuggingFace identifier
that vllm understands can be passed via ``--vlm_model`` / ``--llm_model``.
The default prompt templates target LLaVA-OneVision-style VideoLMs (Qwen2
chat template) and Phi-3-style LLMs, which is what the paper evaluates.
"""

from vllm import LLM, SamplingParams

from PIL import Image


class VideoLMWrapper:
    def __init__(self, model_id, tensor_parallel_size=1, max_model_len=8192,
                 download_dir=None, gpu_memory_utilization=0.9):
        kwargs = dict(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if download_dir:
            kwargs["download_dir"] = download_dir
        self.llm = LLM(**kwargs)
        self.default_params = SamplingParams(
            temperature=0.0, top_k=1, max_tokens=256,
            frequency_penalty=1.0, presence_penalty=0.5,
        )

    def generate(self, prompt, video, sampling_params=None):
        params = sampling_params or self.default_params
        outputs = self.llm.generate(
            {"prompt": prompt, "multi_modal_data": {"video": video}},
            sampling_params=params,
            use_tqdm=False,
        )
        return outputs[0].outputs[0].text.strip()


class LLMWrapper:
    def __init__(self, model_id, tensor_parallel_size=1, max_model_len=8192,
                 download_dir=None, gpu_memory_utilization=0.9):
        kwargs = dict(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if download_dir:
            kwargs["download_dir"] = download_dir
        self.llm = LLM(**kwargs)
        self.default_params = SamplingParams(
            temperature=0.0, top_k=1, max_tokens=512,
            frequency_penalty=1.0, presence_penalty=0.5,
            stop="\n\n",
        )

    def generate(self, prompt, sampling_params=None):
        params = sampling_params or self.default_params
        outputs = self.llm.generate([prompt], params, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()


class VisionCaptioner:
    """Optional per-frame captioner used by the caption-based frame selector."""

    def __init__(self, model_id, tensor_parallel_size=1, max_model_len=4096,
                 download_dir=None, gpu_memory_utilization=0.5):
        kwargs = dict(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if download_dir:
            kwargs["download_dir"] = download_dir
        self.llm = LLM(**kwargs)
        self.default_params = SamplingParams(
            temperature=0.0, top_k=1, max_tokens=64,
        )

    def _prompt(self):
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<image>\nDescribe this frame in one sentence."
            "<|im_end|>\n<|im_start|>assistant\n"
        )

    def caption_frames(self, frames):
        prompt = self._prompt()
        captions = []
        for f in frames:
            img = Image.fromarray(f)
            o = self.llm.generate(
                {"prompt": prompt, "multi_modal_data": {"image": img}},
                sampling_params=self.default_params,
                use_tqdm=False,
            )
            captions.append(o[0].outputs[0].text.strip())
        return captions

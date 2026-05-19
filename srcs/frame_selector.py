"""Frame selectors used during Zoom-in Description Generation.

Two implementations are provided, matching Section 3.3 of the paper:

  * ``ImageFrameSelector`` performs image-to-text matching between candidate
    frames and the feedback question using a vision-language encoder (CLIP).
  * ``CaptionFrameSelector`` performs text-to-text matching between the
    feedback question and frame captions produced by a VisionLM, using a
    pretrained text encoder.

Both expose the same ``select(frames, indices, question, top_k)`` signature.
"""

import numpy as np
import torch
from PIL import Image


class ImageFrameSelector:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def select(self, frames, indices, question, top_k):
        images = [Image.fromarray(f) for f in frames]
        inputs = self.processor(
            text=[question], images=images, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        img_emb = out.image_embeds / out.image_embeds.norm(p=2, dim=-1, keepdim=True)
        txt_emb = out.text_embeds / out.text_embeds.norm(p=2, dim=-1, keepdim=True)
        sims = (img_emb @ txt_emb.T).squeeze(-1)
        k = min(top_k, sims.shape[0])
        top = torch.topk(sims, k=k).indices.cpu().tolist()
        top_sorted = sorted(top)
        return [int(indices[i]) for i in top_sorted]


class CaptionFrameSelector:
    """Text-to-text matching between feedback question and per-frame captions.

    Captions are produced once per video by a VisionLM and cached.
    """

    def __init__(self, captioner, text_encoder_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda"):
        from transformers import AutoModel, AutoTokenizer
        self.captioner = captioner
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        self.text_model = AutoModel.from_pretrained(text_encoder_name).to(device)
        self.text_model.eval()
        self._caption_cache = {}

    @torch.no_grad()
    def _embed(self, texts):
        tok = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        out = self.text_model(**tok)
        mask = tok["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb

    def _captions_for(self, video_id, frames, indices):
        key = str(video_id)
        if key in self._caption_cache:
            return self._caption_cache[key]
        captions = self.captioner.caption_frames(frames)
        self._caption_cache[key] = captions
        return captions

    @torch.no_grad()
    def select(self, frames, indices, question, top_k, video_id=None):
        captions = self._captions_for(video_id if video_id is not None else id(frames), frames, indices)
        cap_emb = self._embed(captions)
        q_emb = self._embed([question])
        sims = (cap_emb @ q_emb.T).squeeze(-1)
        k = min(top_k, sims.shape[0])
        top = torch.topk(sims, k=k).indices.cpu().tolist()
        top_sorted = sorted(top)
        return [int(indices[i]) for i in top_sorted]


def build_frame_selector(kind, captioner=None, device="cuda"):
    if kind == "image":
        return ImageFrameSelector(device=device)
    if kind == "caption":
        if captioner is None:
            raise ValueError("caption-based selector requires a captioner instance")
        return CaptionFrameSelector(captioner=captioner, device=device)
    raise ValueError(f"Unknown frame selector kind: {kind}")

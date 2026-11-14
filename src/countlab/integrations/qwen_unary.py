"""Lộc-inspired Qwen3-VL *region* verifier, integrated as actual model calls.

Preserved source: legacy/loc/notebooks/01_original_pipeline/V4_Qwen3_VL_Verification.ipynb
and 04_validation/V7_5_Blind_Validation_240.ipynb. This extracts the genuine
processor/generate pattern but uses NEW prompts for distinct semantic and whole
checks. It does NOT reproduce V7.6 global veto, V14 CycleGRPO verification,
or past metrics. Exact model revision and preprocessing must be frozen per run.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Callable

from countlab.method.schema import Proposal

_YESNO = re.compile(r"\s*(YES|NO|UNKNOWN)\s*", re.IGNORECASE)


def parse_verdict(text: str) -> str:
    """Fail closed to UNKNOWN on explanation, malformed or ambiguous output."""
    if not isinstance(text, str):
        return "UNKNOWN"
    m = _YESNO.fullmatch(text.strip())
    return m.group(1).upper() if m else "UNKNOWN"


def region_views(image_path: Path, box: tuple[float, float, float, float], *, pad: float=.3):
    """Return highlighted full-image and padded crop; never silently use GT."""
    from PIL import Image, ImageDraw
    with Image.open(image_path) as img:
        full = img.convert("RGB")
    w, h = full.size
    x0,y0,x1,y1 = box
    x0,y0 = max(0., min(w, x0)), max(0., min(h,y0))
    x1,y1 = max(0., min(w, x1)), max(0., min(h,y1))
    if x1<=x0 or y1<=y0:
        raise ValueError("candidate box lies outside image")
    highlighted = full.copy()
    thick = max(2, int(min(w,h)/250))
    ImageDraw.Draw(highlighted).rectangle((x0,y0,x1,y1), outline=(255,0,0), width=thick)
    dx,dy = max(1., x1-x0)*pad, max(1., y1-y0)*pad
    crop = full.crop((max(0,int(x0-dx)), max(0,int(y0-dy)),
                      min(w,int(x1+dx)), min(h,int(y1+dy))))
    return highlighted, crop


def verification_prompts(query: str) -> tuple[str, str]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("nonempty target query required")
    # Every output is a 3-way verdict; UNKNOWN never means NO.
    semantic = ("The red rectangle marks one candidate region in the first image; the second image is its local crop. "
                f"Target category: {query!r}. Is the main object indicated by the RED rectangle actually an instance "
                "of that target category? Ignore other objects outside the rectangle. "
                "Reply with EXACTLY one word: YES, NO, or UNKNOWN (insufficient evidence).")
    whole = ("The red rectangle marks one candidate region in the first image; the second image is its local crop. "
             "Does the marked region correspond to ONE WHOLE physical object, rather than only a body part, "
             "texture, background area, or multiple objects? Partial occlusion may still be a whole instance "
             "if object identity is clear. Reply with EXACTLY one word: YES, NO, or UNKNOWN.")
    return semantic, whole


class QwenRegionUnary:
    """Two GT-blind deterministic generation calls per selected candidate."""
    def __init__(self, *, model_id: str = "Qwen/Qwen3-VL-4B-Instruct", revision: str,
                 generator: Callable | None = None, max_new_tokens: int = 16):
        if not revision or revision in ("main", "latest"):
            raise ValueError("pin an exact Qwen model revision; never use moving main")
        if max_new_tokens < 4 or max_new_tokens > 128:
            raise ValueError("invalid generation cap")
        self.model_id, self.revision = model_id, revision
        self.max_new_tokens = max_new_tokens
        self.calls = 0
        self.fallbacks = 0
        self.generator = generator
        if generator is None:
            self._load_gpu_model()

    def _load_gpu_model(self):
        try:
            import torch
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
        except ImportError as e:
            raise RuntimeError("Install transformers/Qwen3-VL, qwen-vl-utils, torch for online verifier") from e
        self._torch = torch
        self._process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id, revision=self.revision, dtype=torch.float16,
            device_map="auto", low_cpu_mem_usage=True).eval()
        self.generator = self._generate

    def _generate(self, full_image, crop, prompt):
        messages = [{"role":"user", "content":[
            {"type":"image", "image":full_image}, {"type":"image", "image":crop},
            {"type":"text", "text":prompt}]}]
        processor = self.processor
        string = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img, video, kwargs = self._process_vision_info(
            messages, image_patch_size=16, return_video_kwargs=True,
            return_video_metadata=True)
        metas = None
        if video is not None:
            video, metas = zip(*video)
            video, metas = list(video), list(metas)
        inputs = processor(text=[string], images=img, videos=video, video_metadata=metas,
                           padding=True, return_tensors="pt", do_resize=False, **kwargs)
        inputs = inputs.to(next(self.model.parameters()).device)
        with self._torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = [a[len(b):] for b,a in zip(inputs.input_ids, output)]
        return processor.batch_decode(generated, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]

    def verdict(self, image_path: Path, query: str, proposal: Proposal) -> tuple[str,str]:
        full, crop = region_views(image_path, proposal.box)
        semantic_prompt, whole_prompt = verification_prompts(query)
        result = []
        for prompt in (semantic_prompt, whole_prompt):
            self.calls += 1
            try:
                label = parse_verdict(self.generator(full, crop, prompt))
            except Exception:
                label = "UNKNOWN"
            if label == "UNKNOWN":
                self.fallbacks += 1
            result.append(label)
        return result[0], result[1]


def identity_pair_views(image_path: Path, a: Proposal, b: Proposal):
    """Display two proposals in the original scene + comparative crop panel."""
    from PIL import Image, ImageDraw
    with Image.open(image_path) as image:
        full=image.convert("RGB")
    w,h=full.size
    draw=ImageDraw.Draw(full)
    draw.rectangle(a.box,outline=(255,0,0),width=max(2,min(w,h)//250))
    draw.rectangle(b.box,outline=(0,80,255),width=max(2,min(w,h)//250))
    _, c1=region_views(image_path,a.box)
    _, c2=region_views(image_path,b.box)
    # Keep a consistent panel without stretching or scaling the image.
    combined=Image.new("RGB",(c1.width+c2.width,max(c1.height,c2.height)),(255,255,255))
    combined.paste(c1,(0,0));combined.paste(c2,(c1.width,0))
    return full,combined


def verify_identity_pairs(image_path: Path, candidates: list[Proposal], unary: dict[str, tuple[str,str]],
                          verifier: QwenRegionUnary, *, max_pairs: int,
                          min_overlap: float=.1) -> list[dict]:
    """Optional actual Qwen comparison, never geometry-as-identity evidence.

    No guessed SAME edges. Only eligible YES/YES candidates with distinct view
    IDs and meaningful overlap are queried. Failures produce UNKNOWN.
    """
    from countlab.method.schema import box_iou
    if max_pairs < 0 or not 0 <= min_overlap <= 1:
        raise ValueError("invalid pair verification budget")
    eligible=[c for c in candidates if unary.get(c.candidate_id)==('YES','YES')]
    pairs=[]
    for i,a in enumerate(eligible):
        for b in eligible[i+1:]:
            overlap=box_iou(a.box,b.box)
            if a.view_id!=b.view_id and overlap>=min_overlap:
                pairs.append((-overlap,a.candidate_id,b.candidate_id,a,b))
    pairs.sort()
    edges=[]
    for _,_,_,a,b in pairs[:max_pairs]:
        full,panel=identity_pair_views(image_path,a,b)
        prompt=("The RED and BLUE rectangles mark two object proposals in the same ORIGINAL image. "
                "The second image displays their local crops, left=RED, right=BLUE. "
                "Do both proposals refer to the SAME single physical object, not two neighboring objects? "
                "Reply EXACTLY YES, NO, or UNKNOWN when ambiguous.")
        verifier.calls+=1
        try:
            result=parse_verdict(verifier.generator(full,panel,prompt))
        except Exception:
            result='UNKNOWN'
        if result=='UNKNOWN':
            verifier.fallbacks+=1
        relation={'YES':'SAME','NO':'DIFFERENT','UNKNOWN':'UNKNOWN'}[result]
        edges.append({'a':a.candidate_id,'b':b.candidate_id,'relation':relation})
    return edges

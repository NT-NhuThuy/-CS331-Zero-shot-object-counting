"""No upstream patch: short-lived read-only hooks capture BEFORE-fusion features."""
from __future__ import annotations

import argparse
import importlib.metadata
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
from PIL import Image
import torch
from .geometry import cxcywh_to_xyxy, region_tokens
from .io import file_hash, digest


@contextmanager
def working_directory(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class CountGDBackend:
    def __init__(self, cfg):
        self.cfg = cfg
        repo = Path(cfg["repo"])
        if not (repo / "models/GroundingDINO/groundingdino_app.py").is_file():
            raise FileNotFoundError("repo must point to the CountGDPlusPlus checkout")
        if not Path(cfg["bert_dir"]).is_dir() or not Path(cfg["checkpoint"]).is_file():
            raise FileNotFoundError("Check local bert_dir and CountGD++ checkpoint paths")
        self.device = torch.device(cfg["device"])
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; explicitly set device=cpu for diagnostics")
        sys.path.insert(0, str(repo))
        try:
            with working_directory(repo):
                from util.slconfig import SLConfig
                import models.GroundingDINO.groundingdino_app as app
                # Fail clearly if a different installed package owns the generic 'models' namespace.
                if repo not in Path(app.__file__).resolve().parents:
                    raise RuntimeError("Imported wrong models package; run cgground in a fresh process")
                args = argparse.Namespace(**SLConfig.fromfile(str(repo / "cfg_app.py"))._cfg_dict.to_dict())
                args.device = str(self.device)
                args.text_encoder_type = cfg["bert_dir"]
                if args.backbone != "swin_B_384_22k" or args.return_interm_indices != [1, 2, 3]:
                    raise ValueError("This adapter audits the default Swin-B backbone/strides only")
                self.model, _, _ = app.build_groundingdino(args)
                self.mask_fn = app.generate_masks_with_special_tokens_and_transfer_map
                from util.misc import nested_tensor_from_tensor_list
                self.nested_fn = nested_tensor_from_tensor_list
                import datasets.transforms_app as T
                self.transform = T.Compose([T.RandomResize([cfg["resize_short"]], max_size=cfg["resize_max"]),
                    T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        finally:
            # Upstream imports are now cached. Keep path for deferred native imports.
            pass
        # Only load a trusted, locally supplied upstream checkpoint.
        checkpoint = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        if not isinstance(state, dict):
            raise ValueError("Expected a state dict or {'model': state_dict}")
        state = {k.removeprefix("module."): v for k, v in state.items()}
        incompatible = self.model.load_state_dict(state, strict=False)
        allowed_suffixes = ("embeddings.position_ids", "embeddings.token_type_ids")
        missing = [k for k in incompatible.missing_keys if not k.endswith(allowed_suffixes)]
        # Legacy exemplar encoder is inactive in the audited upstream forward.
        legacy_suffixes = (
            "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
            "self_attn.in_proj_weight", "self_attn.in_proj_bias",
            "self_attn.out_proj.weight", "self_attn.out_proj.bias",
            "mlp.linear1.weight", "mlp.linear1.bias",
            "mlp.linear2.weight", "mlp.linear2.bias",
        )
        allowed_legacy = {
            f"feature_map_encoder.layers.{layer}.{suffix}"
            for layer in range(3)
            for suffix in legacy_suffixes
        } | {"feature_map_encoder.norm.weight", "feature_map_encoder.norm.bias"}
        ignored_legacy = sorted(
            k for k in incompatible.unexpected_keys if k in allowed_legacy
        )
        unexpected = [
            k for k in incompatible.unexpected_keys
            if not k.endswith(allowed_suffixes) and k not in allowed_legacy
        ]
        if ignored_legacy:
            print("Ignored inactive encoder checkpoint keys:", ignored_legacy)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch (not silently ignored). Missing={missing[:20]}; unexpected={unexpected[:20]}")
        self.model.to(self.device).eval().requires_grad_(False)
        self.levels = cfg.get("feature_levels", [0, 1])
        if self.levels != [0, 1]:
            raise ValueError("v1 supports feature_levels=[0,1], at strides 8 and 16")
        self.strides = [8, 16]
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
        self.spec = {"checkpoint_sha256": file_hash(cfg["checkpoint"]), "repo_commit": commit,
                     "forward_source_sha256": file_hash(repo / "models/GroundingDINO/groundingdino_app.py"),
                     "bert_config_sha256": file_hash(Path(cfg["bert_dir"]) / "config.json"),
                     "resize_short": cfg["resize_short"], "resize_max": cfg["resize_max"],
                     "grid_size": cfg["grid_size"], "context_scale": cfg["context_scale"],
                     "levels": self.levels, "strides": self.strides,
                     "features": "projected before fusion; no exemplar; single pass",
                     "native_score_rule": "engine_inference raw max sigmoid + all prompt content tokens > text_threshold",
                     "text_threshold": cfg.get('text_threshold', 0.0),
                     "roi": "roi_align aligned=True sampling_ratio=2"}
        self.spec['native_source_hashes'] = {
            str(p.relative_to(repo)): file_hash(p)
            for p in sorted((repo / 'models/GroundingDINO').rglob('*.py'))
        }
        for relative in ('cfg_app.py', 'datasets/transforms_app.py', 'util/misc.py'):
            self.spec['native_source_hashes'][relative] = file_hash(repo / relative)
        self.spec['tokenizer_hashes'] = {
            name: file_hash(Path(cfg['bert_dir']) / name)
            for name in ('vocab.txt', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')
            if (Path(cfg['bert_dir']) / name).is_file()
        }
        self.spec['versions'] = {name: importlib.metadata.version(name)
                                 for name in ('torch', 'torchvision', 'transformers', 'timm')}
        self.spec['extraction_device'] = str(self.device)
        self.spec["id"] = digest(self.spec)

    def _input(self, image_path):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        tensor, _ = self.transform(image, None)
        tensor = tensor.to(self.device)
        return tensor, image.size

    @torch.no_grad()
    def encode_text(self, query):
        caption = query + " ."
        model = self.model
        tokens = model.tokenizer([caption], padding="longest", return_tensors="pt").to(self.device)
        masks, pos, _ = self.mask_fn(tokens, model.specical_tokens, model.tokenizer)
        if tokens.input_ids.shape[1] > model.max_text_len:
            raise ValueError("Prompt too long; refusing silent text truncation")
        encoded_input = {k: v for k, v in tokens.items() if k != "attention_mask"}
        encoded_input.update(attention_mask=masks, position_ids=pos)
        if not model.sub_sentence_present:
            encoded_input = tokens
        values = model.feat_map(model.bert(**encoded_input)["last_hidden_state"])[0]
        ids = tokens.input_ids[0]
        special_ids = set(model.tokenizer.all_special_ids) | set(model.specical_tokens)
        valid = tokens.attention_mask[0].bool()
        for token in special_ids:
            valid &= ids != token
        if not valid.any():
            raise ValueError("Prompt has no content tokens")
        return values[valid].detach().float()

    @torch.no_grad()
    def features(self, image_path):
        tensor, original_size = self._input(image_path)
        nested = self.nested_fn([tensor])
        features, _ = self.model.backbone(nested)
        projected = [self.model.input_proj[level](features[level].tensors) for level in self.levels]
        return projected, (tensor.shape[2], tensor.shape[1]), original_size

    @torch.no_grad()
    def predict(self, image_path, query, check_hooks=False, capture_token_scores=False):
        tensor, original_size = self._input(image_path)
        nested = self.nested_fn([tensor])
        captured, handles = {}, []
        def capture(level):
            def hook(module, args, output):
                # Clone so later in-place modifications cannot corrupt our verifier evidence.
                captured[level] = output.detach().clone()
            return hook
        try:
            for level in self.levels:
                handles.append(self.model.input_proj[level].register_forward_hook(capture(level)))
            output = self.model(nested, nested, [torch.empty((0, 4), device=self.device)], [], [],
                                captions=[query + " ."])
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(self.levels):
            raise RuntimeError("Feature hooks did not run as expected")
        if check_hooks:
            plain = self.model(nested, nested, [torch.empty((0, 4), device=self.device)], [], [],
                               captions=[query + " ."])
            for key in ("pred_logits", "pred_boxes"):
                torch.testing.assert_close(output[key], plain[key], rtol=0, atol=0, equal_nan=True)
        boxes = output["pred_boxes"][0].detach().float()
        probabilities = output["pred_logits"][0].sigmoid()
        scores = probabilities.amax(dim=-1).detach().float()
        # Match the single-class get_count_errs text mask in engine_inference.py.
        ids = output.get('input_ids')
        if ids is None:
            ids = self.model.tokenizer([query + ' .'], return_tensors='pt').input_ids
        separators = torch.where(ids[0] == 1012)[0]
        if not len(separators) or int(separators[0]) <= 1:
            raise RuntimeError('Expected BERT content tokens followed by the period separator')
        end = int(separators[0])
        text_pass = (probabilities[:, 1:end] > self.cfg.get('text_threshold', 0.0)).all(dim=-1)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != boxes.shape[:1]:
            raise RuntimeError("Unexpected CountGD++ output shapes")
        if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
            raise RuntimeError("Nonfinite CountGD++ outputs")
        result = {
            "boxes_cxcywh_normalized": boxes,
            "native_scores": scores,
            "native_text_pass": text_pass,
            "features": [captured[level] for level in self.levels],
            "resized_size": (tensor.shape[2], tensor.shape[1]),
            "original_size": original_size,
        }
        if capture_token_scores:
            result["content_token_scores"] = probabilities[:, 1:end].detach().float().cpu()
            result["content_token_ids"] = ids[0, 1:end].detach().cpu()
        return result

    def tokens(self, features, boxes_original_xyxy, resized_size, original_size):
        ow, oh = original_size
        rw, rh = resized_size
        boxes = torch.as_tensor(boxes_original_xyxy, dtype=torch.float32, device=self.device).reshape(-1, 4)
        boxes = boxes * boxes.new_tensor([rw / ow, rh / oh, rw / ow, rh / oh])
        # Float32 multiplication can place an exact image-edge box a few ULPs outside.
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, rw)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, rh)
        return region_tokens(features, self.strides, boxes, resized_size,
                             self.cfg["grid_size"], self.cfg["context_scale"])

    def proposal_tokens(self, prediction, indices):
        boxes = cxcywh_to_xyxy(prediction["boxes_cxcywh_normalized"][indices])
        w, h = prediction["resized_size"]
        boxes = boxes * boxes.new_tensor([w, h, w, h])
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
        return region_tokens(prediction["features"], self.strides, boxes, (w, h),
                             self.cfg["grid_size"], self.cfg["context_scale"])

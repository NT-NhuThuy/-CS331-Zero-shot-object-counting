import torch
from torch import nn
from torch.nn import functional as F


class GroundingBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, text, visual):
        value, _ = self.attn(text, visual, visual, need_weights=False)
        text = self.ln1(text + value)
        return self.ln2(text + self.ff(text))


class RegionTextVerifier(nn.Module):
    def __init__(self, dim=256, heads=8, layers=2, dropout=0.1, grid_size=7, feature_levels=2,
                 architecture='cross_attention'):
        super().__init__()
        if dim % heads or layers < 1 or grid_size < 1 or feature_levels < 1:
            raise ValueError("Invalid verifier dimensions")
        if architecture not in ('cross_attention', 'mlp'):
            raise ValueError('architecture must be cross_attention or mlp')
        self.architecture = architecture
        self.settings = dict(dim=dim, heads=heads, layers=layers, dropout=dropout,
                             grid_size=grid_size, feature_levels=feature_levels, architecture=architecture)
        self.token_count = feature_levels * 2 * grid_size * grid_size
        if architecture == 'cross_attention':
            self.pos = nn.Parameter(torch.zeros(1, self.token_count, dim))
            nn.init.normal_(self.pos, std=0.02)
        else:
            self.register_buffer('pos', torch.zeros(1, self.token_count, dim))
        self.visual_norm, self.text_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(GroundingBlock(dim, heads, dropout)
                                    for _ in range(layers if architecture == 'cross_attention' else 0))
        self.head = nn.Sequential(nn.Linear(dim * 4, dim), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(dim, 1))
        # Central token(s) in tight ROIs only, across feature levels. Even grids supported.
        centers = sorted({(grid_size - 1) // 2, grid_size // 2})
        indices = [level * 2 * grid_size ** 2 + y * grid_size + x
                   for level in range(feature_levels) for y in centers for x in centers]
        self.register_buffer("center_indices", torch.tensor(indices, dtype=torch.long))

    def forward(self, visual, text, text_valid):
        if visual.ndim != 3 or visual.shape[1] != self.token_count:
            raise ValueError("ROI token count mismatch with model configuration")
        if text.ndim != 3 or text.shape[:2] != text_valid.shape or visual.shape[0] != text.shape[0]:
            raise ValueError("Batch/text mask mismatch")
        if not text_valid.any(dim=1).all():
            raise ValueError("Every sample needs at least one content token")
        center = self.visual_norm(visual.index_select(1, self.center_indices).mean(dim=1))
        if self.architecture == 'mlp':
            # Simple control with the same visual/text inputs and supervised pairs.
            center = self.visual_norm(0.5 * (visual.mean(dim=1)
                         + visual.index_select(1, self.center_indices).mean(dim=1)))
        visual = self.visual_norm(visual + self.pos)
        text = self.text_norm(text)
        for block in self.blocks:
            text = block(text, visual)
        mask = text_valid.to(text.dtype).unsqueeze(-1)
        pooled = (text * mask).sum(dim=1) / mask.sum(dim=1)
        joint = torch.cat([pooled, center, pooled * center, (pooled - center).abs()], dim=-1)
        return self.head(joint).squeeze(-1)


def grounding_loss(logits, labels, region_ids, margin=1.0, cf_weight=0.5):
    """Balance positive/negative BCE; compare prompts ONLY on the SAME physical ROI.

    Unmatched detections never become negative labels here. Multiple valid
    proposals may be positive; semantic matching is separate from deduplication.
    """
    if logits.ndim != 1 or logits.shape != labels.shape or logits.shape != region_ids.shape:
        raise ValueError("Expected aligned flat logits/labels/region IDs")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Grounding labels must be 0/1")
    pieces = [F.binary_cross_entropy_with_logits(logits[labels == v], labels[labels == v])
              for v in (0, 1) if (labels == v).any()]
    if not pieces:
        raise ValueError("Empty supervision batch")
    bce = torch.stack(pieces).mean()
    ranking = []
    for group in region_ids.unique():
        idx = region_ids == group
        pos, neg = logits[idx & (labels == 1)], logits[idx & (labels == 0)]
        if len(pos) and len(neg):
            ranking.append(F.relu(margin - pos[:, None] + neg[None, :]).mean())
    cf = torch.stack(ranking).mean() if ranking else logits.sum() * 0
    return bce + cf_weight * cf, {"bce": bce.detach(), "counterfactual": cf.detach()}

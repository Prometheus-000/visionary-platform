"""
Regional prompting for Krea 2.

This is *not* sd-forge-couple. Forge Couple has two code paths and neither one
reaches Krea 2:

  * `lib_couple/attention_couple.py` drives `set_model_attn2_patch` /
    `set_model_attn2_output_patch`. Only Forge's legacy UNet (`backend/nn/unet.py`)
    reads `transformer_options["patches"]["attn2_patch"]`. Krea 2's
    `SingleStreamDiT` never looks at them, so enabling Couple on Krea 2 is a
    silent no-op — the prompt lines get concatenated and nothing is regionalised.
  * `lib_couple/anima.py` monkeypatches `backend.nn.anima.SelfCrossAttention`,
    a class Krea 2 does not have.

The deeper reason is architectural. Couple's design assumes *cross*-attention:
image queries against a text-only key/value bank, which you can duplicate per
region and blend on the way out. Krea 2 is single-stream — text and image tokens
are concatenated into one sequence and go through plain self-attention
(`backend/nn/krea.py`, `SingleStreamDiT.forward`). There is no attn2 to patch.

Single-stream has its own, cheaper answer: mask the attention. Concatenate every
region's conditioning into the text span, then bias the attention scores so an
image token only sees the text of the regions covering it. Same result as
Couple, one forward pass instead of N, no duplicated batch.

Krea 2's blocks already thread a `mask` argument down to SDPA; upstream just
passes `None`. `forge/backend/nn/krea.py` carries a small vendor patch that
reads a builder out of `transformer_options` instead — see forge/VENDOR.md.

Weights are applied as `log(weight)` added to the scores before softmax, which
multiplies that region's attention by `weight` — the natural way to express
Couple's per-region weight in an additive mask.

Two costs worth knowing about. An explicit mask puts SDPA on its slower path
(no flash kernel), and the bias itself is O(sequence²) — ~70 MB at 1024px. Both
are paid only when regions are actually in use. It also assumes the attention
backend honours a mask, which `attention_pytorch` does; installing
sageattention or flash_attn would push Forge onto a backend that asserts
`mask is None` and falls back per call, so the inference image deliberately
installs neither.
"""

import math

from . import bootstrap

bootstrap.install()

import torch  # noqa: E402

NEG_INF = -1.0e4  # not -inf: an all-masked row would produce NaN under fp16 SDPA


class Region:
    """One prompt over a rectangle in normalised (0..1) canvas coordinates."""

    __slots__ = ("prompt", "x", "y", "width", "height", "weight")

    def __init__(self, prompt: str, x=0.0, y=0.0, width=1.0, height=1.0, weight=1.0):
        self.prompt = prompt
        self.x = max(0.0, min(1.0, float(x)))
        self.y = max(0.0, min(1.0, float(y)))
        self.width = max(0.0, min(1.0 - self.x, float(width)))
        self.height = max(0.0, min(1.0 - self.y, float(height)))
        self.weight = max(0.0, float(weight))

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt, "x": self.x, "y": self.y,
            "width": self.width, "height": self.height, "weight": self.weight,
        }


def columns(prompts: list[str], weight: float = 1.0) -> list[Region]:
    """Split the canvas into equal vertical strips — Couple's Basic/Horizontal."""
    n = max(1, len(prompts))
    return [Region(p, x=i / n, y=0.0, width=1 / n, height=1.0, weight=weight) for i, p in enumerate(prompts)]


def rows(prompts: list[str], weight: float = 1.0) -> list[Region]:
    """Equal horizontal bands — Couple's Basic/Vertical."""
    n = max(1, len(prompts))
    return [Region(p, x=0.0, y=i / n, width=1.0, height=1 / n, weight=weight) for i, p in enumerate(prompts)]


class RegionalMask:
    """
    Builds the attention bias for one sampling run.

    Constructed with the per-region text lengths (in the order their embeddings
    were concatenated) and their rectangles. `sequence_mask` and `text_mask` are
    called from the patched `SingleStreamDiT.forward`; both cache on the token
    geometry, so the cost is paid once per resolution rather than once per block
    per step.
    """

    def __init__(self, lengths: list[int], regions: list[Region], base_length: int = 0, base_weight: float = 1.0):
        assert len(lengths) == len(regions), "one text length per region"
        self.lengths = lengths
        self.regions = regions
        # A leading global prompt applied everywhere, if the caller supplied one.
        self.base_length = base_length
        self.base_weight = max(0.0, float(base_weight))
        self._seq_cache: dict = {}
        self._txt_cache: dict = {}

    @property
    def total_text(self) -> int:
        return self.base_length + sum(self.lengths)

    # region image side

    def _region_coverage(self, h: int, w: int, device) -> torch.Tensor:
        """(num_regions, h*w) float weights — how much each region owns each patch."""
        cov = torch.zeros(len(self.regions), h, w, device=device, dtype=torch.float32)
        for i, r in enumerate(self.regions):
            # Round outward so a strip never leaves a one-patch seam uncovered.
            y0 = int(math.floor(r.y * h))
            y1 = int(math.ceil((r.y + r.height) * h))
            x0 = int(math.floor(r.x * w))
            x1 = int(math.ceil((r.x + r.width) * w))
            y1 = max(y1, y0 + 1)
            x1 = max(x1, x0 + 1)
            cov[i, y0:y1, x0:x1] = r.weight
        return cov.reshape(len(self.regions), h * w)

    def sequence_mask(self, combined, *, txtlen, reflen, h, w, transformer_options):
        """
        Additive attention bias over the full [text | refs | image] sequence.

        Returns a 3-D (1, S, S) or (B, S, S) tensor; `attention_pytorch`
        unsqueezes it to (…, 1, S, S) so it broadcasts across heads. Uncond
        chunks get an all-zero slice — the negative prompt is a single
        un-regionalised span and must keep full attention.

        An image patch covered by no region simply sees no region text (its
        text columns are all masked); the image columns stay open, so the row
        never softmaxes over an entirely masked set.
        """
        batch = combined.shape[0]
        seq = combined.shape[1]
        device = combined.device
        cond_or_uncond = list(transformer_options.get("cond_or_uncond", [0]))

        if txtlen != self.total_text:
            # The text span is not the one we planned for — most likely the
            # uncond pass batched on its own. Leave attention untouched.
            return None

        key = (seq, txtlen, reflen, h, w, batch, tuple(cond_or_uncond), str(device))
        cached = self._seq_cache.get(key)
        if cached is not None:
            return cached

        imglen = h * w
        bias = torch.zeros(seq, seq, device=device, dtype=torch.float32)

        coverage = self._region_coverage(h, w, device)  # (n, imglen)
        img0 = txtlen + reflen
        img1 = img0 + imglen

        # Per-region text spans, laid out in the order the caller concatenated them.
        spans: list[tuple[int, int]] = []
        cursor = self.base_length
        for length in self.lengths:
            spans.append((cursor, cursor + length))
            cursor += length

        # Image queries -> region text keys: allow only where the region covers
        # the patch, biased by log(weight).
        for (t0, t1), cov in zip(spans, coverage):
            allowed = cov > 0
            column = torch.full((imglen,), NEG_INF, device=device, dtype=torch.float32)
            column[allowed] = torch.log(cov[allowed].clamp_min(1e-4))
            bias[img0:img1, t0:t1] = column.unsqueeze(1)

        # Region text queries -> image keys: mirror it, so a region's tokens are
        # informed by its own area rather than the whole canvas.
        for (t0, t1), cov in zip(spans, coverage):
            row = torch.where(
                cov > 0,
                torch.zeros_like(cov),
                torch.full_like(cov, NEG_INF),
            )
            bias[t0:t1, img0:img1] = row.unsqueeze(0)

        # Region text queries -> other regions' text keys: blocked, so the
        # regions stay semantically separate.
        for i, (t0, t1) in enumerate(spans):
            for j, (u0, u1) in enumerate(spans):
                if i != j:
                    bias[t0:t1, u0:u1] = NEG_INF

        # The global prompt (if any) attends to and is attended by everything;
        # its span is left at zero bias except for its own weight.
        if self.base_length and self.base_weight != 1.0:
            bias[img0:img1, 0:self.base_length] = math.log(max(self.base_weight, 1e-4))

        mask = _per_chunk(bias, batch, cond_or_uncond).to(combined.dtype)
        _remember(self._seq_cache, key, mask)
        return mask

    # region text side

    def text_mask(self, context, transformer_options):
        """
        Block-diagonal mask for the text-fusion refiner blocks.

        Without it the refiner mixes every region's tokens together before the
        main blocks ever see them, which quietly undoes the separation the
        sequence mask sets up. Shape (1, L, L), or (B, L, L) when an uncond
        chunk shares the batch and has to be left alone.
        """
        batch, length = context.shape[0], context.shape[1]
        if length != self.total_text:
            return None

        device = context.device
        cond_or_uncond = tuple(transformer_options.get("cond_or_uncond", [0]))
        key = (batch, length, cond_or_uncond, str(device))
        cached = self._txt_cache.get(key)
        if cached is not None:
            return cached

        bias = torch.full((length, length), NEG_INF, device=device, dtype=torch.float32)

        # The global span sees, and is seen by, everything.
        if self.base_length:
            bias[: self.base_length, :] = 0.0
            bias[:, : self.base_length] = 0.0

        cursor = self.base_length
        for span_len in self.lengths:
            bias[cursor : cursor + span_len, cursor : cursor + span_len] = 0.0
            cursor += span_len

        mask = _per_chunk(bias, batch, list(cond_or_uncond)).to(context.dtype)
        _remember(self._txt_cache, key, mask)
        return mask


def _per_chunk(bias: torch.Tensor, batch: int, cond_or_uncond: list[int]) -> torch.Tensor:
    """
    Lift a single (S, S) bias to the batch.

    When every chunk is a cond chunk the bias is returned as (1, S, S) and SDPA
    broadcasts it — worth doing, because a materialised (B, S, S) at 1024px is
    ~70 MB per batch item. Only a mixed cond/uncond batch pays for the copy.
    """
    if 1 not in cond_or_uncond:
        return bias.unsqueeze(0)

    out = bias.unsqueeze(0).expand(batch, -1, -1).clone()
    per_chunk = batch // max(1, len(cond_or_uncond))
    for idx, which in enumerate(cond_or_uncond):
        if which == 1:
            out[idx * per_chunk : (idx + 1) * per_chunk] = 0.0
    return out


def _remember(cache: dict, key, value, limit: int = 4) -> None:
    """Tiny bounded cache — geometry only changes when width/height/steps do."""
    if len(cache) >= limit:
        cache.clear()
    cache[key] = value

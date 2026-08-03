# Vendored Forge backend

Krea 2 inference runs on [sd-webui-forge-classic](https://github.com/Haoming02/sd-webui-forge-classic)
(`neo` branch), not on musubi-tuner. This directory holds the slice of that repo
the inference path actually needs, plus the small amount of code that replaces
the Gradio half.

**Vendored from** `neo` @ `609d3e3af3a9721770468cc6727e1f92fabf7682` (2026-08-03,
"update README"). Nothing outside this directory is needed to build or run — no
clone has to be present, and none is kept. The SHA is here so a future sync can
diff against exactly what was copied rather than guessing.

## What was copied

| Path | Source | Notes |
|---|---|---|
| `backend/` | `backend/` | verbatim, minus `backend/huggingface/` |
| `backend/huggingface/krea/` | same | 5 MB of configs; the other 57 MB are for models we do not load |
| `modules_forge/packages/` | same | `huggingface_guess`, `comfy`, `k_diffusion`, `gguf` — imported by bare name, so `packages/` goes on `sys.path` |
| `modules/sd_schedulers.py` | same | verbatim |

`backend/` is kept whole rather than pruned to the Krea 2 files. `backend/loader.py`
imports every diffusion engine at module scope, so pruning means editing the
loader, and editing the loader means re-editing it on every upstream sync. The
whole tree is ~6 MB.

## What was written

| Path | Replaces |
|---|---|
| `modules/__init__.py`, `shared.py`, `devices.py`, `prompt_parser.py` | the entire webui `modules` package |
| `krea2/` | `modules/processing.py`, `modules/sd_samplers_*.py`, `extensions-builtin/sd_forge_lora/` |

`backend/` reaches into `modules` in exactly three ways — `shared.opts`,
`devices.device`, and `prompt_parser.parse_prompt_attention` — so the shim is
three small files. If a future sync makes `backend/` want something new, the
ImportError is the signal; nothing is stubbed out silently.

## Changes to vendored code

One file. `backend/nn/krea.py`, two lines, marked `VENDOR PATCH` in place:

Upstream `SingleStreamDiT.forward` hardcodes `mask=None` into both the
text-fusion transformer and the single-stream block loop, even though every
layer beneath it already threads a mask down to SDPA. The patch reads a mask
builder out of `transformer_options["krea2_regional"]` instead. With no builder
present — the default, and every non-regional generation — the behaviour is
identical to upstream.

This is what makes regional prompting possible on Krea 2 at all; see the module
docstring in `krea2/regional.py` for why sd-forge-couple itself cannot be used.

## Syncing with upstream

Only worth doing for a Krea 2 fix upstream — this tree is otherwise frozen.
Clone to a scratch directory, copy, throw the clone away:

```bash
git clone --depth 1 -b neo https://github.com/Haoming02/sd-webui-forge-classic /tmp/forge-sync
```

```bash
cd forge && SRC=/tmp/forge-sync && \
rsync -a --exclude __pycache__ --exclude .DS_Store --exclude 'huggingface/' $SRC/backend/ backend/ && \
rsync -a --exclude __pycache__ $SRC/backend/huggingface/krea/ backend/huggingface/krea/ && \
rsync -a --exclude __pycache__ --exclude .DS_Store $SRC/modules_forge/packages/ modules_forge/packages/ && \
cp $SRC/modules/sd_schedulers.py modules/sd_schedulers.py
```

Three things to do afterwards, in order:

1. **Re-apply the `backend/nn/krea.py` patch** — the sync overwrites it. Regional
   prompting silently stops working otherwise; `tools/smoke_krea2.py` asserts the
   patch is present precisely so that failure is loud.
2. **Re-check `modules/shared.py`** against
   `grep -rho 'opts\.[a-zA-Z_0-9]*' backend/ modules/sd_schedulers.py` — a new
   option upstream reads becomes `None` here and fails at the call site.
3. **Run `modal run tools/smoke_krea2.py --gpu --lora any`**, which exercises the
   loader, the LoRA stack and the regional path in one go.

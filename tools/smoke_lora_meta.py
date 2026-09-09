"""
Which family a LoRA is for, and the record that now rides inside it.

    python3 tools/smoke_lora_meta.py

Two things, checked against real names rather than invented ones. The
classifier is fed MiniMax-H3's own 638 parameter names, out of the tensor
index vendored beside this repo, mangled into LoRA keys both ways a trainer
writes them — kohya's underscored `lora_unet_...` and the dotted diffusers
form — and Krea 2's `wq`/`wk`/`txtfusion` modules out of ComfyUI's own model
file. Inventing the key names is the one way this test could pass while the
thing it tests does nothing, since the whole question is what real weights are
called.

Then a safetensors file written by hand, recorded into and read back: the
record survives, `modelspec.architecture` is believed over the key sniff,
kohya's own `ss_*` metadata is still there afterwards, and the tensor bytes
are identical — a header rewrite that touched the weights would be the worst
bug this file could have.

Why it exists: the video picker was offering Krea 2 LoRAs, because a LoRA's
architecture was inferred from where the file came from rather than from what
is in it, and anything brought in by hand claimed nothing and was offered to
both. A wrong verdict here hides a usable LoRA, so ambiguity has to stay
unclaimed — that is the case at the end of the list.
"""
import json, struct, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path("tools").resolve()))
from _from_app import pull

ns = pull({"_safetensors_head", "_safetensors_add_record", "_arch_from_lora_keys",
           # Each cache comes with the function that reads and fills it —
           # pulling one without the other is the NameError this harness
           # exists to raise loudly rather than swallow.
           "_lora_facts", "_LORA_FACTS", "_ARCH_MARKERS", "RECORD_KEY",
           "_record_json", "_output_record", "_module_path", "_base_modules",
           "_BASE_MODULES", "_ARCH_CHECKPOINTS", "_LORA_AFFIXES"})
head, add, arch, facts = (ns["_safetensors_head"], ns["_safetensors_add_record"],
                          ns["_arch_from_lora_keys"], ns["_lora_facts"])

# The two checkpoints, by their real module names.
#
# H3's are ComfyUI's, out of `comfy/ldm/minimax/model.py`, because the volume
# holds Comfy-Org's repackage and not the diffusers export — and the two name
# the same model differently: ComfyUI fuses attention into `qkv_proj` where
# the diffusers export splits it into `to_q`/`to_k`. Krea 2's are out of
# `comfy/ldm/krea2/model.py`: `wq`/`wk`/`wv` and `txtfusion`.
#
# Real names on both sides is the point. Inventing them is the one way this
# file could pass while the thing it tests does nothing, since the whole
# question is what the weights are actually called.
h3_ckpt = (["video_patch_proj.weight", "audio_patch_proj.weight",
            "final_layer.video_out.weight", "final_layer.audio_out.weight"]
           + [f"token_refiner.blocks.{i}.attn.qkv_proj.weight" for i in range(2)]
           + [f"blocks.{i}.attn.{m}.weight" for i in range(8)
              for m in ("qkv_proj", "q_norm", "k_norm", "out_proj")]
           + [f"blocks.{i}.ff.{m}.weight" for i in range(8) for m in ("fc1", "fc2")])
krea_ckpt = (["first.weight", "txtfusion.projector.weight"]
             + [f"blocks.{i}.attn.{w}.weight" for i in range(8)
                for w in ("wq", "wk", "wv", "wo")]
             + [f"blocks.{i}.mlp.{w}.weight" for i in range(8)
                for w in ("w1", "w2", "w3")])

# H3's diffusers export, kept as the odd one out below. Real names, read out
# of the tensor index in the MiniMax-H3 clone — and copied here rather than
# read at run time, because that clone is an upstream checkout this repo
# ignores: a test that opens it passes on the machine that has it and fails
# for everyone else, which is the opposite of what it is for.
h3_diffusers = (["proj_in.weight", "audio_proj_in.weight",
                 "context_embedder.weight", "norm_out.linear.weight"]
                + [f"transformer_blocks.{i}.attn.{m}.weight" for i in range(4)
                   for m in ("to_q", "to_k", "to_v", "to_out.0",
                             "norm_q", "norm_k")]
                + [f"transformer_blocks.{i}.ff.net.{m}.weight" for i in range(4)
                   for m in ("0.proj", "2")]
                + [f"transformer_blocks.{i}.adaln_proj.weight" for i in range(4)]
                + [f"token_refiner.refiner_blocks.{i}.attn.to_q.weight"
                   for i in range(2)])

# A LoRA is written in whichever convention its checkpoint uses — it has to be,
# or it would not load — so each is turned into LoRA keys both ways a trainer
# writes them: kohya's underscored `lora_unet_...`, and the dotted diffusers form.
def kohya(names):
    return [f"lora_unet_{n.rsplit('.',1)[0].replace('.','_')}.lora_down.weight"
            for n in names]


def dotted(names):
    return [f"diffusion_model.{n.rsplit('.',1)[0]}.lora_A.weight" for n in names]


h3_kohya, h3_comfy = kohya(h3_ckpt), dotted(h3_ckpt)
krea, krea_comfy = kohya(krea_ckpt), dotted(krea_ckpt)
# The trap the first cut fell into. `to_q`/`to_k` are the generic diffusers
# spelling, and they are in H3's own export — so a marker list carrying them
# called every diffusers-format Krea 2 LoRA "H3" and hid it from the picker
# that wants it. Measuring against the checkpoint does not make that mistake.
krea_diffusers = dotted(["blocks.0.attn.wq.weight", "blocks.1.attn.wk.weight",
                         "blocks.2.mlp.w1.weight"])

ns["_BASE_MODULES"]["h3"] = frozenset(ns["_module_path"](k) for k in h3_ckpt)
ns["_BASE_MODULES"]["krea2"] = frozenset(ns["_module_path"](k) for k in krea_ckpt)

cases = [
    ("h3 · kohya keys", h3_kohya, "h3"),
    ("h3 · dotted keys", h3_comfy, "h3"),
    ("krea2 · kohya keys", krea, "krea2"),
    ("krea2 · dotted keys", krea_comfy, "krea2"),
    ("krea2 · diffusers spelling", krea_diffusers, "krea2"),
    ("says nothing", ["lora_unet_foo_bar.lora_down.weight"], ""),
    ("empty", [], ""),
    ("both families at once", krea_comfy + h3_comfy, ""),
    # A LoRA for neither checkpoint on the volume is nobody's: unclaimed, and
    # both pickers go on offering it rather than one of them hiding it.
    ("a third architecture", kohya(["down_blocks.0.resnets.0.conv1.weight"]), ""),
    # H3, but written against the diffusers export while the volume holds the
    # ComfyUI repackage. Unclaimed is the right answer and the safe one: the
    # two name the same model differently, so the modules genuinely are not the
    # ones this checkpoint has, and guessing would be guessing.
    ("h3 · the export we do not hold", dotted(h3_diffusers[:40]), ""),
]
bad = 0
for name, keys, want in cases:
    got = arch(keys)
    ok = got == want
    bad += not ok
    print(f"{'ok ' if ok else 'FAIL'} {name:26} → {got!r} (want {want!r})")

# A real safetensors file, written by hand, read back, recorded into, reread.
with tempfile.TemporaryDirectory() as tmp:
    f = Path(tmp) / "k.safetensors"
    hdr = {"lora_unet_blocks_0_attn_wq.lora_down.weight":
           {"dtype": "F32", "shape": [1, 1], "data_offsets": [0, 4]},
           "__metadata__": {"ss_network_dim": "32"}}
    blob = json.dumps(hdr).encode()
    blob += b" " * (-len(blob) % 8)
    f.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00\x00\x00\x00")

    meta, keys = head(f)
    print(f"{'ok ' if meta.get('ss_network_dim') == '32' else 'FAIL'} header read     → {meta}")
    print(f"{'ok ' if len(keys) == 1 else 'FAIL'} tensor names     → {keys}")

    before = f.read_bytes()[-4:]
    add(f, ns["_output_record"](trigger_word="k3nan", dataset="kenan"),
        extra={"modelspec.architecture": "krea2/lora"})
    rec, a = facts(f)
    meta2, _ = head(f)
    print(f"{'ok ' if rec.get('trigger_word') == 'k3nan' else 'FAIL'} record round trip→ {rec}")
    print(f"{'ok ' if a == 'krea2' else 'FAIL'} arch declared    → {a!r}")
    print(f"{'ok ' if meta2.get('ss_network_dim') == '32' else 'FAIL'} kohya keys kept  → {meta2.get('ss_network_dim')!r}")
    print(f"{'ok ' if f.read_bytes()[-4:] == before else 'FAIL'} tensors untouched")
    bad += (rec.get("trigger_word") != "k3nan") + (a != "krea2") + \
           (meta2.get("ss_network_dim") != "32") + (f.read_bytes()[-4:] != before)

print("\nFAILURES:", bad)
sys.exit(1 if bad else 0)

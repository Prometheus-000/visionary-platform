"""
`shared.opts` — the handful of settings `backend/` actually reads.

Upstream this is a live options object backed by `config.json` and the Settings
tab. Headless there is no Settings tab, so it is a plain attribute bag with the
upstream defaults, mutable from Python if a caller wants to change one.

The full set was derived by grepping `opts.<name>` across `backend/`; anything
not listed here is not read by the inference path. Unknown attributes return
None rather than raising, so a forge sync that adds a new `opts.foo` degrades to
"feature off" instead of crashing mid-generation.
"""


class Options:
    # Prompt emphasis: "Original" is the webui default and matches how
    # (word:1.2) weighting behaves in Forge.
    emphasis = "Original"

    # Krea 2 / Anima / Klein reference-image (Edit) modes. Off: this deployment
    # is txt2img only, and Krea 2 reference needs a dedicated Edit LoRA.
    krea2_do_reference = False
    anima_do_reference = False
    klein_no_reference = False

    # Qwen-Image
    qwen_vae_resize = False

    # SDXL — unreachable here, present so `import backend.loader` succeeds.
    sdxl_crop_left = 0
    sdxl_crop_top = 0
    sdxl_refiner_high_aesthetic_score = 6.0
    sdxl_refiner_low_aesthetic_score = 2.5
    sdxl_zero_neg = False

    # Schedulers (read by modules/sd_schedulers.py, which is vendored verbatim).
    hide_schedulers: list = []
    beta_dist_alpha = 0.6
    beta_dist_beta = 0.6
    use_dynamic_shifting = False
    invert_sigmas = False
    use_karras_sigmas = False
    use_exponential_sigmas = False
    use_beta_sigmas = False
    stochastic_sampling = False

    # Neta / token merging / Nunchaku — likewise unreachable, defaults only.
    neta_template_positive = ""
    neta_template_negative = ""
    token_merging_downsample = 2
    token_merging_stride = 2
    token_merging_no_rand = False
    svdq_attention = "nunchaku-fp16"
    svdq_cache_threshold = 0.0
    svdq_cpu_offload = "auto"
    svdq_num_blocks_on_gpu = 1
    svdq_use_pin_memory = False

    def __getattr__(self, name):
        return None


opts = Options()


class _CmdOpts:
    """Only ever consulted for optional paths; never set in this deployment."""

    def __getattr__(self, name):
        return None


cmd_opts = _CmdOpts()

"""
Headless stand-in for sd-webui-forge-classic's `modules` package.

Forge's `backend/` is almost entirely self-contained; the only Gradio-side
things it reaches for are `modules.shared.opts`, `modules.devices.device` and
`modules.prompt_parser.parse_prompt_attention`. Those three files are
reimplemented here so `backend/` can be imported with no Gradio, no webui
bootstrap, and no `launch.py`.

Nothing else from the upstream `modules` package is vendored on purpose: if a
future forge sync makes `backend/` reach for something new, the ImportError is
the signal to look at it rather than a silent behaviour change.
"""

"""
One node, for one reason: ComfyUI cannot carry a list of boxes as a literal.

Krea2RegionalMultiLoRAV12 takes its rectangles on a `BOUNDING_BOX` input, and
the pack's own example workflow fills it from ComfyUI-KJNodes' box builder. We
drive ComfyUI over its API rather than through the canvas, so there is no
builder to draw in and no reason to install a node pack for one wire — the
boxes are already in the request.

What stops us writing them into the graph directly is `execution.py`: an input
whose value is a list is read as a `[node_id, slot_index]` link, and a list of
any other length is rejected as `bad_linked_input`. Two boxes is a length-2
list. So the exact case this feature exists for — two characters, two boxes —
is the one that would be silently interpreted as a wire to node "{'x': 0.0,
...}" instead of failing somewhere legible. A JSON string has no such reading.

There is a second escape hatch upstream (`{"__value__": [...]}`, unwrapped by
`validate_inputs`), and it works at our pinned SHA. It is not used here because
it relies on the validation pass mutating the prompt dict that the execution
pass later reads — two passes, one undocumented interaction, and a failure mode
that only appears once someone bumps COMFY_SHA. Twenty lines of our own is the
cheaper thing to own.

Boxes arrive normalised 0..1, which is what the page already speaks and what
`_coerce_bbox_norm` leaves untouched — it treats any coordinate above 1.0 as
pixels and divides by the canvas, so a normalised box is the one shape that
cannot be misread at either end.
"""

import json


class VisionaryBoxes:
    """JSON string -> BOUNDING_BOX, in the order given."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boxes_json": ("STRING", {
                    "multiline": True,
                    "default": '[{"x": 0.0, "y": 0.0, "width": 0.5, "height": 1.0}]',
                }),
            }
        }

    RETURN_TYPES = ("BOUNDING_BOX",)
    RETURN_NAMES = ("bboxes",)
    FUNCTION = "build"
    CATEGORY = "visionary"

    def build(self, boxes_json):
        try:
            boxes = json.loads(boxes_json or "[]")
        except json.JSONDecodeError as exc:
            raise ValueError(f"boxes_json is not valid JSON: {exc}") from exc
        if not isinstance(boxes, list):
            raise ValueError(f"boxes_json must be a list, got {type(boxes).__name__}")

        out = []
        for i, box in enumerate(boxes):
            if not isinstance(box, dict):
                raise ValueError(f"box {i} must be an object, got {type(box).__name__}")
            try:
                # Rebuilt rather than passed through, so a box carrying extra
                # keys cannot reach _coerce_bbox_norm and take its {x,y,x1,y1}
                # branch by accident — that branch reads a *corner* where this
                # one means a *size*, and the silent result is a box of the
                # wrong shape rather than an error.
                out.append({
                    "x": float(box["x"]), "y": float(box["y"]),
                    "width": float(box["width"]), "height": float(box["height"]),
                })
            except KeyError as exc:
                raise ValueError(f"box {i} is missing {exc.args[0]!r}") from exc
            except (TypeError, ValueError) as exc:
                raise ValueError(f"box {i} has a non-numeric coordinate: {exc}") from exc

        # Index-aligned with regions_json and never filtered: V9's _pair_boxes
        # matches box i to row i of regions_json by ORIGINAL index, so dropping
        # an empty box here would hand every later row the wrong rectangle.
        return (out,)


NODE_CLASS_MAPPINGS = {"VisionaryBoxes": VisionaryBoxes}
NODE_DISPLAY_NAME_MAPPINGS = {"VisionaryBoxes": "Visionary Boxes"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

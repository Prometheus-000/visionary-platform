"""The input shape CLAUDE.md says is real, which is not the one I tested.

Every fragment in the matrix corpus is a complete, well-formed sentence — the
Gucci recreations are elaborate paragraphs. CLAUDE.md is explicit that this is
the wrong distribution: "fragments are the expected input, not the degraded
case. Out of order, incomplete, self-correcting, arriving in pieces over a
minute." A harness that scores a parse on finished prose is a harness measuring
the case that barely occurs.
"""

FRAGMENTS = [
    # telegraphic — the commonest thing anybody types first
    "woman red dress",
    "k3nan, black coat, city behind",
    "close on hands. shaking.",
    "empty diner, 3am",
    "wide. man on a beach. dawn",

    # self-correcting — the shape that has no clean parse at all
    "two guards, no three",
    "she's angry. or maybe just tired",
    "a horse in a field. actually a whole herd",
    "night. no, late afternoon",

    # hedged and incomplete
    "maybe a woman walking through some kind of airport, cinematic, probably evening",
    "the kitchen after the party",
    "something like a courtroom but colder",

    # arriving in pieces, one line at a time
    "old man asleep in a chair\ntelevision on\nnobody else in the room",
    "two kids running\nhallway\nhandheld",

    # a directive mixed into the description
    "two people on a bench in front of a ruin. give it some perspective",
]

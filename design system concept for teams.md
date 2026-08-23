Let me load the diagram module, then lay this out.## The actual failure

Design systems become cages because they bundle three things with wildly different half-lives and govern all of them at the speed of the slowest:

- **Meaning** — what red means, what elevation means, what the spacing rhythm is. Changes every few years.
- **Behavior** — focus management, keyboard, ARIA, controlled/uncontrolled state. Changes almost never.
- **Expression** — how this particular filter panel looks. Changes weekly.

Traditional systems make the *component* the atomic unit, which fuses all three. So a designer who needs a new expression has to renegotiate meaning and behavior to get it — through a governance queue. The rational response is to abandon the system, and now you have drift plus a maintenance tax.

The fix is to split those layers and give each one a different governance rule. Strict where it's cheap to be strict, free where freedom is the point.

## The architecture

```
LAYER 0   CONSTRAINTS        semantic tokens, scales, motion curves
          strict · versioned · machine-verified · never forked

LAYER 1   BEHAVIOR           headless primitives: focus, keyboard, a11y,
          shared · never forked        state machines, portals

LAYER 2   RECIPES            styled compositions of 0 + 1
          forkable by default · eject is a feature

LAYER 3   LOCAL              feature-owned, one-off, unreviewed
          free · no permission required
                    │
                    └──► promotion (earned by usage, not granted)

CROSS-CUTTING
          MACHINE LAYER      MCP server · skills · validators · precedent index
          PROJECTIONS        Figma variables · docs · Storybook  (all generated)
```

**Layer 0 is the only hard contract.** Semantic, not literal — `surface.raised`, `intent.destructive`, not `gray-100`, `red-500`. This is where consistency actually lives. Two screens built by two designers with zero shared components are still coherent if they share Layer 0, because consistency isn't visual sameness, it's *predictability of meaning*. Your own Dream Engine rule — red appears once per screen and it's the commit — is Layer 0 thinking. Once meaning is stable, expression can be wild.

**Layer 1 is never forked because correctness isn't a taste question.** Nobody needs creative freedom over roving tabindex. Radix/Ark/React Aria style. This is the part everyone quietly rebuilds badly when they eject from a monolithic system, so it must be separable from the styling they actually wanted to escape.

**Layer 2 ships with an eject button.** `ds eject Combobox` copies the source into your repo, tokens intact, Layer 1 dependency intact. shadcn proved this model works socially — the moment forking is sanctioned, the adversarial relationship with the DS team evaporates. You don't file a ticket; you take the code. Consistency survives because the eject preserves Layers 0 and 1.

**Layer 3 requires no permission.** This is the load-bearing decision. The reason design systems feel like handicaps is that governance is expensive, so it gets applied *preemptively as restriction*. Make governance cheap and you can permit anything and catch drift afterward.

## The machine layer is what makes the freedom safe

This is the part that's actually new, and it isn't "AI generates components."

**1. Validators, not reviewers.** A linter that fails on raw hex, off-scale spacing, non-semantic token references, and undeclared deviation. An agent can write anything at any volume; the constraint layer catches it at the diff. Deviation stays legal — but it must *declare itself*: `// deviation: custom easing, motion spec doesn't cover physics drag`. Undeclared drift fails CI. Declared drift becomes a promotion signal.

**2. An MCP server, not a docs site.** Agents shouldn't read prose about your system. Expose: live token values, component inventory with real props, and — the important one — **precedent search**. "Has anyone here built a multi-select filter panel?" returns Layer 3 one-offs, not just blessed components. Precedent is what produces consistency without mandate: it makes the well-trodden path visible and cheap, which is more effective than making the off-path expensive.

**3. Taste as a skill, not API docs.** Your `design-first-engineering` and affordance-first work is the right shape for this. The system ships rules with *reasons* — why one commit action per screen, when a component earns copy, the delete-and-look-again test. An agent that has the reasoning generates novel-but-coherent work. An agent that only has the component list generates collage.

**4. Generation targets Layer 3 by default.** Agents build local. Promotion is a separate, deliberate act. Otherwise AI floods Layer 2 with plausible-but-unearned abstractions.

## Kill the two-library problem

Sixty percent of a typical design system team's energy goes to keeping Figma and code in sync. That job shouldn't exist.

**One source (code), two projections.** Tokens flow one direction — code → Figma variables via the Plugin API on CI. Figma becomes a rendering surface and a divergent-thinking space, not a source of truth. Code Connect maps back for handoff. Nobody maintains a parallel component library by hand, ever.

For a designer-engineer this is native. For designers who live in Figma, the deal is: explore freely, and what you ship gets built from the same tokens either way. The sync tax disappears because there's nothing to sync — only to regenerate.

## Promotion economy

The DS team's job changes from *author* to *editor*. Their only real function:

- Instrument Layer 3. When the same shape appears in three features, surface it as a promotion candidate.
- Promote based on demonstrated demand, never speculation.
- Deprecate aggressively — dead components are cheaper than blocked designers, but only if you actually remove them.

This inverts the political economy. Nobody asks permission; they earn adoption. And the system only contains things that were proven necessary by use, which is the only reliable filter against bloat.

Versioning follows: **Layer 0 strict semver, breaking changes rare and loud. Layer 2 loose** — because you can fork, you're never forced to upgrade.

## Where this breaks

Three honest failure modes:

**Everyone ejects and it becomes a snippet dump.** Mitigated by validators — tokens are enforced regardless of ejection — and by promotion telemetry making duplication visible. But watch it. If eject rate on a component exceeds ~40%, the component is wrong, not the users.

**Promotion never happens.** If it's nobody's full-time job, Layer 3 becomes permanent sprawl. This is the single most likely way the model dies.

**AI produces high-volume plausible drift.** Validators catch token violations but not bad judgment. Review has to shift from reading code to reading the *diff against precedent* — "why is this different from the three existing versions of this?" That's a taste question and it stays human.

---

The one-line version: stop distributing a library, start distributing a grammar. The library is a byproduct that accumulates from what people actually built.
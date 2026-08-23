## Building with Visionary

A small, opinionated set of primitives lifted out of the Visionary console. Five
components carry behaviour; everything else you build is **the token vocabulary
plus global classnames** — there is no utility-class framework and no style
props here.

**Dark only.** `styles.css` sets `body{background:var(--bg);color:var(--fg)}` —
`#000` on `#f5f5f5`. There is no light theme and no `prefers-color-scheme`
branch. Let `body` supply the ground; if you paint a panel yourself, use a token
rather than a literal, or it will drift from every component beside it.

**No provider, no theme wrapper.** Nothing reads React context. Load the
stylesheet and render — that is the whole setup.

### Tokens — always `var(--…)`, never a literal

| Group | Tokens |
|---|---|
| Ground | `--bg` `--fg` `--mut` (secondary text) `--dim` (tertiary) |
| Surface ramp | `--wash-1` card · `--wash-2` field · `--wash-3` hover · `--sel` selected |
| Hairlines | `--line` · `--line-2` (raised) |
| Elevation — use all three together | `--lift` `--lift-line` `--lift-cast` |
| Radii, by object | `--r-inner` 9 (inside a control) · `--r-control` 11 (button, input) · `--r-panel` 14 (floating panel, media) · `--r-card` 16 (content card) · `--r-sheet` 20 (modal) · `--r-pill` · `--r-mark` |
| Sizing | `--control-h` 32 · `--head` · `--drawer` · `--ib` (icon-button square) |
| State | `--crit` `--crit-wash` `--crit-line` (destructive) · `--ok` · `--warn` |

Pick a radius by **what the object is**, not by eye — that ramp exists because a
popover, a card, an input and a tile were previously told apart by one pixel.

### Classnames you compose with

- **Buttons** — `button.b` primary (white fill, dark text) · `button.s` compact ·
  `button.t` text-only, muted.
- **`.opt`** — the bar control: 32px, `--wash-2`, hairline, inline-flex. Use for
  anything showing its own value (`16:9`, `Krea 2 Turbo`, `5s`).
- **`.opt.ib`** — icon-only variant, **fixed width `--ib`**. It will squeeze a
  text label to nothing; for a labelled control use plain `.opt`.
- **`.field`** wraps a text input · **`.frow`** is a form row: `<span>` label,
  control, optional `<i>` hint.
- **`.on`** marks a selected control · **`.danger`** a destructive menu row ·
  **`.hint`** a dim clause after a label · **`.sub`** secondary prose.

### The five components

`Popover` (+ `usePopover`) · `Menu` · `Sheet` · `NumInput` (+ `step`) ·
`ErrorNote`, all on `window.Visionary`.

`Popover`, `Menu` and `Sheet` render through a **portal to `document.body`** and
position `fixed`. `Popover` takes the *variant as `className`* — `menu`,
`menu form`, `menu sizer`, `pal` — and does nothing else but place and dismiss.
`anchor` is `HTMLElement | null`, where `null` is the closed state; `usePopover()`
returns exactly that pair.

### Where the truth is

Read `_ds/<folder>/styles.css` and its `@import` closure before styling anything —
it is the real stylesheet, not a summary. Each component's `.prompt.md` and
`.d.ts` carry its API.

```jsx
const { Menu, usePopover } = window.Visionary
const pop = usePopover()

<div style={{ background: 'var(--wash-1)', border: '1px solid var(--line)',
              borderRadius: 'var(--r-card)', padding: 16 }}>
  <button className="opt" onClick={pop.toggle}>16:9 · 720p</button>
  {pop.open && (
    <Menu anchor={pop.anchor} onClose={pop.close} items={[
      { label: 'Reuse settings', run: reuse },
      { sep: true },
      { label: 'Delete', danger: true, run: remove },
    ]} />
  )}
</div>
```

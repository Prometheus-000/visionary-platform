import { useState } from 'react'
import { Menu } from 'visionary-web'

/**
 * The card harness paints its page white; Visionary is a black-only system —
 * `ui.css` sets `body{background:#000;color:#f5f5f5}` and every component here
 * is drawn for that ground. The harness stylesheet loads after `styles.css`, so
 * the preview paints the ground itself rather than the shipped CSS fighting it
 * with `!important` — a real design gets the black from `body` the ordinary way.
 *
 * Normal flow, with a height, rather than `position:fixed` — the harness puts
 * `transform:translateZ(0)` on the cell, which makes it the containing block for
 * fixed descendants, so `inset:0` resolves against a box that has collapsed to
 * nothing. The same transform is what keeps the portalled overlay in the card.
 */
function Dark({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      background: 'var(--bg)', color: 'var(--fg)', minHeight: '100vh', padding: 20,
      font: '14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif',
    }}>{children}</div>
  )
}

/**
 * A menu needs a live anchor, so every export mounts a real button and hands
 * the element straight to `anchor` — the same pair `usePopover` returns, minus
 * the click.
 */
function Anchored({ label, items }: { label: string; items: React.ComponentProps<typeof Menu>['items'] }) {
  const [anchor, setAnchor] = useState<HTMLElement | null>(null)
  return (
    <Dark>
      <button ref={setAnchor} className="opt" type="button">{label}</button>
      {anchor && <Menu anchor={anchor} items={items} onClose={() => {}} />}
    </Dark>
  )
}

/**
 * The LoRA picker: every row a token it will write, ticked when it is already
 * in the prompt, with the trigger phrase as the hint.
 *
 * A style LoRA is near-invisible without its phrase, so the hint is what makes
 * the write legible before it happens.
 */
export function WithTicksAndHints() {
  return (
    <Anchored label="+ LoRA" items={[
      { label: '<lora:k3nan:1>', hint: 'k3nan', on: true, run: () => {} },
      { label: '<lora:film-grain-krea:0.8>', hint: 'shot on 35mm', run: () => {} },
      { label: '<lora:wan22-speed/high:1>', run: () => {} },
      { label: '<lora:wan22-speed/low:1>', run: () => {} },
    ]} />
  )
}

/**
 * The gallery card's menu: a separator, and one item that deletes.
 *
 * `danger` is the only colour in here — everything else is a command that can
 * be taken back.
 */
export function WithSeparatorAndDanger() {
  return (
    <Anchored label="⋯" items={[
      { label: 'Reuse prompt', run: () => {} },
      { label: 'As reference', run: () => {} },
      { sep: true },
      { label: 'Animate', run: () => {} },
      { sep: true },
      { label: 'Delete', danger: true, run: () => {} },
    ]} />
  )
}

/** Plain commands, no state on any row — the common case. */
export function PlainCommands() {
  return (
    <Anchored label="5s" items={[
      { label: 'Still', run: () => {} },
      { label: '3s', run: () => {} },
      { label: '5s', run: () => {} },
      { label: '8s', run: () => {} },
    ]} />
  )
}

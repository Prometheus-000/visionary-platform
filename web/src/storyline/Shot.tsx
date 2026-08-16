import { shares, type Module } from './model'

/**
 * The shot your words add up to, before a take is spent finding out.
 *
 * The need this exists for, stated by the person who has it: *three characters;
 * too many words on how light falls on A, a good deal on B, barely anything on
 * C — show me that A and B are detailed and connected, and that C is ungrouped
 * and now in the background because he only got a few lines.*
 *
 * Three facts, and the heat bar carries only the first:
 *
 * - **Detail** — how much of the picture a thing gets. Extent, which is words.
 * - **Connection** — who stands with whom, which a line cannot draw and two
 *   dimensions can. This is the peer-edge problem dissolving: it was never a
 *   property of relations, it was a property of insisting the display be a line.
 * - **Consequence** — "C is in the background" is a *spatial* result of a
 *   *textual* cause, and no percentage says it. A picture does.
 *
 * **Deliberately a weighting diagram and not a floor plan.** It knows extent and
 * grouping, and it knows nothing about where anybody is standing — so it must
 * not look like it does, or people will read geometry out of an estimate.
 * Shapes, not figures. No ground line, no perspective. That rule is what decides
 * the axes: size and opacity carry extent, horizontal distance carries grouping
 * and nothing else, and order is spent buying the second of those rather than
 * drawn as a fact of its own.
 *
 * **Read-only, and it stays that way.** The moment a shape can be dragged there
 * are two places to compose and they will disagree. It is diagnostic, which is
 * what separates it from the region map CLAUDE.md retired for being
 * navigational — if it ever becomes a way to get somewhere, it has become the
 * map and should go the same way.
 */
export function Shot({ mods }: { mods: Module[] }) {
  // The opening element is the scene's focus, not a subject — the three-axis
  // model says so, and drawing it as one more shape both dilutes the reading
  // and is wrong: "three people in a bar at night" is not a person standing in
  // the frame, it is the frame. So it becomes the ground the rest sit in.
  const [scene, ...rest] = mods
  const bodies = mods.length > 1 ? rest : mods
  const all = shares(mods)
  const top = mods.length > 1 ? all.slice(1) : all
  const max = Math.max(...top, 0.0001)

  // Grouping is symmetric: a tie names one direction and means both. Without
  // that, "B goes with A" would draw A alone and B attached to nothing.
  const groupOf = new Map<string, number>()
  let next = 0
  bodies.forEach((m, i) => {
    const linked = [m.id, ...(m.ties ?? [])]
    const found = linked.map((id) => groupOf.get(id)).find((g) => g !== undefined)
    const g = found ?? next++
    groupOf.set(m.id, g)
    ;(m.ties ?? []).forEach((t) => groupOf.set(t, g))
    // A later element tying backwards has to pull its partner along.
    bodies.slice(0, i).forEach((p) => {
      if ((p.ties ?? []).includes(m.id)) groupOf.set(p.id, g)
    })
  })

  const W = 300
  const H = 110
  const shapes = bodies.map((m, i) => {
    const share = top[i] ?? 0
    const rel = share / max                      // 1 for the biggest thing present
    // Recession: less detail sits back, smaller and fainter. The one reading the
    // bar could not give, and the one the user asked for by name.
    const h = 18 + rel * (H - 46)
    const w = 12 + rel * 30
    return { m, i, rel, h, w, group: groupOf.get(m.id) ?? i }
  })

  // Horizontal position says grouping and nothing else. It used to be a constant
  // gap, which left x carrying sequence plus the residue of centring a row of
  // uneven widths — the first is not a fact about the shot, the second is not a
  // fact at all, and together they read as the floor plan this must never claim.
  // Group members therefore sit adjacent and groups in order of first mention, so
  // proximity is the whole signal and the bracket that used to carry it is gone.
  const TIGHT = 4
  const OPEN = 22
  const order = [...new Set(shapes.map((s) => s.group))]
    .flatMap((g) => shapes.filter((s) => s.group === g))
  const gaps: number[] = order.map((s, i) =>
    i === 0 ? 0 : order[i - 1]!.group === s.group ? TIGHT : OPEN)

  const totalW = order.reduce((n, s) => n + s.w, 0) + gaps.reduce((a, b) => a + b, 0)
  const scale = totalW > W - 16 ? (W - 16) / totalW : 1
  let x = (W - totalW * scale) / 2

  const placed = order.map((s, i) => {
    x += gaps[i]! * scale
    const at = x
    x += s.w * scale
    return { ...s, x: at, w: s.w * scale }
  })

  // Nothing said, nothing drawn. An element with no words still has a shape
  // otherwise — a sliver in an empty frame, which reads as a glitch and is the
  // first thing anyone sees.
  if (!bodies.length || !bodies.some((m) => m.text.trim())) return null

  return (
    <svg className="sl-shot" viewBox={`0 0 ${W} ${H}`} aria-hidden="true"
         role="img" aria-label={scene?.text ?? 'the shot so far'}>
      {/* The scene, as the ground everything stands in rather than a shape
          standing among them. */}
      <rect x="0.5" y="0.5" width={W - 1} height={H - 1} rx="4"
            fill="none" stroke="var(--line)" />
      {placed.map((s) => (
        <rect
          key={s.m.id}
          x={s.x} y={H - 16 - s.h} width={s.w} height={s.h} rx={Math.min(3, s.w / 3)}
          // Opacity carries recession a second time, so the signal survives at
          // small sizes and without colour.
          fill="var(--fg)" opacity={0.18 + s.rel * 0.72}
        />
      ))}
    </svg>
  )
}

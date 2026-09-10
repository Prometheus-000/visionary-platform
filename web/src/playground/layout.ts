/**
 * Computed layout for the node room — the frame is computed, never authored.
 *
 * A hand-rolled layered layout rather than elkjs or dagre, weighed rather than
 * defaulted: a ComfyUI graph is a small DAG (ten to forty nodes), and
 * longest-path layering with barycenter ordering is sixty lines that answer
 * it deterministically, against a dependency that would be the largest thing
 * in the bundle. If graphs ever outgrow this, the seam is this one function.
 *
 * Left to right, because the wire format reads that way: loaders feed
 * samplers feed savers. A cycle cannot happen in a valid graph, but a broken
 * one must not hang the page, so layering is longest-path over a visited set
 * and a cycle simply flattens.
 */
import type { PgGraph } from '../store'

export type Pos = { x: number; y: number }
export type Size = { w: number; h: number }

export const CARD_W = 232
const GAP_X = 72
const GAP_Y = 28

/** The card's height before it is measured: header plus one row per input.
 *  Only used to stack a column; the real card sizes itself. */
export function guessSize(inputCount: number): Size {
  return { w: CARD_W, h: 44 + inputCount * 27 }
}

export function layoutGraph(
  graph: PgGraph,
  sizes: (key: string) => Size,
): Record<string, Pos> {
  const keys = Object.keys(graph)
  const into = new Map<string, string[]>()
  for (const key of keys) into.set(key, [])
  for (const key of keys) {
    for (const val of Object.values(graph[key]?.inputs ?? {})) {
      if (Array.isArray(val) && val.length === 2) {
        const src = String(val[0])
        if (into.has(src)) into.get(key)!.push(src)
      }
    }
  }

  // Longest path from the sources decides the column, so a node sits just
  // right of the last thing it reads.
  const layer = new Map<string, number>()
  const depth = (key: string, seen: Set<string>): number => {
    const known = layer.get(key)
    if (known !== undefined) return known
    if (seen.has(key)) return 0 // a broken cycle flattens instead of hanging
    seen.add(key)
    const feeds = into.get(key) ?? []
    const l = feeds.length
      ? Math.max(...feeds.map((s) => depth(s, seen))) + 1
      : 0
    layer.set(key, l)
    return l
  }
  for (const key of keys) depth(key, new Set())

  const columns: string[][] = []
  for (const key of keys) {
    const l = layer.get(key) ?? 0
    ;(columns[l] ??= []).push(key)
  }

  // Barycenter ordering: a node sits level with the average of what feeds it,
  // so wires run flat instead of crossing the whole column.
  const order = new Map<string, number>()
  columns.forEach((col, ci) => {
    if (ci > 0) {
      col.sort((a, b) => {
        const bary = (k: string) => {
          const feeds = into.get(k) ?? []
          return feeds.length
            ? feeds.reduce((s, p) => s + (order.get(p) ?? 0), 0) / feeds.length
            : 0
        }
        return bary(a) - bary(b)
      })
    }
    col.forEach((k, i) => order.set(k, i))
  })

  const pos: Record<string, Pos> = {}
  columns.forEach((col, ci) => {
    let y = 0
    const tops = new Map<string, number>()
    for (const key of col) {
      tops.set(key, y)
      y += sizes(key).h + GAP_Y
    }
    // Centre each column on the tallest one, so a two-node column floats
    // beside the middle of a ten-node one rather than hanging off its top.
    const height = y - GAP_Y
    for (const key of col) {
      pos[key] = { x: ci * (CARD_W + GAP_X),
                   y: (tops.get(key) ?? 0) - height / 2 }
    }
  })
  return pos
}

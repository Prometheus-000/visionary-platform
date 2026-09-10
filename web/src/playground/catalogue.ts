/**
 * The node catalogue — /object_info, read once and answered from.
 *
 * The server is the authority on what a node takes; the page never restates
 * it (the same rule AppState follows). What this module adds is the *reading*:
 * which inputs are sockets, which are widgets, what control a widget gets —
 * derived from the spec every time rather than tabulated, so a pack installed
 * tomorrow renders correctly today.
 */
import { createContext, useEffect, useState } from 'react'

import { failed } from '../api/client'
import { playgroundNodes } from '../api/routes'
import type { NodeCatalogue, NodeSpec, WorkflowExpose } from '../api/types'
import type { PgGraph, PgNode } from '../store'

/** The catalogue, handed to every node card without threading it through
 *  xyflow's data props — a spec is read-only reference material, not state. */
export const CatContext = createContext<NodeCatalogue | null>(null)

export type InputRow = {
  name: string
  required: boolean
  /** The spec pair: [type-or-enum, config]. */
  type: unknown
  config: Record<string, unknown>
}

/** A link is [node_id, slot]; everything else in an input is a value. */
export function isLink(v: unknown): v is [string | number, number] {
  return Array.isArray(v) && v.length === 2
    && (typeof v[0] === 'string' || typeof v[0] === 'number')
    && typeof v[1] === 'number'
}

export function inputRows(spec: NodeSpec | undefined): InputRow[] {
  const rows: InputRow[] = []
  for (const [required, table] of [
    [true, spec?.input?.required], [false, spec?.input?.optional],
  ] as const) {
    for (const [name, pair] of Object.entries(table ?? {})) {
      const [type, config] = Array.isArray(pair) ? pair : [pair, {}]
      rows.push({
        name, required, type,
        config: (config && typeof config === 'object'
          ? config as Record<string, unknown> : {}),
      })
    }
  }
  return rows
}

/** What control a value input gets. Sockets are the rows this returns null
 *  for: a MODEL or LATENT is only ever wired. */
export function widgetKind(row: InputRow):
    'choice' | 'number' | 'text' | 'toggle' | 'file' | null {
  if (Array.isArray(row.type)) {
    // An enum of input-directory files with image_upload set is ComfyUI's
    // own "this wants a picture" marker — LoadImage and its cousins.
    return row.config.image_upload ? 'file' : 'choice'
  }
  if (row.type === 'INT' || row.type === 'FLOAT') return 'number'
  if (row.type === 'STRING') return 'text'
  if (row.type === 'BOOLEAN') return 'toggle'
  return null
}

/** The graph a fresh node starts with: every widget input written out at the
 *  spec's default. The house rule from the backend, applied in the editor —
 *  an omitted optional input is decided by a signature nobody can see. */
export function defaultInputs(spec: NodeSpec | undefined):
    Record<string, unknown> {
  const inputs: Record<string, unknown> = {}
  for (const row of inputRows(spec)) {
    const kind = widgetKind(row)
    if (!kind || kind === 'file') continue
    if ('default' in row.config) inputs[row.name] = row.config.default
    else if (kind === 'choice' && Array.isArray(row.type) && row.type.length) {
      inputs[row.name] = row.type[0]
    } else if (kind === 'number') inputs[row.name] = 0
    else if (kind === 'text') inputs[row.name] = ''
    else if (kind === 'toggle') inputs[row.name] = false
  }
  return inputs
}

/**
 * The save-time diff that finds what a workflow exposes to the console.
 *
 * New nodes' widget inputs, against the seed the graph grew from: a node the
 * seed already had belongs to the console (it feeds those by inherited key),
 * and a node the user added is exactly the thing the console cannot name —
 * so its dials are what the model menu grows while this workflow is on.
 */
export function diffExposes(
  graph: PgGraph, seed: PgGraph | null, cat: NodeCatalogue | null,
): WorkflowExpose[] {
  // No seed means a dropped or foreign graph — there is no "new against
  // what?", and exposing every widget in it would grow the model menu a
  // whole settings panel. Nothing is exposed, honestly.
  if (!seed) return []
  const out: WorkflowExpose[] = []
  for (const [key, node] of Object.entries(graph)) {
    const was = seed[key]
    if (was && was.class_type === node.class_type) continue
    const spec = cat?.[node.class_type]
    for (const row of inputRows(spec)) {
      const kind = widgetKind(row)
      if (!kind || kind === 'file') continue
      const val = node.inputs?.[row.name]
      if (isLink(val)) continue
      out.push({
        node: key, input: row.name,
        type: kind === 'choice' ? 'choice'
          : kind === 'number' ? 'number'
          : kind === 'toggle' ? 'toggle' : 'text',
        label: `${title(node)} · ${row.name}`,
        default: val,
        ...(kind === 'choice' && Array.isArray(row.type)
          ? { options: row.type.map(String) } : {}),
      })
    }
  }
  return out
}

export function title(node: PgNode): string {
  return node._meta?.title || node.class_type
}

/**
 * Fetched once per mount of the room, `null` while loading, `missing` when
 * the volume has never harvested one — the room offers the refresh instead
 * of failing quietly.
 */
export function useCatalogue(): {
  cat: NodeCatalogue | null; missing: boolean; error: string | null
  reload: () => void
} {
  const [cat, setCat] = useState<NodeCatalogue | null>(null)
  const [missing, setMissing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [tick, setTick] = useState(0)
  useEffect(() => {
    let dead = false
    let timer: number | undefined
    void (async () => {
      const r = await playgroundNodes()
      if (dead) return
      if (failed(r)) { setError(r.error); return }
      const body = r as { missing?: boolean; harvesting?: string }
      if (body.missing) {
        setMissing(true)
        // A missing catalogue is already being harvested — the route starts
        // it — so come back for it rather than leaving the room to press.
        if (body.harvesting) timer = window.setTimeout(() => setTick((t) => t + 1), 5000)
        return
      }
      setMissing(false)
      setError(null)
      setCat(r as NodeCatalogue)
    })()
    return () => { dead = true; window.clearTimeout(timer) }
  }, [tick])
  return { cat, missing, error, reload: () => setTick((t) => t + 1) }
}

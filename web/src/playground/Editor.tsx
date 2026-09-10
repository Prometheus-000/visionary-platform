/**
 * The graph surface — @xyflow/react drawing the store's graph, nothing more.
 *
 * The graph in the store is the single truth and it is the wire format
 * itself; nodes and edges here are *derived* every render, and every gesture
 * (connect, disconnect, delete, drag) writes back through `setPg`. Positions
 * are the one thing that never reaches the store: layout is computed (the
 * frame is computed, never authored), a drag is session-local, and Tidy
 * recomputes the lot.
 */
import {
  Background, BackgroundVariant, ReactFlow,
  type Connection, type Edge, type EdgeChange, type Node, type NodeChange,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { useStore, type PgGraph } from '../store'
import { isLink } from './catalogue'
import { CARD_W, guessSize, layoutGraph, type Pos } from './layout'
import { NodeCard } from './NodeCard'

const nodeTypes = { vsn: NodeCard }

const sizeOf = (graph: PgGraph) => (key: string) =>
  guessSize(Object.keys(graph[key]?.inputs ?? {}).length + 2)

export function Editor({ arrange, errorNode }: {
  /** A counter — bump it to recompute the whole layout. */
  arrange: number
  errorNode: string | null
}) {
  const graph = useStore((s) => s.pg.graph)
  const sel = useStore((s) => s.pg.sel)
  const setPg = useStore((s) => s.setPg)
  const [pos, setPos] = useState<Record<string, Pos>>({})

  // First graph and every Tidy: the whole layout. In between, only a node
  // the map has never seen gets a place — at the right edge, where new work
  // goes — so a drag survives an edit beside it.
  useEffect(() => {
    if (!graph) return
    setPos((old) => {
      const missing = Object.keys(graph).filter((k) => !(k in old))
      if (!missing.length) return old
      if (!Object.keys(old).length) return layoutGraph(graph, sizeOf(graph))
      const edge = Math.max(0, ...Object.values(old).map((p) => p.x))
      const next = { ...old }
      missing.forEach((k, i) => {
        next[k] = { x: edge + CARD_W + 72, y: i * 120 }
      })
      return next
    })
  }, [graph])
  useEffect(() => {
    const g = useStore.getState().pg.graph
    if (g && arrange) setPos(layoutGraph(g, sizeOf(g)))
  }, [arrange])

  const nodes: Node[] = useMemo(() => {
    if (!graph) return []
    return Object.keys(graph).map((k) => ({
      id: k, type: 'vsn' as const,
      position: pos[k] ?? { x: 0, y: 0 },
      data: {},
      selected: k === sel,
      className: k === errorNode ? 'pgerr' : undefined,
    }))
  }, [graph, pos, sel, errorNode])

  // Edge ids are opaque; this map is how a remove finds its way back to the
  // input it has to unwire — parsing names out of the id would break on the
  // first input with the separator in it.
  const [edges, edgeIndex] = useMemo(() => {
    const out: Edge[] = []
    const index = new Map<string, { target: string; input: string }>()
    for (const [k, node] of Object.entries(graph ?? {})) {
      for (const [name, val] of Object.entries(node.inputs ?? {})) {
        if (isLink(val)) {
          const id = `e${out.length}:${k}`
          out.push({
            id, source: String(val[0]), sourceHandle: `out-${val[1]}`,
            target: k, targetHandle: `in-${name}`,
          })
          index.set(id, { target: k, input: name })
        }
      }
    }
    return [out, index] as const
  }, [graph])

  const onNodesChange = useCallback((changes: NodeChange[]) => {
    for (const ch of changes) {
      if (ch.type === 'position' && ch.position) {
        setPos((p) => ({ ...p, [ch.id]: ch.position! }))
      } else if (ch.type === 'select') {
        const cur = useStore.getState().pg.sel
        if (ch.selected) setPg({ sel: ch.id })
        else if (cur === ch.id) setPg({ sel: null })
      } else if (ch.type === 'remove') {
        const s = useStore.getState()
        const g = s.pg.graph
        if (!g?.[ch.id]) continue
        // The node goes, and so does every input that read it: a link to a
        // key not in the graph is the one shape the validator refuses, so a
        // delete must not manufacture it. What a consumer loses is said by
        // its now-empty socket, on screen.
        const next: PgGraph = {}
        for (const [k, node] of Object.entries(g)) {
          if (k === ch.id) continue
          const inputs: Record<string, unknown> = {}
          for (const [name, val] of Object.entries(node.inputs ?? {})) {
            if (isLink(val) && String(val[0]) === ch.id) continue
            inputs[name] = val
          }
          next[k] = { ...node, inputs }
        }
        s.setPg({ graph: next, dirty: true,
                  sel: s.pg.sel === ch.id ? null : s.pg.sel })
      }
    }
  }, [setPg])

  const onEdgesChange = useCallback((changes: EdgeChange[]) => {
    for (const ch of changes) {
      if (ch.type !== 'remove') continue
      const hit = edgeIndex.get(ch.id)
      if (!hit) continue
      const s = useStore.getState()
      const g = s.pg.graph
      const node = g?.[hit.target]
      if (!node) continue
      const inputs = { ...(node.inputs ?? {}) }
      delete inputs[hit.input]
      s.setPg({
        graph: { ...g!, [hit.target]: { ...node, inputs } },
        dirty: true,
      })
    }
  }, [edgeIndex])

  const onConnect = useCallback((c: Connection) => {
    const input = c.targetHandle?.startsWith('in-')
      ? c.targetHandle.slice(3) : null
    const slot = c.sourceHandle?.startsWith('out-')
      ? parseInt(c.sourceHandle.slice(4), 10) : NaN
    if (!c.source || !c.target || !input || !Number.isFinite(slot)) return
    const s = useStore.getState()
    const g = s.pg.graph
    const node = g?.[c.target]
    if (!node) return
    s.setPg({
      graph: {
        ...g!,
        [c.target]: {
          ...node,
          inputs: { ...(node.inputs ?? {}), [input]: [c.source, slot] },
        },
      },
      dirty: true,
    })
  }, [])

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      fitView
      minZoom={0.15}
      deleteKeyCode={['Backspace', 'Delete']}
    >
      <Background variant={BackgroundVariant.Dots} gap={22} size={1} />
    </ReactFlow>
  )
}

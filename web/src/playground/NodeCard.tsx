/**
 * One node, as a card — the value is the control, edited where it is read.
 *
 * No inspector and no properties panel: an input is edited on the row that
 * shows it, which is the console's own rule (a control whose value is its
 * label loses its box). Sockets and widgets are one list in the spec's order,
 * because that is the order ComfyUI's canvas shows and the order a pack's
 * README will name.
 */
import { Handle, Position, useUpdateNodeInternals } from '@xyflow/react'
import { useContext, useEffect, useRef } from 'react'

import { useStore } from '../store'
import {
  CatContext, inputRows, isLink, title, widgetKind, type InputRow,
} from './catalogue'

export function NodeCard({ id }: { id: string }) {
  const node = useStore((s) => s.pg.graph?.[id])
  const cat = useContext(CatContext)
  // Handles are added per input row, and rows change as the catalogue lands
  // or a widget is wired — xyflow has to be told or new ports are
  // unconnectable until the node is dragged.
  const update = useUpdateNodeInternals()
  const rowCount = useRef(0)
  const spec = node ? cat?.[node.class_type] : undefined
  const rows = specRows(node?.inputs ?? {}, spec ? inputRows(spec) : null)
  useEffect(() => {
    if (rows.length !== rowCount.current) {
      rowCount.current = rows.length
      update(id)
    }
  })
  if (!node) return null

  const outputs = spec?.output_name ?? spec?.output ?? inferredOutputs(id)

  const set = (name: string, value: unknown) => {
    const s = useStore.getState()
    const g = s.pg.graph
    if (!g?.[id]) return
    s.setPg({
      graph: {
        ...g,
        [id]: { ...g[id], inputs: { ...(g[id].inputs ?? {}), [name]: value } },
      },
      dirty: true,
    })
  }

  return (
    <div className="pgcard">
      <header className="pghead">
        <span className="pgtitle">{title(node)}</span>
        {node._meta?.title && (
          <span className="pgclass">{node.class_type}</span>
        )}
      </header>
      {rows.map((row) => (
        <Row key={row.name} id={id} row={row}
             value={node.inputs?.[row.name]} set={set} />
      ))}
      {outputs.map((label, i) => (
        <div className="pgrow out" key={`o${i}`}>
          <span className="pgname">{String(label)}</span>
          <Handle type="source" position={Position.Right} id={`out-${i}`}
                  className="pgport" />
        </div>
      ))}
    </div>
  )
}

/** The spec's rows in the spec's order, plus any input the graph carries that
 *  the spec does not name — an unknown catalogue must not hide real wiring. */
function specRows(
  inputs: Record<string, unknown>, fromSpec: InputRow[] | null,
): InputRow[] {
  const rows = fromSpec ? [...fromSpec] : []
  const named = new Set(rows.map((r) => r.name))
  for (const [name, val] of Object.entries(inputs)) {
    if (!named.has(name)) {
      rows.push({
        name, required: false,
        type: isLink(val) ? '*' : typeof val === 'number' ? 'FLOAT'
          : typeof val === 'boolean' ? 'BOOLEAN' : 'STRING',
        config: {},
      })
    }
  }
  return rows
}

/** When the catalogue has never heard of this node, its used output slots are
 *  still visible in the graph — every link that names it names a slot. */
function inferredOutputs(id: string): string[] {
  const g = useStore.getState().pg.graph
  let top = -1
  for (const node of Object.values(g ?? {})) {
    for (const val of Object.values(node.inputs ?? {})) {
      if (isLink(val) && String(val[0]) === id) top = Math.max(top, val[1])
    }
  }
  return Array.from({ length: top + 1 }, (_, i) => `out ${i}`)
}

function Row({ id, row, value, set }: {
  id: string
  row: InputRow
  value: unknown
  set: (name: string, value: unknown) => void
}) {
  const kind = widgetKind(row)
  const wired = isLink(value)
  return (
    <div className={`pgrow${row.required && value === undefined && !kind
      ? ' want' : ''}`}>
      <Handle type="target" position={Position.Left} id={`in-${row.name}`}
              className="pgport" />
      <span className="pgname">{row.name}</span>
      {wired ? null : kind === 'choice' ? (
        <select className="nodrag pgval" value={String(value ?? '')}
                onChange={(e) => set(row.name, e.target.value)}>
          {(Array.isArray(row.type) ? row.type : []).map((opt) => (
            <option key={String(opt)} value={String(opt)}>{String(opt)}</option>
          ))}
        </select>
      ) : kind === 'number' ? (
        <input autoComplete="off" className="nodrag pgval" type="number"
               value={value === undefined ? '' : String(value)}
               step={row.type === 'INT' ? 1 : 'any'}
               onChange={(e) => {
                 const n = row.type === 'INT'
                   ? parseInt(e.target.value, 10) : parseFloat(e.target.value)
                 set(row.name, Number.isFinite(n) ? n : 0)
               }} />
      ) : kind === 'toggle' ? (
        <input className="nodrag pgval" type="checkbox" checked={!!value}
               onChange={(e) => set(row.name, e.target.checked)} />
      ) : kind === 'text' ? (
        <input autoComplete="off" className="nodrag pgval" type="text"
               value={value === undefined ? '' : String(value)}
               onChange={(e) => set(row.name, e.target.value)} />
      ) : kind === 'file' ? (
        <FilePick id={id} name={row.name} value={value} set={set} />
      ) : null}
    </div>
  )
}

/**
 * The picture a LoadImage-style input names. Choosing one stores the bytes in
 * the room's attachments under the file's own name and writes that name into
 * the input — the run stages the file under the same name, so the graph that
 * ran is the graph on screen.
 */
function FilePick({ id: _id, name, value, set }: {
  id: string
  name: string
  value: unknown
  set: (name: string, value: unknown) => void
}) {
  const setPg = useStore((s) => s.setPg)
  return (
    <label className="nodrag pgval pgfile">
      {typeof value === 'string' && value ? value : 'choose…'}
      <input type="file" accept="image/*" hidden
             onChange={(e) => {
               const f = e.target.files?.[0]
               if (!f) return
               const reader = new FileReader()
               reader.onload = () => {
                 const url = String(reader.result || '')
                 const b64 = url.slice(url.indexOf(',') + 1)
                 const s = useStore.getState()
                 setPg({
                   attachments: { ...s.pg.attachments, [f.name]: b64 },
                   dirty: true,
                 })
                 set(name, f.name)
               }
               reader.readAsDataURL(f)
             }} />
    </label>
  )
}

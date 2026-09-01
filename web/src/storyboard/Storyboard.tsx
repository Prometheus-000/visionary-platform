/**
 * The storyboard — the wall.
 *
 * A room behind a door for now, on the Playground's delivery pattern: its own
 * store of one document, the same volume, no change to either console. Its
 * final home is an altitude of the canvas rather than a room (zoom out and
 * the render you were judging recedes into its slot in the sequence), and the
 * door is scaffolding for that, not a destination.
 *
 * What the wall holds is *order*. Panels read left to right and wrap; a panel
 * is a numbered cell — a frame at the board's aspect with the camera's move
 * and the subjects' moves drawn on it, and prose under it; moving one is a
 * splice and says "this happens here instead". There is no duration, no
 * ruler, no take seam anywhere on it — a storyboard that knows seconds is
 * already half an edit, and the director's medium is succession. Seconds are
 * generation territory and live on receipts.
 *
 * **A dragged panel is carried, not marked.** Press its number and move, and
 * the panel itself comes with the hand while the others slide apart to make
 * the slot — the Playground's feel rather than the gallery's, where a
 * hairline said where a thing would land while the thing stayed put. The
 * slot is the nearest cell to the hand, so it snaps by construction and the
 * board is never in a state a save could not describe.
 *
 * Several boards, because a scene put down on Tuesday is picked up on Friday
 * beside the one started on Thursday. Each is a folder on the volume with
 * its pictures beside it; the list is the folder listing.
 */
import { useCallback, useEffect, useRef, useState } from 'react'

import { failed, type ApiError } from '../api/client'
import {
  compile, deleteStoryboard, getStoryboard, listStoryboards, saveStoryboard, uploadStoryboard,
} from '../api/routes'
import { VIDEO_ASPECTS } from '../console/SizeButton'
import { IconPlus, IconStack } from '../icons'
import { useStore } from '../store'
import { ErrorNote } from '../ui/ErrorNote'
import { Menu, type MenuItem } from '../ui/Menu'
import { Sheet } from '../ui/Sheet'
import { useFlip } from './flip'
import {
  createBoard, handoffOf, newPanel, pictureUrl, readBoard, readRows, splice,
  type Board, type BoardRow, type Handoff, type Panel,
} from './model'
import { PanelCard } from './Panel'
import './storyboard.css'

type Drag = {
  id: string
  /** The hand's offset into the panel at the press, so the carried panel
   *  stays under the fingers rather than jumping to its corner. */
  ox: number
  oy: number
  w: number
  h: number
  x: number
  y: number
  /** Set once the press has travelled — before that it is a click. */
  live: boolean
}

export function Storyboard({ onOpen, onGallery }: {
  onOpen: (h: Handoff) => void
  /** Open the gallery to choose a picture for the panel `store.board.pick`
   *  names — the pin lands in it instead of appending. */
  onGallery: () => void
}) {
  const setBoardSlot = useStore((s) => s.setBoard)
  const [rows, setRows] = useState<BoardRow[] | null>(null)
  const [board, setBoard] = useState<Board | null>(null)
  const [err, setErr] = useState<ApiError | null>(null)
  const [sel, setSel] = useState<string[]>([])
  const [drag, setDrag] = useState<Drag | null>(null)
  const [menu, setMenu] = useState<{ anchor: HTMLElement; id: string } | null>(null)
  const [boards, setBoards] = useState<HTMLElement | null>(null)
  const [aspects, setAspects] = useState<HTMLElement | null>(null)
  const [over, setOver] = useState<string | null>(null)
  const [doc, setDoc] = useState<string | null>(null)
  const [busy, setBusy] = useState(0)
  const wallRef = useRef<HTMLDivElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const fileFor = useRef<string | null>(null)

  /* Saving. The whole document, debounced behind the keystroke, and flushed
     when the room unmounts — a mode switch mid-sentence must not lose the
     sentence. `latest` is what the save reads, because a timer fires after
     the render that scheduled it and must not save a stale closure. */
  const latest = useRef<Board | null>(null)
  const dirty = useRef(false)
  const timer = useRef<number | undefined>(undefined)

  const flush = useCallback(async () => {
    window.clearTimeout(timer.current)
    const b = latest.current
    if (!dirty.current || !b) return
    dirty.current = false
    const r = await saveStoryboard(b.name, b)
    if (failed(r)) { setErr(r); dirty.current = true; return }
    setRows((rs) => rs?.map((x) => (x.name === b.name
      ? { ...x, title: b.title, panels: b.panels.length, updated: r.updated ?? x.updated,
          cover: b.panels.find((p) => p.picture)?.picture ?? null }
      : x)) ?? rs)
  }, [])

  const edit = useCallback((fn: (b: Board) => Board) => {
    setBoard((b) => {
      if (!b) return b
      const n = fn(b)
      latest.current = n
      dirty.current = true
      return n
    })
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => void flush(), 600)
  }, [flush])

  const open = useCallback(async (name: string) => {
    await flush()
    const r = await getStoryboard(name)
    if (failed(r)) { setErr(r); return }
    const b = readBoard(r.board, name)
    latest.current = b
    dirty.current = false
    setBoard(b)
    setSel([])
    setErr(null)
    setBoardSlot({ name })
  }, [flush, setBoardSlot])

  // The list, then the board that was open last — or the most recent, which
  // is the same answer nearly always and an honest one when it is not.
  useEffect(() => {
    let alive = true
    void (async () => {
      const r = await listStoryboards()
      if (!alive) return
      if (failed(r)) { setErr(r); setRows([]); return }
      const list = readRows(r)
      setRows(list)
      const want = useStore.getState().board.name
      const first = list.find((x) => x.name === want)?.name ?? list[0]?.name
      if (first) void open(first)
    })()
    return () => { alive = false; void flush() }
  }, [flush, open])

  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      const t = e.target as HTMLElement | null
      if (t?.matches?.('input,textarea')) return
      setSel([])
    }
    document.addEventListener('keydown', key)
    return () => document.removeEventListener('keydown', key)
  }, [])

  const panels = board?.panels ?? []
  useFlip(wallRef, '.sbpanel', panels.map((p) => p.id).join('|'))

  const patch = useCallback((id: string, p: Partial<Panel>) =>
    edit((b) => ({ ...b, panels: b.panels.map((x) => (x.id === id ? { ...x, ...p } : x)) })), [edit])
  const move = (from: number, to: number) =>
    edit((b) => ({ ...b, panels: splice(b.panels, from, to) }))
  const remove = (id: string) => {
    edit((b) => ({ ...b, panels: b.panels.filter((x) => x.id !== id) }))
    setSel((s) => s.filter((x) => x !== id))
  }
  const add = (at?: number, p: Partial<Panel> = {}) =>
    edit((b) => {
      const next = b.panels.slice()
      next.splice(at ?? next.length, 0, newPanel(p))
      return { ...b, panels: next }
    })
  const idx = (id: string) => panels.findIndex((p) => p.id === id)

  /* ---- boards --------------------------------------------------------- */

  const create = async () => {
    await flush()
    const r = await createBoard('')
    if (failed(r)) { setErr(r); return }
    setRows((rs) => [{ name: r.name, title: '', panels: 0, updated: r.updated ?? null, cover: null },
                     ...(rs ?? [])])
    latest.current = r
    dirty.current = false
    setBoard(r)
    setSel([])
    setBoardSlot({ name: r.name })
    requestAnimationFrame(() => document.querySelector<HTMLInputElement>('#sb-title')?.focus())
  }

  const destroy = async () => {
    if (!board) return
    const uploads = board.panels.filter((p) => p.picture && !p.picture.job_id).length
    const what = board.title.trim() || board.name
    if (!confirm(`Delete "${what}"?\n\nIt has ${board.panels.length} panel${board.panels.length === 1 ? '' : 's'}`
                 + (uploads ? ` and ${uploads} uploaded picture${uploads === 1 ? '' : 's'}, which go with it.` : '.')
                 + ' Renders in the gallery are untouched.')) return
    dirty.current = false
    const r = await deleteStoryboard(board.name)
    if (failed(r)) { setErr(r); return }
    const rest = (rows ?? []).filter((x) => x.name !== board.name)
    setRows(rest)
    setBoard(null)
    latest.current = null
    setBoardSlot({ name: null, pick: null })
    if (rest[0]) void open(rest[0].name)
  }

  /* ---- pictures ------------------------------------------------------- */

  const upload = async (files: FileList | File[], target: string | null) => {
    if (!board) return
    const list = Array.from(files).filter((f) => f.type.startsWith('image/'))
    if (!list.length) return
    const fd = new FormData()
    for (const f of list) fd.append('files', f, f.name)
    setBusy((n) => n + 1)
    const r = await uploadStoryboard(board.name, fd)
    setBusy((n) => n - 1)
    if (failed(r)) { setErr(r); return }
    const got = r.files.map((f) => ({ file: f.file }))
    edit((b) => {
      const next = b.panels.slice()
      let at = target ? next.findIndex((p) => p.id === target) : -1
      const rest = got.slice()
      if (at >= 0) {
        // The first replaces the panel that was dropped on; any more become
        // panels after it — several pictures dropped at once is a sequence.
        next[at] = { ...next[at]!, picture: rest.shift()! }
        at += 1
      } else {
        at = next.length
      }
      next.splice(at, 0, ...rest.map((picture) => newPanel({ picture })))
      return { ...b, panels: next }
    })
  }

  const hasFiles = (e: React.DragEvent) => Array.from(e.dataTransfer.types).includes('Files')

  /* ---- carrying a panel ----------------------------------------------- */

  const grab = (e: React.PointerEvent<HTMLElement>, id: string) => {
    if (e.button !== 0) return
    const cell = (e.currentTarget as HTMLElement).closest<HTMLElement>('.sbpanel')
    if (!cell) return
    const r = cell.getBoundingClientRect()
    const start = { x: e.clientX, y: e.clientY }
    const d: Drag = { id, ox: e.clientX - r.left, oy: e.clientY - r.top,
                      w: r.width, h: r.height, x: r.left, y: r.top, live: false }
    let cur: Drag = d
    const onMove = (ev: PointerEvent) => {
      if (!cur.live && Math.hypot(ev.clientX - start.x, ev.clientY - start.y) < 6) return
      cur = { ...cur, live: true, x: ev.clientX - cur.ox, y: ev.clientY - cur.oy }
      setDrag(cur)
      // The slot is the nearest cell to the hand. The cells are measured live
      // because a `whole` panel is taller than its neighbours and rows shift.
      const cells = Array.from(wallRef.current?.querySelectorAll<HTMLElement>('.sbpanel') ?? [])
      let best = -1
      let dist = Infinity
      cells.forEach((c, i) => {
        const cr = c.getBoundingClientRect()
        const dd = Math.hypot(ev.clientX - (cr.left + cr.width / 2), ev.clientY - (cr.top + cr.height / 2))
        if (dd < dist) { dist = dd; best = i }
      })
      const from = cells.findIndex((c) => c.dataset.id === id)
      if (best >= 0 && from >= 0 && best !== from) move(from, best)
    }
    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
      window.removeEventListener('pointercancel', onUp)
      if (!cur.live) {
        // It stayed put: a selection. A second panel makes a pair, in the
        // order they were chosen; a third starts over on that one.
        setSel((s) => (s.includes(id) ? s.filter((x) => x !== id)
                       : s.length >= 2 ? [id] : [...s, id]))
        return
      }
      // Put down: the carried panel rides to its slot, then the slot shows it.
      const cell = wallRef.current?.querySelector<HTMLElement>(`.sbpanel[data-id="${id}"]`)
      const cr = cell?.getBoundingClientRect()
      if (cr) setDrag({ ...cur, x: cr.left, y: cr.top })
      window.setTimeout(() => setDrag(null), 190)
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    window.addEventListener('pointercancel', onUp)
  }

  /* ---- the hand-off ------------------------------------------------------ */

  const animate = () => { if (board) onOpen(handoffOf(board, sel)) }

  const showDoc = async () => {
    if (!board) return
    const h = handoffOf(board, sel.length === 2 ? sel : [])
    const one = h.shots.length === 1
    const r = await compile({
      kind: 'video',
      prompt: h.shots.map((s) => s.line).join('\n'),
      shot: one ? h.shots[0]!.pills : [],
      first_frame: !!h.first,
      last_frame: !!h.last,
      // Several panels is a scene: one shot each, in the composer's own
      // payload shape, so the document read here is the one the run gets.
      ...(!one && {
        scene: {
          style: '', grade: '', sources: {}, cast: [],
          shots: h.shots.map((s) => ({
            line: s.line, beats: 4, pills: s.pills,
            say: { who: [], text: '', lang: 'English', voice: '', carry: false, cutoff: false, offscreen: false },
          })),
        },
      }),
    })
    if (failed(r)) { setErr(r); return }
    setDoc(r.prompt)
  }

  /* ---- menus ---------------------------------------------------------- */

  const panelItems = (id: string): MenuItem[] => {
    const p = panels.find((x) => x.id === id)
    if (!p || !board) return []
    const i = idx(id)
    return [
      { label: 'Animate from this panel', run: () => onOpen(handoffOf(board, [id])) },
      { sep: true },
      { label: 'Choose a picture from the gallery', run: () => { setBoardSlot({ pick: id }); onGallery() } },
      { label: 'Upload a picture…', run: () => { fileFor.current = id; fileRef.current?.click() } },
      ...(p.picture ? [
        p.fit === 'whole'
          ? { label: `Crop to ${board.aspect}`, run: () => patch(id, { fit: 'crop' as const }) }
          : { label: 'Show the whole picture', hint: 'the frame moves on it', run: () => patch(id, { fit: 'whole' as const }) },
        { label: 'Remove the picture', run: () => patch(id, { picture: null, fit: 'crop' as const }) },
      ] : []),
      { sep: true },
      { label: 'Add a panel after this', run: () => add(i + 1) },
      { label: 'Duplicate', run: () => add(i + 1, { ...p, id: undefined, motion: p.motion.map((m) => ({ ...m })) }) },
      { sep: true },
      { label: 'Remove panel', danger: true, run: () => remove(id) },
    ]
  }

  const boardItems = (): MenuItem[] => [
    ...(rows ?? []).map((r) => ({
      label: r.title.trim() || r.name,
      hint: `${r.panels} panel${r.panels === 1 ? '' : 's'}`,
      on: r.name === board?.name,
      run: () => { if (r.name !== board?.name) void open(r.name) },
    })),
    ...(rows?.length ? [{ sep: true as const }] : []),
    { label: 'New storyboard', run: () => void create() },
    ...(board ? [
      { sep: true as const },
      { label: 'Read the H3 document', hint: sel.length === 2 ? 'for the pair' : 'the whole board', run: () => void showDoc() },
      { label: 'Open the scene on the canvas', hint: 'every panel, one shot each', run: () => onOpen(handoffOf(board, [])) },
      { sep: true as const },
      { label: 'Delete this storyboard', danger: true, run: () => void destroy() },
    ] : []),
  ]

  const carried = drag?.live ? panels.find((p) => p.id === drag.id) ?? null : null
  const label = sel.map((id) => idx(id) + 1).filter((n) => n > 0)

  return (
    <div className={`view storyboard${drag?.live ? ' carrying' : ''}`} id="v-storyboard">
      <div className="sbbar">
        <button className="ico" id="sb-boards" type="button" title="Storyboards"
                onClick={(e) => setBoards(e.currentTarget)}>
          <IconStack />
        </button>
        <input className="sbtitle" id="sb-title" placeholder="Untitled storyboard"
               value={board?.title ?? ''} disabled={!board}
               onChange={(e) => edit((b) => ({ ...b, title: e.target.value }))} />
        {board && (
          <button className="opt" id="sb-aspect" type="button" title="The frame every panel is cut to"
                  onClick={(e) => setAspects(e.currentTarget)}>
            {board.aspect}
          </button>
        )}
        <span className="sbcount">
          {board ? `${panels.length} panel${panels.length === 1 ? '' : 's'}` : ''}
          {busy ? ' · uploading…' : ''}
        </span>
        <span className="grow" />
        {board && label.length > 0 && (
          <button className="b" id="sb-animate" type="button"
                  title={label.length === 2
                    ? 'First and last frame: one shot from the first panel to the second'
                    : 'Open this panel on the canvas as the first frame of a take'}
                  onClick={animate}>
            Animate {label.join(' → ')}
          </button>
        )}
        <button className="opt" id="sb-add" type="button" disabled={!board}
                title="A blank panel at the end — prose first, picture later"
                onClick={() => add()}>
          <IconPlus /> Panel
        </button>
      </div>

      {err && <div className="sbnote-line"><ErrorNote err={err} /></div>}

      <div className={`sbwall${over === 'wall' ? ' hot' : ''}`} ref={wallRef}
           onDragOver={(e) => { if (board && hasFiles(e)) { e.preventDefault(); if (!over) setOver('wall') } }}
           onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node)) setOver(null) }}
           onDrop={(e) => {
             if (!hasFiles(e)) return
             e.preventDefault()
             const cell = (e.target as Element).closest<HTMLElement>('.sbpanel')
             setOver(null)
             void upload(e.dataTransfer.files, cell?.dataset.id ?? null)
           }}>
        {rows && !board && rows.length === 0 ? (
          <div className="sbnone">
            No storyboard yet. Start one and add panels — or pin a render from
            the gallery; a card's menu has it.
            <div><button className="b" id="sb-new" type="button" onClick={() => void create()}>New storyboard</button></div>
          </div>
        ) : board && panels.length === 0 ? (
          <div className="sbnone">
            Nothing on this storyboard yet. Add a panel and write what happens,
            drop pictures here, or pin a render from the gallery.
            <div><button className="b" type="button" onClick={() => add()}>Add a panel</button></div>
          </div>
        ) : (
          <div className="sbrow">
            {panels.map((p, i) => (
              <div key={p.id} className={`sbslot${drag?.live && drag.id === p.id ? ' held' : ''}${over === p.id ? ' hot' : ''}`}
                   onDragOver={(e) => { if (hasFiles(e)) { e.preventDefault(); e.stopPropagation(); setOver(p.id) } }}>
                <PanelCard panel={p} i={i} board={board!.name} aspect={board!.aspect}
                           sel={sel.indexOf(p.id)}
                           onPatch={(x) => patch(p.id, x)}
                           onGrab={(e) => grab(e, p.id)}
                           onMenu={(anchor) => setMenu({ anchor, id: p.id })} />
              </div>
            ))}
          </div>
        )}
      </div>

      {carried && drag && (
        <div className="sbghost" style={{ left: drag.x, top: drag.y, width: drag.w }}>
          <div className="sbframe" style={{ aspectRatio: board!.aspect.replace(':', '/') }}>
            {carried.picture && (
              <img src={pictureUrl(board!.name, carried.picture, !!carried.picture.job_id)} alt=""
                   style={{ objectPosition: `${carried.focus[0] * 100}% ${carried.focus[1] * 100}%` }} />
            )}
          </div>
          {carried.prose && <div className="sbghost-prose">{carried.prose}</div>}
        </div>
      )}

      <input ref={fileRef} type="file" accept="image/*" multiple hidden
             onChange={(e) => {
               const files = e.currentTarget.files
               if (files?.length) void upload(files, fileFor.current)
               fileFor.current = null
               e.currentTarget.value = ''
             }} />

      {menu && <Menu anchor={menu.anchor} items={panelItems(menu.id)} onClose={() => setMenu(null)} />}
      {boards && <Menu anchor={boards} items={boardItems()} onClose={() => setBoards(null)} />}
      {aspects && board && (
        <Menu anchor={aspects} onClose={() => setAspects(null)}
              items={VIDEO_ASPECTS.map((a) => ({
                label: a, on: a === board.aspect,
                run: () => edit((b) => ({ ...b, aspect: a })),
              }))} />
      )}
      {doc !== null && (
        <Sheet id="sb-doc" onClose={() => setDoc(null)}>
          <div className="sbdoc">
            <h3>What H3 will read</h3>
            <p className="muted">
              The document the run gets, compiled by the same compiler — the
              prose is yours, the fields and the camera grammar are its.
            </p>
            <textarea readOnly value={doc} rows={Math.min(24, doc.split('\n').length + 2)} />
            <div className="sbdoc-row">
              <button className="b" type="button"
                      onClick={() => void navigator.clipboard?.writeText(doc)}>
                Copy
              </button>
            </div>
          </div>
        </Sheet>
      )}
    </div>
  )
}

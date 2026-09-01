/**
 * The storyboard's record, and the operations that are about the record
 * rather than about the screen.
 *
 * A panel is intent with a replaceable picture: prose, a note to yourself, a
 * *pointer* at a picture — a render's job and file (the gallery's own address)
 * or a bare filename in the board's own folder for one you dropped in — and
 * the two things a storyboard artist draws on top of the sketch: the camera's
 * move, and where a subject goes. Never the bytes: unpinning is free and a
 * deleted render leaves the words standing.
 *
 * The camera is a *pill* — `camera.panr` with an amplitude and a speed — and
 * that is the whole design of the hand-off. A panel carries a shot's own
 * vocabulary, so becoming a shot is a copy, not a translation, and the stencil
 * drawn on the frame is the pill's picture rather than a second language that
 * has to be compiled into the first.
 *
 * There is no duration anywhere on a panel, and that is the design rather
 * than an omission. Order is the only temporal fact the board holds; seconds,
 * seams and seeds belong to generations and live on receipts.
 */
import { failed, type Res } from '../api/client'
import {
  coverUrl, fileUrl, getStoryboard, listStoryboards, saveStoryboard, storyboardFileUrl,
} from '../api/routes'
import type { CameraAmp, CameraSpeed, ShotGroup, ShotPill } from '../api/types'
import { promptOf, type GalleryItem } from '../gallery/types'

export type Picture = { job_id?: string; file: string }
export type Fit = 'crop' | 'whole'
export type Pt = [number, number]
/** A subject's move: two points, fractions of the picture, and who moves. */
export type Motion = { id: string; pts: [Pt, Pt]; label: string }

export type Panel = {
  id: string
  prose: string
  note: string
  picture: Picture | null
  /** Cropped to the board's aspect, or shown whole with the aspect drawn on it
   *  — the tall-picture-and-a-tilt case every storyboard sheet has one of. */
  fit: Fit
  /** Where the crop sits on the picture, or where the aspect frame sits on a
   *  whole one. `[0.5, 0.5]` is centred. */
  focus: Pt
  pills: ShotPill[]
  motion: Motion[]
}

export type Board = {
  name: string
  title: string
  aspect: string
  panels: Panel[]
  updated?: number | null
}

export type BoardRow = {
  name: string
  title: string
  panels: number
  updated: number | null
  cover: Picture | null
}

export const AMPS: CameraAmp[] = ['small', 'medium', 'large']
export const SPEEDS: CameraSpeed[] = ['slow', 'normal', 'fast']

export const ratioOf = (aspect: string): number => {
  const [w, h] = aspect.split(':').map(Number)
  return w && h ? w / h : 16 / 9
}

export const uid = () =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `p${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`

export const newPanel = (p: Partial<Panel> = {}): Panel =>
  ({ id: uid(), prose: '', note: '', picture: null, fit: 'crop', focus: [0.5, 0.5],
     pills: [], motion: [], ...p })

/** Move the panel at `from` so it lands at `to` — a splice, which is the
 *  whole of what "moving" means on a strictly ordered board. */
export function splice(panels: Panel[], from: number, to: number): Panel[] {
  if (from === to || from < 0 || from >= panels.length) return panels
  const next = panels.slice()
  const p = next.splice(from, 1)[0]
  if (!p) return panels
  next.splice(Math.max(0, Math.min(next.length, to)), 0, p)
  return next
}

const clamp01 = (v: unknown) => Math.max(0, Math.min(1, Number(v) || 0))

/** A stored pill, read leniently: an unknown field is dropped, a known one
 *  kept as typed. The server validated it on the way in. */
function readPill(raw: unknown): ShotPill | null {
  const o = (raw ?? {}) as Record<string, unknown>
  if (typeof o.key !== 'string' || !o.key) return null
  const p: ShotPill = { key: o.key }
  if (typeof o.value === 'string') p.value = o.value
  if (typeof o.lang === 'string') p.lang = o.lang
  if (AMPS.includes(o.amp as CameraAmp)) p.amp = o.amp as CameraAmp
  if (SPEEDS.includes(o.speed as CameraSpeed)) p.speed = o.speed as CameraSpeed
  return p
}

/** What the server hands back, read leniently: a document written by an
 *  older page is still a document, and a field this page does not know is
 *  not a reason to refuse the ones it does. */
export function readBoard(raw: Record<string, unknown> | undefined, name: string): Board {
  const panels = Array.isArray(raw?.panels) ? raw.panels : []
  return {
    name,
    title: String(raw?.title ?? ''),
    aspect: typeof raw?.aspect === 'string' && raw.aspect ? raw.aspect : '16:9',
    updated: typeof raw?.updated === 'number' ? raw.updated : null,
    panels: panels.map((p) => {
      const o = (p ?? {}) as Record<string, unknown>
      const pic = o.picture as Record<string, unknown> | null | undefined
      const focus = Array.isArray(o.focus) ? o.focus : []
      const motion = Array.isArray(o.motion) ? o.motion : []
      return newPanel({
        id: String(o.id ?? uid()),
        prose: String(o.prose ?? ''),
        note: String(o.note ?? ''),
        picture: pic && pic.file
          ? { file: String(pic.file), ...(pic.job_id ? { job_id: String(pic.job_id) } : {}) }
          : null,
        fit: o.fit === 'whole' ? 'whole' : 'crop',
        focus: [clamp01(focus[0] ?? 0.5), clamp01(focus[1] ?? 0.5)],
        pills: (Array.isArray(o.pills) ? o.pills : []).map(readPill)
          .filter((x): x is ShotPill => !!x),
        motion: motion.flatMap((m) => {
          const a = (m ?? {}) as Record<string, unknown>
          const pts = Array.isArray(a.pts) ? a.pts : []
          const p0 = pts[0] as unknown[] | undefined
          const p1 = pts[1] as unknown[] | undefined
          if (!p0 || !p1) return []
          return [{
            id: String(a.id ?? uid()),
            pts: [[clamp01(p0[0]), clamp01(p0[1])], [clamp01(p1[0]), clamp01(p1[1])]],
            label: String(a.label ?? ''),
          } satisfies Motion]
        }),
      })
    }),
  }
}

export function readRows(raw: Record<string, unknown> | undefined): BoardRow[] {
  const rows = Array.isArray(raw?.boards) ? raw.boards : []
  return rows.flatMap((r) => {
    const o = (r ?? {}) as Record<string, unknown>
    if (typeof o.name !== 'string' || !o.name) return []
    const pic = o.cover as Record<string, unknown> | null | undefined
    return [{
      name: o.name,
      title: String(o.title ?? ''),
      panels: Number(o.panels) || 0,
      updated: typeof o.updated === 'number' ? o.updated : null,
      cover: pic && pic.file
        ? { file: String(pic.file), ...(pic.job_id ? { job_id: String(pic.job_id) } : {}) }
        : null,
    }]
  })
}

/** Where a picture's bytes are. A render is served the way the gallery serves
 *  it — its cover for the wall, the file for a keyframe; an upload is the
 *  board's own file either way. */
export function pictureUrl(board: string, pic: Picture, cover = false): string {
  if (pic.job_id) return cover ? coverUrl(pic.job_id, pic.file) : fileUrl(pic.job_id, pic.file)
  return storyboardFileUrl(board, pic.file)
}

/* ---- pills ------------------------------------------------------------ */

export const pillIn = (pills: ShotPill[], group: string): ShotPill | null =>
  pills.find((p) => p.key.startsWith(`${group}.`)) ?? null

export const cameraOf = (p: Panel) => pillIn(p.pills, 'camera')

/** Replace the group's pill — every group the storyboard sets is `pick: one`
 *  — or remove it. Order is kept in vocabulary order by the compiler, so it
 *  does not matter here. */
export function withPill(pills: ShotPill[], group: string, pill: ShotPill | null): ShotPill[] {
  const rest = pills.filter((p) => !p.key.startsWith(`${group}.`))
  return pill ? [...rest, pill] : rest
}

export const labelOf = (vocab: ShotGroup[], key: string): string => {
  const [g, ...rest] = key.split('.')
  return vocab.find((x) => x.key === g)?.items.find((it) => it.key === rest.join('.'))?.label ?? key
}

/** `pan right · large · fast` — the tag under the frame, in the words the
 *  document will use, medium and normal omitted as the guide omits them. */
export function cameraTag(vocab: ShotGroup[], pill: ShotPill): string {
  return [labelOf(vocab, pill.key),
          pill.amp && pill.amp !== 'medium' ? pill.amp : '',
          pill.speed && pill.speed !== 'normal' ? pill.speed : ''].filter(Boolean).join(' · ')
}

/* ---- what an arrow says ------------------------------------------------ */

const EDGE = 0.045

function zone([x, y]: Pt): string {
  const c = x < 1 / 3 ? 0 : x < 2 / 3 ? 1 : 2
  const r = y < 1 / 3 ? 0 : y < 2 / 3 ? 1 : 2
  return [
    ['the upper left of the frame', 'the top of the frame', 'the upper right of the frame'],
    ['frame left', 'the centre of the frame', 'frame right'],
    ['the lower left of the frame', 'the bottom of the frame', 'the lower right of the frame'],
  ][r]![c]!
}

function edge([x, y]: Pt): string | null {
  if (x <= EDGE) return 'frame left'
  if (x >= 1 - EDGE) return 'frame right'
  if (y <= EDGE) return 'the top of the frame'
  if (y >= 1 - EDGE) return 'the bottom of the frame'
  return null
}

function heading(a: Pt, b: Pt): string {
  const dx = b[0] - a[0]
  const dy = b[1] - a[1]
  if (Math.abs(dx) >= Math.abs(dy)) return dx > 0 ? 'to the right' : 'to the left'
  return dy > 0 ? 'downward' : 'upward'
}

/**
 * The sentence an arrow writes, in the person's own frame vocabulary.
 *
 * Derived, and shown as derived: it lands in the shot's line at the hand-off
 * and is printed grey under the prose before then, so what the arrow will say
 * is never a surprise. Two points, so three cases — a move between zones, a
 * move too short to leave one, and a move that starts or ends at the edge,
 * which is how an entrance or an exit is drawn.
 */
export function motionClause(m: Motion): string {
  const who = m.label.trim() || 'The subject'
  const [a, b] = m.pts
  const len = Math.hypot(b[0] - a[0], b[1] - a[1])
  const exit = edge(b)
  const entry = edge(a)
  if (entry && exit && entry !== exit) return `${who} crosses the frame from ${entry} to ${exit}.`
  if (entry) return `${who} enters from ${entry} and moves to ${zone(b)}.`
  if (exit) return `${who} moves from ${zone(a)} and exits ${exit}.`
  if (len < 0.08) return `${who} shifts slightly ${heading(a, b)}.`
  if (zone(a) === zone(b)) return `${who} moves ${heading(a, b)} within ${zone(a)}.`
  return `${who} moves from ${zone(a)} to ${zone(b)}.`
}

export const derived = (p: Panel) => p.motion.map(motionClause).join(' ')

const close = (t: string) => {
  const s = t.trim()
  return s && !/[.!?…"']$/.test(s) ? `${s}.` : s
}

/** The prose, closed, with the arrows' sentences after it: the person's words
 *  first and untouched, then what was drawn. */
export const panelLine = (p: Panel) => [close(p.prose), derived(p)].filter(Boolean).join(' ')

/* ---- the hand-off ------------------------------------------------------ */

export type Handoff = {
  board: string
  aspect: string
  shots: { line: string; pills: ShotPill[] }[]
  first: Picture | null
  last: Picture | null
  /** Two panels: first-and-last-frame, one shot describing the path between
   *  them — the guide's FL2VA shape. */
  pair: boolean
}

/**
 * What the canvas is handed.
 *
 * One panel is a shot with a first frame. Two are the FL2VA shape: one shot
 * whose line is the first panel's, then the second panel's prose introduced
 * as where the shot ends — a connective in front of the person's sentence,
 * never a rewrite of it — with the two pictures as the two keyframes. Every
 * panel is a scene, one shot each, opening on the first picture.
 */
export function handoffOf(board: Board, ids: string[]): Handoff {
  const chosen = ids.map((id) => board.panels.find((p) => p.id === id))
    .filter((p): p is Panel => !!p)
  const panels = chosen.length ? chosen : board.panels
  const pair = chosen.length === 2
  if (pair) {
    const [a, b] = panels as [Panel, Panel]
    const end = b.prose.trim()
      ? `The shot ends on the last frame: ${close(b.prose)}`
      : ''
    return {
      board: board.name, aspect: board.aspect, pair,
      shots: [{ line: [panelLine(a), end].filter(Boolean).join(' '), pills: a.pills }],
      first: a.picture, last: b.picture,
    }
  }
  return {
    board: board.name, aspect: board.aspect, pair,
    shots: panels.map((p) => ({ line: panelLine(p), pills: p.pills })),
    first: panels[0]?.picture ?? null,
    last: null,
  }
}

/* ---- boards ------------------------------------------------------------ */

/** A folder name from a title: the folder under storyboard/ that `cat` will
 *  list the film under. Suffixed so two boards called "Scene 1" are two folders. */
export function slugOf(title: string): string {
  const base = title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40)
  return `${base || 'storyboard'}-${Date.now().toString(36).slice(-4)}`
}

export async function createBoard(title: string, aspect = '16:9'): Promise<Res<Board>> {
  const board: Board = { name: slugOf(title), title, aspect, panels: [], updated: null }
  const r = await saveStoryboard(board.name, board)
  if (failed(r)) return r
  return { ...board, updated: r.updated ?? null }
}

/**
 * Pin a render from the gallery: fetch the board, place, save. Done as a
 * round trip rather than through the wall's own state because the gallery
 * is open in Generate, where the wall is not mounted — and the board on the
 * volume is the one record, so the round trip is what keeps two surfaces
 * from holding two sequences.
 *
 * Into the panel that asked for a picture when one did (`pick`), otherwise
 * appended. The prose is seeded from what the person typed for that render —
 * the durable half of the sidecar, never the compiled receipt — because a
 * render's intent is the best first draft of the panel that pins it; a panel
 * that already has words keeps them.
 */
export async function pinToBoard(
  it: GalleryItem,
  at: { name: string | null; pick: string | null },
): Promise<Res<{ name: string }>> {
  const file = it.files[0]
  if (!file) return { error: 'That result has no file to pin.' }
  let name = at.name
  if (!name) {
    const rows = await listStoryboards()
    if (failed(rows)) return rows
    name = readRows(rows)[0]?.name ?? null
  }
  let board: Board
  if (name) {
    const r = await getStoryboard(name)
    if (failed(r)) return r
    board = readBoard(r.board, name)
  } else {
    const made = await createBoard('')
    if (failed(made)) return made
    board = made
  }
  const picture = { job_id: it.job_id, file }
  const target = at.pick ? board.panels.find((p) => p.id === at.pick) : null
  if (target) {
    target.picture = picture
    if (!target.prose.trim()) target.prose = promptOf(it)
  } else {
    board.panels.push(newPanel({ picture, prose: promptOf(it) }))
  }
  const r = await saveStoryboard(board.name, board)
  if (failed(r)) return r
  return { name: board.name }
}

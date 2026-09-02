/**
 * One panel: a numbered cell, a frame at the board's aspect, the layer drawn
 * on it, and the words under it.
 *
 * **What is under the press decides what the drag does.** The frame takes
 * one press and reads it four ways: on an arrow it moves that arrow, on the
 * camera's stencil it changes the move's amplitude, with ⌥ held it reframes
 * the crop, and on bare picture it draws a subject's arrow — start where they
 * are, let go where they end up. Reordering is the *cell's* gesture, off its
 * number strip, so a drag on the picture never has to guess whether it meant
 * "this panel goes there" or "she goes there". That is the regions layer's
 * own rule — a click is not a drag, and the release says which — with the
 * scope resolved toward the smaller object: an arrow before the frame, the
 * stencil before the picture.
 *
 * The tags under the frame are the caption line a storyboard sheet carries —
 * framing, angle, and the camera move with its amplitude and speed — and each
 * is the served vocabulary, picked from the palette's own tiles, so a panel
 * carries exactly what a shot does.
 */
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'

import type { CameraAmp, ShotPill } from '../api/types'
import { IconMore } from '../icons'
import { useStore } from '../store'
import { MotionLayer } from './MotionLayer'
import {
  AMPS, cameraOf, cameraTag, derived, labelOf, pictureUrl, pillIn, ratioOf, uid, withPill,
  type Motion, type Panel, type Pt,
} from './model'
import { PillPicker } from './PillPicker'

const grow = (el: HTMLTextAreaElement | null) => {
  if (!el) return
  el.style.height = '0'
  el.style.height = `${el.scrollHeight}px`
}

const clamp = (v: number) => Math.max(0, Math.min(1, v))

type Gesture =
  | { kind: 'draw'; start: Pt }
  | { kind: 'move'; id: string; start: Pt; pts0: [Pt, Pt] }
  | { kind: 'focus'; start: Pt; focus0: Pt }
  | { kind: 'amp'; d0: number }

/** The pixel size of an element, kept current. */
function useSize(ref: React.RefObject<HTMLElement | null>) {
  const [size, setSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 })
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const read = () => {
      const r = el.getBoundingClientRect()
      setSize((s) => (s.w === r.width && s.h === r.height ? s : { w: r.width, h: r.height }))
    }
    read()
    const ro = new ResizeObserver(read)
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref])
  return size
}

export function PanelCard({ panel, i, board, aspect, sel, onPatch, onGrab, onMenu }: {
  panel: Panel
  i: number
  board: string
  aspect: string
  /** Position in the selection, or -1. */
  sel: number
  onPatch: (patch: Partial<Panel>) => void
  /** A press on the number strip. The wall decides whether it becomes a
   *  reorder or a selection, by whether it travels. */
  onGrab: (e: React.PointerEvent<HTMLElement>) => void
  onMenu: (anchor: HTMLElement) => void
}) {
  const vocab = useStore((s) => s.state?.shot_vocab ?? [])
  const frameRef = useRef<HTMLDivElement>(null)
  const { w, h } = useSize(frameRef)
  const latest = useRef(panel)
  latest.current = panel

  const [nat, setNat] = useState<Pt | null>(null)
  const [gone, setGone] = useState(false)
  const [draft, setDraft] = useState<[Pt, Pt] | null>(null)
  const [arrow, setArrow] = useState<string | null>(null)
  const [pick, setPick] = useState<{ group: string; anchor: HTMLElement } | null>(null)
  const gesture = useRef<Gesture | null>(null)

  const src = panel.picture ? pictureUrl(board, panel.picture, !!panel.picture.job_id) : null
  useEffect(() => { setGone(false); setNat(null) }, [src])

  const ratio = ratioOf(aspect)
  const whole = panel.fit === 'whole' && !!nat
  const natRatio = nat ? nat[0] / nat[1] : ratio
  // The camera's frame on the picture, in pixels: the whole frame when the
  // picture is cropped to the aspect; the aspect box, placed by `focus`, when
  // the picture is shown whole.
  const cam = (() => {
    if (!whole || !w || !h) return { x: 0, y: 0, w, h }
    if (natRatio < ratio) {
      const fh = w / ratio
      return { x: 0, y: (h - fh) * panel.focus[1], w, h: fh }
    }
    const fw = h * ratio
    return { x: (w - fw) * panel.focus[0], y: 0, w: fw, h }
  })()

  const norm = useCallback((e: { clientX: number; clientY: number }): Pt => {
    const r = frameRef.current!.getBoundingClientRect()
    return [clamp((e.clientX - r.left) / r.width), clamp((e.clientY - r.top) / r.height)]
  }, [])

  const down = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return
    const t = e.target as Element
    if (t.closest('button,input,textarea')) return
    const hit = t.closest('[data-arrow]') as SVGElement | null
    const p = latest.current
    const at = norm(e)
    e.currentTarget.setPointerCapture(e.pointerId)
    if (e.altKey && p.picture) {
      gesture.current = { kind: 'focus', start: at, focus0: p.focus }
    } else if (hit) {
      const id = hit.getAttribute('data-arrow')!
      const m = p.motion.find((x) => x.id === id)
      if (!m) return
      setArrow(id)
      gesture.current = { kind: 'move', id, start: at, pts0: m.pts }
    } else if (t.closest('[data-stencil]')) {
      const r = frameRef.current!.getBoundingClientRect()
      const c = [r.left + cam.x + cam.w / 2, r.top + cam.y + cam.h / 2]
      gesture.current = { kind: 'amp', d0: Math.hypot(e.clientX - c[0]!, e.clientY - c[1]!) }
    } else {
      setArrow(null)
      gesture.current = { kind: 'draw', start: at }
    }
    e.preventDefault()
  }

  const moveP = (e: React.PointerEvent<HTMLDivElement>) => {
    const g = gesture.current
    if (!g) return
    const at = norm(e)
    const p = latest.current
    if (g.kind === 'draw') {
      setDraft([g.start, at])
    } else if (g.kind === 'move') {
      const dx = at[0] - g.start[0]
      const dy = at[1] - g.start[1]
      // The whole arrow travels; clamped so neither end leaves the frame,
      // which is what keeps the sentence it writes about *this* picture.
      const lim = (d: number, a: number, b: number) => Math.max(-Math.min(a, b), Math.min(1 - Math.max(a, b), d))
      const ddx = lim(dx, g.pts0[0][0], g.pts0[1][0])
      const ddy = lim(dy, g.pts0[0][1], g.pts0[1][1])
      const pts: [Pt, Pt] = [[g.pts0[0][0] + ddx, g.pts0[0][1] + ddy],
                             [g.pts0[1][0] + ddx, g.pts0[1][1] + ddy]]
      onPatch({ motion: p.motion.map((m) => (m.id === g.id ? { ...m, pts } : m)) })
    } else if (g.kind === 'focus') {
      const dx = at[0] - g.start[0]
      const dy = at[1] - g.start[1]
      if (p.fit === 'whole') {
        // The aspect box rides with the hand along the axis that has slack.
        const slackX = natRatio > ratio ? 1 - ratio / natRatio : 0
        const slackY = natRatio < ratio ? 1 - natRatio / ratio : 0
        onPatch({ focus: [clamp(g.focus0[0] + (slackX ? dx / slackX : 0)),
                          clamp(g.focus0[1] + (slackY ? dy / slackY : 0))] })
      } else {
        // The picture rides with the hand: dragging it right shows more of
        // its left, which is `object-position` moving the other way.
        onPatch({ focus: [clamp(g.focus0[0] - dx * 1.5), clamp(g.focus0[1] - dy * 1.5)] })
      }
    }
  }

  const up = (e: React.PointerEvent<HTMLDivElement>) => {
    const g = gesture.current
    gesture.current = null
    if (!g) return
    const p = latest.current
    if (g.kind === 'draw') {
      const at = norm(e)
      setDraft(null)
      // A press that did not travel is a click, and a click on bare picture
      // puts the arrow's label away. A short travel is a subject who barely
      // moves, which is a thing worth drawing.
      if (Math.hypot(at[0] - g.start[0], at[1] - g.start[1]) < 0.04) return
      const m: Motion = { id: uid(), pts: [g.start, at], label: '' }
      onPatch({ motion: [...p.motion, m] })
      setArrow(m.id)
    } else if (g.kind === 'amp') {
      const r = frameRef.current!.getBoundingClientRect()
      const c = [r.left + cam.x + cam.w / 2, r.top + cam.y + cam.h / 2]
      const d = Math.hypot(e.clientX - c[0]!, e.clientY - c[1]!) - g.d0
      const camera = cameraOf(p)
      if (!camera || Math.abs(d) < 18) return
      const k = AMPS.indexOf((camera.amp ?? 'medium') as CameraAmp)
      const next = AMPS[Math.max(0, Math.min(AMPS.length - 1, k + (d > 0 ? 1 : -1)))]!
      onPatch({ pills: withPill(p.pills, 'camera', { ...camera, amp: next }) })
    }
  }

  // Delete removes the selected arrow, unless a field has the keys.
  useEffect(() => {
    if (!arrow) return
    const key = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      if (t?.matches?.('input,textarea') || t?.isContentEditable) return
      if (e.key === 'Backspace' || e.key === 'Delete') {
        e.preventDefault()
        onPatch({ motion: latest.current.motion.filter((m) => m.id !== arrow) })
        setArrow(null)
      } else if (e.key === 'Escape') {
        setArrow(null)
      }
    }
    document.addEventListener('keydown', key)
    return () => document.removeEventListener('keydown', key)
  }, [arrow, onPatch])

  const camera = cameraOf(panel)
  const framing = pillIn(panel.pills, 'framing')
  const angle = pillIn(panel.pills, 'angle')
  const selArrow = arrow ? panel.motion.find((m) => m.id === arrow) ?? null : null
  const said = derived(panel)
  const group = pick ? vocab.find((g) => g.key === pick.group) : null

  const tag = (key: string, pill: ShotPill | null, text: string) => (
    <button type="button" className={`sbtag sbtag-${key}${pill ? ' set' : ''}`}
            title={pill ? `Change the ${key}` : `Choose a ${key}`}
            onClick={(e) => setPick({ group: key, anchor: e.currentTarget })}>
      {text}
    </button>
  )

  return (
    <div className={`sbpanel${sel >= 0 ? ' sel' : ''}${whole ? ' whole' : ''}`} data-id={panel.id}>
      <div className="sbhead" onPointerDown={onGrab}>
        <span className="sbidx" data-sel={sel >= 0 ? sel + 1 : undefined}>{i + 1}</span>
        <span className="grow" />
        <button className="ico" type="button" title="This panel"
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => onMenu(e.currentTarget)}>
          <IconMore />
        </button>
      </div>

      <div className={`sbframe${panel.picture && !gone ? '' : ' blank'}`} ref={frameRef}
           style={{ aspectRatio: whole ? natRatio : ratio }}
           onPointerDown={down} onPointerMove={moveP} onPointerUp={up}
           onPointerCancel={() => { gesture.current = null; setDraft(null) }}>
        {src && !gone ? (
          <img src={src} alt="" draggable={false}
               style={{ objectFit: whole ? 'contain' : 'cover',
                        objectPosition: `${panel.focus[0] * 100}% ${panel.focus[1] * 100}%` }}
               onLoad={(e) => setNat([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])}
               onError={() => setGone(true)} />
        ) : (
          <span className="sbempty">
            {gone ? 'that render is gone — the words stay' : 'no picture yet — drop one here'}
          </span>
        )}
        {whole && (
          <div className="sbaspect" style={{ left: cam.x, top: cam.y, width: cam.w, height: cam.h }} />
        )}
        <MotionLayer w={w} h={h} camera={camera} motion={panel.motion} draft={draft}
                     sel={arrow} frame={cam} />
        {selArrow && (
          <>
            <input autoComplete="off" className="sblabel" autoFocus placeholder="who moves"
                   value={selArrow.label}
                   style={{ left: `${selArrow.pts[0][0] * 100}%`, top: `${selArrow.pts[0][1] * 100}%` }}
                   onPointerDown={(e) => e.stopPropagation()}
                   onKeyDown={(e) => {
                     if (e.key === 'Enter' || e.key === 'Escape') setArrow(null)
                     // An arrow with no name yet, and Backspace: the arrow was
                     // the thing being taken back, not a letter.
                     if ((e.key === 'Backspace' || e.key === 'Delete') && !selArrow.label) {
                       onPatch({ motion: panel.motion.filter((m) => m.id !== selArrow.id) })
                       setArrow(null)
                     }
                   }}
                   onChange={(e) => onPatch({
                     motion: panel.motion.map((m) => (m.id === selArrow.id ? { ...m, label: e.target.value } : m)),
                   })} />
            <button type="button" className="sbx" title="Remove this arrow"
                    style={{ left: `${selArrow.pts[1][0] * 100}%`, top: `${selArrow.pts[1][1] * 100}%` }}
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={() => { onPatch({ motion: panel.motion.filter((m) => m.id !== selArrow.id) }); setArrow(null) }}>
              ×
            </button>
          </>
        )}
      </div>

      <div className="sbtags">
        {tag('framing', framing, framing ? labelOf(vocab, framing.key) : 'framing')}
        {tag('angle', angle, angle ? labelOf(vocab, angle.key) : 'angle')}
        {tag('camera', camera, camera ? cameraTag(vocab, camera) : 'camera')}
      </div>

      <div className="sbmeta">
        <textarea className="sbprose" rows={1} value={panel.prose}
                  placeholder="What happens"
                  ref={grow}
                  onInput={(e) => grow(e.currentTarget)}
                  onChange={(e) => onPatch({ prose: e.target.value })} />
        {said && <div className="sbsaid" title="What the arrows will say, after your words">{said}</div>}
        <textarea className="sbnote" rows={1} value={panel.note}
                  placeholder="A note to yourself"
                  ref={grow}
                  onInput={(e) => grow(e.currentTarget)}
                  onChange={(e) => onPatch({ note: e.target.value })} />
      </div>

      {pick && group && (
        <PillPicker anchor={pick.anchor} group={group}
                    pill={pillIn(panel.pills, pick.group)}
                    onPick={(p) => onPatch({ pills: withPill(latest.current.pills, pick.group, p) })}
                    onClose={() => setPick(null)} />
      )}
    </div>
  )
}

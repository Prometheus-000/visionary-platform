import { useCallback, useEffect, useRef, useState } from 'react'

import { coverUrl, fileUrl, gallery } from '../api/routes'
import type { GalleryItem } from '../gallery/types'
import { intake } from '../scene/pool'
import { handleOf } from '../scene/model'
import { useStore } from '../store'
import { drawSheet, SHEET_H, SHEET_SLOTS, SHEET_W, type SlotId } from './render'

/**
 * The sheet surface — its own room, entered by a door.
 *
 * **The canvas is the export.** What is drawn here is `drawSheet` at full
 * resolution, which is byte-for-byte what Download saves — a preview with its
 * own arrangement is a preview that can disagree with the file. The six wells
 * under it are the slots: drop a file from the Finder, drag a generation up
 * from the strip, or click for a picker. Empty slots live in the wells and
 * never on the sheet, because the export reflows to filled panels only and a
 * preview showing placeholder rectangles would be showing a different PNG.
 *
 * **The strip is your own work.** The point of a sheet is one `<Picture N>`
 * slot carrying several views of somebody this platform already generated, so
 * the recent image generations are one drag away rather than a round trip
 * through the Finder.
 */

type Held = { url: string; img: HTMLImageElement; own: boolean }

const load = (url: string) =>
  new Promise<HTMLImageElement>((res, rej) => {
    const img = new Image()
    img.onload = () => res(img)
    img.onerror = rej
    img.src = url
  })

export function SheetBuilder() {
  const canvas = useRef<HTMLCanvasElement>(null)
  const input = useRef<HTMLInputElement>(null)
  const pickFor = useRef<SlotId>('main')
  const [name, setName] = useState('')
  const [role, setRole] = useState('')
  const [note, setNote] = useState('')
  const [held, setHeld] = useState<Partial<Record<SlotId, Held>>>({})
  const [hot, setHot] = useState<SlotId | null>(null)
  const [recent, setRecent] = useState<GalleryItem[]>([])

  // The recent generations, images only — a clip has no still to put in a
  // panel. Fetched once on entry; the surface is not a gallery and does not
  // page.
  useEffect(() => {
    void (async () => {
      const r = await gallery()
      if (Array.isArray((r as { items?: unknown }).items)) {
        setRecent((r as { items: GalleryItem[] }).items
          .filter((it) => it.kind === 'image' && it.files.length)
          .slice(0, 30))
      }
    })()
  }, [])

  // Redraw on every change. The images are already decoded — `Held` carries
  // the element — so a full-resolution redraw is milliseconds and the preview
  // never shows a state the export would not.
  useEffect(() => {
    const el = canvas.current
    const ctx = el?.getContext('2d')
    if (!el || !ctx) return
    const images: Partial<Record<SlotId, HTMLImageElement>> = {}
    for (const [k, v] of Object.entries(held)) images[k as SlotId] = v.img
    drawSheet(ctx, { name, role, note, images })
  }, [name, role, note, held])

  const place = useCallback(async (slot: SlotId, url: string, own: boolean) => {
    try {
      const img = await load(url)
      setHeld((h) => {
        const old = h[slot]
        if (old?.own) URL.revokeObjectURL(old.url)
        return { ...h, [slot]: { url, img, own } }
      })
    } catch {
      alert('That image could not be decoded — a PNG or JPEG works.')
      if (own) URL.revokeObjectURL(url)
    }
  }, [])

  const takeFile = useCallback((slot: SlotId, f: File) => {
    if (!f.type.startsWith('image/')) return
    void place(slot, URL.createObjectURL(f), true)
  }, [place])

  /** A dropped URL, fetched into a blob before it touches the canvas.
   *
   *  Two reasons, both real-drag lessons. Fetching means the canvas draws an
   *  object URL and can never be *tainted* — a cross-origin image drawn
   *  directly would poison `toBlob` and Download would fail with a
   *  SecurityError at the end of the person's work rather than at the drop.
   *  And it makes any same-origin route work identically, cover or file. */
  const placeUrl = useCallback(async (slot: SlotId, url: string) => {
    try {
      const r = await fetch(url)
      if (!r.ok) throw new Error(String(r.status))
      const b = await r.blob()
      if (!b.type.startsWith('image/')) throw new Error(b.type)
      void place(slot, URL.createObjectURL(b), true)
    } catch {
      alert('That image could not be fetched — drag it from the strip below, '
            + 'or save it as a file and drop that.')
    }
  }, [place])

  const clear = (slot: SlotId) => {
    setHeld((h) => {
      const old = h[slot]
      if (old?.own) URL.revokeObjectURL(old.url)
      const rest = { ...h }
      delete rest[slot]
      return rest
    })
  }

  const download = () => {
    canvas.current?.toBlob((b) => {
      if (!b) return
      const a = document.createElement('a')
      a.href = URL.createObjectURL(b)
      a.download = `${(name.trim() || 'character').toLowerCase().replace(/[^a-z0-9]+/g, '-')}-sheet.png`
      a.click()
      // On a timer rather than immediately: revoking before the click has
      // been serviced cancels the download in Safari.
      setTimeout(() => URL.revokeObjectURL(a.href), 4000)
    }, 'image/png')
  }

  const filled = SHEET_SLOTS.filter((sl) => held[sl.id]).length

  /**
   * The sheet, into the cast — and this button is the whole announcement that
   * the two surfaces connect. A sentence explaining "you can use this on the
   * video side" is a label; the fix the vetoes ask for is a better gesture, and
   * the better gesture is the output physically travelling to where it works.
   *
   * Creates `@name` if the cast does not have them, attaches the sheet as their
   * reference, seeds the description from the do-not-change note where there is
   * not one, and lands you in front of the rail with their card open — the
   * "she appears" moment, which is the confirmation, so the button needs none.
   */
  const cast = async () => {
    const el = canvas.current
    const handle = handleOf(name)
    if (!el || !handle) return
    const blob = await new Promise<Blob | null>((r) => el.toBlob(r, 'image/png'))
    if (!blob) return
    const got = await intake(new File([blob], `${handle}-sheet.png`, { type: 'image/png' }))
    if (!got) return
    const st = useStore.getState()
    const member = st.scene.cast.find((c) => handleOf(c.name) === handle)
      ?? st.addCast('subject', name)
    st.addFile(got)
    st.attachSlot(member.id, got.id, 'image')
    // Typed, not noted: the app made this sheet, so provenance is certain and
    // the mark costs zero clicks — the compiler writes the citation from here.
    st.patchRef(member.id, got.id, { sheet: true })
    if (!member.note && note.trim()) st.patchCast(member.id, { note: note.trim() })
    st.setKind('video')
    st.setMode('generate')
    st.setRailOpen(member.id)
  }

  return (
    <div className="sheet-stage" id="sheet-stage">
      <div className="sheet-head">
        {/* The header fields are the sheet's header — name, role, the
            visual-ID note the skill asks for. Typed here, drawn there; the
            canvas repaints on each keystroke so the two are never apart. */}
        <input autoComplete="off" className="sh-name" value={name} spellCheck={false}
               placeholder="character name"
               onChange={(e) => setName(e.target.value)} />
        <input autoComplete="off" className="sh-role" value={role} spellCheck={false}
               placeholder="role — protagonist, thief, grandma…"
               onChange={(e) => setRole(e.target.value)} />
        <input autoComplete="off" className="sh-note" value={note} spellCheck={false}
               placeholder="do-not-change traits — age, build, hair, outfit colours, signature props…"
               onChange={(e) => setNote(e.target.value)} />
        <span className="grow" />
        <button type="button" className="b" id="sheet-cast"
                disabled={!filled || !handleOf(name)}
                title={handleOf(name)
                  ? `Cast @${handleOf(name)} — the sheet becomes their reference on the video side`
                  : 'Name the character first'}
                onClick={() => void cast()}>
          {handleOf(name) ? `Cast @${handleOf(name)}` : 'Cast'}
        </button>
        <button type="button" className="b quiet" id="sheet-download"
                disabled={!filled} onClick={download}>
          Download PNG
        </button>
      </div>

      <div className="sheet-view">
        <canvas ref={canvas} width={SHEET_W} height={SHEET_H} id="sheet-canvas" />
      </div>

      {/* The wells are the slots. On the row rather than on the canvas,
          because the canvas shows only what the PNG will hold — an empty slot
          is an invitation, and invitations do not export. */}
      <div className="sheet-wells">
        {SHEET_SLOTS.map((sl) => {
          const h = held[sl.id]
          return (
            <button key={sl.id} type="button"
                    className={`swell${h ? ' set' : ''}${hot === sl.id ? ' hot' : ''}`}
                    title={h ? `${sl.label} — click to clear` : `${sl.label} — drop an image, or click`}
                    onClick={() => {
                      if (h) { clear(sl.id); return }
                      pickFor.current = sl.id
                      input.current?.click()
                    }}
                    onDragOver={(e) => { e.preventDefault(); setHot(sl.id) }}
                    onDragLeave={(e) => {
                      if (!e.currentTarget.contains(e.relatedTarget as Node)) setHot(null)
                    }}
                    onDrop={(e) => {
                      e.preventDefault()
                      setHot(null)
                      const f = [...e.dataTransfer.files].find((x) => x.type.startsWith('image/'))
                      if (f) { takeFile(sl.id, f); return }
                      // A generation dragged up from the strip travels as a
                      // URL. uri-list first, text/plain as the fallback — and
                      // parsed as the format it is: lines, `#` comments legal,
                      // CRLF terminators. The first real line is the picture.
                      const raw = e.dataTransfer.getData('text/uri-list')
                        || e.dataTransfer.getData('text/plain')
                      const url = raw.split(/\r?\n/).find((l) => l.trim() && !l.startsWith('#'))
                      if (url) void placeUrl(sl.id, url.trim())
                    }}>
              {h ? <img src={h.url} alt="" draggable={false} /> : <i>＋</i>}
              <b>{sl.label}</b>
            </button>
          )
        })}
        <input ref={input} type="file" accept="image/*" className="hide"
               onChange={(e) => {
                 const f = e.target.files?.[0]
                 e.target.value = ''
                 if (f) takeFile(pickFor.current, f)
               }} />
      </div>

      {recent.length > 0 && (
        <div className="sheet-strip" id="sheet-strip">
          {recent.map((it) => (
            <img key={it.job_id + it.files[0]!}
                 src={coverUrl(it.job_id, it.files[0]!)} alt="" draggable
                 title="Drag onto a slot — or click for the next empty one"
                 /* The coarse-input path. A drag from strip to well is the
                    precise gesture and stays primary; a click was a dead press,
                    and on glass — where this strip is the only source of your
                    own work — dead was the whole story. It fills the first
                    empty well because that is the order the sheet reads in;
                    aiming at a *particular* well is what the drag is for. */
                 onClick={() => {
                   const slot = SHEET_SLOTS.find((sl) => !held[sl.id])?.id
                   if (slot) void placeUrl(slot, fileUrl(it.job_id, it.files[0]!))
                 }}
                 onDragStart={(e) => {
                   // **Absolute, or it does not survive the drag.** A native
                   // drag round-trips through the OS pasteboard, and Chromium
                   // stores uri-list entries as parsed URLs — a relative path
                   // fails the parse and the entry is silently discarded, and
                   // because setData had replaced the browser's own default
                   // entry, the drop arrived carrying nothing but text/html.
                   // Found by instrumenting 24 of the owner's real drags after
                   // every synthetic dispatch had passed: a synthetic drop
                   // hands over the same live DataTransfer and never crosses
                   // the pasteboard, so it cannot see this. text/plain rides
                   // along for anything that strips uri-list entirely.
                   const abs = new URL(fileUrl(it.job_id, it.files[0]!), window.location.href).href
                   e.dataTransfer.setData('text/uri-list', abs)
                   e.dataTransfer.setData('text/plain', abs)
                   e.dataTransfer.effectAllowed = 'copy'
                 }} />
          ))}
        </div>
      )}
    </div>
  )
}

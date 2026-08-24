import { useRef, useState } from 'react'

import { useStore } from '../store'
import { intake, mediaOf } from './pool'
import { slotFor, type CastMember, type PoolFile } from './model'

/**
 * What you have of a subject, and what each piece of it is for.
 *
 * **One row per reference, and one well that always takes the next file.** This
 * was a single well holding one photograph, which was the right correction to
 * five labelled squares and the wrong stopping point: the guide is explicit that
 * *"One subject may be defined by multiple reference assets, and one reference
 * asset may provide multiple subjects"*, and it writes the relationship as prose
 * rather than as a schema —
 *
 *     <Subject 1> is the woman whose appearance comes from <Picture 1> and whose
 *     walking motion comes from <Video 1>.
 *
 * So a second picture is not a second slot to invent a name for. It is another
 * row, with a field for the one thing only you know: what *this* one is for.
 *
 * **The two notes are different claims and the placeholders have to say so.**
 * `member.note` is what the subject *is* — the description the guide puts right
 * after the label. A `ref.note` is what one asset *provides*, and it is empty in
 * the ordinary case, because a single photograph of somebody needs no gloss.
 *
 * **The file decides the channel, and a file the subject cannot take never
 * lights the well.** Read off `dataTransfer.items` during the drag, which is the
 * one moment the answer is available and still useful — the rejection is the
 * absence of an invitation, and it is the only thing worth keeping from the five
 * squares this replaced.
 */
export function Material({ member }: { member: CastMember }) {
  const s = useStore()
  const input = useRef<HTMLInputElement>(null)
  const [hot, setHot] = useState(false)

  const held = member.refs
    .map((r) => ({ ref: r, file: s.pool[r.fileId] as PoolFile | undefined }))
    .filter((x) => x.file)

  const take = async (f: File) => {
    const got = await intake(f)
    if (!got) {
      // Named, and with a way out — the browser is what failed, so the fix is a
      // format it decodes. Same courtesy `SourceRow` extends.
      alert(`The browser could not read ${f.name || 'that file'} — try a PNG, a`
            + ' JPEG, an MP4 or a WAV.')
      return
    }
    const slot = slotFor(member.kind, got.kind)
    if (!slot) return
    s.addFile(got)
    s.attachSlot(member.id, got.id, slot)
  }

  const welcome = (items: DataTransferItem[]) =>
    items.some((it) => {
      if (it.kind !== 'file') return false
      const m = mediaOf(it.type)
      return !!m && !!slotFor(member.kind, m)
    })

  return (
    <div className="tmat">
      {held.map(({ ref, file }, i) => (
        <div key={ref.fileId} className="tref">
          {file!.kind === 'image'
            ? <img src={file!.url} alt="" draggable={false} />
            // A waveform at 34px is a grey rectangle and a video frame is a
            // guess, so the channel is the thumbnail for both. The filename
            // lives in `title` — it is what you check, not what you read.
            : <span className="tkind" title={file!.name}>
                {file!.kind === 'audio' ? '♪' : '▸'}
              </span>}
          <input className="trefnote" value={ref.note ?? ''} spellCheck={false}
                 // The first one needs no gloss — the subject's own description
                 // already covers it — so only the second row onward asks.
                 placeholder={i === 0
                   ? (file!.kind === 'audio' ? 'how they sound' : 'what this shows')
                   : 'what this one is for — the coat, her posture, the light'}
                 onChange={(e) => { s.patchRef(member.id, ref.fileId, { note: e.target.value }) }} />
          <button type="button" className="x" title={`Remove ${file!.name}`}
                  onClick={() => { s.detachRef(member.id, ref.fileId) }}>×</button>
        </div>
      ))}

      <button type="button"
              className={`twell${hot ? ' hot' : ''}${held.length ? ' more' : ''}`}
              title={held.length
                ? 'Another photograph, or a recording of their voice'
                : 'Drop a photograph, or a recording of their voice'}
              onClick={(e) => { if (e.target !== input.current) input.current?.click() }}
              onDragOver={(e) => {
                if (!welcome([...(e.dataTransfer.items ?? [])])) return
                e.preventDefault()
                setHot(true)
              }}
              // Guarded on relatedTarget: without it every child under the cursor
              // fires its own dragleave and the highlight strobes.
              onDragLeave={(e) => {
                if (!e.currentTarget.contains(e.relatedTarget as Node)) setHot(false)
              }}
              onDrop={(e) => {
                e.preventDefault()
                setHot(false)
                const f = [...e.dataTransfer.files]
                  .find((x) => { const m = mediaOf(x.type); return !!m && !!slotFor(member.kind, m) })
                if (f) void take(f)
              }}>
        <span className="tplus">＋</span>
        {!held.length && <b>a photo, or a voice</b>}
        <input ref={input} type="file" accept="image/*,audio/*,video/*" className="hide"
               onChange={(e) => {
                 const f = e.target.files?.[0]
                 e.target.value = ''
                 if (f) void take(f)
               }} />
      </button>
    </div>
  )
}

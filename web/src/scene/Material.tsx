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
 * after the label. A `ref.note` is what one asset *provides*, and empty is the
 * ordinary case: an unnoted picture is referenced whole. The placeholder still
 * teaches on every row — *what to reference — face, clothing, style…* — because
 * a photograph lending part of itself is a capability nothing else on the page
 * can announce, and hiding it on the first row hid it where everyone starts.
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
      {held.map(({ ref, file }) => (
        <div key={ref.fileId} className="tref">
          {file!.kind === 'image'
            ? <img src={file!.url} alt="" draggable={false} />
            // A waveform at 34px is a grey rectangle and a video frame is a
            // guess, so the channel is the thumbnail for both. The filename
            // lives in `title` — it is what you check, not what you read.
            : <span className="tkind" title={file!.name}>
                {file!.kind === 'audio' ? '♪' : '▸'}
              </span>}
          {/* **The mark is one click, and the marked row shows the derived
              sentence instead of asking for one.** A sheet's citation is
              templated — the compiler writes it — so offering the note field
              on a marked row would be asking for words the run will not read.
              Grey and uneditable is the honest state: derived, always
              visible. Everything intent-shaped stays typed; everything the
              format dictates is written for you. */}
          {file!.kind === 'image' && (
            <button type="button"
                    className={`tsheet${ref.sheet ? ' on' : ''}`}
                    title={ref.sheet
                      ? 'Marked as their character sheet — the citation is written for you. Click to unmark.'
                      : 'Mark as their character sheet — the citation gets written for you.'}
                    onClick={() => { s.patchRef(member.id, ref.fileId, { sheet: !ref.sheet }) }}>
              sheet
            </button>
          )}
          {ref.sheet
            ? <span className="trefnote derived"
                    title="The compiler cites the sheet — its views and written notes define their appearance.">
                views &amp; notes define them — written for you
              </span>
            : <input className="trefnote" value={ref.note ?? ''} spellCheck={false}
                 // One line, and it is teaching rather than labelling: a
                 // photograph can lend *part* of itself — the face, the coat,
                 // the light — and nothing else on the page can say so. The
                 // first row asked nothing for a version, on the argument that
                 // one photo needs no gloss; true of the common case and it
                 // hid the capability exactly where everyone starts. Empty is
                 // still fine — an unnoted picture is referenced whole.
                 placeholder={file!.kind === 'audio'
                   ? 'the voice — tone, pace, accent…'
                   : file!.kind === 'video'
                     ? 'the motion — how they move…'
                     : 'what to reference — face, clothing, style…'}
                 onChange={(e) => { s.patchRef(member.id, ref.fileId, { note: e.target.value }) }} />}
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

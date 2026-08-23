import { useRef, useState } from 'react'

import { useStore } from '../store'
import { intake, mediaOf } from './pool'
import type { CastMember, Media } from './model'

/**
 * One role on one cast member, and the drop target for it.
 *
 * **The slot is the role.** Dropping a photograph on Face *is* the tagging —
 * there is no second gesture and no menu, which is the same reason a region's
 * LoRA is a dropdown on the box rather than a token typed at a caret.
 *
 * **A file over a slot that cannot take it does not highlight.** That is the
 * whole validation, and it is deliberately not a toast: the rejection is the
 * absence of an invitation, which arrives before the drop rather than after it.
 * `DropTile` alerts instead, and it is right to — its tiles all take images, so
 * the wrong file there is a mistake worth naming. Here the row *is* a table of
 * what takes what, so a slot that stays dark has already said it.
 *
 * The kind is read off `dataTransfer.items` during the drag, which carries the
 * MIME type and not the bytes. That is the only moment the answer is available
 * and still useful.
 */
export function Slot({ member, slot, label, takes }: {
  member: CastMember
  slot: string
  label: string
  takes: Media
}) {
  const s = useStore()
  const input = useRef<HTMLInputElement>(null)
  const [hot, setHot] = useState(false)

  const ref = member.refs.find((r) => r.slots.includes(slot))
  const file = ref ? s.pool[ref.fileId] : undefined

  const take = async (f: File) => {
    const got = await intake(f)
    if (!got) {
      // Named, and with a way out — the browser is the thing that failed, so the
      // fix is a format it decodes. Same courtesy `SourceRow` extends.
      alert(`The browser could not read ${f.name || 'that file'} — try a PNG, a`
            + ' JPEG, an MP4 or a WAV.')
      return
    }
    if (got.kind !== takes) return
    s.addFile(got)
    s.attachSlot(member.id, got.id, slot)
  }

  return (
    <button type="button"
            className={['tslot', file ? 'set' : '', hot ? 'hot' : ''].filter(Boolean).join(' ')}
            title={file ? `${label} — ${file.name}. Click to clear.` : `${label} — drop a file`}
            data-slot={slot}
            onClick={(e) => {
              if (e.target === input.current) return
              if (file) s.detachSlot(member.id, slot)
              else input.current?.click()
            }}
            onDragOver={(e) => {
              const items = [...(e.dataTransfer.items ?? [])]
              if (!items.some((it) => it.kind === 'file' && mediaOf(it.type) === takes)) return
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
              const f = [...e.dataTransfer.files].find((x) => mediaOf(x.type) === takes)
              if (f) void take(f)
            }}>
      {file
        ? (file.kind === 'image'
            ? <img src={file.url} alt="" />
            // A clip and a sound file have no thumbnail worth 64px, so they say
            // their role instead — which is the one fact the picture would have
            // been standing in for anyway.
            : <span className="lb full">{label}</span>)
        : <span className="lb">{label}</span>}
      <input ref={input} type="file" accept={`${takes}/*`} className="hide"
             onChange={(e) => {
               const f = e.target.files?.[0]
               e.target.value = ''
               if (f) void take(f)
             }} />
    </button>
  )
}

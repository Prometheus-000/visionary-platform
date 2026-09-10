import { useRef, useState } from 'react'

import { dataUrl } from './files'

/**
 * A 32px tile that takes one picture: a keyframe, a plate, or a region's photo.
 *
 * **`can-drop` is set here rather than written into the markup**, so the class that
 * makes a target glow is applied by the same component that gives it a handler.
 * That is the fault this was written for: the reference tray never had a drop
 * handler at all — it was a click-only picker sitting in a row of tiles that all
 * take drops, and the drag-intent reveal then lit it like the rest. A control that
 * lights up under a dragged file and refuses it is worse than one that stays dark:
 * the dark one is undiscovered, the lit one is a promise the page breaks while you
 * watch.
 *
 * A second click on a filled tile clears it. There is no ✕ because the tile is 32px
 * and a hit target inside it would be smaller than a fingertip. Keyframes could be
 * *replaced* but never removed for a while, so the only route back to
 * text-to-video was switching model and back — harmless while the two halves were
 * unrelated rows, and a corner you cannot get out of now that a keyframe puts the
 * reference tray out of play.
 *
 * `locked` is visible and hoverable so it can say why, which means it is also
 * clickable and droppable unless every entry point checks — a locked tile that
 * opened a file picker and then swallowed the picture would be worse than the
 * hidden tile that preceded it.
 */
export function DropTile({
  label,
  title,
  value,
  glyph,
  onFile,
  onClear,
  /** Out of play this run — dimmed, not hidden. The row's job is to show that
   *  keyframes and references are alternatives, and a control that vanishes when
   *  you fill its neighbour teaches nothing except that the page lost it. */
  off,
  locked,
  accept = 'image/',
  id,
}: {
  label: string
  title: string
  value: string | null
  glyph: React.ReactNode
  onFile: (f: File) => void
  onClear: () => void
  off?: boolean
  locked?: boolean
  accept?: string
  id?: string
}) {
  const input = useRef<HTMLInputElement>(null)
  const [hot, setHot] = useState(false)
  const dead = !!locked || !!off

  return (
    <button id={id} type="button" data-lb={label} title={title}
            className={['drop', 'mini', 'can-drop', value ? 'set' : '', hot ? 'hot' : '',
                        off ? 'off' : '', locked ? 'locked' : ''].filter(Boolean).join(' ')}
            data-drop={label}
            onClick={(e) => {
              if (dead || e.target === input.current) return
              if (!value) input.current?.click()
              else onClear()
            }}
            onDragOver={(e) => {
              if (dead) return
              // Files only, and only the kind this target takes. Without the type
              // check a video dragged onto the picture tray lights it and the drop
              // then silently does nothing — the same broken promise one level down.
              if (![...(e.dataTransfer?.types ?? [])].includes('Files')) return
              e.preventDefault()
              setHot(true)
            }}
            // Guarded on relatedTarget: without it every child under the cursor
            // fires its own dragleave and the highlight strobes.
            onDragLeave={(e) => {
              if (!e.currentTarget.contains(e.relatedTarget as Node)) setHot(false)
            }}
            onDrop={(e) => {
              if (dead) return
              e.preventDefault()
              setHot(false)
              const f = [...(e.dataTransfer?.files ?? [])].find((x) => x.type.startsWith(accept))
              // Said, not swallowed. A file of the wrong kind landing on a tile that
              // just lit up for it has to say why nothing happened.
              if (!f) {
                alert(`That tile takes ${accept === 'image/' ? 'an image' : 'a video'}.`)
                return
              }
              onFile(f)
            }}>
      {/* The lead label comes from `data-lb` via CSS in the vanilla page; here it
          is a real element for the same reason every other one is — a label
          injected only at startup is a label every tile added later goes without. */}
      <span className="lead">{label}</span>
      {value ? <img src={dataUrl(value)} alt="" /> : <span>{glyph}</span>}
      <input ref={input} type="file" accept={`${accept}*`} className="hide"
             onChange={(e) => {
               const f = e.target.files?.[0]
               e.target.value = ''
               if (f) onFile(f)
             }} />
    </button>
  )
}

import { useRef, useState } from 'react'

import { IconFilm, IconFirst, IconLast, IconPhoto } from '../icons'
import { Menu } from '../ui/Menu'
import { usePopover } from '../ui/Popover'
import { DropTile } from '../media/DropTile'
import { dataUrl, shrinkB64, toB64 } from '../media/files'
import { supports, useStore } from '../store'

/**
 * Every picture the model can be given, in one row — and the two halves dim each
 * other.
 *
 * Keyframes and references are the same decision made two ways. They load different
 * transformers, so one excludes the other, and they used to be two rows: keyframes
 * parked at the right of the strip among the numeric controls, references in their own
 * row below. Two pairs of unlabelled 36px dashed tiles, one row apart, telling each
 * other apart by tooltip.
 *
 * What that cost was not tidiness. The keyframe tiles were never found at all, and
 * dropping photos into the reference tray *looked* like filling keyframe slots that
 * kept growing — which is exactly what the tray does and exactly what a fixed pair
 * must never look like. Side by side with a rule between them, the tray that grows and
 * the two slots that do not are told apart by shape, which is the thing a tooltip
 * could not do.
 *
 * Whichever half is out of play goes dim rather than disappearing: the row's job is to
 * show that these are alternatives, and a control that vanishes when you fill its
 * neighbour teaches nothing except that the page lost it. **References win when both
 * are attached**, because that is what the run does, so they are the half that stays
 * live — dimming both, which the symmetric version did, left the gallery's "As
 * reference" hand-off in a state where a keyframe could not be cleared and a second
 * reference could not be added: the two controls locked each other out.
 */
export function SourceRow() {
  const s = useStore()
  const sup = supports(s)
  const role = usePopover()
  const roleFor = useRef(0)
  const [addRef, setAddRef] = useState<'img' | 'vid' | null>(null)

  const n = s.refs.length + s.refVids.length
  const framed = !!(s.keyframe.first || s.keyframe.last)
  const maxRefs = s.state?.max_refs ?? 9
  const maxVids = s.state?.max_ref_videos ?? 3

  const take = async (kindOf: 'img' | 'vid', files: File[]) => {
    // Re-read the bucket and the cap at drop time rather than closing over them:
    // both arrays are replaced wholesale by Reuse and by a model change dropping
    // references it cannot take, so a captured array would be pushed into a
    // detached one.
    const st = useStore.getState()
    const isImg = kindOf === 'img'
    const bucket = isImg ? st.refs : st.refVids
    const max = isImg ? maxRefs : maxVids
    if (bucket.length >= max) {
      alert(`${max} ${isImg ? 'image' : 'video'} references is the model's limit.`)
      return
    }
    const out = [...bucket]
    for (const f of files.slice(0, max - bucket.length)) {
      // Images go through the same shrink the region photos do, for the payload half
      // of the reason H3_REF_MAX_SIDE gives: nine photographs straight off a phone is
      // tens of megabytes of base64 in one JSON body. The server caps them again on
      // arrival and that is the copy that binds — this is the one that keeps the
      // request from being the slowest part of pressing Generate. Videos are sent
      // whole: there is nothing here that can re-encode one.
      const b = await (isImg ? shrinkB64(f) : toB64(f))
      // `null` is a file the canvas could not decode. Pushed anyway it would be a chip
      // with no picture and a base64 of "null" in the request, which fails on the GPU
      // rather than on the file it came from.
      if (b) out.push(b)
      else alert('Could not read that file.')
    }
    if (isImg) st.setRefs(out)
    else st.setRefVids(out)
  }

  return (
    <div className="opts" id="v-src-sec">
      <DropTile id="v-drop-first" label="First frame" value={s.keyframe.first} off={!!n}
                glyph={<IconFirst />}
                title="The clip starts on this image. Drop or click; click again to clear."
                onFile={async (f) => s.setKeyframe('first', await toB64(f))}
                onClear={() => s.setKeyframe('first', null)} />
      {sup.last_frame && (
        <DropTile id="v-drop-last" label="Last frame" value={s.keyframe.last} off={!!n}
                  glyph={<IconLast />}
                  title="The clip ends on this image. Drop or click; click again to clear."
                  onFile={async (f) => s.setKeyframe('last', await toB64(f))}
                  onClear={() => s.setKeyframe('last', null)} />
      )}

      <span className="wrap" id="v-refs">
        {s.refs.map((b, i) => {
          const spec = s.state?.shot_roles?.find((x) => x.key === s.refRoles[i])
          return (
            <div className="ref" key={`p${i}`}>
              <img src={dataUrl(b)} alt="" />
              <b>P{i + 1}</b>
              <button className="x" title="Remove" type="button"
                      onClick={() => {
                        // The roles are positional — index i is <Picture i+1> — so
                        // removing the second chip has to remove the second role with
                        // it. Left alone, every role after the gap would have silently
                        // moved onto the wrong picture.
                        s.setRefs(s.refs.filter((_, k) => k !== i),
                                  s.refRoles.filter((_, k) => k !== i))
                      }}>×</button>
              {/* What this picture defines. The role compiles into the prompt's
                  `subject_definitions` — "<Subject 1> is the person in <Picture 1>" —
                  which is the whole answer to "you should not describe the picture you
                  attached, but there is nowhere else to put it so everyone does". Now
                  there is somewhere, and it is a menu rather than a sentence. */}
              <button className={`role${spec ? '' : ' none'}`} type="button"
                      title="What this picture defines. It goes into the prompt as a subject, so you never have to describe the photograph itself."
                      onClick={(e) => { roleFor.current = i; role.toggle(e) }}>
                {spec ? spec.label : 'role'}
              </button>
            </div>
          )
        })}
        {/* Same media fragment as the gallery card, and for the same reason: a
            reference tile with no frame painted is an unlabelled black square, which
            defeats the point of showing the reference you attached. No role bar — the
            compiler builds subjects out of <Picture n> only, and a menu that set
            something nothing reads is worse than no menu. */}
        {s.refVids.map((b, i) => (
          <div className="ref" key={`v${i}`}>
            <video src={`${dataUrl(b, 'video/mp4')}#t=0.04`} muted />
            <b>V{i + 1}</b>
            <button className="x" title="Remove" type="button"
                    onClick={() => s.setRefVids(s.refVids.filter((_, k) => k !== i))}>×</button>
          </div>
        ))}
      </span>

      {/* Named for the token they produce, not for what they take: what you attach
          here is what the prompt then calls <Picture 1> / <Video 1>, and the chips are
          already lettered P1/V1 to match. */}
      {sup.references && (
        <>
          <AddTile id="v-add-ref" label="Picture" accept="image/" off={framed && !n}
                   glyph={<IconPhoto />}
                   title="Add an image reference — the subject, redrawn in a new shot. The prompt refers to it as <Picture 1>."
                   onFiles={(f) => void take('img', f)}
                   picking={addRef === 'img'} onPick={() => setAddRef('img')}
                   onDone={() => setAddRef(null)} />
          <AddTile id="v-add-vid" label="Video" accept="video/" off={framed && !n}
                   glyph={<IconFilm />}
                   title="Add a video reference. The prompt refers to it as <Video 1>."
                   onFiles={(f) => void take('vid', f)}
                   picking={addRef === 'vid'} onPick={() => setAddRef('vid')}
                   onDone={() => setAddRef(null)} />
          {/* Reference tokens ride through every sampling step, so this is a per-step
              price. Both options are bounded — see H3_REF_MAX_SIDE — which is the only
              reason "max detail" is offered at all. */}
          <div className="opt" id="v-ref-size-wrap">
            <select value={s.vid.refSize}
                    title='How much of each reference the model reads. "match canvas" scales every picture to the clip&apos;s own pixel area; "max detail" hands over as much of it as the run allows — 1536px on the long side — and buys likeness at several times the sampling time.'
                    onChange={(e) => s.setVid({ refSize: e.target.value as 'match' | 'max' })}>
              <option value="match">match canvas</option>
              <option value="max">max detail</option>
            </select>
          </div>
        </>
      )}

      {role.open && (
        <Menu anchor={role.anchor} onClose={role.close} items={[
          { label: 'No role', on: !s.refRoles[roleFor.current],
            run: () => s.setRefRoles(withAt(s.refRoles, roleFor.current, '')) },
          ...(s.state?.shot_roles ?? []).map((r) => ({
            label: r.label,
            on: s.refRoles[roleFor.current] === r.key,
            run: () => s.setRefRoles(withAt(s.refRoles, roleFor.current, r.key)),
          })),
        ]} />
      )}
    </div>
  )
}

const withAt = (list: string[], i: number, v: string) => {
  const out = [...list]
  while (out.length <= i) out.push('')
  out[i] = v
  return out
}

/** The tray's two add buttons: a tile with no picture in it, that takes several files
 *  at once. Distinct from `DropTile` because it never holds a value — filling it
 *  appends a chip beside it, which is exactly the difference in shape the row exists
 *  to show. */
function AddTile({ id, label, title, accept, glyph, off, onFiles, picking, onPick, onDone }: {
  /** Kept because `tools/ui-checks/check_drop.py` addresses every target by id, and a
   *  target it cannot find is a target nobody is checking accepts a drop. */
  id: string
  label: string
  title: string
  accept: string
  glyph: React.ReactNode
  off: boolean
  onFiles: (files: File[]) => void
  picking: boolean
  onPick: () => void
  onDone: () => void
}) {
  const input = useRef<HTMLInputElement>(null)
  const [hot, setHot] = useState(false)
  return (
    <button id={id} type="button" data-lb={label} title={title} data-drop={`${label} reference`}
            className={['drop', 'mini', 'can-drop', hot ? 'hot' : '', off ? 'off' : '']
              .filter(Boolean).join(' ')}
            onClick={(e) => {
              if (off || e.target === input.current) return
              onPick()
              input.current?.click()
            }}
            onDragOver={(e) => {
              if (off) return
              if (![...(e.dataTransfer?.types ?? [])].includes('Files')) return
              e.preventDefault()
              setHot(true)
            }}
            onDragLeave={(e) => {
              if (!e.currentTarget.contains(e.relatedTarget as Node)) setHot(false)
            }}
            onDrop={(e) => {
              if (off) return
              e.preventDefault()
              setHot(false)
              const files = [...(e.dataTransfer?.files ?? [])].filter((f) => f.type.startsWith(accept))
              if (!files.length) {
                alert(`That tile takes ${accept === 'image/' ? 'an image' : 'a video'}.`)
                return
              }
              onFiles(files)
            }}>
      <span className="lead">{label}</span>
      <span>{glyph}</span>
      <input ref={input} type="file" accept={`${accept}*`} multiple className="hide"
             onChange={(e) => {
               const files = [...(e.target.files ?? [])]
               e.target.value = ''
               if (picking && files.length) onFiles(files)
               onDone()
             }} />
    </button>
  )
}

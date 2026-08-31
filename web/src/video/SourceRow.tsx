import { useRef, useState } from 'react'

import { IconFilm, IconFirst, IconLast, IconPhoto } from '../icons'
import { Menu } from '../ui/Menu'
import { usePopover } from '../ui/Popover'
import { DropTile } from '../media/DropTile'
import { dataUrl, shrinkB64, toB64 } from '../media/files'
import { live, type PoolFile, type SourceKind } from '../scene/model'
import { intake } from '../scene/pool'
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
  // **The composer changes what this row's tiles mean, and the flat trays go
  // dark the moment it is live** — `videoBody` sends the cast's files then,
  // never the trays, so a chip added here would render, count for nothing and
  // upload nothing: a control that looks live and is quietly ignored, which is
  // the exact fault the dimming rule below exists to prevent. What each tile
  // becomes instead is per-tile — see each one.
  const composed = live(s.scene)
  const srcRole = usePopover()
  // The clip-level source video, if one is set — continue and edit are one
  // slot with two meanings, mutually exclusive in the validator, so the chip
  // is one chip with a role button rather than two tiles.
  const srcKind: SourceKind | null = s.scene.sources.continue?.length
    ? 'continue' : s.scene.sources.edit?.length ? 'edit' : null
  const srcFile: PoolFile | undefined = srcKind
    ? s.pool[s.scene.sources[srcKind]![0]!] : undefined

  const takeSource = async (files: File[]) => {
    const got = await intake(files[0]!)
    if (!got) {
      alert(`The browser could not read ${files[0]?.name || 'that video'} — an`
            + ' MP4 works everywhere; convert it and drop it again.')
      return
    }
    if (got.kind !== 'video') {
      alert('The source tile takes a video — the clip continues from it, or edits it.')
      return
    }
    const st = useStore.getState()
    st.addFile(got)
    // Continue by default: chaining is what this product has measured and
    // built for, and the chip's role button is one click from `edit`. The
    // *drop* is the choice of role here — the tile's own label says what it
    // makes — so this is the slot-is-the-role rule, not the model inferring a
    // task from mere presence, which the guide warns against.
    st.setSource(srcKind ?? 'continue', got.id)
  }

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
      // Named, and with a way out. "Could not read that file." was the whole
      // message and it left the person holding a file and no next move — the
      // browser is the thing that failed to decode it, so the fix is a format it
      // does decode, and `isImg` already says which list that is.
      else alert(isImg
        ? `The browser could not read ${f.name || 'that image'} — save it as a PNG or`
          + ' JPEG and drop it again.'
        : `The browser could not read ${f.name || 'that video'} — an MP4 works`
          + ' everywhere; convert it and drop it again.')
    }
    if (isImg) st.setRefs(out)
    else st.setRefVids(out)
  }

  const motion = !!s.continueFrom
  return (
    <div className="opts" id="v-src-sec">
      {/* **The Motion tile is the lever, and it is visible where every other
          anchor lives.** Set by Continue when the finished take saved its
          sampler latent: the next run opens with the previous clip's actual
          motion and audio pinned as context, instead of restarting from a
          still. One click clears it back to the frame — the tile row is the
          whole choice of fidelity: motion, frame, or a cut, each one gesture
          apart, none hidden inside the Continue button. Not a DropTile:
          nothing can be dropped on it, because its value is a fact about the
          last take rather than a file. */}
      {motion && (
        <button type="button" className="drop mini set" id="v-motion"
                title={`Motion and audio continue from take ${s.continueFrom ?? ''}.`
                       + ' Click to fall back to its last frame.'}
                onClick={() => s.setContinueFrom(null)}>
          <span className="lead">Motion ›</span>
        </button>
      )}
      <DropTile id="v-drop-first" label="First frame" value={s.keyframe.first}
                off={!!n || motion}
                glyph={<IconFirst />}
                title={composed
                  ? 'The clip opens on this image. It travels with the cast '
                    + 'as the keyframe anchor.'
                  : 'The clip starts on this image. Drop or click; click again to clear.'}
                onFile={async (f) => s.setKeyframe('first', await toB64(f))}
                onClear={() => s.setKeyframe('first', null)} />
      {sup.last_frame && (
        <DropTile id="v-drop-last" label="Last frame" value={s.keyframe.last}
                  off={!!n || motion || composed}
                  glyph={<IconLast />}
                  title={composed
                    // The first frame rides the cast run as grammar — see
                    // `readScene` — and there is no grammar slot that says
                    // *ends on*, so this tile is the one keyframe a cast run
                    // genuinely cannot take. Dim with the reason, never a
                    // silent no-op: it was live here while the generator
                    // ignored it, which reads as the model failing.
                    ? 'A cast run anchors its opening only — the last frame '
                      + 'cannot be pinned with references attached.'
                    : 'The clip ends on this image. Drop or click; click again to clear.'}
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
        {/* The clip-level source video — the writer `scene.sources` never
            had. One chip, because continue and edit are two answers to what
            this clip *is* and the validator refuses the pair; the role button
            swaps between them rather than a second chip appearing. */}
        {composed && srcFile && (
          <div className="ref" key="src">
            <video src={`${dataUrl(srcFile.b64, 'video/mp4')}#t=0.04`} muted />
            <b>SRC</b>
            <button className="x" title="Remove" type="button"
                    onClick={() => s.setSource(srcKind!, null)}>×</button>
            <button className="role" type="button"
                    title="What this video is to the clip — continued from, or edited into the target."
                    onClick={(e) => srcRole.toggle(e)}>
              {srcKind === 'edit' ? 'edited' : 'continues'}
            </button>
          </div>
        )}
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
          <AddTile id="v-add-ref" label="Picture" accept="image/"
                   off={(framed && !n) || composed}
                   glyph={<IconPhoto />}
                   title={composed
                     // With a cast, the flat tray never travels — <Picture N>
                     // numbers the cast's own files — so a picture has to
                     // belong to somebody before the run can read it. The
                     // rejection is the absence of an invitation, with the
                     // way in named.
                     ? 'With a cast, a picture belongs to somebody — drop it '
                       + 'on a member\u2019s card, or make it its own subject '
                       + 'with @.'
                     : 'Add an image reference — the subject, redrawn in a new shot. The prompt refers to it as <Picture 1>.'}
                   onFiles={(f) => void take('img', f)}
                   picking={addRef === 'img'} onPick={() => setAddRef('img')}
                   onDone={() => setAddRef(null)} />
          {/* The same tile, two meanings, told apart by label: flat tray when
              there is no composer, the clip-level source door when there is —
              which is where continue and edit finally get a gesture. The slot
              is the role. */}
          <AddTile id="v-add-vid" label={composed ? 'Source' : 'Video'} accept="video/"
                   // `framed && !n` is the flat-tray exclusivity — keyframes
                   // against references, one transformer or the other. Under a
                   // composer the first frame *rides along* as grammar, so a
                   // keyframe no longer excludes anything, and leaving the old
                   // gate here locked the Source door the moment a storyboard
                   // frame was dropped — found by doing exactly that.
                   off={composed ? false : framed && !n}
                   glyph={<IconFilm />}
                   title={composed
                     ? 'A source video for the clip — it continues from it, or '
                       + 'edits it. The chip\u2019s role button says which.'
                     : 'Add a video reference. The prompt refers to it as <Video 1>.'}
                   onFiles={(f) => void (composed ? takeSource(f) : take('vid', f))}
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

      {srcRole.open && srcFile && (
        <Menu anchor={srcRole.anchor} onClose={srcRole.close} items={[
          { label: 'The clip continues from it', on: srcKind === 'continue',
            run: () => s.setSource('continue', srcFile.id) },
          { label: 'The clip is an edit of it', on: srcKind === 'edit',
            run: () => s.setSource('edit', srcFile.id) },
        ]} />
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

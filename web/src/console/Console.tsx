import { useEffect, useLayoutEffect, useRef } from 'react'

import { ErrorNote } from '../ui/ErrorNote'
import { usePopover } from '../ui/Popover'
import { LoraBox } from '../lora/LoraBox'
import { LoraButton } from '../lora/LoraButton'
import { loraNote } from '../lora/note'
import { MotionDoor } from '../motion/MotionDoor'
import { MotionPanel } from '../motion/MotionPanel'
import { Palette } from '../shot/Palette'
import { Peek } from '../shot/Peek'
import { ShotDoor } from '../shot/ShotDoor'
import { Rail } from '../shot/Rail'
import { SourceRow } from '../video/SourceRow'
import { motionLive, supports, useStore } from '../store'
import { Field } from './Field'
import { autoGrow } from './fieldMax'
import { dropUnsupported, videoReady } from './resolve'
import { ImageSampling, VideoSampling } from './SamplingButton'
import { SizeButton, VideoSizeButton } from './SizeButton'
import type { RunState } from '../canvas/useGenerate'
import type { VideoRun } from '../video/useVideo'

/**
 * The bar under the canvas.
 *
 * **Sorted by how often you reach for something, and by nothing else.** The row holds
 * what you touch constantly — the dimensions and the LoRAs. One button holds what you
 * touch rarely: the model, the sampler, steps, CFG, shift, the seed, the batch count.
 * Shot and Regions were in this row and are not any more: they are not settings, they
 * are the two things the app is for, and each now lives on the surface it acts on.
 *
 * The near miss worth recording is *scope* — per-generation against per-session —
 * because it sounds right and fails on its own examples. Almost nothing in the row
 * genuinely changes every take: you do not pick a new aspect ratio per render. And CFG
 * is not a thing you set once a session either; it is a thing you almost never set.
 * Scope does not predict where a hand goes. Frequency does, and it is the one you can
 * answer by watching yourself work.
 *
 * **The console has a budget — 30% of the viewport — and the prompt is what yields to
 * it.** Everything else in here is fixed or conditional: the strip is one row, the rail
 * is one row, and the boxes cost nothing — they are on the canvas. See `fieldMax.ts`.
 */
export function Console({
  run,
  vidRun,
  onGenerate,
  onStop,
  lastShot,
}: {
  run: RunState
  vidRun: VideoRun
  onGenerate: () => void
  onStop: () => void
  /** The last generation, as a thumbnail. A utility lives with the controls — reaching the
   *  gallery is part of making something, like writing a prompt — so it goes here, 15px off
   *  the bottom, where the hand that just pressed Generate already is. */
  lastShot: React.ReactNode
}) {
  const s = useStore()
  const box = useRef<HTMLDivElement>(null)
  const pal = usePopover()
  const mo = usePopover()

  // The console has to watch itself, because the prompt is not the only thing that
  // grows: arming Regions adds a bar and picking pills adds a rail, and both happen
  // long after the last keystroke. Without this the field keeps whatever height it won
  // when it was the only claimant — measured, a long prompt sat at 30.0% and climbed to
  // 38.1% when the others arrived.
  //
  // It converges in one pass because `fieldMax` subtracts the field's own height, so
  // `other` does not move when the field does.
  useLayoutEffect(() => {
    const con = box.current
    if (!con) return
    const grow = () => {
      const field = con.querySelector<HTMLTextAreaElement>('.field.on-neg #neg')
        ?? con.querySelector<HTMLTextAreaElement>('#prompt')
      autoGrow(field, con)
    }
    grow()
    const ro = new ResizeObserver(grow)
    ro.observe(con)
    window.addEventListener('resize', grow)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', grow)
    }
  }, [])

  /* A model that cannot take what is already attached would fail at submit. Dropped
     here, where the section it came from is visibly gone, which is the only version of
     this that does not look like the request lost it. */
  useEffect(() => {
    const drop = dropUnsupported(s)
    if (!drop) return
    if (drop.refs) s.setRefs(drop.refs, drop.refRoles)
    if (drop.refVids) s.setRefVids(drop.refVids)
    if (drop.last === null) s.setKeyframe('last', null)
  }, [s])

  /* The first ready family, once. Kept across polls: only the picker's own labels are
     rewritten, and the resolver re-runs against whatever is still selected. */
  useEffect(() => {
    if (s.vid.model || !s.state) return
    const ready = s.state.video_models.filter((m) => m.ready)
    s.setVid({ model: (ready[0] ?? s.state.video_models[0])?.key ?? '' })
  }, [s])

  const image = s.kind === 'image'
  const note = loraNote(s)
  const busy = image ? run.running : vidRun.running
  const err = image ? run.error : vidRun.error

  // Which model this run needs and whether it is on the volume. On the image side that
  // is the two Krea 2 checkpoints; on the video side it is per task, so a t2v run is
  // never told to download the i2v pair it will not load.
  const anyImageModel = ['turbo', 'raw'].some((k) =>
    s.state?.models.find((m) => m.key === k)?.present)
  const vid = videoReady(s)
  const ready = image ? anyImageModel : vid.ready
  // `anyImageModel` is false both when the volume is genuinely empty and when
  // `state` has not arrived yet — `s.state?.…?.present` short-circuits to
  // undefined before the fetch returns. Left conflated, the second case paints
  // "No DiT on the volume" on first render, telling you to download weights the
  // app has not yet looked for. So say nothing until there is a real answer.
  const modelNote = !s.state
    ? ''
    : image
    ? (anyImageModel
        ? (s.state.models.find((m) => m.key === 'vae')?.present
           && s.state.models.find((m) => m.key === 'text_encoder')?.present
            ? '' : 'The VAE and text encoder are also required.')
        // The gear is the only route to the fix, so the note names it.
        : 'No DiT on the volume — download Krea 2 Turbo under Settings.')
    : ''

  return (
    <div className="console" ref={box}>
      {/* `ErrorNote` rather than a bare err-box, so the server's own report rides
          under the sentence instead of being the sentence — see `ui/ErrorNote`. */}
      <ErrorNote err={err} />

      <Field consoleEl={box} onSubmit={() => { if (ready && !busy) onGenerate() }}>
        <div className={image ? '' : 'hide'} id="c-image">
          <div className="opts">
            <SizeButton />
            <LoraButton id="add-lora" />
            {/* Regions left this row for good — it is a canvas verb, and you place a
                character by drawing on the empty frame. Shot stayed, next to `+ LoRA`,
                because both write into the prompt rather than beside it; what changed
                is that it carries a word now instead of being a 34px opaque mark. See
                `ShotDoor` for why it is here rather than on the rail. */}
            <ShotDoor id="g-shot" kind="image" on={!!s.shot.length} onClick={pal.toggle} />
            <span className="actions">
              <span className="muted" id="gen-model-line">{modelNote}</span>
              <ImageSampling />
              {lastShot}
              <GenerateButton id="go-gen" busy={busy} ready={ready}
                              onGenerate={onGenerate} onStop={onStop} />
            </span>
          </div>
        </div>

        <div className={image ? 'hide' : ''} id="c-video">
          <div className="opts">
            <VideoSizeButton />
            {/* Wan only, and the same picker the image side uses. The one thing the
                A14B pair forces — which expert a LoRA patches — rides in the token as a
                third field, read off the filename when the matched pair names it. */}
            {supports(s).loras && <LoraButton id="v-add-lora" />}
            {/* The motion door replaces the shot palette on this side — the
                palette's tiles were phrases in a vacuum, and behind this door
                the model has looked at the frame. The old door survives as the
                degrade: a server with no `motion_groups` gets the app exactly
                as it was, so turning the feature off is deleting one key. */}
            {s.state?.motion_groups?.length
              ? <MotionDoor id="v-shot"
                            on={motionLive(s) || !!s.shot.length}
                            onClick={mo.toggle} />
              : <ShotDoor id="v-shot" kind="video" on={!!s.shot.length} onClick={pal.toggle} />}
            <span className="actions">
              <span className="muted" id="v-model-line">{vid.note}</span>
              <VideoSampling />
              {lastShot}
              <GenerateButton id="go-vid" busy={busy} ready={ready}
                              onGenerate={onGenerate} onStop={onStop} />
            </span>
          </div>
        </div>
      </Field>

      {/* One palette, shared by both strips' doors — the vocabulary is one table with
          three compilers behind it, so a second popover per side would be a second
          place for the dimming rules to drift. (With motion_groups served, only the
          image door opens it; the video door opens the motion panel below.) */}
      {pal.open && <Palette anchor={pal.anchor} onClose={pal.close} />}
      {mo.open && <MotionPanel anchor={mo.anchor} onClose={mo.close} />}

      {/* Every picture the model can be given. Lifted out of the video strip when that
          moved into the field: this is a row of pictures, not a row of controls, and it
          is the one thing that could not follow. */}
      {!image && <SourceRow />}

      <Rail />
      <LoraBox />
      <Peek />

      {/* Only ever says what is wrong. A line confirming the LoRAs you can already read
          in the prompt above it would be the page telling you what you can see. The
          rewrite note was the one exception — a press rather than a state, since you
          could not look at the composer and tell whether Enhance had decided or died
          — and it went with the feature.

          The line is *reserved* — one row of height whether or not there is a note.
          Mounted on demand, each note that appeared grew the console, and the console
          is pinned to the bottom of the stage, so everything in it shifted the moment
          a warning arrived — which is mid-typing for the LoRA notes and mid-attach for
          the keyframe one, exactly when a hand is over Generate. A button that moves
          as you reach for it costs more than the 17px this holds at rest. */}
      <p className="muted warn" id="console-notes">
        <span id="lora-note">{note}</span>
        {!image && <VideoNote />}
      </p>


      {/* **A run reports itself once, and not here.** This block was a second
          progress bar under the canvas's own hairline and a second way to stop
          under the Generate button's own Stop — two of each for one run, a
          screen apart.

          The bar was the visible half: two white lines crossing the window,
          top and bottom, for one number. The Cancel was the worse half. It did
          exactly what Stop does — the job checks a flag between steps and
          unwinds — but it sat in a block detached from the button that had just
          become Stop, so pressing it changed a word in the wrong corner and the
          render carried on to the next step boundary. That reads as a control
          that does nothing, which is worse than not having one.

          What replaced both: the hairline along the canvas's top edge, and
          `#gen-meta`, where the live phase takes the render summary's place for
          as long as there is one. */}

    </div>
  )
}

/**
 * The id follows the side — `go-gen` on images, `go-vid` on video.
 *
 * Both strips are in the DOM at once and one is merely `.hide`, so a single
 * hardcoded id put two `#go-gen`s on the page. That is invalid HTML, but the
 * reason it matters here is narrower: `querySelector('#go-gen')` returns the
 * first, so anything reaching for the video button — a check, a shortcut, the
 * Enter binding — silently drove the hidden image one instead.
 */
function GenerateButton({ id, busy, ready, onGenerate, onStop }: {
  id: string; busy: boolean; ready: boolean; onGenerate: () => void; onStop: () => void
}) {
  return busy
    ? <button className="b" type="button" onClick={onStop}>Stop</button>
    : (
      <button className="b" id={id} type="button" disabled={!ready} onClick={onGenerate}>
        Generate
      </button>
    )
}

/** "Keyframes are ignored" only when there is a keyframe to ignore. Said
 *  unconditionally, it was the page's one mention of a control this layout had
 *  already made unfindable — a warning about something you do not have, pointing
 *  at somewhere you cannot see. A span in the reserved note line, so appearing
 *  costs no layout. */
function VideoNote() {
  const s = useStore()
  const n = s.refs.length + s.refVids.length
  const framed = !!(s.keyframe.first || s.keyframe.last)
  const text = n
    ? (framed ? `${n} reference${n > 1 ? 's' : ''} — keyframes are ignored for this run.` : '')
    : (s.keyframe.first
        ? `${supports(s).references ? '' : 'Image-to-video. '}`
          + 'Canvas follows the first frame’s aspect ratio.'
        : '')
  return <span id="vid-note">{text}</span>
}

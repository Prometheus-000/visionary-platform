import { useEffect, useMemo, useRef, useState } from 'react'

import { everyMs, failed } from '../api/client'
import {
  caption, captionDataset, clipUrl, deleteCaptionPreset, getState, imageUrl, prependTrigger,
  removeImage, replaceCaptions, saveCaptionPreset, status, stop, thumbUrl,
} from '../api/routes'
import { fmtFileSize } from '../format'
import { IconSliders, IconTag, IconUpload } from '../icons'
import { useNearViewport } from '../media/inview'
import { Thumb } from '../media/thumb'
import { useStore } from '../store'
import { ErrorNote } from '../ui/ErrorNote'
import { Masonry } from '../ui/Masonry'
import { useBusy } from '../ui/useBusy'
import { Duplicates } from './Duplicates'
import { Insight } from './Insight'
import type { DatasetImage, useDatasets } from './useDatasets'
import { SESSION } from './session'

/** A tile's height that is not the picture: the caption box's 66px floor and the
 *  tile's own border. Only the packer reads it — a column of landscape tiles
 *  packed as though their captions were free is a column that comes up short. */
const TILE_CHROME = 68

type Ds = ReturnType<typeof useDatasets>

/**
 * The contact sheet, which is the thing that scrolls.
 *
 * There are two ways to get a set and they are not equal: mostly you drag one in. So the
 * drop target is the screen when nothing is chosen, rather than a bar you find after
 * creating something to put it in — and the open sheet takes a drop too, because adding to
 * a set you already have was otherwise reachable only through the + Images button. The
 * obvious gesture did nothing, or worse: the browser's own default is to drop the file into
 * whichever caption box happens to be under the cursor.
 */
export function Editor({ ds, onLightbox, lead }: {
  ds: Ds
  onLightbox: (src: string) => void
  /** Whatever the screen around this needs at the head of the bar — today, the
   *  way back to the sessions board. It goes *inside* the sticky bar rather
   *  than above it, because a set is a long scroll and a back button that
   *  scrolls away is one you have to scroll up to find. */
  lead?: React.ReactNode
}) {
  const state = useStore((s) => s.state)
  const fileInput = useRef<HTMLInputElement>(null)
  const [hot, setHot] = useState(false)
  const [uploading, setUploading] = useState(0)
  const [filter, setFilter] = useState<'all' | 'uncap' | 'notrig' | 'img' | 'vid'>('all')
  /* A view of this set rather than a place: the filter row is where you look to
     change what is on the sheet, so the door to the duplicate review is in it.
     What it swaps in is not a filtered grid — a duplicate is a statement about
     a *pair*, so the arrangement has to be groups. */
  const [dupes, setDupes] = useState(false)
  const [density, setDensity] = useState(2)
  const [isolated, setIsolated] = useState<{ names: string[]; label: string } | null>(null)
  const [insightOpen, setInsightOpen] = useState(false)
  const [saveName, setSaveName] = useState('')
  const { busy, run } = useBusy()

  const flags = (img: DatasetImage) => {
    const cap = (img.caption || '').trim()
    const trig = ds.trigger.trim()
    return {
      cap,
      noTrig: !!trig && !cap.toLowerCase().includes(trig.toLowerCase()),
      thin: !!ds.insight?.thin.includes(img.name),
    }
  }

  const visible = useMemo(() => ds.images.filter((i) => {
    if (isolated) return isolated.names.includes(i.name)
    const f = flags(i)
    if (filter === 'uncap') return !f.cap
    if (filter === 'notrig') return f.noTrig
    // Absent means image: a set listed by an older deploy has no `kind` on its
    // tiles, and the filter that hid all of them would look like a set that
    // had lost its contents.
    if (filter === 'img') return i.kind !== 'video'
    if (filter === 'vid') return i.kind === 'video'
    return true
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [ds.images, ds.trigger, ds.insight, filter, isolated])

  /* ---- upload ---------------------------------------------------------- */
  /* XHR, not fetch — fetch reports no upload progress, and an eighty-image set is a
     transfer whose only honest readout is bytes moving. */
  const upload = async (list: FileList | File[]) => {
    const keep = [...list].filter(
      (f) => /\.(png|jpe?g|webp|bmp|avif|mp4|mov|webm|mkv|m4v|zip|txt)$/i.test(f.name))
    if (!keep.length || uploading) return
    let name = ds.open
    if (!name) {
      name = await ds.create(ds.suggestName(keep))
      if (!name) return
    }
    setUploading(1)
    ds.setEditError(null)
    const fd = new FormData()
    keep.forEach((f) => fd.append('files', f, f.name))
    fd.append('dataset', name)
    fd.append('session', SESSION)

    const x = new XMLHttpRequest()
    x.open('POST', '/api/upload')
    x.upload.onprogress = (e) => {
      if (e.lengthComputable) setUploading(Math.max(1, Math.round((e.loaded / e.total) * 100)))
    }
    x.onload = () => {
      setUploading(0)
      let r: { error?: string } = {}
      try {
        r = JSON.parse(x.responseText)
      } catch { /* a 500 from Modal is a plain-text traceback */ }
      if (r.error || x.status >= 400) {
        // The sentence first, the body underneath — the split `client.ts` already makes
        // for every other route, which this one missed by building its own message. It
        // read `Upload failed (500): ` and then 300 characters sliced out of the middle
        // of a Modal traceback: unreadable as a headline, and truncated exactly where
        // the person who *could* read it needed the rest. The clamp goes with it, since
        // the disclosure is closed and costs a line whatever is inside.
        ds.setEditError({
          error: r.error
            || `The upload did not finish (${x.status}) — nothing was added to the set.`
              + ' Try the same files again.',
          // Only when the server did not already give us a sentence, or the disclosure
          // is the same words a second time.
          detail: r.error ? undefined : x.responseText.trim() || undefined,
        })
        return
      }
      // The rail as well as the sheet: the image count Start training reads lives in the
      // list, so refreshing only the tiles left the button disabled on a set you were
      // looking at the contents of.
      void ds.loadTiles(name!).then(() => ds.load())
    }
    x.onerror = () => {
      setUploading(0)
      // XHR hands over nothing here — no status, no body — so the old "Network error
      // during upload." was accurate and a dead end. It has the same three causes as
      // `api()`'s fetch failure (server not running, dev proxy on the wrong port,
      // dropped wifi) and the page cannot tell them apart, so it says the one thing
      // true of all three and what to do next. The files never left, so the retry is
      // the same drop again.
      ds.setEditError('The upload was cut off before it finished — check that the server'
        + ' is still running, then drop the same files again.')
    }
    x.send(fd)
  }

  /* ---- captions -------------------------------------------------------- */
  const saveCaption = async (img: DatasetImage, value: string) => {
    if (!ds.open || img.caption === value) return
    const r = await captionDataset(ds.open, { image: img.name, caption: value })
    if (failed(r)) return ds.setEditError(r.error)
    ds.setImages(ds.images.map((i) => (i.name === img.name ? { ...i, caption: value } : i)))
    await ds.refreshInsight(ds.open, ds.trigger.trim())
  }

  const drop = (e: React.DragEvent) => {
    e.preventDefault()
    setHot(false)
    void upload(e.dataTransfer.files)
  }

  if (!ds.open) {
    return (
      <>
      {/* On the drop target too: this is where "+ New set" from the session
          form lands, and with no set open there is no bar to carry it. */}
      {!!lead && <div className="opts" style={{ marginTop: 0, marginBottom: 12 }}>{lead}</div>}
      <div className={`drop hero can-drop${hot ? ' hot' : ''}`} id="drop"
           onClick={() => { if (!uploading) fileInput.current?.click() }}
           onDragOver={(e) => { e.preventDefault(); setHot(true) }}
           onDragLeave={() => setHot(false)}
           onDrop={drop}>
        <input ref={fileInput} type="file" id="files" multiple className="hide"
               accept="image/*,video/*,.zip,.txt"
               onChange={(e) => {
                 void upload(e.target.files ?? [])
                 e.target.value = ''
               }} />
        {/* No name field. Naming was the toll on the one action this screen exists for,
            and it is charged for a decision — is this set worth keeping — that you cannot
            make before seeing the images. It moved to Save, on the sheet, where the
            images are. */}
        <div className="drop-face">
          <div className="glyph" id="drop-glyph"><IconUpload /></div>
          <b id="drop-title">{uploading ? 'Uploading…' : 'Drop images, clips or a .zip'}</b>
          <div className="muted" id="drop-sub" style={{ marginTop: 7 }}>
            {ds.error ?? ''}
          </div>
        </div>
        {!!uploading && (
          <div id="up-prog" style={{ marginTop: 18, width: 'min(420px,70%)' }}>
            <div className="bar"><i style={{ width: `${uploading}%` }} /></div>
          </div>
        )}
      </div>
      </>
    )
  }

  const cols = [10, 8, 6, 5, 3][Math.max(0, Math.min(4, density))] ?? 6
  /* One tally, counted once, from the array the tiles are drawn from.
     The bar used to carry this count and the insight's server-computed pair a few
     centimetres apart — "12 images · 5 captioned" next to "19/24 captioned" — and
     neither was wrong. `/insight` is a second request over `_dataset_images` (stills
     only, no clips), answered with the *committed* trigger word, so it trails the tiles
     by a round trip and by every keystroke in the trigger box that has not blurred yet.
     Two right answers to the same question still read as a broken page, and labelling
     one "as of the last analysis" would have been asking the person to reconcile them.
     So the bar states the count once and spends the second slot on the one fact the
     count does not carry: how many of those captions actually contain the trigger. Both
     come off `ds.images` and the live `ds.trigger`, matched the way the server matches
     (substring, case-folded — triggers are deliberately unwordlike), so the bar cannot
     disagree with itself or with the borders on the tiles below it. `ds.insight` still
     feeds the panel, where it is framed as an analysis and has nothing beside it to
     contradict. */
  const trig = ds.trigger.trim().toLowerCase()
  const captions = ds.images.map((i) => (i.caption || '').trim()).filter(Boolean)
  const captioned = captions.length
  const withTrigger = trig ? captions.filter((c) => c.toLowerCase().includes(trig)).length : 0
  // Counted apart, and never summed. A set of 24 images and 6 clips is not a
  // set of 30 of anything — one of those numbers is what the trainer will read
  // and the other is not, and a single total hides which is which.
  const clips = ds.images.filter((i) => i.kind === 'video').length
  const stills = ds.images.length - clips
  const shown = isolated
    ? ` · ${isolated.names.length} of ${ds.images.length} · ${isolated.label}`
    : visible.length === ds.images.length ? '' : ` · showing ${visible.length}`

  return (
    <div id="ds-sheet" className={hot ? 'hot' : ''} data-drop="Add to this set"
         onDragOver={(e) => { e.preventDefault(); setHot(true) }}
         onDragLeave={(e) => {
           if (!e.currentTarget.contains(e.relatedTarget as Node)) setHot(false)
         }}
         onDrop={drop}>
      <div className="opts sheet-bar">
        {lead}
        {/* A saved set states its name; a draft is still asking for one, so for a draft
            the name field *is* the title rather than sitting next to a heading that
            repeats it. */}
        {ds.saved ? (
          <h2 id="ds-title">{ds.open}</h2>
        ) : (
          <>
            <div className="opt mid" id="ds-name-wrap">
              <IconTag />
              {/* A zip dropped as `portraits.zip` already suggested `portraits`; carrying
                  it in means Save is usually one click rather than one click and some
                  typing. */}
              <input autoComplete="off" id="ds-save-name" placeholder="name to save it" spellCheck={false}
                     value={saveName || ds.open}
                     onChange={(e) => setSaveName(e.target.value)}
                     onKeyDown={(e) => {
                       if (e.key !== 'Enter') return
                       e.preventDefault()
                       void run('save', () => ds.save(saveName || ds.open!))
                     }} />
            </div>
            {/* Save is a rename on the volume, a re-list and a re-open, and it painted
                nothing for the length of all three — so the button looked unpressed and
                got pressed again, which is two saves racing to move the same draft out
                of `drafts/`. Enter in the name field goes through `run` for the same
                reason: holding the key is the fastest double-submit on the page. */}
            <button className="s" id="ds-save" title="Keep this set in datasets/"
                    type="button" disabled={!!busy}
                    onClick={() => void run('save', () => ds.save(saveName || ds.open!))}>
              {busy === 'save' ? 'Saving…' : 'Save'}
            </button>
          </>
        )}
        <span className="muted" id="ds-count">
          {stills} image{stills === 1 ? '' : 's'}
          {clips ? ` · ${clips} clip${clips === 1 ? '' : 's'}` : ''}
          {' · '}{captioned} captioned{shown}
        </span>
        {/* The two kind filters appear only when the set holds both kinds. A
            "Clips" button on a set of photographs is a control that can only
            ever empty the grid, and the count line above already says there
            are none. */}
        {/* **One control, not five buttons.** Which subset of the grid you are
            looking at is a single choice, and it was drawn as a row of pills
            identical to the row of pills beside it that *does* things — so a
            filter and an action were the same object, and the only thing
            separating the chosen filter from the rest was an inline border
            colour. A segmented group says "pick one of these" by its shape,
            which is what leaves the buttons outside it free to mean "do this". */}
        <span className="seg" id="ds-filters">
          {([['all', 'All', 'Show everything'],
             ...(clips && stills
               ? [['img', 'Images', 'Only the still images — what the trainer reads today'] as const,
                  ['vid', 'Clips', 'Only the video files'] as const]
               : []),
             ['uncap', 'Uncaptioned', 'Only files with no caption'],
             ['notrig', 'No trigger', 'Only captions missing the trigger word']] as const)
            .map(([k, label, title]) => (
              // `f-{key}`, matching app.py. Unlike the page, the active one is
              // marked: three buttons that all look identical while one of them
              // is filtering the grid is a view you cannot account for.
              <button key={k} id={`f-${k}`} title={title} type="button"
                      className={`s${filter === k && !isolated && !dupes ? ' on' : ''}`}
                      onClick={() => { setIsolated(null); setDupes(false); setFilter(k) }}>
                {label}
              </button>
            ))}
          {/* A word rather than a glyph, for the reason the strip's two icons
              each took one: this is a destination, and an icon can carry a
              control whose home you are already in but cannot announce a room
              you have never been in. */}
          <button id="f-dupes" className={`s${dupes ? ' on' : ''}`} type="button"
                  title="Group images that are the same picture, and choose which copy to keep"
                  onClick={() => { setIsolated(null); setFilter('all'); setDupes((v) => !v) }}>
            Duplicates
          </button>
        </span>
        {/* Its own group, because it is its own control: two ends of one
            stepper rather than two more filters.
            Named on the face of it. A bare − / + between the filters and the tallies is
            two glyphs that could plausibly step a page, a count or a zoom, and the only
            thing that said which was a hover tooltip — which is no answer at all for
            anyone reading this with a keyboard or a screen reader, and a slow one for
            everybody else. The word costs one slot in a bar that has room for it. */}
        <span className="seg" id="ds-density" role="group" aria-label="Thumbnail size">
          <span className="muted" style={{ padding: '0 6px' }}>Size</span>
          <button className="s" id="dens-down" type="button"
                  title="Smaller thumbnails" aria-label="Smaller thumbnails"
                  onClick={() => setDensity((d) => Math.max(0, d - 1))}>−</button>
          <button className="s" id="dens-up" type="button"
                  title="Larger thumbnails" aria-label="Larger thumbnails"
                  onClick={() => setDensity((d) => Math.min(4, d + 1))}>+</button>
        </span>
        <span className="actions">
          {/* A ratio over captions, not over images: "0/0 trigger" on a set nobody has
              captioned yet reads as a failure rather than a not-yet, which is the same
              reason the panel holds this stat back. */}
          <span className="muted" id="ins-summary"
                title="Captions containing the trigger word">
            {!!trig && !!captioned && `${withTrigger}/${captioned} trigger`}
          </span>
          {/* Named, not just iconed. The sliders glyph is already spoken for — the two
              Advanced toggles wear it — so one glyph standing for three unrelated panels
              is naming none of them. */}
          <button className={`t${insightOpen ? ' on' : ''}`} id="ins-toggle"
                  data-lb="Captions" type="button"
                  title="Write captions with a vision model, and see what this set is actually teaching."
                  onClick={() => setInsightOpen((v) => !v)}>
            <IconSliders />
            <span className="lead">Captions</span>
          </button>
          <button className="s" id="ds-add" type="button"
                  onClick={() => fileInput.current?.click()}>
            + Images
          </button>
        </span>
      </div>

      <input ref={fileInput} type="file" multiple className="hide" accept="image/*,video/*,.zip,.txt"
             onChange={(e) => {
               void upload(e.target.files ?? [])
               e.target.value = ''
             }} />

      <ErrorNote err={ds.editError} />
      {!!uploading && (
        <div style={{ margin: '10px 0' }}>
          <div className="bar"><i style={{ width: `${uploading}%` }} /></div>
        </div>
      )}

      {/* Captioning is an action on these images, so it lives with them rather than in the
          training console below. */}
      {insightOpen && (
        <div id="ins-panel" className="adv" style={{ margin: '0 0 14px' }}>
          <Captioner ds={ds} state={state} />
          <FindReplace ds={ds} visible={visible} />
          <Insight data={ds.insight}
                   onIsolate={(names, label) => setIsolated({ names, label })}
                   onIsolatePhrase={(phrase) => {
                     const p = phrase.toLowerCase()
                     setIsolated({
                       names: ds.images.filter((i) => (i.caption || '').toLowerCase().includes(p))
                         .map((i) => i.name),
                       label: `“${phrase}”`,
                     })
                   }} />
        </div>
      )}

      {dupes ? (
        <Duplicates ds={ds} onLightbox={onLightbox} onExit={() => setDupes(false)} />
      ) : (
      /* Packed rather than gridded, for the reason `.tile .ph`'s own comment gives:
         you cannot judge a crop you cannot see. A square cell with `contain`
         showed the whole frame and spent the difference on bars — a portrait set
         at ten across was two thirds `--wash-2`. The cell is the picture's shape
         now and the columns absorb the ragged bottoms; the density slider still
         decides how many there are, because that is a person saying how big they
         want to look at these. */
      <Masonry id="tiles" items={visible} cols={cols} gap={10} chrome={TILE_CHROME}
               ratio={(i) => (i.width && i.height ? i.height / i.width : 1)}
               render={(i) => {
          const f = flags(i)
          const sz = fmtFileSize(i.bytes)
          // The badge is clipped by the tile at high density, so what it can
          // always show stays first and the rest is on the tile itself. Kept in
          // one string so the two never disagree about the same file.
          const facts = [i.width && i.height ? `${i.width}×${i.height}` : '',
                         i.width && i.height ? `${(i.width * i.height / 1e6).toFixed(1)} MP` : '',
                         sz,
                         (i.name.split('.').pop() || '').toUpperCase()].filter(Boolean).join(' · ')
          return (
            <div key={i.name}
                 title={facts}
                 className={['tile', i.kind === 'video' ? 'vid' : '',
                             f.thin ? 'thin' : '', f.noTrig ? 'notrig' : '',
                             isolated ? 'sel' : ''].filter(Boolean).join(' ')}>
              {/* Overrides `.tile .ph`'s square. An image the server could not
                  measure keeps it and letterboxes, which is the same fallback the
                  packer above is using for it. */}
              <div className="ph"
                   style={i.width && i.height ? { aspectRatio: `${i.width}/${i.height}` } : undefined}>
                <Frame set={ds.open!} item={i} onLightbox={onLightbox} />
                <div className="dim">{i.width ? `${i.width}×${i.height} · ` : ''}{sz}</div>
                <button className="rm" title="Delete" type="button"
                        onClick={async () => {
                          const r = await removeImage(ds.open!, { image: i.name })
                          if (failed(r)) return ds.setEditError(r.error)
                          ds.setImages(ds.images.filter((x) => x.name !== i.name))
                          await ds.refreshInsight(ds.open!, ds.trigger.trim())
                        }}>×</button>
              </div>
              <CaptionBox image={i} onSave={(v) => void saveCaption(i, v)} />
            </div>
          )
        }} />
      )}
    </div>
  )
}

/**
 * What a tile paints, which is not the same route for the two kinds.
 *
 * An image has a thumbnail cached beside the dataset. A clip has none —
 * `web_image` has no ffmpeg, so there is nothing on the server that can make a
 * picture of one — so the tile does what a gallery card does: loads metadata
 * only and seeks to `#t=` so the browser paints a frame out of bytes it already
 * had to fetch. Not `#t=0`: seeking to exactly zero is not required to decode a
 * frame and some browsers leave the canvas blank.
 *
 * Gated behind the observer, because that request is unconditional and a grid
 * of them at once is what jams the volume — see `useNearViewport`.
 *
 * The corner mark is the treatment that tells the two apart at a glance, and it
 * is a mark rather than a border tint because the tile already spends its
 * border on two caption defects.
 */
function Frame({ set, item, onLightbox }: {
  set: string
  item: DatasetImage
  onLightbox: (src: string) => void
}) {
  const box = useRef<HTMLDivElement>(null)
  const isVideo = item.kind === 'video'
  const near = useNearViewport(box, isVideo)
  if (!isVideo) {
    // Not a bare <img>: the Thumb queues the fetch and retries a failure —
    // see media/thumb.tsx for the two faults that made it exist.
    return (
      <Thumb url={thumbUrl(set, item.name)}
             onClick={() => onLightbox(imageUrl(set, item.name))} />
    )
  }
  return (
    <div className="clip" ref={box}>
      {near && (
        <video src={`${clipUrl(set, item.name)}#t=0.04`} preload="metadata" muted playsInline
               onClick={() => onLightbox(clipUrl(set, item.name))} />
      )}
      <span className="kind" title="A clip. Nothing trains on these yet.">▶</span>
    </div>
  )
}

/** Autosave per caption on blur, with the pending state visible meanwhile. Enter commits
 *  rather than adding a line, because a caption is a sentence and the box is one row. */
function CaptionBox({ image, onSave }: { image: DatasetImage; onSave: (v: string) => void }) {
  const [value, setValue] = useState(image.caption ?? '')
  const box = useRef<HTMLTextAreaElement>(null)
  // Synced when fresh text arrives — a run finishing, a replace, a refetch.
  // `useState` alone seeds once and never again, so the tile kept painting
  // old text over a fresh listing: the page could be right and still look
  // stale, which is half of why "did the replace take?" needed a full
  // reload. The box being typed in is exempt — focus is the claim.
  useEffect(() => {
    if (document.activeElement !== box.current) setValue(image.caption ?? '')
  }, [image.caption])
  const dirty = value !== (image.caption ?? '')
  return (
    <textarea ref={box} className={dirty ? 'dirty' : ''} placeholder="No caption" spellCheck={false}
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onBlur={() => onSave(value.trim())}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  e.currentTarget.blur()
                }
              }} />
  )
}

/**
 * Write every caption with a vision model.
 *
 * **A caption preset is a decision about what to leave out, and a refusal is not an
 * error.** Both menus exist for reasons the other cannot cover: the preset decides what
 * the LoRA learns is free to vary, and the model decides whether it will describe a real
 * person at all. A stock instruct model declines often enough to matter, and on a
 * character set that is *every* image — the decline is fluent prose that passes every
 * check downstream, lands in a `.txt` sidecar and trains, so the symptom is a LoRA that
 * learned to say it cannot describe someone. The count of refusals is reported by name
 * and points at the other menu, because "run finished, nineteen of twenty-four
 * captioned" is not a diagnosis.
 */
function Captioner({ ds, state }: { ds: Ds; state: ReturnType<typeof useStore.getState>['state'] }) {
  const [preset, setPreset] = useState('')
  const [model, setModel] = useState('')
  const [mode, setMode] = useState('skip')
  // null means "follow the preset": the textarea shows the preset's own text
  // until the first keystroke, and switching presets snaps back to following.
  const [instr, setInstr] = useState<string | null>(null)
  // A blank preset being written. Binary on purpose: the house prompt runs
  // whole and its reply is cut up by code, so a copy of it that is "mostly
  // editable" is a trap — rename a label and the extraction quietly changes.
  // Either the house prompt exactly, or your own from nothing.
  const [draft, setDraft] = useState(false)
  const [adv, setAdv] = useState(false)
  const [maxTokens, setMaxTokens] = useState(320)
  const [temperature, setTemperature] = useState(0.6)
  const [topP, setTopP] = useState(0.9)
  const [prog, setProg] = useState<{ pct: number; note: string } | null>(null)
  // The running job, so Cancel has something to name. A stop is cooperative —
  // the loop reads the flag between images — so the readout says "Stopping…"
  // until the record comes back, rather than pretending the button was
  // instant.
  const [capJob, setCapJob] = useState<string | null>(null)
  const [stopping, setStopping] = useState(false)
  // `runBusy`, because `run` here is already the captioning run — which has `prog` for
  // its readout and is a poll rather than a round trip, so it keeps it.
  const { busy, run: runBusy } = useBusy()

  const presets = state?.caption_presets ?? []
  const models = state?.caption_models ?? []
  const curPreset = preset || state?.caption_defaults.preset || presets[0]?.key || ''
  const curModel = model || state?.caption_defaults.model || models[0]?.key || ''
  const presetSpec = presets.find((p) => p.key === curPreset)
  // A built-in is the house prompt and is read-only; a saved preset of your
  // own is editable in place; a draft is yours from blank.
  const house = !draft && !presetSpec?.custom
  const shownInstr = draft ? (instr ?? '') : (instr ?? presetSpec?.instruction ?? '')
  const edited = !draft && !!presetSpec?.custom && instr !== null
    && instr.trim() !== (presetSpec?.instruction ?? '').trim()
  // Clicking into the house prompt does not edit it — it opens a blank of your
  // own, with the menu saying so before the first keystroke.
  const startDraft = () => { setDraft(true); setInstr('') }
  const endDraft = () => { setDraft(false); setInstr(null) }
  // Only the captioner's note survives. The preset's used to summarise an
  // instruction the page did not show; the whole text is in the box now, and
  // a summary of a thing beside the thing is noise. What the captioner menu
  // still cannot say for itself is that the first run is a 17 GB pull.
  const modelNote = models.find((m) => m.key === curModel)?.note ?? ''

  // The store's state is the served vocabulary, so saving or deleting a preset
  // re-asks the server rather than patching a local copy that could drift.
  const reloadState = async () => {
    const r = await getState()
    if (!failed(r)) useStore.getState().setState(r)
  }

  const savePreset = async () => {
    const name = window.prompt('Name this preset', presetSpec?.custom && !draft ? presetSpec.label : '')
    if (!name?.trim()) return
    const r = await saveCaptionPreset(name.trim(), shownInstr)
    if (failed(r)) return ds.setEditError(r.error)
    await reloadState()
    if (r.key) setPreset(r.key)
    endDraft()
  }

  const removePreset = async () => {
    if (!presetSpec?.custom) return
    if (!confirm(`Delete the preset “${presetSpec.label}”?`)) return
    const r = await deleteCaptionPreset(presetSpec.key)
    if (failed(r)) return ds.setEditError(r.error)
    setPreset('')
    setInstr(null)
    await reloadState()
  }

  const run = async () => {
    if (!ds.open) return
    setProg({ pct: 0, note: 'Queued…' })
    ds.setEditError(null)
    const r = await caption({
      dataset: ds.open, trigger_word: ds.trigger.trim(),
      preset: curPreset, model: curModel, write_mode: mode,
      // Sent only when the text is yours — a draft or an edited preset — so
      // an untouched box runs exactly what picking the preset always ran, and
      // the server can tell the house prompt (extracted) from your own (kept
      // as written) by whether this is empty.
      instruction: draft || edited ? shownInstr : '',
      max_tokens: maxTokens, temperature, top_p: topP,
    })
    if (failed(r)) {
      setProg(null)
      return ds.setEditError(r.error)
    }
    setCapJob(r.job_id)
    setStopping(false)
    let stopPressed = false
    const t = everyMs(async () => {
      const st = await status(r.job_id)
      if (failed(st)) return
      stopPressed = stopPressed || !!st.stop
      setProg({
        pct: Number(st.percent ?? 0),
        // Named while it loads. The first minute of a run against a captioner that is not
        // in the cache yet is a 17 GB pull, and a bare "Loading captioner…" for twenty
        // minutes is indistinguishable from a hang.
        note: stopPressed
          ? `Stopping after ${st.step ?? 0}…`
          : st.step
            ? `Captioning ${st.step}/${String(st.total_steps ?? '?')}`
            : `Loading ${String(st.model_label ?? 'captioner')}…`,
      })
      // Refresh mid-run so captions land visibly rather than all at the end.
      if (st.step && Number(st.step) % 5 === 0) void ds.loadTiles(ds.open!)
      if (st.status === 'completed') {
        clearInterval(t)
        setProg(null)
        setCapJob(null)
        await ds.loadTiles(ds.open!)
        const refused = Number(st.refused ?? 0)
        if (refused) {
          ds.setEditError(
            `${refused} image${refused === 1 ? '' : 's'} the captioner declined to describe`
            + (st.model === 'qwen3vl' ? '. Try the uncensored captioner.' : '.'),
          )
        }
      } else if (st.status === 'failed') {
        clearInterval(t)
        setProg(null)
        setCapJob(null)
        // Two dead ends in one line: the job's own error was the headline, so a run that
        // died on an OOM shouted a CUDA traceback, and when the job carried no error at
        // all it said "Captioning failed" and stopped there. The captions already
        // written are still on the volume, and the two knobs that get a stuck run
        // through are the token cap under Advanced and the model — so the sentence
        // says both, and the traceback rides underneath for whoever wants it.
        ds.setEditError({
          error: 'The captioning run stopped before it finished. Whatever it had already'
            + ' written is kept — run it again, and if it keeps stopping try a lower'
            + ' max tokens or a different captioner.',
          detail: st.error ? String(st.error) : undefined,
        })
      }
    }, 2500)
  }

  // Cooperative: the flag is set here, the loop reads it before the next
  // image, and the run completes with what it wrote. Nothing is discarded, so
  // no confirm.
  const cancel = async () => {
    if (!capJob || stopping) return
    setStopping(true)
    setProg((p) => p && { ...p, note: 'Stopping…' })
    const r = await stop(capJob)
    if (failed(r)) {
      setStopping(false)
      ds.setEditError(r.error)
    }
  }

  return (
    <>
      <div className="opts" style={{ marginTop: 0 }}>
        <div className="opt mid">
          <IconTag />
          <input autoComplete="off" id="ds-trig" placeholder="trigger word" spellCheck={false}
                 value={ds.trigger}
                 onChange={(e) => ds.setTrigger(e.target.value)}
                 onBlur={() => void ds.commitTrigger(ds.trigger)} />
        </div>
        {/* Two round trips behind one press — the rewrite of every sidecar that lacks
            the word, then a reload of every tile in the set — and it had no pending
            state at all, so on a set of eighty the only evidence anything happened was
            the grid repainting some seconds later. A second press in that gap cannot
            double a prefix — the route is idempotent on purpose — but it does rewrite
            every sidecar and reload every tile again, which is how the wait doubles. */}
        <button className="s" id="do-prepend" type="button" disabled={!!prog || !!busy}
                title="Put the trigger word at the front of every caption that lacks it"
                onClick={() => void runBusy('prepend', async () => {
                  const trig = ds.trigger.trim()
                  if (!trig) return ds.setEditError('Set a trigger word first.')
                  const r = await prependTrigger(ds.open!, { trigger_word: trig })
                  if (failed(r)) return ds.setEditError(r)
                  await ds.loadTiles(ds.open!)
                })}>
          {busy === 'prepend' ? 'Fixing…' : 'Fix'}
        </button>
        {/* Two menus, no labels: "Character" and "JoyCaption Beta One" each name
            themselves, and neither could be mistaken for the other. A third —
            Short/Medium/Long — sat between them until 2026-09-02 and changed
            nothing in the captions it was supposed to size. */}
        <div className="opt">
          <select id="cap-preset" value={draft ? '__draft__' : curPreset}
                  onChange={(e) => { setPreset(e.target.value); endDraft() }}>
            {presets.map((p) => <option key={p.key} value={p.key}>{p.label}</option>)}
            {draft && <option value="__draft__">New preset</option>}
          </select>
        </div>
        {presetSpec?.custom && (
          <button className="s" id="cap-preset-del" type="button"
                  title="Delete this preset" onClick={() => void removePreset()}>✕</button>
        )}
        <div className="opt">
          <select id="cap-model" value={curModel} onChange={(e) => setModel(e.target.value)}>
            {models.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
          </select>
        </div>
        {/* Beside the menu it describes. It used to close the whole row, and
            once the "What the model sees" fold opened above it, it read as the
            last paragraph of what the model sees. */}
        {modelNote && (
          <span className="muted" id="cap-note" style={{ alignSelf: 'center' }}>{modelNote}</span>
        )}
        {/* The four honest answers to "this image already has a caption", replacing
            a checkbox that could only say two of them. Skip leads because it is the
            only one that cannot lose work. */}
        <div className="opt">
          <select id="cap-mode" value={mode} onChange={(e) => setMode(e.target.value)}
                  title="What happens to images that already have a caption">
            <option value="skip">Skip captioned</option>
            <option value="append">Append</option>
            <option value="prepend">Prepend</option>
            <option value="replace">Replace</option>
          </select>
        </div>
        <span className="actions">
          <button className={`s${adv ? ' on' : ''}`} id="cap-adv" type="button"
                  title="Max tokens, temperature, top-p"
                  onClick={() => setAdv((v) => !v)}>
            Advanced
          </button>
          <button className="s" id="do-caption" type="button"
                  disabled={!!prog || !!busy || (draft && !shownInstr.trim())}
                  onClick={() => void run()}>
            Caption
          </button>
        </span>
      </div>
      {/* The instruction itself, always shown. The house prompt is read-only,
          and a click into it opens a blank of your own instead. A saved preset
          of yours edits in place, and Save keeps it. Nothing composes around
          any of it on the server: `{NAME}` becomes the trigger word, and the
          house prompt's reply is cut to what follows its final CAPTION:; yours
          is saved as written. This comment used to say the opposite, and the sixty-word
          rulebook it was covering for went unnoticed for a month. */}
      {/* Sized to the text, capped: the preset is a form of a dozen lines, and a
          three-row box showed one sentence of it — a field you cannot see is a
          field you cannot edit. */}
      <textarea id="cap-instr" value={shownInstr} spellCheck={false} readOnly={house}
                placeholder={draft ? 'Write your own instruction.' : undefined}
                rows={Math.min(18, Math.max(4, shownInstr.split('\n').length))}
                style={{ width: '100%', marginTop: 8, resize: 'vertical',
                         opacity: house ? 0.7 : 1 }}
                onMouseDown={house ? startDraft : undefined}
                onFocus={house ? startDraft : undefined}
                onChange={(e) => { if (!house) setInstr(e.target.value) }} />
      {/* One line under the box, and it is the owner's sentence for the
          house prompt. The other states say how the reply is handled, because
          that is the one fact the writer of a preset cannot see anywhere. */}
      {house && (
        <p className="muted" id="cap-parse-note" style={{ margin: '6px 2px 0' }}>
          Editing is disabled, output is extracted in code.
        </p>
      )}
      {draft && (
        <div className="row" style={{ gap: 8, marginTop: 6 }}>
          <span className="muted" style={{ fontSize: 12 }}>
            Your own — the reply is saved as the model writes it.
          </span>
          <button className="s" id="cap-save-preset" type="button"
                  disabled={!shownInstr.trim()} onClick={() => void savePreset()}>
            Save as preset
          </button>
          <button className="s" id="cap-instr-reset" type="button"
                  title="Back to the house prompt" onClick={endDraft}>
            Discard
          </button>
        </div>
      )}
      {!house && !draft && !edited && (
        <p className="muted" id="cap-parse-note" style={{ margin: '6px 2px 0' }}>
          Your preset — the reply is saved as the model writes it.
        </p>
      )}
      {edited && (
        <div className="row" style={{ gap: 8, marginTop: 6 }}>
          <span className="muted" style={{ fontSize: 12 }}>
            Edited — this run uses your text.
          </span>
          <button className="s" id="cap-save-preset" type="button"
                  onClick={() => void savePreset()}>
            Save
          </button>
          <button className="s" id="cap-instr-reset" type="button"
                  title="Back to the preset as saved"
                  onClick={() => setInstr(null)}>
            Reset
          </button>
        </div>
      )}
      {adv && (
        <div className="opts" id="cap-adv-row" style={{ marginTop: 8 }}>
          {/* Named, every one — "320" is a token cap, a seed or a size with equal
              plausibility, and a bare number is not a value. */}
          <div className="opt" data-lb="Max tokens">
            <span className="lead">Max tokens</span>
            <input autoComplete="off" type="number" id="cap-max-tokens" min={16} max={1024} step={16}
                   style={{ width: 64 }} value={maxTokens}
                   onChange={(e) => setMaxTokens(Number(e.target.value) || 320)} />
          </div>
          <div className="opt" data-lb="Temperature">
            <span className="lead">Temperature</span>
            <input autoComplete="off" type="number" id="cap-temp" min={0} max={1.5} step={0.05}
                   style={{ width: 64 }} value={temperature}
                   onChange={(e) => setTemperature(Number(e.target.value))} />
          </div>
          <div className="opt" data-lb="Top-p">
            <span className="lead">Top-p</span>
            <input autoComplete="off" type="number" id="cap-top-p" min={0.05} max={1} step={0.05}
                   style={{ width: 64 }} value={topP}
                   onChange={(e) => setTopP(Number(e.target.value))} />
          </div>
        </div>
      )}
      {prog && (
        <div id="cap-prog" style={{ marginTop: 9 }}>
          <div className="bar"><i style={{ width: `${prog.pct}%` }} /></div>
          <div className="row" style={{ gap: 8, marginTop: 6, alignItems: 'center' }}>
            <p className="muted" style={{ margin: 0 }}>{prog.note}</p>
            <button className="s" id="cap-cancel" type="button" disabled={stopping}
                    title="Stop after the image it is on; captions already written are kept"
                    onClick={() => void cancel()}>
              {stopping ? 'Stopping…' : 'Cancel'}
            </button>
          </div>
        </div>
      )}
    </>
  )
}

/**
 * Find & replace across the captions on screen.
 *
 * Scoped to the current filtered view rather than the whole set, so the
 * filters are the targeting tool — and the count is on the button, because a
 * number that changes when you touch a filter is only trustworthy if you can
 * see it change before you press. The count is computed the same way the
 * server matches (literal string, optional case fold), so what the button
 * promises is what the route does.
 */
function FindReplace({ ds, visible }: { ds: Ds; visible: DatasetImage[] }) {
  const [find, setFind] = useState('')
  const [repl, setRepl] = useState('')
  const [caseOn, setCaseOn] = useState(false)
  const [note, setNote] = useState<string | null>(null)

  const hits = useMemo(() => {
    if (!find) return 0
    const f = caseOn ? find : find.toLowerCase()
    return visible.filter((i) => {
      const cap = i.caption || ''
      return (caseOn ? cap : cap.toLowerCase()).includes(f)
    }).length
  }, [visible, find, caseOn])

  const run = async () => {
    if (!ds.open || !find || !hits) return
    const r = await replaceCaptions(ds.open, {
      find, replace: repl, match_case: caseOn,
      images: visible.map((i) => i.name),
    })
    if (failed(r)) return ds.setEditError(r.error)
    const n = Number(r.changed ?? 0)
    setNote(`${n} caption${n === 1 ? '' : 's'} changed.`)
    // The reply carries the new text, so the tiles are patched from it
    // directly. The refetch this replaces used to read the same stale mount
    // the replace had just written through, and repainted the text you had
    // just replaced — which is what made "did it take?" unanswerable.
    const caps = (r as { captions?: Record<string, string> }).captions ?? {}
    ds.setImages(ds.images.map((i) => {
      const c = caps[i.name]
      return c === undefined ? i : { ...i, caption: c }
    }))
    await ds.refreshInsight(ds.open, ds.trigger.trim())
  }

  return (
    <div className="opts" id="cap-replace" style={{ marginTop: 10 }}>
      <div className="opt mid">
        <input autoComplete="off" id="fr-find" placeholder="find in captions" spellCheck={false}
               value={find} onChange={(e) => { setFind(e.target.value); setNote(null) }}
               onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); void run() } }} />
      </div>
      <div className="opt mid">
        <input autoComplete="off" id="fr-replace" placeholder="replace with" spellCheck={false}
               value={repl} onChange={(e) => setRepl(e.target.value)}
               onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); void run() } }} />
      </div>
      <label className="row" style={{ gap: 7, margin: 0, color: '#ddd', fontSize: 13 }}>
        <input type="checkbox" id="fr-case" style={{ width: 'auto' }}
               checked={caseOn} onChange={(e) => setCaseOn(e.target.checked)} />
        Match case
      </label>
      <span className="actions">
        {note && <span className="muted" id="fr-note" style={{ fontSize: 12 }}>{note}</span>}
        <button className="s" id="fr-run" type="button" disabled={!hits}
                onClick={() => void run()}>
          Replace in {hits}
        </button>
      </span>
    </div>
  )
}

import { useMemo, useRef, useState } from 'react'

import { everyMs, failed } from '../api/client'
import {
  caption, captionDataset, clipUrl, imageUrl, prependTrigger, removeImage, status, thumbUrl,
} from '../api/routes'
import { fmtFileSize } from '../format'
import { IconSliders, IconTag, IconUpload } from '../icons'
import { useNearViewport } from '../media/inview'
import { useStore } from '../store'
import { Duplicates } from './Duplicates'
import { Insight } from './Insight'
import type { DatasetImage, useDatasets } from './useDatasets'
import { SESSION } from './session'

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
        const detail = r.error || x.responseText.slice(0, 300) || 'no response body'
        ds.setEditError(`Upload failed (${x.status}): ${detail}`)
        return
      }
      // The rail as well as the sheet: the image count Start training reads lives in the
      // list, so refreshing only the tiles left the button disabled on a set you were
      // looking at the contents of.
      void ds.loadTiles(name!).then(() => ds.load())
    }
    x.onerror = () => {
      setUploading(0)
      ds.setEditError('Network error during upload.')
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
  const captioned = ds.images.filter((i) => (i.caption || '').trim()).length
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
          <b id="ds-title" style={{ fontSize: 14 }}>{ds.open}</b>
        ) : (
          <>
            <div className="opt mid" id="ds-name-wrap">
              <IconTag />
              {/* A zip dropped as `portraits.zip` already suggested `portraits`; carrying
                  it in means Save is usually one click rather than one click and some
                  typing. */}
              <input id="ds-save-name" placeholder="name to save it" spellCheck={false}
                     value={saveName || ds.open}
                     onChange={(e) => setSaveName(e.target.value)}
                     onKeyDown={(e) => {
                       if (e.key !== 'Enter') return
                       e.preventDefault()
                       void ds.save(saveName || ds.open!)
                     }} />
            </div>
            <button className="s" id="ds-save" title="Keep this set in datasets/"
                    type="button" onClick={() => void ds.save(saveName || ds.open!)}>
              Save
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
        {([['all', 'All', 'Show everything'],
           ...(clips && stills
             ? [['img', 'Images', 'Only the still images — what the trainer reads today'] as const,
                ['vid', 'Clips', 'Only the video files'] as const]
             : []),
           ['uncap', 'Uncaptioned', 'Only files with no caption'],
           ['notrig', 'No trigger', 'Only captions missing the trigger word']] as const)
          .map(([k, label, title]) => (
            // `f-{key}`, matching app.py. Unlike the page, the active one is
            // marked: three buttons that all look identical while one of them is
            // filtering the grid is a view you cannot account for.
            <button key={k} id={`f-${k}`} className="s" title={title} type="button"
                    style={{ borderColor: filter === k && !isolated && !dupes ? 'rgba(255,255,255,.45)' : '' }}
                    onClick={() => { setIsolated(null); setDupes(false); setFilter(k) }}>
              {label}
            </button>
          ))}
        {/* A word rather than a glyph, for the reason the strip's two icons each
            took one: this is a destination, and an icon can carry a control
            whose home you are already in but cannot announce a room you have
            never been in. */}
        <button id="f-dupes" className="s" type="button"
                title="Group images that are the same picture, and choose which copy to keep"
                style={{ borderColor: dupes ? 'rgba(255,255,255,.45)' : '' }}
                onClick={() => { setIsolated(null); setFilter('all'); setDupes((v) => !v) }}>
          Duplicates
        </button>
        <button className="s" id="dens-down" title="Smaller tiles" type="button"
                onClick={() => setDensity((d) => Math.max(0, d - 1))}>−</button>
        <button className="s" id="dens-up" title="Larger tiles" type="button"
                onClick={() => setDensity((d) => Math.min(4, d + 1))}>+</button>
        <span className="actions">
          <span className="muted" id="ins-summary">
            {ds.insight && (
              (ds.insight.trigger_word && ds.insight.captioned
                ? `${ds.insight.with_trigger}/${ds.insight.captioned} trigger · ` : '')
              + `${ds.insight.captioned}/${ds.insight.images} captioned`
            )}
          </span>
          {/* Named, not just iconed. The sliders glyph is already spoken for — the two
              Advanced toggles wear it — so one glyph standing for three unrelated panels
              is naming none of them. */}
          <button className={`opt ib${insightOpen ? ' on' : ''}`} id="ins-toggle"
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

      {ds.editError && <div className="err-box">{ds.editError}</div>}
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
      <div className="tiles" id="tiles"
           style={{ gridTemplateColumns: `repeat(${cols},minmax(0,1fr))` }}>
        {visible.map((i) => {
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
              <div className="ph">
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
        })}
      </div>
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
    return (
      <img loading="lazy" src={thumbUrl(set, item.name)} alt=""
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
  const dirty = value !== (image.caption ?? '')
  return (
    <textarea className={dirty ? 'dirty' : ''} placeholder="No caption" spellCheck={false}
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
  const [length, setLength] = useState('medium')
  const [over, setOver] = useState(false)
  const [prog, setProg] = useState<{ pct: number; note: string } | null>(null)

  const presets = state?.caption_presets ?? []
  const models = state?.caption_models ?? []
  const curPreset = preset || state?.caption_defaults.preset || presets[0]?.key || ''
  const curModel = model || state?.caption_defaults.model || models[0]?.key || ''
  // What the two words cannot say: which half of the picture a preset throws away, and
  // that the second captioner is a download. Both come from the server, so the note and
  // the instruction behind it can never disagree.
  const note = [presets.find((p) => p.key === curPreset)?.note,
                models.find((m) => m.key === curModel)?.note].filter(Boolean).join(' · ')

  const run = async () => {
    if (!ds.open) return
    setProg({ pct: 0, note: 'Queued…' })
    ds.setEditError(null)
    const r = await caption({
      dataset: ds.open, trigger_word: ds.trigger.trim(),
      preset: curPreset, model: curModel, length, overwrite: over,
    })
    if (failed(r)) {
      setProg(null)
      return ds.setEditError(r.error)
    }
    const t = everyMs(async () => {
      const st = await status(r.job_id)
      if (failed(st)) return
      setProg({
        pct: Number(st.percent ?? 0),
        // Named while it loads. The first minute of a run against a captioner that is not
        // in the cache yet is a 17 GB pull, and a bare "Loading captioner…" for twenty
        // minutes is indistinguishable from a hang.
        note: st.step
          ? `Captioning ${st.step}/${String(st.total_steps ?? '?')}`
          : `Loading ${String(st.model_label ?? 'captioner')}…`,
      })
      // Refresh mid-run so captions land visibly rather than all at the end.
      if (st.step && Number(st.step) % 5 === 0) void ds.loadTiles(ds.open!)
      if (st.status === 'completed') {
        clearInterval(t)
        setProg(null)
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
        ds.setEditError(st.error || 'Captioning failed')
      }
    }, 2500)
  }

  return (
    <>
      <div className="opts" style={{ marginTop: 0 }}>
        <div className="opt mid">
          <IconTag />
          <input id="ds-trig" placeholder="trigger word" spellCheck={false}
                 value={ds.trigger}
                 onChange={(e) => ds.setTrigger(e.target.value)}
                 onBlur={() => void ds.commitTrigger(ds.trigger)} />
        </div>
        <button className="s" id="do-prepend" type="button"
                title="Put the trigger word at the front of every caption that lacks it"
                onClick={async () => {
                  const trig = ds.trigger.trim()
                  if (!trig) return ds.setEditError('Set a trigger word first.')
                  const r = await prependTrigger(ds.open!, { trigger_word: trig })
                  if (failed(r)) return ds.setEditError(r.error)
                  await ds.loadTiles(ds.open!)
                }}>
          Fix
        </button>
        {/* Three menus, no labels: "Character", "Medium" and "Qwen3-VL 8B" each name
            themselves, and none could be mistaken for another. */}
        <div className="opt">
          <select id="cap-preset" value={curPreset} onChange={(e) => setPreset(e.target.value)}>
            {presets.map((p) => <option key={p.key} value={p.key}>{p.label}</option>)}
          </select>
        </div>
        <div className="opt">
          <select id="cap-len" value={length} onChange={(e) => setLength(e.target.value)}>
            <option value="short">Short</option>
            <option value="medium">Medium</option>
            <option value="long">Long</option>
          </select>
        </div>
        <div className="opt">
          <select id="cap-model" value={curModel} onChange={(e) => setModel(e.target.value)}>
            {models.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
          </select>
        </div>
        <label className="row" style={{ gap: 7, margin: 0, color: '#ddd', fontSize: 13 }}>
          <input type="checkbox" id="cap-over" style={{ width: 'auto' }}
                 checked={over} onChange={(e) => setOver(e.target.checked)} />
          Replace existing
        </label>
        <span className="actions">
          <button className="s" id="do-caption" type="button" disabled={!!prog}
                  onClick={() => void run()}>
            Caption
          </button>
        </span>
      </div>
      <p className="muted" id="cap-note" style={{ margin: '8px 2px 0' }}>{note}</p>
      {prog && (
        <div id="cap-prog" style={{ marginTop: 9 }}>
          <div className="bar"><i style={{ width: `${prog.pct}%` }} /></div>
          <p className="muted" style={{ marginTop: 6 }}>{prog.note}</p>
        </div>
      )}
    </>
  )
}

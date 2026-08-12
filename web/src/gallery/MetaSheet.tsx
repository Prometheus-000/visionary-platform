import { useState } from 'react'

import { IconClose } from '../icons'
import { Sheet } from '../ui/Sheet'
import { shotItem, useStore } from '../store'
import { reuse } from './reuse'
import type { GalleryItem } from './types'

/**
 * Everything the sidecar kept about one run.
 *
 * The prompt appears twice when the compiler did something, and in this order. What you
 * wrote is what you recognise the run by; what ran is the six-field document, and it is
 * here because the only other way to find out what the encoder was given is to render
 * again.
 *
 * Copy takes the typed one — what the box above the button shows. Copying the compiled
 * document from under a button sitting beside a textarea containing the typed sentence
 * is the button doing something other than what you can see.
 */
export function MetaSheet({ item, onClose }: { item: GalleryItem; onClose: () => void }) {
  const vocab = useStore((s) => s.state?.shot_vocab) ?? []
  const roles = useStore((s) => s.state?.shot_roles) ?? []
  const [copied, setCopied] = useState(false)
  const typed = item.prompt_typed || item.prompt || ''

  const rows: [string, string | number | undefined][] = [
    ['Kind', item.kind],
    ['Model', item.model],
    ['Job', item.job_id],
    ['Size', item.width ? `${item.width}×${item.height}` : ''],
    ['Seed', (item.seeds ?? []).join(', ') || item.seed],
    ['Steps', item.steps],
    ['Sampler', item.sampler],
    ['Scheduler', item.scheduler],
    ['CFG', item.cfg_scale],
    ['Shift', item.shift],
    ['Expert switch', item.switch_at ? `step ${item.switch_at}` : ''],
    ['Length', item.seconds ? `${item.seconds}s · ${item.frames} frames · ${item.fps} fps` : ''],
    ['References', item.references || item.ref_videos
      ? `${item.references ?? 0} image, ${item.ref_videos ?? 0} video` : ''],
    // `expert` only exists on the video stack, `applied` only on the image one.
    ['LoRAs', (item.loras ?? []).map((l) =>
      `${l.name} @ ${l.unet}`
      + (l.expert && l.expert !== 'both' ? ` (${l.expert} noise)` : '')
      + (l.applied === false ? ' (not applied)' : '')).join(', ')],
    ['Regions', (item.regions ?? []).map((r) => r.prompt).filter(Boolean).join(' | ')],
    // The pills by name, so a run is readable without decompiling its document — and
    // readable a year later, when the labels may have moved but `camera.pushin` still
    // says what was picked.
    ['Shot', (item.shot ?? []).map((p) => {
      const i = shotItem(vocab, p.key)
      return (i ? i.label : p.key) + (p.value ? `: “${p.value}”` : '')
    }).join(', ')],
    ['Reference roles', (item.ref_roles ?? []).map((r, n) => {
      const spec = roles.find((x) => x.key === r)
      return spec ? `P${n + 1} ${spec.label}` : ''
    }).filter(Boolean).join(', ')],
    ['Files', item.files.join(', ')],
    ['Created', item.created ? new Date(item.created * 1000).toLocaleString() : ''],
  ]

  return (
    <Sheet onClose={onClose}>
      <div className="sheet-head">
        <h1 className="grow">Metadata</h1>
        <button className="ico" type="button" onClick={onClose}><IconClose /></button>
      </div>
      <label>Prompt</label>
      <textarea rows={5} readOnly value={typed} />
      {item.prompt_typed && item.prompt_typed !== item.prompt && (
        <>
          <label style={{ marginTop: 12 }}>What the model read</label>
          <textarea rows={7} readOnly value={item.prompt ?? ''} />
        </>
      )}
      {item.negative_prompt && (
        <>
          <label style={{ marginTop: 12 }}>Negative</label>
          <textarea rows={2} readOnly value={item.negative_prompt} />
        </>
      )}
      <dl className="kv" style={{ marginTop: 18 }}>
        {rows.filter(([, v]) => v !== undefined && v !== null && v !== '').map(([k, v]) => (
          <div key={k} style={{ display: 'contents' }}>
            <dt>{k}</dt>
            <dd>{String(v)}</dd>
          </div>
        ))}
      </dl>
      <div className="row" style={{ gap: 8, marginTop: 20 }}>
        <button className="s" type="button"
                onClick={async () => {
                  await navigator.clipboard.writeText(typed)
                  setCopied(true)
                }}>
          {copied ? 'Copied' : 'Copy prompt'}
        </button>
        <button className="s" type="button" onClick={() => { onClose(); reuse(item) }}>
          Reuse settings
        </button>
        <span className="grow" />
        <button className="s" type="button" onClick={onClose}>Close</button>
      </div>
    </Sheet>
  )
}

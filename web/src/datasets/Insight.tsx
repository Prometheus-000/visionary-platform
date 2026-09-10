import type { Insight as InsightData } from '../api/types'

/**
 * What this dataset is actually teaching the model.
 *
 * Trigger coverage first, because a caption without the trigger word trains a LoRA you
 * cannot summon — and it is a ratio *over captions*, so it says nothing until there are
 * captions: "0/0 have the trigger" reads as a failure rather than a not-yet.
 *
 * Human-error checks come next. Identical captions and tag-style captions are *defects*;
 * the clause list below them is not — repeated clauses are what the set is teaching, and
 * whether that is right is a judgement only you can make. That is why the clauses are
 * ranked rather than flagged.
 *
 * **What a caption names is what the model learns is free to vary; what it never names is
 * what the trigger word ends up owning.** Which is why clicking a clause isolates the
 * images that carry it: the question is never "is this phrase frequent", it is "is this
 * phrase on the pictures I meant".
 */
export function Insight({ data, onIsolate, onIsolatePhrase }: {
  data: InsightData | null
  onIsolate: (names: string[], label: string) => void
  /** By phrase rather than by name, because which images carry it is a question about
   *  the captions on screen — which the editor holds and this panel does not. */
  onIsolatePhrase: (phrase: string) => void
}) {
  if (!data) return null
  const pct = (n: number, t: number) => (t ? Math.round((n / t) * 100) : 0)

  const meter = (n: number, t: number) => {
    const p = pct(n, t)
    const cls = p >= 100 ? '' : p >= 80 ? 'warn' : 'bad'
    return <div className="meter"><i className={cls} style={{ width: `${p}%` }} /></div>
  }

  const dupImgs = (data.duplicates ?? []).flatMap((x) => x.images)
  const max = data.phrases?.[0]?.count || 1

  return (
    <div id="ins-body" style={{ marginTop: 13 }}>
      {data.trigger_word && data.captioned > 0 && (
        <>
          <div className="stat">
            <b>{data.with_trigger}/{data.captioned}</b><span>have the trigger</span>
          </div>
          {meter(data.with_trigger, data.captioned)}
        </>
      )}
      <div className="stat">
        <b>{data.captioned}/{data.images}</b><span>captioned</span>
      </div>
      {meter(data.captioned, data.images)}

      {!!data.median_words && (
        <p className="muted" style={{ margin: '-6px 0 14px', fontSize: 12 }}>
          median {data.median_words} words
          {data.thin.length ? ` · ${data.thin.length} thin` : ''}
        </p>
      )}

      {!!dupImgs.length && (
        <button className="ph-row" type="button" style={{ marginBottom: 6 }}
                onClick={() => onIsolate(dupImgs, `${dupImgs.length} identical captions`)}>
          <span className="ph-bar">
            <i className="bad" style={{ ['--w' as string]: '100%' }} />
            <span>{dupImgs.length} identical captions</span>
          </span>
        </button>
      )}
      {!!data.tag_style?.length && (
        <button className="ph-row" type="button" style={{ marginBottom: 6 }}
                onClick={() => onIsolate(data.tag_style, `${data.tag_style.length} written as tags`)}>
          <span className="ph-bar">
            <i className="warn" style={{ ['--w' as string]: '100%' }} />
            {/* Prose, not tags — the text encoders these models use parse grammar. */}
            <span>{data.tag_style.length} written as tags</span>
          </span>
        </button>
      )}

      {!!data.phrases?.length && (
        <>
          <label style={{ marginTop: 4 }}>Repeated clauses</label>
          {data.phrases.map((p) => (
            <button className="ph-row" type="button" key={p.phrase}
                    onClick={() => onIsolatePhrase(p.phrase)}>
              <span className="ph-bar">
                <i className={p.share >= 0.6 ? 'hot' : ''}
                   style={{ ['--w' as string]: `${Math.round((p.count / max) * 100)}%` }} />
                <span>{p.phrase}</span>
              </span>
              <span className="ph-n">{p.count}</span>
            </button>
          ))}
          <p className="muted" style={{ marginTop: 8, fontSize: 11 }}>
            Click to see where it repeats.
          </p>
        </>
      )}
    </div>
  )
}

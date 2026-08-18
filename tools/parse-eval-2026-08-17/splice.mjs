import { readFileSync } from 'node:fs'

// ── the page's real gate, unchanged ─────────────────────────────────────────
const merge = (m) => { const s=[...m].filter(([a,b])=>b>a).sort((x,y)=>x[0]-y[0]); const o=[]
  for (const [a,b] of s){ const l=o[o.length-1]; if(l&&a<=l[1]) l[1]=Math.max(l[1],b); else o.push([a,b]) } return o }
const EDGES=/^[\s.,;:!?]+|[\s.,;:!?]+$/g, edges=(s)=>s.replace(EDGES,'')
const fold=(s)=>{const l=s.toLowerCase();return l.length===s.length?l:s}
function gaps(marks,len){const o=[];let at=0
  for(const [a,b] of marks){if(a>at)o.push([at,a]);at=Math.max(at,b)} if(at<len)o.push([at,len]);return o}
function insertionOnly(typed,text,marks){const hay=fold(typed),taken=[]
  for(const [a,b] of gaps(marks,text.length)){const run=edges(fold(text.slice(a,b)));if(!run)continue
    let start=0
    for(;;){const i=hay.indexOf(run,start);if(i<0)return false;const end=i+run.length
      if(!taken.some(([x,y])=>x<end&&i<y)){taken.push([i,end]);break} start=i+1}}
  return true}

// ── the proposal: construct the insertion, never rejoin ─────────────────────
//
// The prose targets get the person's sentence back, byte for byte, with the
// model's invented clauses spliced in at the anchor they follow. The output is
// an insertion *by construction* rather than by luck, which is the whole point:
// `insertionOnly` cannot fail on a string built this way, and
// `_document_matches` compares against this same function so it matches too.
const flat = (els) => els.flatMap((e) => [e, ...flat(e.children ?? [])])

let mismarked = 0
function splice(typed, elements) {
  const hay = fold(typed)
  const adds = []          // [offsetInTyped, textToInsert]
  let cursor = 0           // how far through the prose the document has got
  for (const e of flat(elements)) {
    const text = e.text ?? ''
    if (!text) continue
    const inv = e.invented ?? []
    const wholly = inv.length === 1 && inv[0][0] === 0 && inv[0][1] === text.length
    // **A run marked invented whose words are already in the prose is not
    // invention.** It is the model mis-marking its own copying, and both
    // candidates did it — "in a chair" and "walks toward two guards" are the
    // person's words, marked `invented`, and splicing them duplicates the
    // clause. `insertionOnly` cannot catch this because it only checks the
    // derived gaps: whatever is marked invented is waved through by definition,
    // so the gate trusts exactly the claim the model is worst at.
    if (wholly && hay.includes(fold(edges(text)))) { 
      const i = hay.indexOf(fold(edges(text)), cursor)
      if (i >= 0) cursor = i + edges(text).length
      mismarked++
      continue
    }
    if (wholly) { adds.push([cursor, text]); continue }
    // A derived (or mixed) element anchors: find it and advance the cursor. A
    // mixed clause's invented runs ride along inside it, which is why only the
    // wholly-invented case needs splicing — the rest is already the prose.
    const i = hay.indexOf(fold(edges(text)), cursor)
    if (i >= 0) cursor = i + edges(text).length
  }
  if (!adds.length) return { text: typed, invented: [] }

  let out = '', at = 0
  const invented = []
  for (const [pos, add] of adds.sort((a, b) => a[0] - b[0])) {
    out += typed.slice(at, pos)
    const sep = out.trim().endsWith(',') || !out.trim() ? ' ' : ', '
    const start = out.length + sep.length
    out += sep + add
    invented.push([start, out.length])
    at = pos
  }
  out += typed.slice(at)
  return { text: out, invented: merge(invented) }
}

// ── measure it on the real answers already collected ────────────────────────
let n=0, pass=0, hadInv=0, keptInv=0
const show = []
for (const file of ['funnel_fragments.json','funnel_finished.json']) {
  for (const r of JSON.parse(readFileSync(new URL(file, import.meta.url), 'utf8'))) {
    const el = r.elements || []
    if (!el.length) continue
    n++
    const s = splice(r.prose, el)
    const ok = insertionOnly(r.prose, s.text, s.invented)
    pass += ok
    const anyInv = flat(el).some((e) => (e.invented ?? []).length)
    hadInv += anyInv
    keptInv += anyInv && s.invented.length > 0
    if (!ok || (anyInv && show.length < 3)) show.push([ok, r.prose, s.text])
  }
}
console.log(`  documents with a parse            ${n}`)
console.log(`  insertionOnly passes on splice    ${pass}/${n}  (${Math.round(100*pass/n)}%)`)
console.log(`  had invented content              ${hadInv}/${n}`)
console.log(`  invention survived the splice     ${keptInv}/${hadInv || 1}`)
console.log(`  runs marked invented that were    ${mismarked}   <- dropped, not spliced`)
console.log(`    already the person's words`)
console.log('\n  examples')
for (const [ok, typed, text] of show) {
  console.log(`    ${ok ? 'pass' : 'FAIL'}  typed: ${typed.replace(/\n/g,' ⏎ ').slice(0,60)}`)
  console.log(`          out:   ${text.replace(/\n/g,' ⏎ ').slice(0,60)}`)
}

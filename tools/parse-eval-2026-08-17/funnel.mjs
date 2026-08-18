// The page's own gate, transcribed from web/src/console/marks.ts so the funnel
// is measured against what actually ships rather than a paraphrase of it.
const merge = (marks) => {
  const sorted = [...marks].filter(([a,b]) => b > a).sort((x,y) => x[0]-y[0])
  const out = []
  for (const [a,b] of sorted) {
    const last = out[out.length-1]
    if (last && a <= last[1]) last[1] = Math.max(last[1], b); else out.push([a,b])
  }
  return out
}
function documentMarks(prompt, elements) {
  const invented = [], spans = []; let at = 0
  const walk = (e) => {
    const i = e.text ? prompt.indexOf(e.text, at) : -1
    if (i >= 0) { at = i + e.text.length; spans.push([i, at])
      for (const [a,b] of e.invented ?? []) invented.push([i+a, i+b]) }
    ;(e.children ?? []).forEach(walk)
  }
  elements.forEach(walk)
  return { invented: merge(invented), spans: merge(spans) }
}
const EDGES = /^[\s.,;:!?]+|[\s.,;:!?]+$/g
const edges = (s) => s.replace(EDGES, '')
const fold  = (s) => { const l = s.toLowerCase(); return l.length === s.length ? l : s }
function gaps(marks, len) {
  const out = []; let at = 0
  for (const [a,b] of marks) { if (a > at) out.push([at,a]); at = Math.max(at,b) }
  if (at < len) out.push([at,len]); return out
}
function insertionOnly(typed, text, marks) {
  const hay = fold(typed); const taken = []
  for (const [a,b] of gaps(marks, text.length)) {
    const run = edges(fold(text.slice(a,b))); if (!run) continue
    let start = 0
    for (;;) {
      const i = hay.indexOf(run, start); if (i < 0) return false
      const end = i + run.length
      if (!taken.some(([x,y]) => x < end && i < y)) { taken.push([i,end]); break }
      start = i + 1
    }
  }
  return true
}

import { readFileSync } from 'node:fs'
const rows = JSON.parse(readFileSync(process.argv[2], 'utf8'))
let fired = 0, written = 0, grey = 0, reaches = 0
console.log(`\n  ${'fires'.padEnd(6)}${'writes'.padEnd(8)}${'grey'.padEnd(6)}${'renders'.padEnd(9)} fragment`)
console.log('  ' + '─'.repeat(72))
for (const r of rows) {
  const el = r.elements || []
  const f = el.length > 0
  const marks = f ? documentMarks(r.text || '', el) : { invented: [], spans: [] }
  // The box write is the gate. When it passes, the box holds the compiled text
  // and `_document_matches` then matches by construction, so the document
  // reaches the render. When it is refused, `doc.for` stays the typed prose and
  // the server's round-trip check drops the document at generate time.
  const w = f && insertionOnly(r.prose, r.text || '', marks.invented)
  const g = f && marks.invented.length > 0
  fired += f; written += w; grey += g; reaches += w
  const tick = (b) => b ? ' yes' : ' —  '
  console.log(`  ${tick(f).padEnd(6)}${tick(w).padEnd(8)}${tick(g).padEnd(6)}${tick(w).padEnd(9)} ${JSON.stringify(r.prose).slice(0,44)}`)
}
const n = rows.length, pc = (x) => `${x}/${n}  (${Math.round(100*x/n)}%)`
console.log(`\n  parse returns a document   ${pc(fired)}`)
console.log(`  the box accepts the write  ${pc(written)}`)
console.log(`  a grey run is visible      ${pc(grey)}`)
console.log(`  the document reaches a render ${pc(reaches)}`)

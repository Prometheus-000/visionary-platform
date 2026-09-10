/**
 * Read the workflow out of a PNG, the way ComfyUI writes it.
 *
 * Every Playground render embeds its API-format graph in a `prompt` tEXt
 * chunk — ComfyUI's own key — so the file *is* the workflow and dropping it
 * back onto the room reopens the graph that made it. This is the reader for
 * that drop, and for PNGs made by a stock ComfyUI, which writes the same key.
 *
 * Hand-rolled chunk walk rather than a dependency: PNG chunks are
 * length/type/data/crc and this needs exactly two text keys out of them.
 * Compressed iTXt/zTXt are not read — ComfyUI writes neither — and a graph
 * that only exists compressed reports itself rather than parsing as absent.
 */
import type { PgGraph } from '../store'

const SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]

function textChunks(buf: ArrayBuffer): Record<string, string> {
  const view = new DataView(buf)
  const bytes = new Uint8Array(buf)
  if (bytes.length < 8 || SIGNATURE.some((b, i) => bytes[i] !== b)) {
    throw new Error('Not a PNG — the embedded-workflow drop reads PNGs only.')
  }
  const out: Record<string, string> = {}
  let at = 8
  while (at + 8 <= bytes.length) {
    const len = view.getUint32(at)
    const type = String.fromCharCode(...bytes.subarray(at + 4, at + 8))
    const data = bytes.subarray(at + 8, at + 8 + len)
    if (type === 'tEXt') {
      const sep = data.indexOf(0)
      if (sep > 0) {
        out[String.fromCharCode(...data.subarray(0, sep))] =
          new TextDecoder('latin1').decode(data.subarray(sep + 1))
      }
    } else if (type === 'iTXt') {
      const sep = data.indexOf(0)
      if (sep > 0 && data[sep + 1] === 0) {
        // keyword \0 compression=0 \0 method \0 lang \0 translated \0 text
        let p = sep + 3
        for (let fields = 0; fields < 2 && p < data.length; p++) {
          if (data[p] === 0) fields++
        }
        out[String.fromCharCode(...data.subarray(0, sep))] =
          new TextDecoder().decode(data.subarray(p))
      }
    } else if (type === 'IEND') {
      break
    }
    at += 12 + len
  }
  return out
}

/** The graph, or an error that says what the file actually carries. */
export async function graphFromPng(file: File): Promise<PgGraph> {
  const chunks = textChunks(await file.arrayBuffer())
  const raw = chunks['prompt']
  if (!raw) {
    if (chunks['workflow']) {
      // The UI-format export: node positions and widget arrays, no
      // class_type inputs. Converting it is a project of its own, so the
      // error names the fix instead of pretending the file is empty.
      throw new Error(
        `${file.name} carries a UI-format workflow (the "workflow" key). `
        + 'The Playground reads API-format graphs — in ComfyUI, load the '
        + 'file and use Export (API), or drop a render made here.')
    }
    throw new Error(
      `${file.name} has no workflow in it — no "prompt" chunk. Renders made `
      + 'in the Playground and API-format ComfyUI PNGs both carry one.')
  }
  let graph: unknown
  try {
    graph = JSON.parse(raw)
  } catch {
    throw new Error(`${file.name}'s workflow chunk is not valid JSON.`)
  }
  if (!graph || typeof graph !== 'object' || Array.isArray(graph)
      || !Object.keys(graph as object).length) {
    throw new Error(`${file.name}'s workflow chunk is not a graph.`)
  }
  return graph as PgGraph
}

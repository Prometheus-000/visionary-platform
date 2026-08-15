/**
 * Typed prose split into elements — a **stand-in** for `/api/parse`.
 *
 * Deliberately dumb: it breaks on sentence ends and line breaks and nothing
 * else. The real parse is what decides which elements are anchors, what hangs
 * off what, and what was invented rather than derived — and that instruction is
 * blocked on experiments, so faking intelligence here would be inventing an
 * answer to the one question still open. What this gives is the structure made
 * visible and manipulable; the smart version replaces this file alone.
 */
import { mod, type Module } from './model'

export function segment(text: string): Module[] {
  const parts = text
    .split(/\n+/)
    .flatMap((line) => line.split(/(?<=[.!?])\s+/))
    .map((s) => s.trim())
    .filter(Boolean)
  return parts.length ? parts.map((p) => mod(p)) : [mod('')]
}

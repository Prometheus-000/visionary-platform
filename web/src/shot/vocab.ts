/**
 * Which pills the thing in front of you can actually read.
 *
 * Two different reasons a pill might not be read, and they are worth different
 * words: the image side has no camera and no soundtrack, and Wan is silent.
 *
 * **Per item as well as per group**, because one item disagrees with its group and
 * it is the one that matters: on-screen text is a picture of words and every model
 * can draw it, while a spoken line is audio. Given a group and no item, the answer
 * is whether *any* of its items is live — which is what decides whether the whole
 * section is dim.
 */
import type { ShotGroup, ShotItem } from '../api/types'
import type { Store } from '../store'
import { supports, videoModel } from '../store'

export function shotLive(
  s: Pick<Store, 'state' | 'kind' | 'vid'>,
  g: ShotGroup,
  it?: ShotItem,
): boolean {
  if (s.kind === 'image') return !!g.image
  if (!it) return g.items.some((x) => shotLive(s, g, x))
  const need = it.needs ?? g.needs
  return !need || !!supports(s)[need]
}

export function shotWhy(s: Pick<Store, 'state' | 'kind' | 'vid'>): string {
  if (s.kind === 'image') return 'video only'
  const m = videoModel(s)
  return m ? `${m.label} is silent` : ''
}

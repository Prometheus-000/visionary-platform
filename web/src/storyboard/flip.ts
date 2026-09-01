import { useLayoutEffect, useRef, type RefObject } from 'react'

/**
 * Make a reorder *travel*.
 *
 * When the panels' order changes, React moves the elements and every panel
 * jumps to its new slot in one frame — which reads as the board flickering
 * rather than a panel being put down somewhere. FLIP: remember where each
 * element was, let React lay out the new order, then start each one at its
 * old position and let a transition carry it to the new one. Keyed by the
 * `data-id` on the element, so a panel's identity survives its index changing.
 *
 * Held here rather than pulled from the `motion` package because the wall's
 * layout is a wrapping row and the reorderable list that package offers is
 * one axis only — and forty lines is cheaper to own than to work around.
 */
export function useFlip(root: RefObject<HTMLElement | null>, selector: string, dep: unknown) {
  const prev = useRef<Map<string, DOMRect>>(new Map())
  useLayoutEffect(() => {
    const el = root.current
    if (!el) return
    const next = new Map<string, DOMRect>()
    el.querySelectorAll<HTMLElement>(selector).forEach((node) => {
      const id = node.dataset.id
      if (!id) return
      const r = node.getBoundingClientRect()
      next.set(id, r)
      const was = prev.current.get(id)
      if (!was) return
      const dx = was.left - r.left
      const dy = was.top - r.top
      if (!dx && !dy) return
      node.style.transition = 'none'
      node.style.transform = `translate(${dx}px,${dy}px)`
      // Two frames: the first commits the offset, the second starts the ride.
      requestAnimationFrame(() => requestAnimationFrame(() => {
        node.style.transition = 'transform .2s cubic-bezier(.2,.7,.2,1)'
        node.style.transform = ''
      }))
    })
    prev.current = next
  }, [root, selector, dep])
}

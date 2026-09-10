import { useCallback, useState } from 'react'

/**
 * One pending flag for a component's mutations, keyed so several buttons can share it.
 *
 * Eight mutations across Settings, Sets, Train and the gallery did their work between a
 * click and a reply with nothing on screen in between: the token save, both LoRA and set
 * and session deletes, the set save, the trigger prepend, and the gallery's delete and
 * purge. A press that appears to do nothing is pressed again — which for the set save
 * meant two writes, and for a delete meant a second confirm dialog for a file already on
 * its way out.
 *
 * The shape is not new — the rewrite button did exactly this with its own flag since
 * it shipped — one key in flight, the others shut, the live one saying so — and the
 * downloads, captioning and duplicates panels each grew their own version of it. This is
 * that pattern with a name, so the ninth one does not have to be invented again.
 *
 * Keyed rather than boolean because the components that need it have more than one
 * button: Settings can add a caption model and remove one, and a row of ✕ buttons is a
 * delete per card. The key says *which* is running, so the live button can say
 * "Deleting…" while its neighbours only go quiet.
 *
 * `run` refuses while anything is in flight, which is the double-submit guard: the check
 * is on the flag rather than on the button being disabled, because `disabled` is a paint
 * and the keyboard, a double-click and a queued event can all get past a paint.
 */
export function useBusy() {
  const [busy, setBusy] = useState<string | null>(null)

  const run = useCallback(async (key: string, fn: () => Promise<unknown>) => {
    // Reading the flag rather than a ref: React batches the setState, but every caller
    // here awaits a network round trip, so the re-render lands long before the reply.
    if (busy) return
    setBusy(key)
    try {
      await fn()
    } finally {
      // Always, including on a throw. A pending flag that survives its own failure is a
      // panel that can never be used again without a reload.
      setBusy(null)
    }
  }, [busy])

  return { busy, run }
}

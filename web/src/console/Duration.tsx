import { warm } from '../api/routes'
import { Menu } from '../ui/Menu'
import { usePopover } from '../ui/Popover'
import { useStore } from '../store'

/**
 * How long. **Still is zero, and video is what happens above it.**
 *
 * This replaces `#kind-toggle`, and the replacement is the point rather than a
 * relabelling. That control was a mode switch: a 20px glyph, no word, showing the
 * state you were *in* rather than the one you would get — so its tooltip had to say
 * both ("Image — tap for video") and everybody clicked once to find out what it did,
 * on a control that rebuilds the entire strip. Half the application sat behind it and
 * nothing on the page said the word "video" until after you had switched.
 *
 * Measured, the two strips differ by exactly one control: a duration picker. Size,
 * LoRA, Shot, Enhance, the model button and Generate are common to both and only
 * carry different values. So the mode was a mode for one parameter, and CLAUDE.md
 * already names the way out:
 *
 * > **Duration starts at zero.** A still is the default and time is added. Someone
 * > who wants one image should finish and leave without learning that motion exists.
 *
 * Which is what this is. At `Still` the run is Krea 2; pick a number and it is H3.
 * Video stops being a place you navigate to and becomes a value you set — and
 * that is what makes it findable, because a control reading "Still" invites a press
 * in a way an unlabelled photo glyph never did, while somebody who only wants a
 * picture never has to learn what the other values are for.
 *
 * It is also the stronger reading of the rule the chip was written for — *"which one
 * you get is a property of what you are making, not an address you navigate to"*.
 * Duration is that property. The chip was an address wearing its costume.
 *
 * **Leftmost, where the chip was.** It decides what every control to its right
 * *means* — which sizes exist, which model loads, whether LoRAs load at all — so it
 * reads before them, and the one control in the row with that power is the one that
 * should be read first.
 */
export function Duration() {
  const s = useStore()
  const pop = usePopover()

  const m = s.state?.video_models.find((x) => x.key === s.vid.model)
  const lengths = m?.lengths ?? []
  // The same fallback `SecondsSelect` made, kept because the store holds whatever was
  // last picked and a model swap can leave a length the new model does not offer.
  const cur = lengths.map(String).includes(s.vid.seconds)
    ? s.vid.seconds
    : String(m?.defaults.seconds ?? lengths[0] ?? '')

  const still = s.kind === 'image'
  // Nothing to offer above zero, so the control states the one thing that is true and
  // does not open onto an empty list. Disabled rather than absent: the strip must not
  // reflow when a video model finishes downloading, and a door that is visibly shut
  // still says the feature exists.
  const none = !lengths.length

  return (
    <>
      <button className={`opt ib${pop.open ? ' on' : ''}`} id="g-duration" type="button"
              disabled={none}
              title="How long. Still is a photograph; anything above it is a clip."
              onClick={pop.toggle}>
        {still ? 'Still' : `${cur}s`}
      </button>
      {pop.open && (
        <Menu anchor={pop.anchor} onClose={pop.close}
              items={[
                {
                  label: 'Still',
                  on: still,
                  run: () => {
                    s.setKind('image')
                    // The switch is the earliest honest signal of which
                    // container this session is about to want, and Enhance
                    // lives on that container. Warming here is what turns the
                    // first press after a switch into a press rather than a
                    // wait. Fire and forget; a repeat is a no-op on a warm one.
                    void warm('image')
                  },
                },
                ...lengths.map((n) => ({
                  label: `${n}s`,
                  on: !still && String(n) === cur,
                  run: () => {
                    // Both, and in this order: the seconds are what the video side
                    // will read on its first render, so writing them after the kind
                    // would paint one frame of the previous length.
                    s.setVid({ seconds: String(n) })
                    s.setKind('video')
                    void warm('video')
                  },
                })),
              ]} />
      )}
    </>
  )
}

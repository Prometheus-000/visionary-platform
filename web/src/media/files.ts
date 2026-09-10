/**
 * Turning a file into the base64 the routes take.
 *
 * Reading a File is needed by six entry points — the two keyframe tiles, the two
 * plates, a region's photo, the reference tray, and the hand-off from finished work
 * — so it lives here rather than six times over.
 */

export const toB64 = (f: Blob): Promise<string | null> =>
  new Promise((res) => {
    const r = new FileReader()
    r.onload = () => res(String(r.result).split(',')[1] ?? null)
    r.onerror = () => res(null)
    r.readAsDataURL(f)
  })

/**
 * 1536px on the long side and JPEG when it resizes, because **nine** photographs
 * in one JSON body is the payload this feature invites — nine being H3's own
 * maximum, so it is the case to size for rather than the unlucky one.
 *
 * On the image side this is the only cap there is: the node's own `ref_max_side` is
 * set to 0 in the graph so the resizing happens in exactly one place and the two
 * cannot end up fighting over which one shrank the picture. The video references
 * reuse it and are not that case — `H3_REF_MAX_SIDE` caps the staged file again, at
 * the same number, because the gallery's "Use as reference" hand-off never passes
 * through here and H3 has no `ref_max_side` to switch off. The two agreeing is not
 * what makes that correct; **the server binding is.** If they ever drift, the
 * picture is still capped on the side that decides.
 */
export const REF_MAX = 1536

export function shrinkB64(file: File): Promise<string | null> {
  return new Promise((res) => {
    const img = new Image()
    const url = URL.createObjectURL(file)
    img.onload = () => {
      URL.revokeObjectURL(url)
      // Under the cap the bytes go verbatim, because the canvas only knows how to
      // hand back PNG and a 224 KB JPEG that needed no resizing came out of it at
      // 1.9 MB — 8.6x, for a picture nothing had asked to change. Nine of those is
      // 17 MB of base64 to save 2, which is the fix costing more than the thing it
      // fixes. Same rule the volume uses for the orientation tag and
      // `_fit_reference` for the size: rewrite only what is out of spec, pass the
      // rest through untouched.
      if (img.width <= REF_MAX && img.height <= REF_MAX) {
        void toB64(file).then(res)
        return
      }
      const k = REF_MAX / Math.max(img.width, img.height)
      const c = document.createElement('canvas')
      c.width = Math.round(img.width * k)
      c.height = Math.round(img.height * k)
      c.getContext('2d')?.drawImage(img, 0, 0, c.width, c.height)
      // **JPEG, and the note above is why.** That note measured PNG at 8.6x a
      // photograph's own encoding and drew the right conclusion for the
      // pass-through case only — the resize path kept emitting PNG, which is
      // the path every phone photo takes. Nine references is H3's maximum and
      // somebody hitting it sent 40-60 MB of base64 in one body, through the
      // web container, through Modal's blob store and back down to the GPU.
      //
      // Lossless buys nothing here: every reference is read at `LoadImage`'s
      // index 0 and the MASK output is taken by no graph in this file, so alpha
      // is discarded downstream whatever is sent. 0.92 on a 1536px photograph
      // is a few hundred KB against several MB.
      res(c.toDataURL('image/jpeg', 0.92).split(',')[1] ?? null)
    }
    img.onerror = () => {
      URL.revokeObjectURL(url)
      res(null)
    }
    img.src = url
  })
}

export const dataUrl = (b64: string, mime = 'image/png') => `data:${mime};base64,${b64}`

/** Fetch bytes back off the volume. The canvas stills are a streamed `<img src>`,
 *  so the base64 the video side needs does not exist client-side — a gallery card
 *  hands off the same bytes through the same route. */
export async function fileToB64(url: string): Promise<string | null> {
  try {
    return await toB64(await (await fetch(url)).blob())
  } catch {
    return null
  }
}

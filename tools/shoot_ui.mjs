/**
 * The README's screenshots, taken the same way every time.
 *
 * A headless browser rather than a screen capture, which is the distinction
 * that matters: `screencapture` has no concept of a tab, so it photographs
 * whatever is genuinely on the display and catches whatever window happens to
 * be in front. This addresses the page, so a shot is reproducible, needs nobody
 * at the keyboard, and cannot pick up anything that is not the app.
 *
 *   node tools/shoot_ui.mjs https://…modal.run
 *
 * `deviceScaleFactor: 2` because the images it replaces are retina and a README
 * that mixes the two looks like a mistake rather than a choice.
 */
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'

const URL = process.argv[2] || 'http://localhost:5173'
// Re-shooting one frame should not cost the other six, several of which
// generate. `node tools/shoot_ui.mjs <url> generate` takes just that one.
const ONLY = process.argv.slice(3)
const OUT = 'docs'
const VIEW = { width: 1440, height: 900 }

const FRAGMENT = 'empty diner, 3am'

const VIDEO =
  'Two steel battleships trade broadsides across a heavy grey sea, the nearer ' +
  'one close enough that the red letters painted along its bow fill a third of ' +
  'the frame. Its guns fire a full salvo; the far ship is struck amidships, ' +
  'erupts in flame and begins to list.'

const SOUND =
  'cannon fire and its concussion rolling across open water, heavy swell ' +
  'against steel, wind over the deck, fire roaring on the far hull'

const VIDEO_STILL =
  'A woman at a rain-streaked window in a dim room, turning to look straight ' +
  'down the lens, one hard light from the street outside raking across her ' +
  'face, editorial fashion photography, shallow depth of field.'

const REGIONAL_STILL =
  'Two friends side by side on a fire escape at dusk, city behind them going ' +
  'violet, shot on 35mm with a hard flash, editorial portrait.'

const HERO =
  'High-fashion studio editorial photography. A model wearing a massive, ' +
  'sculptural avant-garde gown made entirely of liquid-like molten chrome and ' +
  'deep crimson velvet. The chrome fabric perfectly reflects a grid of neon ' +
  'pink and electric blue studio softboxes. High-contrast studio lighting, ' +
  'sharp reflections contrasting against the matte, light-absorbing properties ' +
  'of the velvet. Clean, solid charcoal gray background. Editorial composition, ' +
  'sharp focus, 8k resolution.'

const wait = (ms) => new Promise((r) => setTimeout(r, ms))

const SHOTS = [
  {
    // **The two shots want opposite prompts, which is why they are two runs.**
    // This one has to show a fragment *becoming* a prompt, so it starts from
    // four words and stops before Generate: an empty canvas keeps the eye on
    // the text, which is the whole event.
    name: 'enhance',
    async run(page, shoot) {
      await page.fill('#prompt', FRAGMENT)
      await page.waitForTimeout(300)
      // Scoped to `#c-image`, because both strips are in the DOM at once and
      // one is merely `.hide` — the same trap the duplicate `#go-gen` id was.
      const strip = page.locator('#c-image')
      await strip.getByRole('button', { name: 'Enhance' }).click()
      // The first press of a container's life loads the weights; later ones are
      // seconds. Wait on the affordance that only exists afterwards.
      await strip.getByRole('button', { name: 'Undo' }).waitFor({ timeout: 300000 })
      await page.waitForTimeout(600)
      await shoot('enhance')
    },
  },
  {
    // **The hero is the claim the rest of the README lives up to**, so it is
    // written rather than fragmentary and it renders. An empty grid sells the
    // opposite of "the canvas is the largest thing on screen".
    name: 'generate',
    async run(page, shoot) {
      await page.fill('#prompt', HERO)
      await page.waitForTimeout(300)
      await page.locator('#c-image').getByRole('button', { name: 'Generate' }).click()
      await page.locator('#canvas img, .frame img').first()
        .waitFor({ timeout: 600000 })
      await page.waitForTimeout(1500)
      await shoot('generate')
    },
  },
  {
    name: 'shot-palette',
    async run(page, shoot) {
      await page.fill('#prompt', 'a lone fisherman hauling a net')
      await page.waitForTimeout(300)
      await page.locator('#c-image').getByRole('button', { name: 'Shot' }).click()
      await page.waitForTimeout(700)
      await shoot('shot-palette')
    },
  },
  {
    // The video side, which differs from the image side by one control — so the
    // shot has to be of a duration actually chosen, with pills on the rail and
    // the compiled document open. A strip at rest would look identical.
    name: 'video',
    async run(page, shoot) {
      // **The console is what makes this shot worth having, not the render.**
      // The video side differs from the image side by one control, so a strip
      // at rest is indistinguishable — what is video-only is a camera move, and
      // what a closed vocabulary cannot contain is a line of on-screen text or
      // a written soundscape. Both are valued pills, both are here, and between
      // them they say more about the palette than any framing choice does.
      await page.fill('#prompt', VIDEO)
      await page.waitForTimeout(300)
      await page.click('#g-duration')
      await page.waitForTimeout(500)
      await page.getByRole('button', { name: '5s', exact: true }).click()
      await page.waitForTimeout(900)

      const strip = page.locator('#c-video')
      await strip.getByRole('button', { name: 'Shot' }).click()
      await page.waitForTimeout(800)
      for (const pill of ['wide', 'low', 'hard sun', 'anamorphic', 'track side']) {
        const t = page.getByRole('button', { name: pill, exact: true })
        if (await t.count()) { await t.first().click(); await page.waitForTimeout(320) }
      }
      // A valued pill opens an `input.v` rather than toggling — the whole reason
      // the rail can carry a line of dialogue at all.
      const valued = async (label, text) => {
        const t = page.getByRole('button', { name: label, exact: true })
        if (!(await t.count())) return
        await t.first().click()
        await page.waitForTimeout(600)
        const field = page.locator('input.v:visible').first()
        if (await field.count()) {
          await field.fill(text)
          await field.press('Enter')
          await page.waitForTimeout(600)
        }
      }
      await valued('on-screen text', 'KAOS')
      await valued('other', SOUND)
      await page.waitForTimeout(500)

      await page.keyboard.press('Escape')
      await page.waitForTimeout(600)
      await strip.getByRole('button', { name: 'Generate' }).click()
      await page.locator('#canvas video, .frame video').first()
        .waitFor({ timeout: 900000 })
      await page.waitForTimeout(3000)

      // With the panel, over the take it produced.
      await strip.getByRole('button', { name: 'Shot' }).click()
      await page.waitForTimeout(900)
      await page.mouse.wheel(0, -1200)
      await page.waitForTimeout(500)
      await shoot('video-shot')

      // And without — **and without the compiled document open**, which is the
      // correction. That disclosure is several lines of monospace against a
      // console capped at 30% of the viewport, and the canvas yields the
      // difference: opening it to show what a shot compiles to buried the take
      // it compiled to.
      await page.keyboard.press('Escape')
      await page.waitForTimeout(800)
      await shoot('video')
    },
  },
  {
    // **Boxes are drawn before the render and revealed after it.** A render
    // puts them away every time — `off` is re-entered on every land, which is
    // what keeps a finished picture clean — so a plain drag afterwards is not
    // "draw a box", it is a click on bare canvas. Geometry comes back with
    // ⌘, the same modifier that already meant "a new box, here".
    name: 'regional',
    async run(page, shoot) {
      await page.fill('#prompt', REGIONAL_STILL)
      await page.waitForTimeout(400)
      const frame = page.locator('#canvas, .frame').first()
      const box = await frame.boundingBox()
      if (!box) throw new Error('no frame to draw on')
      const at = (fx, fy) => [box.x + box.width * fx, box.y + box.height * fy]
      const drag = async (x1, y1, x2, y2) => {
        await page.mouse.move(...at(x1, y1))
        await page.mouse.down()
        await page.mouse.move(...at(x2, y2), { steps: 18 })
        await page.mouse.up()
        await page.waitForTimeout(500)
      }
      // On the empty frame a plain drag places one — this is the gesture the
      // empty canvas describes in words.
      await drag(0.10, 0.16, 0.47, 0.9)
      await drag(0.53, 0.16, 0.90, 0.9)

      await page.locator('#c-image').getByRole('button', { name: 'Generate' }).click()
      await page.locator('#canvas img, .frame img').first().waitFor({ timeout: 600000 })
      await page.waitForTimeout(1500)

      // ⌘-click to bring geometry back over the result, which is the whole
      // reason the boxes survive a render: you adjust them against the picture
      // you actually got.
      await page.keyboard.down('Meta')
      await page.mouse.click(...at(0.28, 0.5))
      await page.keyboard.up('Meta')
      await page.waitForTimeout(900)
      // Then open one, so the card is showing — a rectangle says nothing about
      // who is in it.
      await page.mouse.click(...at(0.28, 0.5))
      await page.waitForTimeout(900)
      await shoot('regional')
    },
  },
  {
    // The set, with the duplicate scan run — the README describes grouping and
    // its screenshot has never shown any.
    name: 'dataset',
    async run(page, shoot) {
      await page.getByRole('button', { name: 'Train' })
        .or(page.getByRole('link', { name: 'Train' })).first().click()
      await page.waitForTimeout(1500)
      await page.getByRole('button', { name: 'Sets' }).first().click()
      await page.waitForTimeout(1800)
      const set = page.getByText('editorial', { exact: false }).first()
      if (await set.count()) { await set.click(); await page.waitForTimeout(2000) }
      const scan = page.getByRole('button', { name: /duplicate|scan/i }).first()
      if (await scan.count()) {
        await scan.click()
        // The scan is resumable and the page calls again until it stops making
        // progress, so this waits on the result rather than a fixed sleep.
        await page.waitForTimeout(12000)
      }
      await shoot('dataset')
    },
  },
  {
    name: 'gallery',
    async run(page, shoot) {
      await page.click('#drawer-toggle, [title*="allery" i]').catch(() => {})
      await page.waitForTimeout(900)
      await shoot('gallery')
    },
  },
]

const browser = await chromium.launch()
const ctx = await browser.newContext({
  viewport: VIEW, deviceScaleFactor: 2, colorScheme: 'dark',
})
const page = await ctx.newPage()
mkdirSync(OUT, { recursive: true })

for (const shot of SHOTS) {
  if (ONLY.length && !ONLY.includes(shot.name)) continue
  await page.goto(URL, { waitUntil: 'networkidle' })
  await wait(1200)
  const shoot = async (name) => {
    const path = `${OUT}/${name}.png`
    await page.screenshot({ path })
    console.log(`  wrote ${path}`)
  }
  try {
    await shot.run(page, shoot)
  } catch (err) {
    console.log(`  skip ${shot.name}: ${String(err).split('\n')[0].slice(0, 90)}`)
  }
}
await browser.close()

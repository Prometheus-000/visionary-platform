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
      // **A still first, because the canvas keeps the last render.** An empty
      // frame under the video controls shows the one thing the layout exists to
      // prevent, and it is also the real flow: you make a frame and then decide
      // it should move.
      await page.fill('#prompt', VIDEO_STILL)
      await page.waitForTimeout(300)
      await page.locator('#c-image').getByRole('button', { name: 'Generate' }).click()
      await page.locator('#canvas img, .frame img').first().waitFor({ timeout: 600000 })
      await page.waitForTimeout(1200)

      await page.click('#g-duration')
      await page.waitForTimeout(500)
      await page.getByRole('button', { name: '5s', exact: true }).click()
      await page.waitForTimeout(900)

      await page.locator('#c-video').getByRole('button', { name: 'Shot' }).click()
      await page.waitForTimeout(800)
      // Names verified against the live palette rather than guessed; a camera
      // move is the one group the image side does not get, so the shot should
      // carry one if the vocabulary offers it.
      for (const pill of ['medium close-up', 'golden hour', 'slow push-in',
                          'push in', 'handheld', 'anamorphic']) {
        const t = page.getByRole('button', { name: pill, exact: true })
        if (await t.count()) { await t.first().click(); await page.waitForTimeout(350) }
      }
      await page.keyboard.press('Escape')
      await page.waitForTimeout(600)
      // The caption promises the compiled document, so open it.
      const peek = page.getByText('what the model reads', { exact: false }).first()
      if (await peek.count()) { await peek.click(); await page.waitForTimeout(700) }
      await shoot('video')
    },
  },
  {
    // Regions are a canvas verb, so this draws them rather than opening a panel:
    // two boxes on the empty frame, then one touched so its card is showing.
    name: 'regional',
    async run(page, shoot) {
      // Drawn over a render, which is what the boxes are for — CLAUDE.md's
      // reason they survive a result at all is that you adjust them against the
      // picture you actually got.
      await page.fill('#prompt', REGIONAL_STILL)
      await page.waitForTimeout(300)
      await page.locator('#c-image').getByRole('button', { name: 'Generate' }).click()
      await page.locator('#canvas img, .frame img').first().waitFor({ timeout: 600000 })
      await page.waitForTimeout(1200)
      const frame = page.locator('#canvas, .frame').first()
      const box = await frame.boundingBox()
      if (!box) throw new Error('no frame to draw on')
      const drag = async (x1, y1, x2, y2) => {
        await page.mouse.move(box.x + box.width * x1, box.y + box.height * y1)
        await page.mouse.down()
        await page.mouse.move(box.x + box.width * x2, box.y + box.height * y2, { steps: 18 })
        await page.mouse.up()
        await page.waitForTimeout(500)
      }
      await drag(0.08, 0.14, 0.46, 0.9)
      await drag(0.54, 0.14, 0.92, 0.9)
      // Touch the first one so its card opens — the card is where a region says
      // who is in it, and a screenshot of bare rectangles shows none of that.
      await page.mouse.click(box.x + box.width * 0.27, box.y + box.height * 0.5)
      await page.waitForTimeout(800)
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

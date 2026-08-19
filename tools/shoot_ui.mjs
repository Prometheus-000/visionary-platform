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
const OUT = 'docs'
const VIEW = { width: 1440, height: 900 }

const FRAGMENT = 'empty diner, 3am'

const HERO =
  'Black and white studio portrait, a single hard light raking from the left ' +
  'across a bare shoulder and the line of a jaw, the head turned three-quarters ' +
  'away, deep unbroken black behind. A calla lily held close to the collarbone, ' +
  'its edge catching the same light. Large-format detail, fine silver grain, ' +
  'skin rendered as sculpture, shadow doing most of the work.'

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

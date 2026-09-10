/**
 * The design-system entry: the five primitives that are not about this app.
 *
 * `web/` is an application, not a library, so nothing here was exported before.
 * This barrel exists so `tools/`-adjacent consumers — and claude.ai/design —
 * can build with the same parts the console does, against the same stylesheet.
 *
 * What is in it is decided by one test: does the component know anything about
 * Visionary? Popover, Menu, Sheet, NumInput and ErrorNote do not — they take
 * children, items, values and a message. Console, Canvas, Card and the rest
 * read the store, the API client and the job lifecycle, so they are the app
 * rather than the system, and exporting them would be exporting a screenshot.
 *
 * `ui.css` comes with them because these components carry no styling of their
 * own: every one of them is a classname the stylesheet draws. It is imported
 * here rather than copied, because `tools/extract_css.py` regenerates it
 * byte-for-byte and a copy would be stale the next time that runs.
 *
 * `root.css` is deliberately absent — it is `#root` flex geometry for the app
 * shell, which is the one thing in the stylesheet that is about this page
 * rather than about the parts.
 */
import '../styles/ui.css'

export { Popover, usePopover } from '../ui/Popover'
export { Menu } from '../ui/Menu'
export type { MenuItem } from '../ui/Menu'
export { Sheet } from '../ui/Sheet'
export { NumInput, step } from '../ui/NumInput'
export { ErrorNote } from '../ui/ErrorNote'
export type { ApiError } from '../api/client'

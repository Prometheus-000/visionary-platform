import { useEffect } from 'react'
import { createPortal } from 'react-dom'

/**
 * A modal panel: the metadata sheet, and Settings.
 *
 * Escape and a click on the scrim close it. The scrim test is `e.target === the
 * scrim itself` rather than "not inside the sheet", because the sheet is a
 * scroller and a drag that starts on its scrollbar and ends outside it is not a
 * click on the backdrop.
 */
export function Sheet({
  onClose,
  children,
  id,
}: {
  onClose: () => void
  children: React.ReactNode
  id?: string
}) {
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', key)
    return () => document.removeEventListener('keydown', key)
  }, [onClose])

  return createPortal(
    <div className="scrim" id={id}
         onClick={(e) => {
           if (e.target === e.currentTarget) onClose()
         }}>
      <div className="sheet">{children}</div>
    </div>,
    document.body,
  )
}

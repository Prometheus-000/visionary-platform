/**
 * A modal panel: the metadata sheet, and Settings.
 *
 * Escape and a click on the scrim close it. The scrim test is `e.target === the
 * scrim itself` rather than "not inside the sheet", because the sheet is a
 * scroller and a drag that starts on its scrollbar and ends outside it is not a
 * click on the backdrop.
 */
export declare function Sheet({ onClose, children, id, }: {
    onClose: () => void;
    children: React.ReactNode;
    id?: string;
}): import("react").ReactPortal;

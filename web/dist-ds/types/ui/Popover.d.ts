/**
 * The one floating element on this page, and the one place that knows how a
 * popover behaves.
 *
 * The LoRA picker, the shot palette, both size pickers, both sampling forms and
 * every overflow menu are this component with different children. That is not
 * tidiness: `floatBy` in the vanilla page was factored out because the
 * scroll-close guard below is a bug that was fixed once, and a second floating
 * element with its own copy of it is a second place for it to come back.
 *
 * Four behaviours, each earned:
 *
 * **Anchored, and clamped to the viewport.** A menu tall enough to need flipping
 * above its button is also tall enough for `top - h` to land off the top of the
 * window, and the rows that go past the edge are unreachable — the scrollbar is
 * inside the menu, so nothing scrolls them back.
 *
 * **Closes on outside mousedown, in the capture phase.** Not click: an expanded
 * pill's input has focus and its own focusout rebuilds the rail, so by the time
 * a click would fire the element it was aimed at has been replaced.
 *
 * **Closes on scroll — except its own.** A menu anchored to a button inside a
 * scrolling pane has to close when that pane moves under it. But the menu is
 * itself a scroller when the LoRA list is long, and its own scroll reaches a
 * capture-phase listener the same way. Without the guard the list closed on the
 * first wheel notch, which reads as "the picker cannot scroll" rather than "the
 * picker is closing".
 *
 * **Measures before it paints.** Position needs the rendered size, so the first
 * frame is drawn with `visibility:hidden` and `useLayoutEffect` places it before
 * the browser gets the chance to show it in the wrong spot.
 */
export declare function Popover({ anchor, className, onClose, children, }: {
    anchor: HTMLElement | null;
    /** `menu`, `menu sizer`, `menu form` or `pal` — the stylesheet selects on
     *  these, and they are kept verbatim because ui.css is regenerated. */
    className: string;
    onClose: () => void;
    children: React.ReactNode;
}): import("react").ReactPortal;
/**
 * The anchor half of every popover, held once.
 *
 * Every caller needs the same two things — which element the popover hangs off,
 * and whether it is open — and they are one value: an anchor is the open state.
 * Returning them separately invites the third state where a popover is open with
 * no anchor, which places it at 0,0 in the top-left corner of the window.
 */
export declare function usePopover(): {
    anchor: HTMLElement | null;
    open: boolean;
    close: () => void;
    /** Pass as `onClick`. Toggles, so a second press on the button that opened it
     *  closes it — which is what anyone does when they meant to look and not to
     *  choose. */
    toggle: (e: React.MouseEvent<HTMLElement>) => void;
    /** Open against an element you already have. The gallery's card menu needs this: the
     *  anchor is the card's own ⋯ button, but *which item* the menu is about is state the
     *  card cannot hold, so the handler sets both and cannot go through `toggle`'s
     *  event. */
    at: (el: HTMLElement) => void;
};

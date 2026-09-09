/**
 * A list of things you can do to one object.
 *
 * One floating menu, refilled — a menu rendered inside every card would put a
 * hundred hidden subtrees in a grid that is already a hundred images, and only
 * one of them can ever be open.
 *
 * `on` is a tick rather than a highlight, and the tick is what makes a second
 * click legible as a removal rather than a click that did nothing. It is used by
 * the LoRA picker (already in the prompt) and the reference role menu (this
 * picture's role), which are the two menus where an item has a state.
 *
 * `drag` makes an item a *source* as well as a command — the LoRA picker's rows,
 * which can be dropped on a box, the prompt or the bare frame instead of being
 * clicked. It is opt-in per item rather than a property of the menu because the
 * other two menus here act on an object that is already selected: there is
 * nowhere to carry them to.
 */
export type MenuItem = {
    sep: true;
} | {
    label: string;
    run: () => void;
    on?: boolean;
    danger?: boolean;
    sep?: false;
    /** A dim clause after the label — what picking this will actually write
     *  beyond the label itself. The LoRA picker puts the trigger phrase here,
     *  because a row that is about to write words into your sentence should
     *  show them first. */
    hint?: string;
    /** Written onto the drag as one private MIME type. See `lora/drag.ts`. */
    drag?: (e: React.DragEvent) => void;
};
export declare function Menu({ anchor, items, onClose, }: {
    anchor: HTMLElement | null;
    items: MenuItem[];
    onClose: () => void;
}): import("react").JSX.Element;

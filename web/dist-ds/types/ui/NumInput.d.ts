export declare function step(value: string, dir: 1 | -1, coarse: boolean, { fine, bigStep, base }?: {
    fine?: number;
    bigStep?: number;
    base?: number;
}): string | null;
/**
 * A numeric box in the console.
 *
 * Every one of them takes the arrows, which in the vanilla page was a delegated
 * listener on three sections — because region rows are built long after load,
 * and a handler attached only at startup is one every row added later silently
 * goes without. In React the component *is* the delegation.
 *
 * Enter blurs and then generates, because you have just typed a seed or a step
 * count and reaching for the mouse to commit it is the wrong ending. The blur is
 * first so a width still on its way to being snapped is snapped before the run
 * reads it.
 */
export declare function NumInput({ value, onValue, fine, bigStep, base, onEnter, onCommit, ...rest }: {
    value: string;
    onValue: (v: string) => void;
    fine?: number;
    bigStep?: number;
    /** What an empty box counts from — the checkpoint's own number. */
    base?: number;
    onEnter?: () => void;
    /** Blur and Enter, for the boxes that snap on the way out. */
    onCommit?: () => void;
} & Omit<React.InputHTMLAttributes<HTMLInputElement>, 'value' | 'onChange' | 'step'>): import("react").JSX.Element;

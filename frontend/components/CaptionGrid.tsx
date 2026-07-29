"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Dataset review — see every image, read its caption, fix it before training.
 *
 * Captions are the highest-leverage thing in a LoRA dataset and the easiest to
 * get silently wrong, so this makes them visible and editable rather than a
 * side effect of pressing Train.
 *
 * Edits are held locally and flushed on a debounce; the caption job writes .txt
 * sidecars next to the images, which is exactly what training reads.
 */

export type Backend = {
  uploadUrl: string;
  trainUrl: string;
  statusUrl: string;
  stopUrl: string;
  datasetUrl: string;
  captionUrl: string;
  saveCaptionsUrl: string;
};

type Item = { name: string; caption: string; thumb: string | null };

const STYLES = [
  { id: "descriptive", label: "Descriptive" },
  { id: "descriptive_casual", label: "Casual" },
  { id: "tags", label: "Tags" },
] as const;

const LENGTHS = [
  { id: "short", label: "Short" },
  { id: "medium", label: "Medium" },
  { id: "long", label: "Long" },
] as const;

export default function CaptionGrid({
  backend,
  accessKey,
  jobId,
  triggerWord,
  onBack,
  onTrain,
}: {
  backend: Backend;
  accessKey: string;
  jobId: string;
  triggerWord: string;
  onBack: () => void;
  onTrain: () => void;
}) {
  const [items, setItems] = useState<Item[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [captioning, setCaptioning] = useState(false);
  const [progress, setProgress] = useState<{ step: number; total: number } | null>(null);
  const [dirty, setDirty] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState<number | null>(null);
  const [style, setStyle] = useState<string>("descriptive");
  const [length, setLength] = useState<string>("medium");
  const [overwrite, setOverwrite] = useState(false);
  const [showOptions, setShowOptions] = useState(false);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const saveRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const post = useCallback(
    async (url: string, body: Record<string, unknown>) => {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...body, access_key: accessKey }),
      });
      const data = await res.json();
      if (data.status === "error") throw new Error(data.error);
      return data;
    },
    [accessKey],
  );

  const load = useCallback(async () => {
    try {
      const data = await post(backend.datasetUrl, { job_id: jobId });
      setItems(data.images ?? []);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    }
  }, [backend.datasetUrl, jobId, post]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(
    () => () => {
      if (pollRef.current) clearInterval(pollRef.current);
      if (saveRef.current) clearTimeout(saveRef.current);
    },
    [],
  );

  /** Flush pending edits. Also called before captioning and before training. */
  const flush = useCallback(
    async (edits: Record<string, string>) => {
      if (!Object.keys(edits).length) return;
      setSaving(true);
      try {
        await post(backend.saveCaptionsUrl, { job_id: jobId, captions: edits });
        setDirty((d) => {
          const next = { ...d };
          for (const k of Object.keys(edits)) if (next[k] === edits[k]) delete next[k];
          return next;
        });
        setSavedAt(Date.now());
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setSaving(false);
      }
    },
    [backend.saveCaptionsUrl, jobId, post],
  );

  const edit = (name: string, caption: string) => {
    setItems((prev) =>
      prev ? prev.map((i) => (i.name === name ? { ...i, caption } : i)) : prev,
    );
    const next = { ...dirty, [name]: caption };
    setDirty(next);
    if (saveRef.current) clearTimeout(saveRef.current);
    saveRef.current = setTimeout(() => flush(next), 1200);
  };

  const runCaptioning = async () => {
    if (captioning) return;
    if (saveRef.current) clearTimeout(saveRef.current);
    await flush(dirty); // never let the job overwrite unsaved hand edits

    setCaptioning(true);
    setError(null);
    setProgress(null);
    try {
      const { call_id } = await post(backend.captionUrl, {
        job_id: jobId,
        trigger_word: triggerWord,
        style,
        length,
        overwrite,
      });

      pollRef.current = setInterval(async () => {
        try {
          const s = await post(backend.statusUrl, { call_id, job_id: jobId });
          if (s.status === "running") {
            if (typeof s.step === "number")
              setProgress({ step: s.step, total: s.total_steps ?? 0 });
            return;
          }
          if (pollRef.current) clearInterval(pollRef.current);
          setCaptioning(false);
          setProgress(null);
          if (s.status === "completed") await load();
          else setError(s.error ?? "Captioning failed.");
        } catch {
          /* transient — keep polling */
        }
      }, 2500);
    } catch (err) {
      setError((err as Error).message);
      setCaptioning(false);
    }
  };

  const startTraining = async () => {
    if (saveRef.current) clearTimeout(saveRef.current);
    await flush(dirty);
    onTrain();
  };

  const total = items?.length ?? 0;
  const captioned = items?.filter((i) => i.caption.trim()).length ?? 0;
  const pendingEdits = Object.keys(dirty).length;

  return (
    <div>
      {/* Header */}
      <div className="mb-4 flex items-center justify-between gap-3">
        <button
          onClick={onBack}
          className="flex items-center gap-1 text-[13px] text-neutral-500 transition hover:text-neutral-300"
        >
          <BackIcon className="h-3.5 w-3.5" />
          Back
        </button>
        <span className="text-[12px] tabular-nums text-neutral-500">
          {captioned} / {total} captioned
        </span>
      </div>

      {/* Caption controls */}
      <div className="mb-4 rounded-2xl border border-white/10 bg-white/[0.02] p-4">
        <button
          onClick={runCaptioning}
          disabled={captioning || !total}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-white px-4 py-3 text-[14px] font-medium text-black transition active:scale-[0.985] disabled:cursor-not-allowed disabled:bg-white/25 disabled:text-black/40"
        >
          {captioning ? (
            <>
              <SpinnerIcon className="h-4 w-4 animate-spin" />
              {progress
                ? `Captioning ${progress.step}/${progress.total}…`
                : "Starting captioner…"}
            </>
          ) : (
            <>
              <WandIcon className="h-4 w-4" />
              Auto-caption
            </>
          )}
        </button>

        {captioning && progress && progress.total > 0 && (
          <div className="mt-3 h-0.5 w-full overflow-hidden rounded-full bg-white/10">
            <div
              className="h-full rounded-full bg-white/70 transition-all duration-500"
              style={{ width: `${(progress.step / progress.total) * 100}%` }}
            />
          </div>
        )}

        <button
          type="button"
          onClick={() => setShowOptions((v) => !v)}
          className="mt-2.5 flex items-center gap-1.5 text-[12px] text-neutral-500 transition hover:text-neutral-300"
        >
          <ChevronIcon
            className={`h-3 w-3 transition-transform ${showOptions ? "rotate-90" : ""}`}
          />
          Caption options
        </button>

        {showOptions && (
          <div className="mt-3 space-y-3">
            <Segmented label="Style" options={STYLES} value={style} onChange={setStyle} />
            <Segmented label="Length" options={LENGTHS} value={length} onChange={setLength} />
            <label className="flex items-center gap-2.5 pt-0.5">
              <input
                type="checkbox"
                checked={overwrite}
                onChange={(e) => setOverwrite(e.target.checked)}
                className="h-4 w-4 rounded border-white/20 bg-white/10 accent-white"
              />
              <span className="text-[13px] text-neutral-300">
                Replace existing captions
              </span>
            </label>
          </div>
        )}
      </div>

      {error && (
        <div className="mb-4 rounded-2xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-[13px] text-red-300">
          {error}
        </div>
      )}

      {/* Grid */}
      {items === null ? (
        <div className="py-12 text-center text-[13px] text-neutral-600">Loading…</div>
      ) : (
        <div className="space-y-3">
          {items.map((item) => (
            <div
              key={item.name}
              className="flex gap-3 rounded-2xl border border-white/10 bg-white/[0.02] p-3"
            >
              {item.thumb ? (
                // Data URI from our own backend; next/image would need remote
                // config and buys nothing for a 320px inline thumbnail.
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={item.thumb}
                  alt={item.name}
                  className="h-24 w-24 shrink-0 rounded-xl object-cover"
                />
              ) : (
                <div className="flex h-24 w-24 shrink-0 items-center justify-center rounded-xl bg-white/5 text-[11px] text-neutral-600">
                  no preview
                </div>
              )}
              <div className="flex min-w-0 flex-1 flex-col">
                <div className="mb-1 flex items-center justify-between gap-2">
                  <span className="truncate font-mono text-[11px] text-neutral-600">
                    {item.name}
                  </span>
                  {dirty[item.name] !== undefined && (
                    <span className="shrink-0 text-[10px] text-amber-400/70">edited</span>
                  )}
                </div>
                <textarea
                  value={item.caption}
                  onChange={(e) => edit(item.name, e.target.value)}
                  placeholder="No caption"
                  rows={3}
                  className="w-full flex-1 resize-none rounded-lg border border-white/10 bg-white/[0.04] px-2.5 py-2 text-[12px] leading-relaxed text-neutral-200 outline-none transition placeholder:text-neutral-600 focus:border-white/25"
                />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Footer */}
      <div className="sticky bottom-0 -mx-5 mt-5 border-t border-white/10 bg-neutral-950/95 px-5 pb-1 pt-4 backdrop-blur">
        <button
          onClick={startTraining}
          disabled={!total || captioning}
          className="w-full rounded-2xl bg-white px-4 py-3.5 text-[15px] font-medium text-black transition active:scale-[0.985] disabled:cursor-not-allowed disabled:bg-white/25 disabled:text-black/40"
        >
          Start training
        </button>
        <p className="mt-2 h-4 text-center text-[11px] text-neutral-600">
          {saving
            ? "Saving…"
            : pendingEdits
              ? `${pendingEdits} unsaved`
              : savedAt
                ? "Captions saved"
                : captioned < total
                  ? `${total - captioned} without a caption`
                  : ""}
        </p>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */

function Segmented<T extends readonly { id: string; label: string }[]>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: T;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div>
      <span className="mb-1.5 block text-[12px] text-neutral-500">{label}</span>
      <div className="flex gap-1.5">
        {options.map((o) => (
          <button
            key={o.id}
            onClick={() => onChange(o.id)}
            className={`flex-1 rounded-lg px-2 py-1.5 text-[12px] transition ${
              value === o.id
                ? "bg-white text-black"
                : "bg-white/[0.06] text-neutral-400 hover:text-neutral-200"
            }`}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

type IconProps = { className?: string };

const SpinnerIcon = ({ className }: IconProps) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
    <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
  </svg>
);

const WandIcon = ({ className }: IconProps) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <path d="M9 2l1.2 3.3L13.5 6.5 10.2 7.7 9 11 7.8 7.7 4.5 6.5 7.8 5.3 9 2z" />
    <path d="M18 11l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8.8-2.2z" />
    <path
      d="M4 21L14 11"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      fill="none"
    />
  </svg>
);

const ChevronIcon = ({ className }: IconProps) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path
      d="M9 6l6 6-6 6"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const BackIcon = ({ className }: IconProps) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path
      d="M15 6l-6 6 6 6"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

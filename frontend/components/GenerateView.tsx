"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Generate — Krea 2 inference.
 *
 * Layout follows the Krea reference: black canvas carrying the results, with a
 * single floating prompt bar pinned near the bottom. Every secondary control is
 * a pill inside that bar, so the canvas stays entirely for the images.
 *
 * Turbo is the default model: 8 steps at guidance 1.0 (mu 1.15) against RAW's
 * 28 at 5.5. The backend picks those per-model defaults, so switching models
 * does not silently leave you with 8-step RAW output.
 */

type Lora = { name: string; trigger_word: string; files: string[]; path: string };
type GenImage = { name: string; data: string; path: string };

const ASPECTS = [
  { id: "1:1", w: 1024, h: 1024 },
  { id: "4:3", w: 1152, h: 896 },
  { id: "3:2", w: 1216, h: 832 },
  { id: "16:9", w: 1344, h: 768 },
  { id: "2:3", w: 832, h: 1216 },
  { id: "9:16", w: 768, h: 1344 },
] as const;

export default function GenerateView() {
  const [urls, setUrls] = useState<Record<string, string> | null>(null);
  const [accessKey, setAccessKey] = useState("");
  const [prompt, setPrompt] = useState("");
  const [model, setModel] = useState<"turbo" | "raw">("turbo");
  const [aspect, setAspect] = useState<string>("1:1");
  const [numImages, setNumImages] = useState(1);
  const [loras, setLoras] = useState<Lora[]>([]);
  const [lora, setLora] = useState<string>("");
  const [loraStrength, setLoraStrength] = useState(1);
  const [seed, setSeed] = useState<string>("");

  const [open, setOpen] = useState<"model" | "aspect" | "lora" | "more" | null>(null);
  const [busy, setBusy] = useState(false);
  const [images, setImages] = useState<GenImage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    setAccessKey(localStorage.getItem("visionary.access_key") ?? "");
    fetch("/api/train")
      .then((r) => r.json())
      .then((d) => (d.status === "ok" ? setUrls(d) : setError(d.error)))
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(
    () => () => {
      if (pollRef.current) clearInterval(pollRef.current);
      if (tickRef.current) clearInterval(tickRef.current);
    },
    [],
  );

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

  // Load the LoRA list once the backend URLs and key are both known.
  useEffect(() => {
    if (!urls?.lorasUrl || !accessKey) return;
    post(urls.lorasUrl, {})
      .then((d) => setLoras(d.loras ?? []))
      .catch(() => {});
  }, [urls, accessKey, post]);

  const generate = async () => {
    if (!urls || busy || !prompt.trim()) return;
    if (!accessKey) return setError("Set your access key in Settings first.");

    setBusy(true);
    setError(null);
    setElapsed(0);
    if (tickRef.current) clearInterval(tickRef.current);
    tickRef.current = setInterval(() => setElapsed((s) => s + 1), 1000);

    const a = ASPECTS.find((x) => x.id === aspect) ?? ASPECTS[0];
    try {
      const { call_id, job_id } = await post(urls.generateUrl, {
        prompt: prompt.trim(),
        model,
        width: a.w,
        height: a.h,
        num_images: numImages,
        ...(lora ? { lora_path: lora, lora_multiplier: loraStrength } : {}),
        ...(seed.trim() ? { seed: Number(seed) } : {}),
      });

      pollRef.current = setInterval(async () => {
        try {
          const s = await post(urls.statusUrl, { call_id, job_id });
          if (s.status === "running") return;
          if (pollRef.current) clearInterval(pollRef.current);
          if (tickRef.current) clearInterval(tickRef.current);
          setBusy(false);
          if (s.status === "completed") setImages(s.images ?? []);
          else setError(s.error ?? "Generation failed.");
        } catch {
          /* transient — keep polling */
        }
      }, 3000);
    } catch (err) {
      if (tickRef.current) clearInterval(tickRef.current);
      setError((err as Error).message);
      setBusy(false);
    }
  };

  const activeLora = loras.find((l) => l.path === lora);

  return (
    <div className="relative flex h-dvh flex-col">
      {/* Header */}
      <div className="px-6 pt-5 text-[14px] text-neutral-500">
        Model <span className="font-medium text-neutral-100">Krea 2 {model === "turbo" ? "Turbo" : "RAW"}</span>
      </div>

      {/* Canvas */}
      <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-56 pt-4">
        {error && (
          <div className="mx-auto mb-4 max-w-2xl rounded-2xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-[13px] text-red-300">
            {error}
          </div>
        )}

        {images.length > 0 ? (
          <div
            className={`mx-auto grid max-w-4xl gap-3 ${
              images.length > 1 ? "grid-cols-2" : "grid-cols-1 max-w-xl"
            }`}
          >
            {images.map((img) => (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                key={img.name}
                src={img.data}
                alt={img.name}
                className="w-full rounded-2xl border border-white/10"
              />
            ))}
          </div>
        ) : (
          <div className="flex h-full min-h-[40vh] flex-col items-center justify-center">
            <div className="flex items-center gap-3 opacity-25">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-sky-400 to-indigo-500">
                <svg viewBox="0 0 24 24" fill="none" className="h-6 w-6 text-white">
                  <rect x="3" y="4" width="18" height="16" rx="2.5" stroke="currentColor" strokeWidth="2.2" />
                  <circle cx="8.5" cy="9.5" r="1.6" fill="currentColor" />
                  <path d="M4 17l5-4.5 4 3.5 3-2.5 4 3.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
              <span className="text-[34px] font-semibold tracking-tight">Image</span>
            </div>
            {busy && (
              <p className="mt-6 text-[13px] tabular-nums text-neutral-500">
                Generating… {elapsed}s
              </p>
            )}
          </div>
        )}
      </div>

      {/* Prompt bar */}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 px-4 pb-5">
        <div className="pointer-events-auto mx-auto w-full max-w-3xl rounded-3xl border border-white/10 bg-neutral-900/80 p-3 backdrop-blur-xl">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) generate();
            }}
            placeholder="Describe an image and click generate…"
            rows={2}
            className="w-full resize-none bg-transparent px-2 py-1.5 text-[15px] text-neutral-100 outline-none placeholder:text-neutral-500"
          />

          <div className="mt-1 flex flex-wrap items-center gap-1.5">
            <Chip active={open === "model"} onClick={() => setOpen(open === "model" ? null : "model")}>
              <SparkIcon className="h-3.5 w-3.5" />
              Krea 2 {model === "turbo" ? "Turbo" : "RAW"}
            </Chip>

            <Chip
              active={open === "lora"}
              onClick={() => setOpen(open === "lora" ? null : "lora")}
              highlight={Boolean(lora)}
            >
              <LoraIcon className="h-3.5 w-3.5" />
              {activeLora ? activeLora.name : "Lora"}
            </Chip>

            <Chip active={open === "aspect"} onClick={() => setOpen(open === "aspect" ? null : "aspect")}>
              <FrameIcon className="h-3.5 w-3.5" />
              {aspect}
            </Chip>

            <Chip active={open === "more"} onClick={() => setOpen(open === "more" ? null : "more")}>
              <SlidersIcon className="h-3.5 w-3.5" />
            </Chip>

            <button
              onClick={generate}
              disabled={busy || !prompt.trim()}
              aria-label="Generate"
              className="ml-auto flex h-9 w-9 items-center justify-center rounded-full bg-white text-black transition active:scale-95 disabled:bg-white/25 disabled:text-black/40"
            >
              {busy ? (
                <SpinnerIcon className="h-4 w-4 animate-spin" />
              ) : (
                <SparkIcon className="h-4 w-4" />
              )}
            </button>
          </div>

          {/* Popovers */}
          {open === "model" && (
            <Popover>
              {(["turbo", "raw"] as const).map((m) => (
                <Row
                  key={m}
                  active={model === m}
                  onClick={() => {
                    setModel(m);
                    setOpen(null);
                  }}
                  title={`Krea 2 ${m === "turbo" ? "Turbo" : "RAW"}`}
                  sub={m === "turbo" ? "8 steps · fast" : "28 steps · higher quality"}
                />
              ))}
            </Popover>
          )}

          {open === "aspect" && (
            <Popover>
              <div className="grid grid-cols-3 gap-1.5">
                {ASPECTS.map((a) => (
                  <button
                    key={a.id}
                    onClick={() => {
                      setAspect(a.id);
                      setOpen(null);
                    }}
                    className={`rounded-lg px-2 py-2 text-[12px] transition ${
                      aspect === a.id ? "bg-white text-black" : "bg-white/[0.06] text-neutral-300 hover:bg-white/10"
                    }`}
                  >
                    {a.id}
                  </button>
                ))}
              </div>
            </Popover>
          )}

          {open === "lora" && (
            <Popover>
              {loras.length === 0 ? (
                <p className="px-1 py-2 text-[13px] text-neutral-500">
                  No trained LoRAs yet.
                </p>
              ) : (
                <>
                  <Row
                    active={!lora}
                    onClick={() => {
                      setLora("");
                      setOpen(null);
                    }}
                    title="None"
                  />
                  {loras.map((l) => (
                    <Row
                      key={l.path}
                      active={lora === l.path}
                      onClick={() => {
                        setLora(l.path);
                        if (l.trigger_word && !prompt.includes(l.trigger_word))
                          setPrompt((p) => (p ? `${l.trigger_word}, ${p}` : l.trigger_word));
                        setOpen(null);
                      }}
                      title={l.name}
                      sub={l.trigger_word || undefined}
                    />
                  ))}
                  {lora && (
                    <div className="mt-2 border-t border-white/10 px-1 pt-3">
                      <div className="mb-1.5 flex justify-between text-[12px] text-neutral-500">
                        <span>Strength</span>
                        <span className="tabular-nums text-neutral-300">
                          {loraStrength.toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={1.5}
                        step={0.05}
                        value={loraStrength}
                        onChange={(e) => setLoraStrength(Number(e.target.value))}
                        className="w-full accent-white"
                      />
                    </div>
                  )}
                </>
              )}
            </Popover>
          )}

          {open === "more" && (
            <Popover>
              <div className="mb-3">
                <span className="mb-1.5 block text-[12px] text-neutral-500">Images</span>
                <div className="flex gap-1.5">
                  {[1, 2, 3, 4].map((n) => (
                    <button
                      key={n}
                      onClick={() => setNumImages(n)}
                      className={`flex-1 rounded-lg py-1.5 text-[12px] transition ${
                        numImages === n ? "bg-white text-black" : "bg-white/[0.06] text-neutral-300"
                      }`}
                    >
                      {n}
                    </button>
                  ))}
                </div>
              </div>
              <label className="block">
                <span className="mb-1.5 block text-[12px] text-neutral-500">Seed</span>
                <input
                  value={seed}
                  onChange={(e) => setSeed(e.target.value.replace(/[^\d]/g, ""))}
                  placeholder="random"
                  inputMode="numeric"
                  className="w-full rounded-lg border border-white/10 bg-white/[0.04] px-2.5 py-2 text-[13px] tabular-nums text-neutral-100 outline-none focus:border-white/25"
                />
              </label>
            </Popover>
          )}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */

function Chip({
  children,
  active,
  highlight,
  onClick,
}: {
  children: React.ReactNode;
  active?: boolean;
  highlight?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[12.5px] transition ${
        active
          ? "bg-white/[0.16] text-neutral-100"
          : highlight
            ? "bg-white/[0.10] text-neutral-100"
            : "bg-white/[0.06] text-neutral-300 hover:bg-white/[0.10]"
      }`}
    >
      {children}
    </button>
  );
}

function Popover({ children }: { children: React.ReactNode }) {
  return (
    <div className="mt-2 rounded-2xl border border-white/10 bg-neutral-900 p-2">{children}</div>
  );
}

function Row({
  title,
  sub,
  active,
  onClick,
}: {
  title: string;
  sub?: string;
  active?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex w-full items-center justify-between rounded-lg px-2.5 py-2 text-left transition ${
        active ? "bg-white/[0.10]" : "hover:bg-white/[0.06]"
      }`}
    >
      <span className="min-w-0">
        <span className="block truncate text-[13px] text-neutral-100">{title}</span>
        {sub && <span className="block truncate font-mono text-[11px] text-neutral-500">{sub}</span>}
      </span>
      {active && <CheckIcon className="ml-2 h-3.5 w-3.5 shrink-0 text-neutral-300" />}
    </button>
  );
}

type IconProps = { className?: string };

const SparkIcon = ({ className }: IconProps) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <path d="M12 2l2.2 6.2L20.4 10l-6.2 2.2L12 18.4 9.8 12.2 3.6 10l6.2-1.8L12 2z" />
  </svg>
);

const SpinnerIcon = ({ className }: IconProps) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
    <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
  </svg>
);

const CheckIcon = ({ className }: IconProps) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path d="M4 12.5l5 5L20 6.5" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const LoraIcon = ({ className }: IconProps) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <circle cx="12" cy="12" r="8.5" stroke="currentColor" strokeWidth="2" />
    <circle cx="12" cy="12" r="3" fill="currentColor" />
  </svg>
);

const FrameIcon = ({ className }: IconProps) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <rect x="6" y="3" width="12" height="18" rx="2.5" stroke="currentColor" strokeWidth="2" />
  </svg>
);

const SlidersIcon = ({ className }: IconProps) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path d="M4 8h10M18 8h2M4 16h4M12 16h8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    <circle cx="16" cy="8" r="2" stroke="currentColor" strokeWidth="2" />
    <circle cx="10" cy="16" r="2" stroke="currentColor" strokeWidth="2" />
  </svg>
);

"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Dispatch, DragEvent, ReactNode, SetStateAction } from "react";


/**
 * TrainerMVP — drop images, name it, train.
 *
 * Upload goes browser -> Modal directly (4 GiB ceiling). It deliberately does
 * not pass through the Next route, which would 413 at 4.5 MB.
 *
 * Visual language follows the Krea reference: near-black canvas, one soft
 * panel, muted borders, a single white pill carrying the action. Advanced
 * hyperparameters stay folded away until asked for.
 */

import CaptionGrid, { type Backend } from "./CaptionGrid";

type Phase =
  | "idle"
  | "uploading"
  | "review"
  | "queued"
  | "running"
  | "stopping"
  | "done"
  | "error";

/** Live telemetry published by the training container, plus the final result. */
type Result = {
  status?: string;
  lora_name?: string;
  output_dir?: string;
  files?: string[];
  images?: number;
  duration_s?: number;
  log_tail?: string[];
  error?: string;
  note?: string;
  // progress
  phase?: string;
  percent?: number;
  step?: number;
  total_steps?: number;
  elapsed?: string;
  eta?: string;
  rate?: string;
  loss?: string;
  epoch?: number;
  total_epochs?: number;
};

/** Backend phase labels -> what a human wants to read. */
const PHASE_LABEL: Record<string, string> = {
  queued: "Queued",
  starting: "Starting up",
  "cache latents": "Caching latents",
  "cache text encoder outputs": "Caching text encoder",
  train: "Training",
  stopping: "Stopping",
};

const IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif"];

/** Defaults mirror the krea2 reference command. */
const DEFAULTS = {
  resolution: 1024,
  batch_size: 1,
  num_repeats: 1,
  network_dim: 32,
  network_alpha: 32,
  learning_rate: 1e-4,
  max_train_epochs: 30,
  save_every_n_epochs: 1,
  // 0 = save on epoch boundaries only. Raising the frequency shrinks how much
  // work a Stop throws away, since musubi cannot flush a checkpoint on demand.
  save_every_n_steps: 0,
  // Keep only the newest N intermediate checkpoints (~0.5 GB each). The final
  // model is written separately and is never pruned. 0 = keep every one.
  save_last_n_epochs: 3,
  discrete_flow_shift: 2.5,
  seed: 42,
};

type Advanced = typeof DEFAULTS;

const isImage = (name: string) =>
  IMAGE_EXTS.some((e) => name.toLowerCase().endsWith(e));
const isZip = (name: string) => name.toLowerCase().endsWith(".zip");
const isCaption = (name: string) => name.toLowerCase().endsWith(".txt");

export default function TrainerMVP() {
  const [backend, setBackend] = useState<Backend | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);

  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [loraName, setLoraName] = useState("");
  const [triggerWord, setTriggerWord] = useState("");

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [adv, setAdv] = useState<Advanced>(DEFAULTS);

  const [showSettings, setShowSettings] = useState(false);
  const [accessKey, setAccessKey] = useState("");
  const [hfToken, setHfToken] = useState("");
  // Auto-caption during training is now the fallback path; the normal route is
  // the review screen, where you can see what the captioner actually wrote.
  const [autoCaptionInTrain] = useState(false);

  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState(0);
  const [statusLine, setStatusLine] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [live, setLive] = useState<Result | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);

  const inputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const imageCount = useMemo(
    () => files.filter((f) => isImage(f.name)).length,
    [files],
  );
  const zipCount = useMemo(() => files.filter((f) => isZip(f.name)).length, [files]);
  const captionCount = useMemo(
    () => files.filter((f) => isCaption(f.name)).length,
    [files],
  );
  const totalBytes = useMemo(
    () => files.reduce((sum, f) => sum + f.size, 0),
    [files],
  );

  const busy =
    phase === "uploading" ||
    phase === "queued" ||
    phase === "running" ||
    phase === "stopping";
  const ready = (imageCount > 0 || zipCount > 0) && loraName.trim() && triggerWord.trim();

  // Backend URLs come from the server so they live in one place.
  useEffect(() => {
    fetch("/api/train")
      .then((r) => r.json())
      .then((d) => (d.status === "ok" ? setBackend(d) : setConfigError(d.error)))
      .catch((e) => setConfigError(String(e)));
  }, []);

  useEffect(() => {
    setAccessKey(localStorage.getItem("visionary.access_key") ?? "");
    setHfToken(localStorage.getItem("visionary.hf_token") ?? "");
  }, []);

  useEffect(
    () => () => {
      if (pollRef.current) clearInterval(pollRef.current);
      if (tickRef.current) clearInterval(tickRef.current);
    },
    [],
  );

  const addFiles = useCallback((incoming: FileList | File[]) => {
    const kept = Array.from(incoming).filter(
      (f) => isImage(f.name) || isZip(f.name) || isCaption(f.name),
    );
    if (!kept.length) return;
    setFiles((prev) => {
      const seen = new Set(prev.map((f) => `${f.name}:${f.size}`));
      return [...prev, ...kept.filter((f) => !seen.has(`${f.name}:${f.size}`))];
    });
    setError(null);
  }, []);

  const onDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setDragging(false);
      if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files);
    },
    [addFiles],
  );

  /** XHR, not fetch — fetch gives no upload progress events. */
  const uploadDataset = (b: Backend, key: string): Promise<string> =>
    new Promise((resolve, reject) => {
      const form = new FormData();
      files.forEach((f) => form.append("files", f, f.name));

      const xhr = new XMLHttpRequest();
      xhr.open("POST", b.uploadUrl);
      xhr.setRequestHeader("X-Visionary-Key", key);

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) setProgress(Math.round((e.loaded / e.total) * 100));
      };
      xhr.onload = () => {
        let data: Record<string, unknown> = {};
        try {
          data = JSON.parse(xhr.responseText);
        } catch {
          return reject(new Error(`Upload returned non-JSON (${xhr.status}).`));
        }
        if (xhr.status >= 200 && xhr.status < 300 && data.job_id) {
          resolve(String(data.job_id));
        } else {
          reject(new Error(String(data.error ?? `Upload failed (${xhr.status}).`)));
        }
      };
      xhr.onerror = () => reject(new Error("Network error during upload."));
      xhr.send(form);
    });

  const pollStatus = (b: Backend, key: string, callId: string, job: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(b.statusUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ call_id: callId, job_id: job, access_key: key }),
        });
        const data: Result = await res.json();

        if (data.status === "running") {
          setLive(data);
          setPhase((p) => (p === "stopping" ? "stopping" : "running"));
          return;
        }

        if (pollRef.current) clearInterval(pollRef.current);
        if (tickRef.current) clearInterval(tickRef.current);

        // "stopped" is a success path, not an error — checkpoints were kept.
        if (data.status === "completed" || data.status === "stopped") {
          setResult(data);
          setPhase("done");
        } else {
          setError(data.error ?? "Training failed.");
          setResult(data);
          setPhase("error");
        }
      } catch {
        /* transient network blip — keep polling */
      }
    }, 3000);
  };

  const stop = async () => {
    if (!backend || !jobId || phase === "stopping") return;
    setPhase("stopping");
    try {
      const res = await fetch(backend.stopUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: jobId, access_key: accessKey.trim() }),
      });
      const data = await res.json();
      if (data.status === "error") throw new Error(data.error);
      // The poll loop carries it the rest of the way to a final result.
    } catch (err) {
      setError(`Could not stop: ${(err as Error).message}`);
      setPhase("running");
    }
  };

  const start = async () => {
    if (!ready || busy) return;
    if (!backend) return setError("Backend URLs are not configured.");
    const key = accessKey.trim();
    if (!key) {
      setShowSettings(true);
      return setError("Enter your access key in Settings first.");
    }

    setError(null);
    setResult(null);
    setLive(null);
    setJobId(null);
    setProgress(0);
    setElapsed(0);
    setPhase("uploading");
    setStatusLine("Uploading dataset…");

    try {
      // Upload only. Training is deliberately a second, separate action so the
      // captions get looked at before an A100 spends an hour learning them.
      const job = await uploadDataset(backend, key);
      setJobId(job);
      setPhase("review");
    } catch (err) {
      setError((err as Error).message);
      setPhase("error");
    }
  };

  /** Called from the review screen once captions are settled. */
  const beginTraining = async () => {
    if (!backend || !jobId) return;
    const key = accessKey.trim();

    setError(null);
    setElapsed(0);
    setPhase("queued");
    setStatusLine("Queueing…");

    if (tickRef.current) clearInterval(tickRef.current);
    tickRef.current = setInterval(() => setElapsed((s) => s + 1), 1000);

    try {
      const res = await fetch(backend.trainUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_id: jobId,
          lora_name: loraName.trim(),
          trigger_word: triggerWord.trim(),
          auto_caption: autoCaptionInTrain,
          access_key: key,
          ...(hfToken.trim() ? { hf_token: hfToken.trim() } : {}),
          ...adv,
        }),
      });
      const data = await res.json();
      if (data.status !== "queued") throw new Error(data.error ?? "Could not queue.");

      setPhase("running");
      pollStatus(backend, key, data.call_id, jobId);
    } catch (err) {
      if (tickRef.current) clearInterval(tickRef.current);
      setError((err as Error).message);
      setPhase("error");
    }
  };

  const reset = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    if (tickRef.current) clearInterval(tickRef.current);
    setPhase("idle");
    setFiles([]);
    setResult(null);
    setLive(null);
    setJobId(null);
    setError(null);
    setProgress(0);
    setElapsed(0);
  };

  const persist = (k: string, v: string, set: (s: string) => void) => {
    set(v);
    if (v.trim()) localStorage.setItem(k, v.trim());
    else localStorage.removeItem(k);
  };

  return (
    <div className="min-h-dvh bg-black px-4 py-10 text-neutral-100 antialiased sm:px-6 sm:py-14">
      <div className="mx-auto w-full max-w-lg">
        {/* Header */}
        <div className="mb-7 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 to-blue-500">
              <SparkIcon className="h-4 w-4 text-white" />
            </div>
            <div className="leading-tight">
              <h1 className="text-[15px] font-medium tracking-tight">Train LoRA</h1>
              <p className="text-[13px] text-neutral-500">Krea 2 · RAW</p>
            </div>
          </div>
          <button
            onClick={() => setShowSettings((v) => !v)}
            aria-label="Settings"
            className="rounded-xl border border-white/10 bg-white/[0.04] p-2.5 text-neutral-400 transition hover:text-neutral-100"
          >
            <GearIcon className="h-4 w-4" />
          </button>
        </div>

        {configError && (
          <div className="mb-4 rounded-2xl border border-amber-500/20 bg-amber-500/10 px-4 py-3 text-[13px] text-amber-300">
            {configError}
          </div>
        )}

        {showSettings && (
          <div className="mb-4 rounded-3xl border border-white/10 bg-neutral-950/80 p-5">
            <Field label="Access key">
              <input
                type="password"
                value={accessKey}
                onChange={(e) => persist("visionary.access_key", e.target.value, setAccessKey)}
                placeholder="VISIONARY_API_KEY"
                autoComplete="off"
                className={inputCls + " font-mono"}
              />
            </Field>
            <div className="h-3" />
            <Field label="Hugging Face token">
              <input
                type="password"
                value={hfToken}
                onChange={(e) => persist("visionary.hf_token", e.target.value, setHfToken)}
                placeholder="hf_••••  (optional if set in Modal)"
                autoComplete="off"
                className={inputCls + " font-mono"}
              />
            </Field>
            <p className="mt-2.5 text-[12px] leading-relaxed text-neutral-600">
              Stored in this browser only.
            </p>
          </div>
        )}

        {/* Main panel */}
        <div className="rounded-3xl border border-white/10 bg-neutral-950/80 p-5 shadow-2xl shadow-black/40">
          {phase === "review" && backend && jobId ? (
            <CaptionGrid
              backend={backend}
              accessKey={accessKey.trim()}
              jobId={jobId}
              triggerWord={triggerWord.trim()}
              onBack={() => setPhase("idle")}
              onTrain={beginTraining}
            />
          ) : phase === "done" && result ? (
            <DonePanel result={result} onReset={reset} />
          ) : (
            <>
              {/* Dropzone */}
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                onClick={() => !busy && inputRef.current?.click()}
                className={`cursor-pointer rounded-2xl border border-dashed p-7 text-center transition ${
                  dragging
                    ? "border-white/40 bg-white/[0.07]"
                    : "border-white/15 bg-white/[0.02] hover:border-white/25"
                } ${busy ? "pointer-events-none opacity-60" : ""}`}
              >
                <input
                  ref={inputRef}
                  type="file"
                  multiple
                  accept={[...IMAGE_EXTS, ".zip", ".txt"].join(",")}
                  className="hidden"
                  onChange={(e) => e.target.files && addFiles(e.target.files)}
                />
                {files.length === 0 ? (
                  <>
                    <UploadIcon className="mx-auto mb-2.5 h-6 w-6 text-neutral-600" />
                    <p className="text-[14px] text-neutral-300">Drop images or a .zip</p>
                    <p className="mt-1 text-[12px] text-neutral-600">or tap to browse</p>
                  </>
                ) : (
                  <>
                    <p className="text-[14px] text-neutral-200">
                      {zipCount > 0 && `${zipCount} zip${zipCount > 1 ? "s" : ""}`}
                      {zipCount > 0 && imageCount > 0 && " · "}
                      {imageCount > 0 && `${imageCount} image${imageCount > 1 ? "s" : ""}`}
                    </p>
                    <p className="mt-1 text-[12px] text-neutral-600">
                      {(totalBytes / 1e6).toFixed(1)} MB
                      {captionCount > 0 && ` · ${captionCount} captions`}
                    </p>
                    {!busy && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setFiles([]);
                        }}
                        className="mt-2 text-[12px] text-neutral-500 underline-offset-2 hover:text-neutral-300 hover:underline"
                      >
                        Clear
                      </button>
                    )}
                  </>
                )}
              </div>

              {/* Names */}
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <Field label="LoRA name">
                  <input
                    value={loraName}
                    onChange={(e) => setLoraName(e.target.value)}
                    disabled={busy}
                    placeholder="my_style"
                    autoCapitalize="none"
                    spellCheck={false}
                    className={inputCls}
                  />
                </Field>
                <Field label="Trigger word">
                  <input
                    value={triggerWord}
                    onChange={(e) => setTriggerWord(e.target.value)}
                    disabled={busy}
                    placeholder="ohwx_style"
                    autoCapitalize="none"
                    spellCheck={false}
                    className={inputCls}
                  />
                </Field>
              </div>

              {/* Advanced */}
              <button
                type="button"
                onClick={() => setShowAdvanced((v) => !v)}
                className="mt-3 flex items-center gap-1.5 rounded-lg px-1 py-1 text-[13px] text-neutral-500 transition hover:text-neutral-300"
              >
                <ChevronIcon
                  className={`h-3.5 w-3.5 transition-transform ${showAdvanced ? "rotate-90" : ""}`}
                />
                Advanced
              </button>

              {showAdvanced && (
                <div className="mt-2 grid grid-cols-2 gap-3 rounded-2xl border border-white/10 bg-white/[0.02] p-4">
                  <Num label="Rank (dim)" k="network_dim" adv={adv} set={setAdv} busy={busy} />
                  <Num label="Alpha" k="network_alpha" adv={adv} set={setAdv} busy={busy} />
                  <Num label="Epochs" k="max_train_epochs" adv={adv} set={setAdv} busy={busy} />
                  <Num label="Learning rate" k="learning_rate" adv={adv} set={setAdv} busy={busy} step="0.00001" />
                  <Num label="Resolution" k="resolution" adv={adv} set={setAdv} busy={busy} step="64" />
                  <Num label="Repeats" k="num_repeats" adv={adv} set={setAdv} busy={busy} />
                  <Num label="Batch size" k="batch_size" adv={adv} set={setAdv} busy={busy} />
                  <Num label="Save every (epochs)" k="save_every_n_epochs" adv={adv} set={setAdv} busy={busy} />
                  <Num label="Save every (steps)" k="save_every_n_steps" adv={adv} set={setAdv} busy={busy} />
                  <Num label="Keep last N ckpts" k="save_last_n_epochs" adv={adv} set={setAdv} busy={busy} />
                  <Num label="Flow shift" k="discrete_flow_shift" adv={adv} set={setAdv} busy={busy} step="0.1" />
                  <Num label="Seed" k="seed" adv={adv} set={setAdv} busy={busy} />
                  <button
                    onClick={() => setAdv(DEFAULTS)}
                    className="col-span-2 mt-1 rounded-xl border border-white/10 py-2 text-[12px] text-neutral-500 transition hover:text-neutral-300"
                  >
                    Reset to defaults
                  </button>
                </div>
              )}

              {/* Action */}
              {busy ? (
                <ProgressPanel
                  phase={phase}
                  uploadPercent={progress}
                  uploadLabel={statusLine}
                  live={live}
                  elapsed={elapsed}
                  onStop={stop}
                  canStop={Boolean(jobId) && phase !== "uploading"}
                />
              ) : (
                <button
                  onClick={start}
                  disabled={!ready}
                  className="mt-4 flex w-full items-center justify-center gap-2 rounded-2xl bg-white px-4 py-3.5 text-[15px] font-medium text-black transition active:scale-[0.985] disabled:cursor-not-allowed disabled:bg-white/25 disabled:text-black/40"
                >
                  Upload &amp; review
                </button>
              )}

              {phase === "error" && error && (
                <div className="mt-4 rounded-2xl border border-red-500/20 bg-red-500/10 px-4 py-3">
                  <p className="text-[13px] leading-relaxed text-red-300">{error}</p>
                  {result?.log_tail?.length ? <LogTail lines={result.log_tail} /> : null}
                </div>
              )}
            </>
          )}
        </div>

        <p className="mt-5 text-center text-[12px] text-neutral-700">
          Writes to /workspace/models/Lora
        </p>
      </div>

      <style>{`
        @keyframes slide { 0% { transform: translateX(-100%);} 100% { transform: translateX(300%);} }
      `}</style>
    </div>
  );
}

/* ------------------------------------------------------------------ */

const inputCls =
  "w-full rounded-xl border border-white/10 bg-white/[0.04] px-3.5 py-2.5 text-[14px] text-neutral-100 outline-none transition placeholder:text-neutral-600 focus:border-white/25 disabled:opacity-50";

/**
 * Live run state.
 *
 * Determinate bar only when the backend reports real step counts. During upload
 * it tracks bytes; during setup phases (weights, caching) there is genuinely no
 * total to divide by, so it stays indeterminate rather than inventing a number.
 */
function ProgressPanel({
  phase,
  uploadPercent,
  uploadLabel,
  live,
  elapsed,
  onStop,
  canStop,
}: {
  phase: Phase;
  uploadPercent: number;
  uploadLabel: string;
  live: Result | null;
  elapsed: number;
  onStop: () => void;
  canStop: boolean;
}) {
  const isUpload = phase === "uploading";
  const isStopping = phase === "stopping";
  const training = live?.phase === "train" && typeof live?.step === "number";

  const pct = isUpload ? uploadPercent : training ? (live?.percent ?? 0) : null;

  const headline = isStopping
    ? "Stopping…"
    : isUpload
      ? uploadLabel
      : (PHASE_LABEL[live?.phase ?? ""] ?? live?.phase ?? "Preparing…");

  return (
    <div className="mt-4 rounded-2xl border border-white/10 bg-white/[0.02] p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <SpinnerIcon className="h-3.5 w-3.5 shrink-0 animate-spin text-neutral-400" />
          <span className="truncate text-[14px] text-neutral-100">{headline}</span>
        </div>
        {pct !== null && (
          <span className="shrink-0 text-[14px] font-medium tabular-nums text-neutral-100">
            {pct}%
          </span>
        )}
      </div>

      <div className="h-1 w-full overflow-hidden rounded-full bg-white/10">
        {pct !== null ? (
          <div
            className="h-full rounded-full bg-white/80 transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        ) : (
          <div className="h-full w-1/3 animate-[slide_1.4s_ease-in-out_infinite] rounded-full bg-white/50" />
        )}
      </div>

      {/* Detail grid — only render facts the backend actually sent. */}
      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 text-[12px]">
        {training && (
          <Stat
            label="Step"
            value={`${live?.step?.toLocaleString()} / ${live?.total_steps?.toLocaleString()}`}
          />
        )}
        {typeof live?.epoch === "number" && (
          <Stat label="Epoch" value={`${live.epoch}${live.total_epochs ? ` / ${live.total_epochs}` : ""}`} />
        )}
        {live?.eta && live.eta !== "?" && <Stat label="ETA" value={live.eta} />}
        {live?.rate && !live.rate.startsWith("?") && <Stat label="Rate" value={live.rate} />}
        {live?.loss && <Stat label="Loss" value={Number(live.loss).toFixed(4)} />}
        <Stat label="Elapsed" value={fmtTime(elapsed)} />
      </dl>

      {canStop && (
        <button
          onClick={onStop}
          disabled={isStopping}
          className="mt-4 w-full rounded-xl border border-red-500/25 bg-red-500/10 px-4 py-2.5 text-[13px] font-medium text-red-300 transition hover:bg-red-500/15 disabled:opacity-50"
        >
          {isStopping ? "Stopping…" : "Stop training"}
        </button>
      )}
      {canStop && !isStopping && (
        <p className="mt-2 text-center text-[11px] leading-relaxed text-neutral-600">
          Keeps checkpoints already saved
        </p>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <dt className="text-neutral-600">{label}</dt>
      <dd className="tabular-nums text-neutral-300">{value}</dd>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-[12px] font-medium text-neutral-500">{label}</span>
      {children}
    </label>
  );
}

function Num({
  label,
  k,
  adv,
  set,
  busy,
  step,
}: {
  label: string;
  k: keyof Advanced;
  adv: Advanced;
  set: Dispatch<SetStateAction<Advanced>>;
  busy: boolean;
  step?: string;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-[12px] text-neutral-500">{label}</span>
      <input
        type="number"
        step={step}
        value={adv[k]}
        disabled={busy}
        onChange={(e) =>
          set((p) => ({ ...p, [k]: e.target.value === "" ? p[k] : Number(e.target.value) }))
        }
        className="w-full rounded-lg border border-white/10 bg-white/[0.04] px-2.5 py-2 text-[13px] tabular-nums text-neutral-100 outline-none focus:border-white/25 disabled:opacity-50"
      />
    </label>
  );
}

function LogTail({ lines }: { lines: string[] }) {
  return (
    <pre className="mt-3 max-h-40 overflow-auto whitespace-pre-wrap rounded-xl bg-black/50 p-3 font-mono text-[11px] leading-relaxed text-neutral-500">
      {lines.join("\n")}
    </pre>
  );
}

function DonePanel({ result, onReset }: { result: Result; onReset: () => void }) {
  const [copied, setCopied] = useState(false);
  const path = result.output_dir ?? "";
  const final = result.files?.find((f) => !/-\d{6}\.safetensors$/.test(f));
  const wasStopped = result.status === "stopped";
  const empty = !result.files?.length;

  return (
    <div>
      <div className="mb-4 flex items-center gap-2.5">
        <div
          className={`flex h-6 w-6 items-center justify-center rounded-full ${
            wasStopped ? "bg-amber-500/15" : "bg-emerald-500/15"
          }`}
        >
          {wasStopped ? (
            <StopIcon className="h-3 w-3 text-amber-400" />
          ) : (
            <CheckIcon className="h-3.5 w-3.5 text-emerald-400" />
          )}
        </div>
        <span className="text-[15px] font-medium">
          {wasStopped ? "Stopped" : "Training complete"}
        </span>
      </div>

      {result.note && (
        <p className="mb-4 rounded-xl border border-amber-500/20 bg-amber-500/10 px-3 py-2.5 text-[12px] leading-relaxed text-amber-300">
          {result.note}
        </p>
      )}

      <dl className="space-y-3">
        <Row label="LoRA" value={result.lora_name ?? "—"} mono />
        <Row
          label="Trained on"
          value={`${result.images ?? "?"} images · ${fmtTime(Math.round(result.duration_s ?? 0))}`}
        />
        {final && <Row label="Final checkpoint" value={final} mono />}
        {empty && <Row label="Checkpoints" value="none written" />}
        <div>
          <dt className="mb-1 text-[12px] text-neutral-500">Folder</dt>
          <dd className="flex items-start gap-2">
            <code className="min-w-0 flex-1 break-all rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2 text-[12px] leading-relaxed text-neutral-300">
              {path}
            </code>
            <button
              onClick={async () => {
                await navigator.clipboard.writeText(path);
                setCopied(true);
                setTimeout(() => setCopied(false), 1600);
              }}
              aria-label="Copy path"
              className="shrink-0 rounded-xl border border-white/10 bg-white/[0.04] p-2.5 text-neutral-400 transition hover:text-neutral-100"
            >
              {copied ? (
                <CheckIcon className="h-3.5 w-3.5 text-emerald-400" />
              ) : (
                <CopyIcon className="h-3.5 w-3.5" />
              )}
            </button>
          </dd>
        </div>
        {result.files && result.files.length > 1 && (
          <Row label="Checkpoints" value={`${result.files.length} saved`} />
        )}
      </dl>

      <button
        onClick={onReset}
        className="mt-5 w-full rounded-2xl border border-white/10 bg-white/[0.04] px-4 py-3 text-[14px] font-medium text-neutral-200 transition hover:bg-white/[0.07] active:scale-[0.985]"
      >
        Train another
      </button>
    </div>
  );
}

function Row({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <dt className="mb-1 text-[12px] text-neutral-500">{label}</dt>
      <dd className={`break-all text-[13px] text-neutral-200 ${mono ? "font-mono" : ""}`}>
        {value}
      </dd>
    </div>
  );
}

function fmtTime(s: number) {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

/* ---------------------------- icons ------------------------------- */

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

const CopyIcon = ({ className }: IconProps) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <rect x="9" y="9" width="12" height="12" rx="2.5" stroke="currentColor" strokeWidth="2" />
    <path d="M5 15V5.5A2.5 2.5 0 0 1 7.5 3H15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
  </svg>
);

const StopIcon = ({ className }: IconProps) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <rect x="5" y="5" width="14" height="14" rx="2.5" />
  </svg>
);

const ChevronIcon = ({ className }: IconProps) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path d="M9 6l6 6-6 6" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const UploadIcon = ({ className }: IconProps) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path d="M12 16V4m0 0L7 9m5-5l5 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
  </svg>
);

const GearIcon = ({ className }: IconProps) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <circle cx="12" cy="12" r="3.2" stroke="currentColor" strokeWidth="2" />
    <path
      d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9v0a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"
      stroke="currentColor"
      strokeWidth="1.6"
    />
  </svg>
);

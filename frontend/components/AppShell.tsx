"use client";

import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import TrainerMVP from "./TrainerMVP";
import GenerateView from "./GenerateView";

/**
 * App shell — fixed sidebar on desktop, top tab bar on phones.
 *
 * Follows the Krea reference: ~256px sidebar, compact nav rows with small
 * coloured glyphs, a muted section label, and one white pill pinned to the
 * bottom. The canvas stays pure black so generated images carry the page.
 */

export type View = "train" | "generate";

const NAV: { id: View; label: string; icon: ReactNode; tint: string }[] = [
  {
    id: "train",
    label: "Train LoRA",
    tint: "from-violet-500 to-blue-500",
    icon: (
      <svg viewBox="0 0 24 24" fill="currentColor" className="h-3 w-3">
        <path d="M12 2l2.2 6.2L20.4 10l-6.2 2.2L12 18.4 9.8 12.2 3.6 10l6.2-1.8L12 2z" />
      </svg>
    ),
  },
  {
    id: "generate",
    label: "Image",
    tint: "from-sky-400 to-indigo-500",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" className="h-3 w-3">
        <rect x="3" y="4" width="18" height="16" rx="2.5" stroke="currentColor" strokeWidth="2.4" />
        <circle cx="8.5" cy="9.5" r="1.6" fill="currentColor" />
        <path d="M4 17l5-4.5 4 3.5 3-2.5 4 3.5" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
];

export default function AppShell() {
  const [view, setView] = useState<View>("train");
  const [showSettings, setShowSettings] = useState(false);
  const [accessKey, setAccessKey] = useState("");
  const [hfToken, setHfToken] = useState("");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setAccessKey(localStorage.getItem("visionary.access_key") ?? "");
    setHfToken(localStorage.getItem("visionary.hf_token") ?? "");
    setReady(true);
  }, []);

  const persist = (k: string, v: string, set: (s: string) => void) => {
    set(v);
    if (v.trim()) localStorage.setItem(k, v.trim());
    else localStorage.removeItem(k);
  };

  return (
    <div className="flex min-h-dvh flex-col bg-black text-neutral-100 antialiased md:flex-row">
      {/* Sidebar (desktop) */}
      <aside className="hidden w-64 shrink-0 flex-col border-r border-white/[0.07] px-3 py-4 md:flex">
        <div className="mb-5 flex items-center gap-2.5 rounded-xl bg-white/[0.05] px-3 py-2.5">
          <div className="flex h-6 w-6 items-center justify-center rounded-md bg-gradient-to-br from-violet-500 to-blue-500">
            <svg viewBox="0 0 24 24" fill="currentColor" className="h-3 w-3 text-white">
              <path d="M12 2l2.2 6.2L20.4 10l-6.2 2.2L12 18.4 9.8 12.2 3.6 10l6.2-1.8L12 2z" />
            </svg>
          </div>
          <span className="text-[14px] font-medium">Visionary</span>
        </div>

        <p className="mb-1.5 px-3 text-[11px] font-medium text-neutral-600">Tools</p>
        <nav className="flex flex-col gap-0.5">
          {NAV.map((n) => (
            <button
              key={n.id}
              onClick={() => setView(n.id)}
              className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-left text-[14px] transition ${
                view === n.id
                  ? "bg-white/[0.08] text-neutral-100"
                  : "text-neutral-400 hover:bg-white/[0.04] hover:text-neutral-200"
              }`}
            >
              <span
                className={`flex h-5 w-5 items-center justify-center rounded-[6px] bg-gradient-to-br ${n.tint} text-white`}
              >
                {n.icon}
              </span>
              {n.label}
            </button>
          ))}
        </nav>

        <div className="flex-1" />

        <button
          onClick={() => setShowSettings(true)}
          className="flex items-center gap-2.5 rounded-lg px-3 py-2 text-left text-[13px] text-neutral-500 transition hover:bg-white/[0.04] hover:text-neutral-300"
        >
          <GearIcon className="h-4 w-4" />
          Settings
          {ready && !accessKey && (
            <span className="ml-auto h-1.5 w-1.5 rounded-full bg-amber-400" />
          )}
        </button>
      </aside>

      {/* Tab bar (mobile) */}
      <header className="flex items-center gap-1 border-b border-white/[0.07] px-3 py-2.5 md:hidden">
        {NAV.map((n) => (
          <button
            key={n.id}
            onClick={() => setView(n.id)}
            className={`flex items-center gap-2 rounded-lg px-3 py-1.5 text-[13px] transition ${
              view === n.id ? "bg-white/[0.08] text-neutral-100" : "text-neutral-500"
            }`}
          >
            <span
              className={`flex h-4 w-4 items-center justify-center rounded-[5px] bg-gradient-to-br ${n.tint} text-white`}
            >
              {n.icon}
            </span>
            {n.label}
          </button>
        ))}
        <button
          onClick={() => setShowSettings(true)}
          aria-label="Settings"
          className="ml-auto rounded-lg p-2 text-neutral-500"
        >
          <GearIcon className="h-4 w-4" />
          {ready && !accessKey && (
            <span className="absolute mt-[-14px] ml-1.5 h-1.5 w-1.5 rounded-full bg-amber-400" />
          )}
        </button>
      </header>

      {/* Canvas */}
      <main className="min-w-0 flex-1">
        {view === "train" ? <TrainerMVP /> : <GenerateView />}
      </main>

      {showSettings && (
        <div
          className="fixed inset-0 z-50 flex items-end justify-center bg-black/70 p-4 backdrop-blur-sm sm:items-center"
          onClick={() => setShowSettings(false)}
        >
          <div
            className="w-full max-w-sm rounded-3xl border border-white/10 bg-neutral-950 p-5"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-[15px] font-medium">Settings</h2>
              <button
                onClick={() => setShowSettings(false)}
                className="rounded-lg p-1 text-neutral-500 transition hover:text-neutral-200"
                aria-label="Close"
              >
                <svg viewBox="0 0 24 24" fill="none" className="h-4 w-4">
                  <path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
                </svg>
              </button>
            </div>

            <label className="mb-1.5 block text-[12px] font-medium text-neutral-500">
              Access key
            </label>
            <input
              type="password"
              value={accessKey}
              onChange={(e) => persist("visionary.access_key", e.target.value, setAccessKey)}
              placeholder="VISIONARY_API_KEY"
              autoComplete="off"
              className="mb-4 w-full rounded-xl border border-white/10 bg-white/[0.04] px-3.5 py-2.5 font-mono text-[13px] text-neutral-100 outline-none transition placeholder:text-neutral-600 focus:border-white/25"
            />

            <label className="mb-1.5 block text-[12px] font-medium text-neutral-500">
              Hugging Face token
            </label>
            <input
              type="password"
              value={hfToken}
              onChange={(e) => persist("visionary.hf_token", e.target.value, setHfToken)}
              placeholder="hf_••••  (optional if set in Modal)"
              autoComplete="off"
              className="w-full rounded-xl border border-white/10 bg-white/[0.04] px-3.5 py-2.5 font-mono text-[13px] text-neutral-100 outline-none transition placeholder:text-neutral-600 focus:border-white/25"
            />
            <p className="mt-2.5 text-[12px] text-neutral-600">Stored in this browser only.</p>
          </div>
        </div>
      )}
    </div>
  );
}

const GearIcon = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <circle cx="12" cy="12" r="3.2" stroke="currentColor" strokeWidth="2" />
    <path
      d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9v0a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"
      stroke="currentColor"
      strokeWidth="1.6"
    />
  </svg>
);

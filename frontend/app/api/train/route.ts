import { NextResponse } from "next/server";

/**
 * GET /api/train  —  backend discovery.
 *
 * Purpose changed from the first pass, deliberately. It used to proxy the
 * training POST. It cannot proxy uploads: Vercel functions cap request bodies
 * at 4.5 MB and answer 413 FUNCTION_PAYLOAD_TOO_LARGE, while a dataset zip is
 * comfortably larger. Modal web endpoints accept 4 GiB, so the browser uploads
 * straight there.
 *
 * Once the dataset bypasses this route, proxying only the trigger call would
 * split auth across two places for no gain. So this route now does one job:
 * hand the browser the backend URLs from server env, so they are configured in
 * one place and changing them needs no rebuild.
 *
 * The access key is NOT served here — it is entered once in the UI and lives in
 * the browser. Serving it would publish it to anyone who loads the page.
 *
 * Env (Vercel project settings, or .env.local):
 *   MODAL_WORKSPACE=your-modal-workspace-slug
 * or set the three URLs explicitly:
 *   MODAL_UPLOAD_URL, MODAL_TRAIN_URL, MODAL_STATUS_URL
 */

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const workspace = process.env.MODAL_WORKSPACE;

  // Modal's deployed endpoint URLs follow
  // https://{workspace}--{label}.modal.run  — labels are set in main.py.
  // Modal deploys each labelled endpoint at https://{workspace}--{label}.modal.run
  // Labels are set in backend/main.py; keep these two lists in step.
  const LABELS = {
    uploadUrl: "upload-dataset",
    trainUrl: "trigger-train",
    statusUrl: "train-status",
    stopUrl: "stop-train",
    datasetUrl: "dataset",
    captionUrl: "caption",
    saveCaptionsUrl: "save-captions",
    generateUrl: "generate",
    lorasUrl: "loras",
  } as const;

  const OVERRIDES: Record<keyof typeof LABELS, string | undefined> = {
    uploadUrl: process.env.MODAL_UPLOAD_URL,
    trainUrl: process.env.MODAL_TRAIN_URL,
    statusUrl: process.env.MODAL_STATUS_URL,
    stopUrl: process.env.MODAL_STOP_URL,
    datasetUrl: process.env.MODAL_DATASET_URL,
    captionUrl: process.env.MODAL_CAPTION_URL,
    saveCaptionsUrl: process.env.MODAL_SAVE_CAPTIONS_URL,
    generateUrl: process.env.MODAL_GENERATE_URL,
    lorasUrl: process.env.MODAL_LORAS_URL,
  };

  const urls: Record<string, string> = {};
  const missing: string[] = [];
  for (const [key, label] of Object.entries(LABELS)) {
    const value =
      OVERRIDES[key as keyof typeof LABELS] ??
      (workspace ? `https://${workspace}--${label}.modal.run` : undefined);
    if (value) urls[key] = value;
    else missing.push(key);
  }

  if (missing.length) {
    return NextResponse.json(
      {
        status: "error",
        error:
          "Backend URLs are not configured. Set MODAL_WORKSPACE, or set each of: " +
          missing.join(", "),
      },
      { status: 500 },
    );
  }

  return NextResponse.json({ status: "ok", ...urls });
}

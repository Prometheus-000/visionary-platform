import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "Visionary — Train LoRA",
  description: "Krea 2 LoRA training on Modal.",
};

export const viewport: Viewport = {
  themeColor: "#000000",
  // The UI is built phone-first; lock the scale so the dropzone and number
  // inputs do not trigger zoom-on-focus in mobile Safari.
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

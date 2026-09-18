import type { Metadata, Viewport } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Realtime Translator",
  description: "Субтитры и перевод разговора в реальном времени",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#020617",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ru" className="dark">
      <body className="min-h-[100dvh] bg-slate-950 text-slate-100 antialiased">{children}</body>
    </html>
  );
}

import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono, Saira_Condensed } from "next/font/google";
import Script from "next/script";
import "./globals.css";
import { THEME_INIT_SCRIPT } from "@/lib/theme";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const monitorNumerals = Saira_Condensed({
  variable: "--font-bm-num",
  weight: ["500", "600"],
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "VITAL - Vision-Based Intelligent Triage & Lifesign Analytics",
  description: "Vision-based intelligent triage and autonomous lifesign analytics dashboard for clinicians.",
};

export const viewport: Viewport = {
  themeColor: [
    { color: "#fbf8f5" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      data-theme="light"
      suppressHydrationWarning
      className={`${geistSans.variable} ${geistMono.variable} ${monitorNumerals.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        {/* Applies the saved/OS theme to <html> before first paint (no flash of the wrong theme). */}
        <Script id="vital-theme-init" strategy="beforeInteractive">{THEME_INIT_SCRIPT}</Script>
        {children}
      </body>
    </html>
  );
}

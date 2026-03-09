import type { ReactNode } from "react";
import "./globals.css";

export const metadata = {
  title: "Enterprise Support Copilot",
  description: "Minimal UI for uploading documents and asking questions.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}


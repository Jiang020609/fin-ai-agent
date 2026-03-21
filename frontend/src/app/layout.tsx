import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FinAI - 智能金融问答",
  description: "基于大模型的金融资产问答系统",
  viewport: "width=device-width, initial-scale=1, viewport-fit=cover",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}

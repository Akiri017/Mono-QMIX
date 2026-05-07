import type { Metadata } from "next"
import "./globals.css"
import { ThemeProvider } from "@/contexts/ThemeContext"

export const metadata: Metadata = {
  title: "Civiq - Hierarchical Multi-Agent Coordination Framework",
  description: "A Hierarchical Multi-Agent Coordination Framework using QMIX for Urban Optimization",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="font-inter">
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  )
}

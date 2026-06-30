import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import { ThemeProvider } from "@/contexts/ThemeContext"
import { RoleProvider } from "@/contexts/RoleContext"
import { RoleSelectModal } from "@/components/RoleSelectModal"

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
})

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
    <html lang="en" className={inter.variable}>
      <body>
        <ThemeProvider>
          <RoleProvider>
            {children}
            <RoleSelectModal />
          </RoleProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}

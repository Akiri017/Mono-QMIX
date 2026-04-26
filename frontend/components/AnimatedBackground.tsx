'use client'

import { useTheme } from '@/contexts/ThemeContext'

export const AnimatedBackground = () => {
  const { theme } = useTheme()
  const isDark = theme === 'dark'

  const darkBlobs = [
    { width: 640, height: 640, background: 'radial-gradient(circle at 38% 38%, #7c3aed 0%, #4c1d95 55%, transparent 100%)', top: '-200px', left: '-170px', animation: 'blob1 14s ease-in-out infinite', animationDelay: '0s' },
    { width: 580, height: 580, background: 'radial-gradient(circle at 60% 38%, #06b6d4 0%, #0e4f6b 55%, transparent 100%)', top: '-120px', right: '-180px', animation: 'blob2 17s ease-in-out infinite', animationDelay: '-4s' },
    { width: 520, height: 520, background: 'radial-gradient(circle at 50% 58%, #db2777 0%, #7d1050 55%, transparent 100%)', bottom: '4%', left: '12%', animation: 'blob3 15s ease-in-out infinite', animationDelay: '-7s' },
    { width: 480, height: 480, background: 'radial-gradient(circle at 44% 54%, #2563eb 0%, #1e3a8a 55%, transparent 100%)', bottom: '8%', right: '4%', animation: 'blob4 18s ease-in-out infinite', animationDelay: '-2s' },
    { width: 400, height: 400, background: 'radial-gradient(circle at 50% 50%, #6d28d9 0%, #3b0764 55%, transparent 100%)', top: '40%', left: '40%', transform: 'translate(-50%, -50%)', animation: 'blob5 20s ease-in-out infinite', animationDelay: '-10s' },
  ]

  // Saturated cores fading to transparent — visible against a light page
  const lightBlobs = [
    { width: 700, height: 700, background: 'radial-gradient(circle at 38% 38%, #8b5cf6 0%, rgba(139,92,246,0.35) 45%, transparent 100%)', top: '-220px', left: '-190px', animation: 'blob1 14s ease-in-out infinite', animationDelay: '0s' },
    { width: 640, height: 640, background: 'radial-gradient(circle at 60% 38%, #06b6d4 0%, rgba(6,182,212,0.3) 48%, transparent 100%)',  top: '-140px', right: '-200px', animation: 'blob2 17s ease-in-out infinite', animationDelay: '-4s' },
    { width: 580, height: 580, background: 'radial-gradient(circle at 50% 58%, #ec4899 0%, rgba(236,72,153,0.28) 48%, transparent 100%)', bottom: '2%',  left: '10%',   animation: 'blob3 15s ease-in-out infinite', animationDelay: '-7s' },
    { width: 520, height: 520, background: 'radial-gradient(circle at 44% 54%, #3b82f6 0%, rgba(59,130,246,0.28) 48%, transparent 100%)',  bottom: '6%',  right: '2%',   animation: 'blob4 18s ease-in-out infinite', animationDelay: '-2s' },
    { width: 440, height: 440, background: 'radial-gradient(circle at 50% 50%, #a855f7 0%, rgba(168,85,247,0.22) 48%, transparent 100%)',  top: '40%',   left: '40%',   transform: 'translate(-50%, -50%)', animation: 'blob5 20s ease-in-out infinite', animationDelay: '-10s' },
  ]

  const blobs = isDark ? darkBlobs : lightBlobs

  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none" style={{ zIndex: 0 }}>
      {blobs.map((blob, i) => (
        <div key={i} className="absolute rounded-full" style={{ ...blob, filter: 'blur(80px)', opacity: isDark ? 0.55 : 0.65 }} />
      ))}
    </div>
  )
}

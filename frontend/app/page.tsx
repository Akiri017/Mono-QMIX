'use client'

import { useState, useEffect, useRef } from 'react'
import { AnimatedBackground } from '@/components/AnimatedBackground'
import { SimulationControls } from '@/components/SimulationControls'
import { useTheme } from '@/contexts/ThemeContext'

const researchers = [
  { name: 'Kristian Bautista', role: 'Project Manager',  email: 'kristiandavidbautista@gmail.com',  avatar: '/images/researchers/Kristian_David_R_Bautista.jpg' },
  { name: 'Angel Letada',      role: 'SUMO Engineer',     email: 'angel.letada1205@gmail.com',        avatar: '/images/researchers/Angel_A_Letada.png' },
  { name: 'Michael Pascual',   role: 'SUMO Engineer',     email: 'michaelkevinpascual47@gmail.com',   avatar: '/images/researchers/Michael_Kevin_D_Pascual.png' },
  { name: 'Marianne Santos',   role: 'Software Engineer', email: 'mariannesantos174@gmail.com',       avatar: '/images/researchers/Marianne_Angelika_B_Santos.png' },
]

// ── Hooks ──────────────────────────────────────────────────────────────────────

function useReveal(threshold = 0.12) {
  const ref = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    const el = ref.current; if (!el) return
    const obs = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) { setVisible(true); obs.disconnect() } },
      { threshold }
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [threshold])
  return { ref, visible }
}

function useCountUp(target: number, active: boolean, duration = 1400) {
  const [val, setVal] = useState(0)
  useEffect(() => {
    if (!active) return
    setVal(0)
    const t0 = performance.now(); let id: number
    const step = (now: number) => {
      const p = Math.min((now - t0) / duration, 1)
      setVal((1 - (1 - p) ** 3) * target)
      if (p < 1) id = requestAnimationFrame(step); else setVal(target)
    }
    id = requestAnimationFrame(step)
    return () => cancelAnimationFrame(id)
  }, [target, active, duration])
  return val
}

function useScrolled(threshold = 100) {
  const [scrolled, setScrolled] = useState(false)
  useEffect(() => {
    const fn = () => setScrolled(window.scrollY > threshold)
    fn()
    window.addEventListener('scroll', fn, { passive: true })
    return () => window.removeEventListener('scroll', fn)
  }, [threshold])
  return scrolled
}

function useActiveSection(ids: string[]) {
  const [active, setActive] = useState('')
  useEffect(() => {
    const observers = ids.map(id => {
      const el = document.getElementById(id); if (!el) return null
      const obs = new IntersectionObserver(
        ([e]) => { if (e.isIntersecting) setActive(id) },
        { threshold: 0.2, rootMargin: '-80px 0px -55% 0px' }
      )
      obs.observe(el); return obs
    })
    return () => observers.forEach(o => o?.disconnect())
  }, []) // eslint-disable-line react-hooks/exhaustive-deps
  return active
}

// ── Shared sub-components ─────────────────────────────────────────────────────

function GlassCard({ children, className = '', style, ...rest }: { children: React.ReactNode } & React.HTMLAttributes<HTMLDivElement>) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'
  const base: React.CSSProperties = {
    background: isDark
      ? 'linear-gradient(155deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.04) 100%)'
      : 'linear-gradient(155deg, rgba(255,255,255,0.72) 0%, rgba(255,255,255,0.52) 100%)',
    backdropFilter: 'blur(28px)', WebkitBackdropFilter: 'blur(28px)',
    border: `1px solid ${isDark ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.7)'}`,
    borderRadius: '16px',
    boxShadow: isDark
      ? 'inset 0 1px 0 rgba(255,255,255,0.1), 0 8px 32px rgba(0,0,0,0.28)'
      : 'inset 0 1px 0 rgba(255,255,255,0.9), 0 4px 24px rgba(99,102,241,0.07), 0 1px 4px rgba(15,23,42,0.06)',
  }
  return (
    <div className={className} style={{ ...base, ...style }} {...rest}>{children}</div>
  )
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'
  return (
    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full mb-4" style={{
      background: isDark ? 'rgba(6,182,212,0.1)' : 'rgba(255,255,255,0.6)',
      border: `1px solid ${isDark ? 'rgba(6,182,212,0.25)' : 'rgba(2,132,199,0.3)'}`,
      backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)',
    }}>
      <div className="w-1.5 h-1.5 rounded-full" style={{ background: isDark ? '#06B6D4' : '#0284C7' }} />
      <span className="text-[10px] font-bold tracking-widest uppercase" style={{ color: isDark ? '#06B6D4' : '#0369a1' }}>{children}</span>
    </div>
  )
}

function ThemeToggle() {
  const { theme, toggle } = useTheme()
  const isDark = theme === 'dark'
  const [spinKey, setSpinKey] = useState(0)

  const handleToggle = () => {
    setSpinKey(k => k + 1)
    toggle()
  }

  return (
    <button
      onClick={handleToggle}
      aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      className="w-8 h-8 rounded-full flex items-center justify-center transition-all duration-200"
      style={{
        background: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.55)',
        border: `1px solid ${isDark ? 'rgba(255,255,255,0.14)' : 'rgba(255,255,255,0.8)'}`,
        backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)',
        color: isDark ? 'rgba(255,255,255,0.6)' : '#374151',
        boxShadow: isDark ? 'none' : '0 1px 4px rgba(15,23,42,0.08)',
      }}
      onMouseEnter={(e) => {
        const el = e.currentTarget as HTMLElement
        el.style.background = isDark ? 'rgba(255,255,255,0.14)' : 'rgba(255,255,255,0.85)'
        el.style.color = isDark ? '#fff' : '#111827'
      }}
      onMouseLeave={(e) => {
        const el = e.currentTarget as HTMLElement
        el.style.background = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.55)'
        el.style.color = isDark ? 'rgba(255,255,255,0.6)' : '#374151'
      }}
    >
      <div key={spinKey} style={{ animation: spinKey > 0 ? 'spin-icon 0.42s cubic-bezier(0.34, 1.3, 0.64, 1) forwards' : 'none', display: 'flex' }}>
        {isDark ? (
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="5" />
            <line x1="12" y1="1" x2="12" y2="3" /><line x1="12" y1="21" x2="12" y2="23" />
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" /><line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
            <line x1="1" y1="12" x2="3" y2="12" /><line x1="21" y1="12" x2="23" y2="12" />
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" /><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
          </svg>
        )}
      </div>
    </button>
  )
}

// ── Sticky nav (appears on scroll) ────────────────────────────────────────────

function StickyNav({ visible, activeSection }: { visible: boolean; activeSection: string }) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'
  const NAV = [
    { label: 'About',        href: '#about',    id: 'about' },
    { label: 'Contact Us',   href: '#contact',  id: 'contact' },
  ]
  return (
    <div
      aria-hidden={!visible}
      style={{
        position: 'fixed', top: 0, left: 0, right: 0, zIndex: 1000,
        transform: visible ? 'translateY(0)' : 'translateY(-110%)',
        transition: 'transform 0.35s cubic-bezier(0.4, 0, 0.2, 1)',
        background: isDark ? 'rgba(6,1,18,0.85)' : 'rgba(240,246,255,0.88)',
        backdropFilter: 'blur(20px)', WebkitBackdropFilter: 'blur(20px)',
        borderBottom: `1px solid ${isDark ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.6)'}`,
        boxShadow: isDark ? '0 4px 24px rgba(0,0,0,0.45)' : '0 4px 20px rgba(15,23,42,0.09)',
        padding: '10px clamp(14px, 4vw, 28px)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}
    >
      <div className="flex items-center gap-2.5">
        <img src="/icons/civiq-logo.png" alt="Civiq"
          className={`w-5 h-5 object-contain flex-shrink-0 ${isDark ? 'brightness-0 invert opacity-80' : 'opacity-90'}`} />
        <span className="font-bold text-[13px] tracking-widest" style={{ color: isDark ? 'rgba(255,255,255,0.75)' : '#1e293b' }}>CIVIQ</span>
      </div>
      <div className="flex items-center gap-0.5 sm:gap-1">
        {NAV.map(({ label, href, id }) => {
          const isActive = id && activeSection === id
          return (
            <a key={label} href={href}
              className="px-2.5 sm:px-4 py-1.5 rounded-full text-[11px] sm:text-[12px] font-medium transition-all duration-150 whitespace-nowrap"
              style={{
                color: isActive ? (isDark ? '#38BDF8' : '#1d4ed8') : (isDark ? 'rgba(255,255,255,0.52)' : '#374151'),
                background: isActive ? (isDark ? 'rgba(56,189,248,0.1)' : 'rgba(29,78,216,0.08)') : 'transparent',
              }}
              onMouseEnter={(e) => {
                if (isActive) return
                const el = e.currentTarget as HTMLElement
                el.style.color = isDark ? 'rgba(255,255,255,0.9)' : '#111827'
                el.style.background = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(255,255,255,0.6)'
              }}
              onMouseLeave={(e) => {
                if (isActive) return
                const el = e.currentTarget as HTMLElement
                el.style.color = isDark ? 'rgba(255,255,255,0.52)' : '#374151'
                el.style.background = 'transparent'
              }}>
              {label}
            </a>
          )
        })}
        <div className="ml-2"><ThemeToggle /></div>
      </div>
    </div>
  )
}

// ── Hero navbar (inside OBU) ──────────────────────────────────────────────────

function StatusBar({ activeSection }: { activeSection: string }) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'
  const NAV = [
    { label: 'About',        href: '#about',    id: 'about' },
    { label: 'Contact Us',   href: '#contact',  id: 'contact' },
  ]
  return (
    <div className="flex items-center justify-between flex-shrink-0" style={{
      padding: '10px clamp(14px, 4vw, 28px)',
      background: isDark ? 'rgba(0,0,0,0.35)' : 'rgba(255,255,255,0.45)',
      borderBottom: `1px solid ${isDark ? 'rgba(255,255,255,0.06)' : 'rgba(255,255,255,0.6)'}`,
      backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)',
    }}>
      <div className="flex items-center gap-2 sm:gap-2.5 min-w-0">
        <img src="/icons/civiq-logo.png" alt="Civiq"
          className={`w-5 h-5 object-contain flex-shrink-0 ${isDark ? 'brightness-0 invert opacity-80' : 'opacity-90'}`} />
        <span className="font-bold text-[13px] tracking-widest flex-shrink-0" style={{ color: isDark ? 'rgba(255,255,255,0.75)' : '#1e293b' }}>CIVIQ</span>
        <span className="hidden md:inline text-[10px] font-medium truncate" style={{ color: isDark ? 'rgba(255,255,255,0.3)' : '#64748b' }}>
          ·&nbsp; A Hierarchical Multi-Agent Coordination Framework
        </span>
      </div>
      <div className="flex items-center gap-0.5 sm:gap-1 flex-shrink-0">
        {NAV.map(({ label, href, id }) => {
          const isActive = id && activeSection === id
          return (
            <a key={label} href={href}
              className="px-2.5 sm:px-4 py-1.5 rounded-full text-[11px] sm:text-[12px] font-medium transition-all duration-150 whitespace-nowrap"
              style={{
                color: isActive ? (isDark ? '#38BDF8' : '#1d4ed8') : (isDark ? 'rgba(255,255,255,0.52)' : '#374151'),
                background: isActive ? (isDark ? 'rgba(56,189,248,0.1)' : 'rgba(29,78,216,0.08)') : 'transparent',
              }}
              onMouseEnter={(e) => {
                if (isActive) return
                const el = e.currentTarget as HTMLElement
                el.style.color = isDark ? 'rgba(255,255,255,0.9)' : '#111827'
                el.style.background = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(255,255,255,0.6)'
              }}
              onMouseLeave={(e) => {
                if (isActive) return
                const el = e.currentTarget as HTMLElement
                el.style.color = isDark ? 'rgba(255,255,255,0.52)' : '#374151'
                el.style.background = 'transparent'
              }}>
              {label}
            </a>
          )
        })}
        <div className="ml-2"><ThemeToggle /></div>
      </div>
    </div>
  )
}

// ── KPI Card ──────────────────────────────────────────────────────────────────

interface KPICardProps {
  rawValue: number
  format: (n: number) => string
  unit: string
  label: string
  sub: string
  darkColor: string
  lightColor: string
  rgb: string
  icon: React.ReactNode
  sectionActive: boolean
  staggerMs: number
}

function KPICard({ rawValue, format, unit, label, sub, darkColor, lightColor, rgb, icon, sectionActive, staggerMs }: KPICardProps) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'
  const [active, setActive] = useState(false)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    if (!sectionActive) return
    const t = setTimeout(() => { setActive(true); setVisible(true) }, staggerMs)
    return () => clearTimeout(t)
  }, [sectionActive, staggerMs])

  const animVal = useCountUp(rawValue, active)
  const color = isDark ? darkColor : lightColor
  const cardBg = isDark ? `rgba(${rgb},0.10)` : 'rgba(255,255,255,0.9)'
  const cardBorder = isDark ? `rgba(${rgb},0.25)` : `rgba(${rgb},0.28)`
  const textPrimary = isDark ? 'rgba(255,255,255,0.88)' : '#111827'
  const textMuted = isDark ? 'rgba(255,255,255,0.35)' : '#6b7280'

  return (
    <GlassCard className="p-6" style={{
      border: `1px solid ${cardBorder}`,
      background: cardBg,
      opacity: visible ? 1 : 0,
      transform: visible ? 'translateY(0)' : 'translateY(24px)',
      transition: 'opacity 0.55s ease, transform 0.55s ease',
    }}>
      <div className="flex items-center justify-between mb-4">
        <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: `rgba(${rgb},0.15)`, color }}>
          {icon}
        </div>
        <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full"
          style={{ background: `rgba(${rgb},0.12)`, color, border: `1px solid ${cardBorder}` }}>Civiq</span>
      </div>
      <div className="flex items-baseline gap-1.5 mb-1">
        <span className="text-[34px] font-black tabular-nums leading-none" style={{ color }}>{format(animVal)}</span>
        <span className="text-[13px] font-semibold" style={{ color: textMuted }}>{unit}</span>
      </div>
      <div className="text-[13px] font-semibold mb-0.5" style={{ color: textPrimary }}>{label}</div>
      <div className="text-[11px]" style={{ color: textMuted }}>{sub}</div>
    </GlassCard>
  )
}

// ── Researcher Card ───────────────────────────────────────────────────────────

function ResearcherCard({ name, role, email, avatar, staggerMs, sectionActive }: {
  name: string; role: string; email: string; avatar: string; staggerMs: number; sectionActive: boolean
}) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'
  const [visible, setVisible] = useState(false)
  const [hovered, setHovered] = useState(false)
  const [emailHov, setEmailHov] = useState(false)

  useEffect(() => {
    if (!sectionActive) return
    const t = setTimeout(() => setVisible(true), staggerMs)
    return () => clearTimeout(t)
  }, [sectionActive, staggerMs])

  const textPrimary = isDark ? 'rgba(255,255,255,0.88)' : '#111827'
  const textMuted = isDark ? 'rgba(255,255,255,0.35)' : '#6b7280'
  const avatarShadow = isDark ? '0 4px 20px rgba(0,0,0,0.4)' : '0 4px 16px rgba(15,23,42,0.12)'
  const avatarRing = isDark ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.7)'
  const emailColor = emailHov ? (isDark ? '#38BDF8' : '#1d4ed8') : (isDark ? 'rgba(255,255,255,0.3)' : '#6b7280')

  return (
    <div style={{
      opacity: visible ? 1 : 0,
      transform: visible ? 'none' : 'translateY(20px)',
      transition: 'opacity 0.5s ease, transform 0.5s ease',
    }}>
      <GlassCard
        className="p-5 flex flex-col items-center text-center gap-3 cursor-default"
        style={{
          transition: 'transform 0.22s ease, box-shadow 0.22s ease',
          transform: hovered ? 'translateY(-5px)' : 'none',
          boxShadow: hovered
            ? isDark
              ? 'inset 0 1px 0 rgba(255,255,255,0.1), 0 20px 48px rgba(0,0,0,0.5)'
              : 'inset 0 1px 0 rgba(255,255,255,0.9), 0 16px 40px rgba(99,102,241,0.18), 0 4px 16px rgba(15,23,42,0.12)'
            : undefined,
        }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        <div className="w-20 h-20 rounded-full overflow-hidden"
          style={{ boxShadow: avatarShadow, outline: `2px solid ${avatarRing}`, outlineOffset: '2px' }}>
          <img src={avatar} alt={name} className="w-full h-full object-cover" />
        </div>
        <div>
          <p className="text-[13px] font-bold mb-0.5" style={{ color: textPrimary }}>{name}</p>
          <p className="text-[11px] italic mb-2" style={{ color: textMuted }}>{role}</p>
          <a href={`mailto:${email}`} className="text-[10px] break-all"
            style={{ color: emailColor, transition: 'color 0.15s' }}
            onMouseEnter={() => setEmailHov(true)}
            onMouseLeave={() => setEmailHov(false)}>
            {email}
          </a>
        </div>
      </GlassCard>
    </div>
  )
}

// ── Footer link with animated arrow ──────────────────────────────────────────

function FooterLink({ label, href, isDark }: { label: string; href: string; isDark: boolean }) {
  const [hov, setHov] = useState(false)
  const textSecondary = isDark ? 'rgba(255,255,255,0.55)' : '#374151'
  const textPrimary = isDark ? 'rgba(255,255,255,0.88)' : '#111827'
  return (
    <a href={href}
      className="flex items-center gap-1 text-[13px]"
      style={{ color: hov ? textPrimary : textSecondary, transition: 'color 0.15s' }}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}>
      {label}
      <span style={{
        display: 'inline-block',
        opacity: hov ? 1 : 0,
        transform: hov ? 'translateX(0)' : 'translateX(-4px)',
        transition: 'opacity 0.15s, transform 0.15s',
        fontSize: '12px',
      }}>→</span>
    </a>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Home() {
  const { theme } = useTheme()
  const isDark = theme === 'dark'

  // Scroll & section state
  const heroScrolled = useScrolled(600)
  const activeSection = useActiveSection(['about', 'researchers', 'contact'])

  // Reveal observers
  const aboutReveal       = useReveal(0.1)
  const kpiReveal         = useReveal(0.1)
  const researchersReveal = useReveal(0.1)

  // Hero chevron fade
  const chevronHidden = useScrolled(120)

  const c = {
    pageBg:          isDark ? '#060112' : '#dde9f8',
    pageBgGrad:      isDark
      ? 'linear-gradient(135deg, #060112 0%, #0b0320 40%, #040c1c 100%)'
      : 'linear-gradient(135deg, #dde9f8 0%, #d5e3f5 40%, #e0edf9 100%)',
    heroBg:          isDark ? 'rgba(8,14,32,0.48)'  : 'rgba(255,255,255,0.22)',
    heroInner:       isDark ? 'rgba(6,11,26,0.62)'  : 'rgba(210,228,255,0.22)',
    heroBorder:      isDark ? 'rgba(255,255,255,0.11)' : 'rgba(255,255,255,0.75)',
    heroShadow:      isDark
      ? '0 32px 80px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.14), inset 0 -1px 0 rgba(0,0,0,0.3)'
      : '0 32px 80px rgba(99,102,241,0.1), 0 8px 32px rgba(15,23,42,0.06), inset 0 1px 0 rgba(255,255,255,0.9)',
    heroInnerBorder: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.6)',
    heroInsetTop:    isDark ? 'rgba(255,255,255,0.05)' : 'rgba(255,255,255,0.5)',
    heroDivider:     isDark ? 'rgba(255,255,255,0.09)' : 'rgba(99,102,241,0.15)',
    heroCtrlBg:      isDark ? 'rgba(255,255,255,0.04)' : 'rgba(255,255,255,0.82)',
    heroCtrlBorder:  isDark ? 'rgba(255,255,255,0.08)' : 'rgba(15,23,42,0.1)',
    heroCtrlInset:   isDark ? 'rgba(255,255,255,0.05)' : 'rgba(255,255,255,0.95)',
    h1Grad:          isDark
      ? 'linear-gradient(140deg, #93C5FD 0%, #60A5FA 40%, #38BDF8 100%)'
      : 'linear-gradient(140deg, #1e3a8a 0%, #1d4ed8 45%, #0369a1 100%)',
    headingGrad:     isDark
      ? 'linear-gradient(140deg, #93C5FD 0%, #38BDF8 100%)'
      : 'linear-gradient(140deg, #1e3a8a 0%, #0369a1 100%)',
    textBody:        isDark ? 'rgba(255,255,255,0.52)' : '#374151',
    textPrimary:     isDark ? 'rgba(255,255,255,0.88)' : '#111827',
    textSecondary:   isDark ? 'rgba(255,255,255,0.55)' : '#374151',
    textMuted:       isDark ? 'rgba(255,255,255,0.35)' : '#6b7280',
    textUltraMuted:  isDark ? 'rgba(255,255,255,0.25)' : '#9ca3af',
    accent:          isDark ? '#38BDF8' : '#1d4ed8',
    accentLight:     isDark ? '#06B6D4' : '#0284C7',
    greenEmphasis:   isDark ? '#4ADE80' : '#15803d',
    sectionBorder:   isDark ? 'rgba(255,255,255,0.06)' : 'rgba(255,255,255,0.5)',
    footerBg:        isDark ? 'rgba(0,0,0,0.25)' : 'rgba(255,255,255,0.2)',
    socialBg:        isDark ? 'rgba(255,255,255,0.06)' : 'rgba(255,255,255,0.5)',
    socialBorder:    isDark ? 'rgba(255,255,255,0.1)'  : 'rgba(255,255,255,0.7)',
    socialColor:     isDark ? 'rgba(255,255,255,0.45)' : '#4b5563',
    socialBgHover:   isDark ? 'rgba(6,182,212,0.12)'  : 'rgba(255,255,255,0.8)',
    dotBg:           isDark
      ? 'linear-gradient(135deg,rgba(255,255,255,0.18),rgba(255,255,255,0.04))'
      : 'linear-gradient(135deg,rgba(255,255,255,0.7),rgba(255,255,255,0.3))',
    dotShadow:       isDark ? 'inset 0 1px 2px rgba(0,0,0,0.6)' : 'inset 0 1px 2px rgba(15,23,42,0.06)',
    ringTrack:       isDark ? 'rgba(255,255,255,0.08)' : 'rgba(15,23,42,0.1)',
    avatarShadow:    isDark ? '0 4px 20px rgba(0,0,0,0.4)' : '0 4px 16px rgba(15,23,42,0.12)',
    avatarRing:      isDark ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.7)',
  }

  const kpiData = [
    {
      rawValue: 2.0, format: (n: number) => n.toFixed(1),
      unit: 'min', label: 'Average Travel Time', sub: 'Consistent across free-flow to forced-flow',
      darkColor: '#38BDF8', lightColor: '#0369a1', rgb: '56,189,248',
      icon: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>,
    },
    {
      rawValue: 2273, format: (n: number) => Math.round(n).toLocaleString(),
      unit: 'veh/hr', label: 'Peak Network Throughput', sub: '37.4% more than QMIX at max congestion',
      darkColor: '#A78BFA', lightColor: '#6d28d9', rgb: '167,139,250',
      icon: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>,
    },
    {
      rawValue: 19.3, format: (n: number) => n.toFixed(1),
      unit: 'sec', label: 'Wait Time at Peak Load', sub: '18.8% less waiting than QMIX under congestion',
      darkColor: '#4ADE80', lightColor: '#15803d', rgb: '74,222,128',
      icon: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>,
    },
  ]

  return (
    <main className="w-full" style={{ background: c.pageBg }}>
      <div className="fixed inset-0 pointer-events-none" style={{ background: c.pageBgGrad, zIndex: -1 }} />
      <AnimatedBackground />

      {/* Sticky navbar — slides in after scrolling past hero */}
      <StickyNav visible={heroScrolled} activeSection={activeSection} />

      {/* ── HERO ── */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden"
        style={{ zIndex: 2, padding: 'clamp(16px, 3vw, 48px)' }}>
        <div className="relative w-full mx-auto" style={{
          maxWidth: 'min(1160px, 100%)',
          background: c.heroBg, backdropFilter: 'blur(20px)', WebkitBackdropFilter: 'blur(20px)',
          borderRadius: '24px', padding: '12px',
          border: `1px solid ${c.heroBorder}`,
          boxShadow: c.heroShadow,
        }}>
          {(['left-2', 'right-2'] as const).map(side => (
            <div key={side} className={`absolute ${side} top-1/2 -translate-y-1/2 flex flex-col gap-2.5 pointer-events-none`}>
              {[0, 1, 2].map(i => <div key={i} className="w-2 h-2 rounded-full" style={{ background: c.dotBg, boxShadow: c.dotShadow }} />)}
            </div>
          ))}
          <div className="relative w-full flex flex-col overflow-hidden" style={{
            background: c.heroInner, backdropFilter: 'blur(24px)', WebkitBackdropFilter: 'blur(24px)',
            borderRadius: '14px', border: `1px solid ${c.heroInnerBorder}`,
            boxShadow: isDark ? 'inset 0 1px 0 rgba(255,255,255,0.06)' : 'inset 0 1px 0 rgba(255,255,255,0.8)',
          }}>
            <div className="absolute inset-x-0 top-0 h-24 pointer-events-none rounded-t-[14px]"
              style={{ background: `linear-gradient(180deg, ${c.heroInsetTop} 0%, transparent 100%)` }} />
            <StatusBar activeSection={activeSection} />
            <div className="flex flex-col items-center gap-5"
              style={{ padding: 'clamp(20px, 3vw, 40px) clamp(16px, 4vw, 48px) clamp(24px, 3vw, 44px)' }}>
              <div className="text-center w-full" style={{ maxWidth: '780px' }}>
                <h1 className="font-extrabold leading-[1.1] mb-4 select-none" style={{
                  fontSize: 'clamp(22px, 3vw, 42px)',
                  backgroundImage: c.h1Grad,
                  WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
                }}>
                  Solving Urban Congestion Through Hierarchical Coordination.
                </h1>
                <p className="leading-[1.75] mb-6" style={{ fontSize: 'clamp(12px, 1.1vw, 14px)', color: c.textBody }}>
                  An undergraduate thesis applying{' '}
                  <span className="font-semibold" style={{ color: c.accent }}>Hierarchical Multi-Agent Reinforcement Learning</span>{' '}
                  to urban traffic management. Vehicles act as intelligent agents that cooperatively learn routing decisions — reducing congestion across a simulated road network.
                </p>
                {/* CTA button with animated arrow */}
                <CTAButton isDark={isDark} />
              </div>
              <div className="w-full" style={{ height: '1px', background: `linear-gradient(90deg, transparent, ${c.heroDivider}, transparent)` }} />
              <div className="w-full" style={{
                maxWidth: '860px', background: c.heroCtrlBg, backdropFilter: 'blur(12px)',
                WebkitBackdropFilter: 'blur(12px)', borderRadius: '18px',
                border: `1px solid ${c.heroCtrlBorder}`,
                boxShadow: isDark ? `inset 0 1px 0 ${c.heroCtrlInset}` : `inset 0 1px 0 ${c.heroCtrlInset}, 0 4px 20px rgba(15,23,42,0.08)`,
                padding: 'clamp(16px, 2vw, 24px)',
              }}>
                <SimulationControls darkMode={isDark} />
              </div>
            </div>
          </div>
        </div>

        {/* Scroll-down chevron — outer fades on scroll, inner bounces */}
        <div
          aria-hidden="true"
          style={{
            position: 'absolute', bottom: '28px', left: '50%',
            transform: 'translateX(-50%)',
            opacity: chevronHidden ? 0 : 1,
            transition: 'opacity 0.4s ease',
            pointerEvents: 'none',
          }}
        >
          <div style={{ animation: 'bounce-chevron 2s ease-in-out infinite', color: isDark ? 'rgba(255,255,255,0.3)' : 'rgba(15,23,42,0.3)' }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </div>
        </div>
      </section>

      {/* ── ABOUT ── */}
      <section id="about" className="relative w-full py-24" style={{ zIndex: 2 }}>
        <div className="max-w-6xl mx-auto px-6 md:px-8 space-y-20">

          {/* About Civiq */}
          <div ref={aboutReveal.ref} style={{
            opacity: aboutReveal.visible ? 1 : 0,
            transform: aboutReveal.visible ? 'none' : 'translateY(28px)',
            transition: 'opacity 0.65s ease, transform 0.65s ease',
          }}>
            <SectionLabel>About Civiq</SectionLabel>
            <h2 className="text-[32px] md:text-[36px] font-extrabold mb-6 leading-tight" style={{
              backgroundImage: c.headingGrad,
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
            }}>
              Redefining Urban Traffic Management
            </h2>
            <p className="text-[15px] leading-relaxed max-w-3xl" style={{ color: c.textSecondary }}>
              Civiq is a hierarchical software framework designed to redefine urban traffic management through cooperative intelligence. Built around the QMIX algorithm, Civiq addresses the scalability challenges of traditional Multi-Agent Reinforcement Learning by utilizing a three-level architecture: individual vehicle agents, local coordination via Roadside Units, and a central server for global optimization. By integrating directly with the SUMO environment, Civiq transforms selfish routing behaviors into a synchronized network, significantly improving throughput and reducing urban congestion.
            </p>
          </div>

          {/* KPI cards */}
          <div ref={kpiReveal.ref}>
            <div style={{
              opacity: kpiReveal.visible ? 1 : 0,
              transform: kpiReveal.visible ? 'none' : 'translateY(20px)',
              transition: 'opacity 0.5s ease, transform 0.5s ease',
            }}>
              <SectionLabel>Smart Traffic Routing</SectionLabel>
              <h3 className="text-[24px] md:text-[28px] font-bold mb-8" style={{ color: c.textPrimary }}>
                Scalable by Design, Fast in Practice
              </h3>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5 mb-6">
              {kpiData.map((card, i) => (
                <KPICard key={card.label} {...card} sectionActive={kpiReveal.visible} staggerMs={i * 110} />
              ))}
            </div>
            <div style={{
              opacity: kpiReveal.visible ? 1 : 0,
              transform: kpiReveal.visible ? 'none' : 'translateY(16px)',
              transition: 'opacity 0.6s ease 0.35s, transform 0.6s ease 0.35s',
            }}>
              <GlassCard className="p-7" style={!isDark ? {
                background: 'linear-gradient(155deg, rgba(255,255,255,0.36) 0%, rgba(255,255,255,0.18) 100%)',
                border: '1px solid rgba(255,255,255,0.56)',
                boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.78), 0 4px 18px rgba(15,23,42,0.05)',
              } : {}}>
                <p className="text-[14px] leading-relaxed" style={{ color: c.textSecondary }}>
                  Civiq transforms congested urban traffic into a coordinated and efficient network by enabling vehicles to operate cooperatively rather than competitively. Under maximum congestion (LOS E), Civiq achieves{' '}
                  <span className="font-semibold" style={{ color: c.accent }}>2,273 vehicles per hour</span> — a 37.4% improvement over Monolithic QMIX — while keeping average wait time to just 19.3 seconds, 18.8% lower than QMIX at the same conditions. Across all traffic levels, CiViQ maintains a stable average travel time of approximately 2 minutes.
                </p>
              </GlassCard>
            </div>
          </div>

        </div>
      </section>

      {/* ── RESEARCHERS ── */}
      <section id="researchers" className="relative w-full py-24"
        style={{ zIndex: 2, borderTop: `1px solid ${c.sectionBorder}` }}>
        <div className="max-w-5xl mx-auto px-6 md:px-8">
          <div ref={researchersReveal.ref} className="text-center mb-14" style={{
            opacity: researchersReveal.visible ? 1 : 0,
            transform: researchersReveal.visible ? 'none' : 'translateY(20px)',
            transition: 'opacity 0.55s ease, transform 0.55s ease',
          }}>
            <SectionLabel>The Team</SectionLabel>
            <h2 className="text-[32px] md:text-[36px] font-extrabold" style={{
              backgroundImage: c.headingGrad,
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
            }}>The Researchers</h2>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-5 md:gap-6">
            {researchers.map((r, i) => (
              <ResearcherCard key={r.name} {...r} staggerMs={i * 80} sectionActive={researchersReveal.visible} />
            ))}
          </div>
        </div>
      </section>

      {/* ── FOOTER ── */}
      <footer id="contact" className="relative w-full py-14"
        style={{ zIndex: 2, borderTop: `1px solid ${c.sectionBorder}`, background: c.footerBg, backdropFilter: 'blur(16px)', WebkitBackdropFilter: 'blur(16px)' }}>
        <div className="max-w-5xl mx-auto px-6 md:px-8">
          <div className="flex flex-col md:flex-row items-start justify-between gap-10 md:gap-12 mb-12">
            <div className="max-w-xs">
              <div className="flex items-center gap-2.5 mb-3">
                <img src="/icons/civiq-logo.png" alt="Civiq"
                  className={`w-6 h-6 object-contain ${isDark ? 'brightness-0 invert opacity-75' : 'opacity-85'}`} />
                <span className="font-bold text-[14px] tracking-widest" style={{ color: c.textPrimary }}>CIVIQ</span>
              </div>
              <p className="text-[12px] leading-relaxed" style={{ color: c.textMuted }}>
                A Hierarchical Multi-Agent Coordination Framework for urban traffic management.
              </p>
              <div className="flex gap-3 mt-4">
                {[
                  { label: 'GitHub',   href: 'https://github.com',   path: 'M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.6.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z' },
                  { label: 'LinkedIn', href: 'https://linkedin.com', path: 'M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.225 0z' },
                ].map(({ label, href, path }) => (
                  <a key={label} href={href} target="_blank" rel="noopener noreferrer" aria-label={label}
                    className="w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-150"
                    style={{ background: c.socialBg, border: `1px solid ${c.socialBorder}`, color: c.socialColor }}
                    onMouseEnter={(e) => { const el = e.currentTarget as HTMLElement; el.style.background = c.socialBgHover; el.style.color = c.accentLight }}
                    onMouseLeave={(e) => { const el = e.currentTarget as HTMLElement; el.style.background = c.socialBg; el.style.color = c.socialColor }}>
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d={path} /></svg>
                  </a>
                ))}
              </div>
            </div>
            <div className="flex gap-12 md:gap-16">
              <div>
                <h4 className="text-[10px] font-bold uppercase tracking-widest mb-4" style={{ color: c.textMuted }}>Resources</h4>
                <ul className="space-y-2.5">
                  {[
                    { label: 'About Civiq',     href: '/#about' },
                    { label: 'The Researchers', href: '/#researchers' },
                  ].map(({ label, href }) => (
                    <li key={label}>
                      <FooterLink label={label} href={href} isDark={isDark} />
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
          <div style={{ borderTop: `1px solid ${c.sectionBorder}`, paddingTop: '24px' }}>
            <p className="text-[12px] text-center" style={{ color: c.textUltraMuted }}>
              © {new Date().getFullYear()} A Priori Group. All rights reserved.
            </p>
          </div>
        </div>
      </footer>
    </main>
  )
}

// ── CTA Button (extracted to avoid inline hook) ───────────────────────────────

function CTAButton({ isDark }: { isDark: boolean }) {
  const [hov, setHov] = useState(false)
  return (
    <div className="flex gap-3 justify-center">
      <a href="#about"
        className="inline-flex items-center gap-2 font-semibold rounded-full transition-all duration-200 text-white"
        style={{
          padding: 'clamp(8px,1vw,10px) clamp(16px,2vw,24px)',
          fontSize: 'clamp(11px,1vw,13.5px)',
          background: 'linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%)',
          boxShadow: hov ? '0 6px 28px rgba(59,130,246,0.65)' : '0 4px 20px rgba(59,130,246,0.45)',
          transform: hov ? 'translateY(-1px)' : 'none',
        }}
        onMouseEnter={() => setHov(true)}
        onMouseLeave={() => setHov(false)}>
        <span>Explore CiViQ</span>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
          style={{ transform: hov ? 'translateY(2px)' : 'none', transition: 'transform 0.2s ease' }}>
          <path d="M12 5v14m-7-7l7 7 7-7" />
        </svg>
      </a>
    </div>
  )
}

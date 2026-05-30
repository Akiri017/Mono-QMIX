'use client'

import { Suspense, useState, useEffect, useRef, memo, createContext, useContext } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import dynamic from 'next/dynamic'
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Bar, Cell, ReferenceArea,
} from 'recharts'
import { AnimatedBackground } from '@/components/AnimatedBackground'
import { SimulationControls } from '@/components/SimulationControls'
import { useTheme } from '@/contexts/ThemeContext'

// ─── Dashboard theme context ───────────────────────────────────────────────────

interface DashColors {
  isDark: boolean
  pageBg: string; pageBgGrad: string
  obuBg: string; obuBorder: string; obuShadow: string
  screenBg: string; screenBorder: string; screenShadow: string; screenGloss: string
  contentBg: string; contentBorder: string
  sidebarBg: string; sidebarBorder: string
  statusBg: string; statusBorder: string
  glassBg: string; glassBorder: string; glassShadow: string
  tp: string; ts: string; tm: string; tu: string   // text primary/secondary/muted/ultra-muted
  divider: string
  badgeBg: string; badgeBorder: string; badgeColor: string
  itemBg: string; itemBgMd: string; itemBgHi: string
  chartGrid: string
  chartAxis: { fill: string; fontSize: number }
  chartAxisLine: { stroke: string }
  chartTickLine: { stroke: string }
  chartLabel: string
  barTrack: string; ringTrack: string
  tooltipBg: string; tooltipBorder: string; tooltipColor: string; tooltipShadow: string
  screwBg: string; screwShadow: string
  blob1: string; blob2: string; blob3: string
  logoClass: string
  hoverBg: string; hoverBgMd: string; hoverColor: string
}

function buildDashColors(isDark: boolean): DashColors {
  return isDark ? {
    isDark: true,
    pageBg: '#060112',
    pageBgGrad: 'linear-gradient(135deg, #060112 0%, #0b0320 40%, #040c1c 100%)',
    obuBg: 'rgba(8,14,32,0.48)', obuBorder: 'rgba(255,255,255,0.11)',
    obuShadow: '0 32px 80px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.14), inset 0 -1px 0 rgba(0,0,0,0.3)',
    screenBg: 'rgba(6,11,26,0.45)', screenBorder: 'rgba(255,255,255,0.08)',
    screenShadow: 'inset 0 1px 0 rgba(255,255,255,0.06)',
    screenGloss: 'linear-gradient(180deg, rgba(255,255,255,0.05) 0%, transparent 100%)',
    contentBg: 'rgba(4,9,22,0.82)', contentBorder: 'rgba(255,255,255,0.07)',
    sidebarBg: 'rgba(0,0,0,0.2)', sidebarBorder: 'rgba(255,255,255,0.06)',
    statusBg: 'rgba(0,0,0,0.35)', statusBorder: 'rgba(255,255,255,0.06)',
    glassBg: 'linear-gradient(155deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.04) 100%)',
    glassBorder: 'rgba(255,255,255,0.14)',
    glassShadow: 'inset 0 1px 0 rgba(255,255,255,0.13), inset 0 -1px 0 rgba(0,0,0,0.15), 0 8px 32px rgba(0,0,0,0.32)',
    tp: 'rgba(255,255,255,0.9)', ts: 'rgba(255,255,255,0.72)',
    tm: 'rgba(255,255,255,0.55)', tu: 'rgba(255,255,255,0.45)',
    divider: 'rgba(255,255,255,0.10)',
    badgeBg: 'rgba(255,255,255,0.07)', badgeBorder: 'rgba(255,255,255,0.14)', badgeColor: 'rgba(255,255,255,0.60)',
    itemBg: 'rgba(255,255,255,0.04)', itemBgMd: 'rgba(255,255,255,0.07)', itemBgHi: 'rgba(255,255,255,0.12)',
    chartGrid: 'rgba(255,255,255,0.07)',
    chartAxis: { fill: 'rgba(255,255,255,0.52)', fontSize: 10 },
    chartAxisLine: { stroke: 'rgba(255,255,255,0.10)' },
    chartTickLine: { stroke: 'rgba(255,255,255,0.10)' },
    chartLabel: 'rgba(255,255,255,0.52)',
    barTrack: 'rgba(255,255,255,0.1)', ringTrack: 'rgba(255,255,255,0.08)',
    tooltipBg: 'rgba(4,9,22,0.97)', tooltipBorder: 'rgba(255,255,255,0.12)',
    tooltipColor: 'rgba(255,255,255,0.88)', tooltipShadow: '0 12px 24px rgba(0,0,0,0.45)',
    screwBg: 'linear-gradient(135deg,rgba(255,255,255,0.18),rgba(255,255,255,0.04))',
    screwShadow: 'inset 0 1px 2px rgba(0,0,0,0.6)',
    blob1: 'radial-gradient(circle at 40% 40%, rgba(6,182,212,0.28) 0%, transparent 70%)',
    blob2: 'radial-gradient(circle at 60% 55%, rgba(139,92,246,0.24) 0%, transparent 70%)',
    blob3: 'radial-gradient(circle at 50% 50%, rgba(37,99,235,0.18) 0%, transparent 70%)',
    logoClass: 'brightness-0 invert opacity-80',
    hoverBg: 'rgba(255,255,255,0.07)', hoverBgMd: 'rgba(255,255,255,0.14)', hoverColor: 'rgba(255,255,255,0.9)',
  } : {
    isDark: false,
    pageBg: '#e8f0fb',
    pageBgGrad: 'linear-gradient(135deg, #e8f0fb 0%, #dce8f8 42%, #f1f6fd 100%)',
    obuBg: 'rgba(255,255,255,0.62)', obuBorder: 'rgba(148,163,184,0.38)',
    obuShadow: '0 24px 60px rgba(15,23,42,0.14), 0 8px 26px rgba(15,23,42,0.08), inset 0 1px 0 rgba(255,255,255,0.95)',
    screenBg: 'rgba(255,255,255,0.76)', screenBorder: 'rgba(148,163,184,0.32)',
    screenShadow: 'inset 0 1px 0 rgba(255,255,255,0.94)',
    screenGloss: 'linear-gradient(180deg, rgba(255,255,255,0.8) 0%, transparent 100%)',
    contentBg: 'rgba(255,255,255,0.74)', contentBorder: 'rgba(148,163,184,0.35)',
    sidebarBg: 'rgba(255,255,255,0.68)', sidebarBorder: 'rgba(148,163,184,0.32)',
    statusBg: 'rgba(255,255,255,0.72)', statusBorder: 'rgba(148,163,184,0.35)',
    glassBg: 'linear-gradient(155deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.78) 100%)',
    glassBorder: 'rgba(148,163,184,0.42)',
    glassShadow: 'inset 0 1px 0 rgba(255,255,255,0.98), 0 6px 22px rgba(15,23,42,0.1), 0 1px 3px rgba(15,23,42,0.08)',
    tp: '#0b1220', ts: '#1f2d3d', tm: '#334155', tu: '#516274',
    divider: 'rgba(15,23,42,0.2)',
    badgeBg: 'rgba(255,255,255,0.9)', badgeBorder: 'rgba(148,163,184,0.4)', badgeColor: '#334155',
    itemBg: 'rgba(255,255,255,0.82)', itemBgMd: 'rgba(255,255,255,0.9)', itemBgHi: 'rgba(255,255,255,0.98)',
    chartGrid: 'rgba(15,23,42,0.14)',
    chartAxis: { fill: '#334155', fontSize: 10 },
    chartAxisLine: { stroke: 'rgba(15,23,42,0.2)' },
    chartTickLine: { stroke: 'rgba(15,23,42,0.16)' },
    chartLabel: '#475569',
    barTrack: 'rgba(15,23,42,0.08)', ringTrack: 'rgba(15,23,42,0.1)',
    tooltipBg: 'rgba(15,23,42,0.97)', tooltipBorder: 'rgba(255,255,255,0.08)',
    tooltipColor: 'rgba(255,255,255,0.88)', tooltipShadow: '0 12px 24px rgba(15,23,42,0.2)',
    screwBg: 'linear-gradient(135deg,rgba(255,255,255,0.95),rgba(255,255,255,0.62))',
    screwShadow: 'inset 0 1px 2px rgba(15,23,42,0.14)',
    blob1: 'radial-gradient(circle at 40% 40%, rgba(6,182,212,0.14) 0%, transparent 70%)',
    blob2: 'radial-gradient(circle at 60% 55%, rgba(139,92,246,0.12) 0%, transparent 70%)',
    blob3: 'radial-gradient(circle at 50% 50%, rgba(37,99,235,0.1) 0%, transparent 70%)',
    logoClass: 'opacity-90',
    hoverBg: 'rgba(219,234,254,0.86)', hoverBgMd: 'rgba(191,219,254,0.78)', hoverColor: '#0f172a',
  }
}

function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(false)
  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)')
    const update = () => setReduced(media.matches)
    update()
    media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [])
  return reduced
}

function usePageEntrance() {
  const reducedMotion = usePrefersReducedMotion()
  const [ready, setReady] = useState(false)
  useEffect(() => {
    if (reducedMotion) {
      setReady(true)
      return
    }
    const id = requestAnimationFrame(() => setReady(true))
    return () => cancelAnimationFrame(id)
  }, [reducedMotion])
  return { reducedMotion, ready }
}

const DashColorsContext = createContext<DashColors>(buildDashColors(true))
const useDashColors = () => useContext(DashColorsContext)

// react-gauge-chart uses D3 and must be client-only (no SSR)
const GaugeComponent = dynamic(() => import('react-gauge-chart'), { ssr: false })

// ─── Status Bar ───────────────────────────────────────────────────────────────

function DashThemeToggle() {
  const { theme, toggle } = useTheme()
  const c = useDashColors()
  const isDark = theme === 'dark'
  const [spinKey, setSpinKey] = useState(0)
  const handleToggle = () => { setSpinKey(k => k + 1); toggle() }
  return (
    <button
      onClick={handleToggle}
      aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      className="w-8 h-8 rounded-full flex items-center justify-center transition-all duration-200 flex-shrink-0"
      style={{ background: c.badgeBg, border: `1px solid ${c.badgeBorder}`, color: c.badgeColor }}
      onMouseEnter={(e) => { const el = e.currentTarget as HTMLElement; el.style.background = c.hoverBgMd; el.style.color = c.hoverColor }}
      onMouseLeave={(e) => { const el = e.currentTarget as HTMLElement; el.style.background = c.badgeBg; el.style.color = c.badgeColor }}
    >
      <div key={spinKey} style={{ animation: spinKey > 0 ? 'spin-icon 0.42s cubic-bezier(0.34,1.3,0.64,1) forwards' : 'none', display: 'flex' }}>
        {isDark ? (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="5"/>
            <line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
            <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
          </svg>
        ) : (
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
          </svg>
        )}
      </div>
    </button>
  )
}

const SURVEY_URL = 'https://docs.google.com/forms/d/e/1FAIpQLSfO1J0zUOVJdmIoTM4vjY5uVCS0_Ju93zY8xXkVTnxGcvP6fg/viewform'

const StatusBar = () => {
  const router = useRouter()
  const c = useDashColors()
  const NAV_LINKS = [
    { label: 'About',        href: '/#about' },
    { label: 'Contact Us',   href: '/#contact' },
  ]
  return (
    <div className="flex items-center justify-between px-7 py-2.5 flex-shrink-0"
      style={{ background: c.statusBg, borderBottom: `1px solid ${c.statusBorder}` }}>
      <button
        onClick={() => router.push('/')}
        className="flex items-center gap-2.5 transition-opacity duration-150 hover:opacity-80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 rounded-md"
      >
        <img src="/icons/civiq-logo.png" alt="Civiq" className={`w-5 h-5 object-contain ${c.logoClass}`} />
        <span className="font-bold text-[13px] tracking-widest" style={{ color: c.tp }}>CIVIQ</span>
        <span className="text-[10px] font-medium" style={{ color: c.tu }}>
          ·&nbsp; A Hierarchical Multi-Agent Coordination Framework
        </span>
      </button>
      <div className="flex items-center gap-1">
        {NAV_LINKS.map(({ label, href }) => (
          <button
            key={label}
            onClick={() => router.push(href)}
            className="px-4 py-1.5 rounded-full text-[12px] font-medium transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
            style={{ color: c.ts, background: 'transparent' }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLElement).style.color = c.hoverColor
              ;(e.currentTarget as HTMLElement).style.background = c.hoverBg
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLElement).style.color = c.ts
              ;(e.currentTarget as HTMLElement).style.background = 'transparent'
            }}
          >
            {label}
          </button>
        ))}
        <a href={SURVEY_URL} target="_blank" rel="noopener noreferrer"
          className="ml-2 px-4 py-1.5 rounded-full text-[12px] font-bold transition-all duration-150"
          style={{ background: 'transparent', color: '#06B6D4', border: '1.5px solid #06B6D4' }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(6,182,212,0.1)' }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'transparent' }}>
          Evaluate
        </a>
        <div className="ml-2"><DashThemeToggle /></div>
      </div>
    </div>
  )
}

// ─── Data ─────────────────────────────────────────────────────────────────────

type AlgoKey = 'civiq' | 'qmix' | 'selfish'
type Page = 'summary' | AlgoKey | 'road_closures'

// Sparkline series type — swap in real API data once backend is connected
export interface KpiSeries {
  travelTime: number[]
  waitTime: number[]
  throughput: number[]
  speed: number[]
}
export interface KpiChanges {
  travelTime: number
  waitTime: number
  throughput: number
  speed: number
}

// Per-episode data for detail chart — replace with real API data when backend is ready
// Each entry: { episode, value, lo, hi } where lo/hi are confidence band bounds across seeds
export interface EpisodePoint {
  episode: number
  value: number
  lo: number    // lower confidence bound
  hi: number    // upper confidence bound
  ma: number    // 10-episode moving average
}
export interface EpisodeSeries {
  travelTime: EpisodePoint[]
  waitTime:   EpisodePoint[]
  throughput: EpisodePoint[]
  speed:      EpisodePoint[]
}

// Training curve: cumulative reward per episode across seeds
export interface TrainingPoint {
  episode: number
  reward: number
  lo: number   // lower CI across seeds
  hi: number   // upper CI across seeds
  ma: number   // moving average
}

// CPU utilisation per time step (%) — plug in system monitor output
export interface CpuPoint {
  step: number
  cpu: number
  ma: number
}

export interface SystemSeries {
  training: TrainingPoint[]   // empty for selfish (no training)
  cpu:      CpuPoint[]
}

// Queue length per intersection per simulation step
export interface QueuePoint { step: number; int1: number; int2: number; int3: number; int4: number }
// Vehicle density / road occupancy
export interface DensityPoint { step: number; density: number; ma: number }
// Composite congestion index (0–1 normalised)
export interface CongestionPoint { step: number; index: number; ma: number }
// MARL training diagnostics per episode (empty for non-learning algos)
export interface MarlPoint {
  episode: number
  reward: number; rewardLo: number; rewardHi: number; rewardMa: number
  tdLoss: number
  gradNorm: number
  qTakenMean: number; qTakenLo: number; qTakenHi: number; targetMean: number
}

interface AlgoData {
  id: AlgoKey
  label: string
  sublabel: string
  rank: number
  color: string
  colorDim: string
  border: string
  travelTime: number
  waitTime: number
  throughput: number
  speed: number
  co2: number
  fuel: number
  computeTime: number
  cpuMean: number    // mean CPU % across all training steps (100 = 1 full core)
  cpuPeak: number    // peak CPU % observed during training
  convergence: number | null
  reward: number | null
  description: string
  strengths: string[]
  scores: number[]
  // Pluggable time-series — replace with real API data when backend is ready
  sparklines: KpiSeries
  changes: KpiChanges
  changes2?: KpiChanges
  episodes: EpisodeSeries
  system: SystemSeries
  queue: QueuePoint[]
  density: DensityPoint[]
  congestion: CongestionPoint[]
  marl: MarlPoint[]   // empty for selfish (no training)
}

// Generate dummy episode series with noise, trend, and confidence band
// Replace calls to this with real API data when backend is ready
function makeSeries(
  start: number, end: number, noiseAmp: number,
  band: number, episodes = 200, seed = 1
): EpisodePoint[] {
  const pts: EpisodePoint[] = []
  let r = seed
  const rand = () => { r = (r * 1664525 + 1013904223) & 0xffffffff; return (r >>> 0) / 0xffffffff - 0.5 }
  const raw: number[] = []
  for (let i = 0; i < episodes; i++) {
    const t = i / (episodes - 1)
    raw.push(start + (end - start) * t + rand() * noiseAmp)
  }
  const W = 10
  for (let i = 0; i < episodes; i++) {
    const slice = raw.slice(Math.max(0, i - W + 1), i + 1)
    const ma = slice.reduce((a, b) => a + b, 0) / slice.length
    const noise = rand() * noiseAmp * 0.4
    pts.push({ episode: i + 1, value: +raw[i].toFixed(2), lo: +(raw[i] - band + noise).toFixed(2), hi: +(raw[i] + band + noise).toFixed(2), ma: +ma.toFixed(2) })
  }
  return pts
}

// Training curve: reward from ~start → end, with logistic ease-in (slow start, fast rise, plateau)
function makeTraining(startR: number, endR: number, band: number, episodes = 200, seed = 1): TrainingPoint[] {
  let r = seed
  const rand = () => { r = (r * 1664525 + 1013904223) & 0xffffffff; return (r >>> 0) / 0xffffffff - 0.5 }
  const W = 10
  const raw: number[] = []
  for (let i = 0; i < episodes; i++) {
    // logistic curve: slow at first, rapid middle, plateau near end
    const t = i / (episodes - 1)
    const logistic = 1 / (1 + Math.exp(-10 * (t - 0.35)))
    raw.push(startR + (endR - startR) * logistic + rand() * (band * 1.4))
  }
  const pts: TrainingPoint[] = []
  for (let i = 0; i < episodes; i++) {
    const slice = raw.slice(Math.max(0, i - W + 1), i + 1)
    const ma = slice.reduce((a, b) => a + b, 0) / slice.length
    const bNoise = rand() * band * 0.3
    pts.push({ episode: i + 1, reward: +raw[i].toFixed(1), lo: +(raw[i] - band + bNoise).toFixed(1), hi: +(raw[i] + band + bNoise).toFixed(1), ma: +ma.toFixed(1) })
  }
  return pts
}

// CPU utilization: rises at start, peaks, then stabilises
function makeCpu(basePercent: number, peakPercent: number, noiseAmp: number, steps = 120, seed = 1): CpuPoint[] {
  let r = seed
  const rand = () => { r = (r * 1664525 + 1013904223) & 0xffffffff; return (r >>> 0) / 0xffffffff - 0.5 }
  const W = 8
  const raw: number[] = []
  for (let i = 0; i < steps; i++) {
    const t = i / (steps - 1)
    // bell-ish: rises quickly to peak then settles at base
    const shaped = basePercent + (peakPercent - basePercent) * Math.exp(-4 * Math.pow(t - 0.15, 2)) * (1 - t * 0.4)
    raw.push(Math.min(100, Math.max(0, shaped + rand() * noiseAmp)))
  }
  const pts: CpuPoint[] = []
  for (let i = 0; i < steps; i++) {
    const slice = raw.slice(Math.max(0, i - W + 1), i + 1)
    const ma = slice.reduce((a, b) => a + b, 0) / slice.length
    pts.push({ step: i + 1, cpu: +raw[i].toFixed(1), ma: +ma.toFixed(1) })
  }
  return pts
}

// Queue length (vehicles) per intersection over simulation steps
function makeQueue(baseLevels: [number,number,number,number], noiseAmp: number, steps = 120, seed = 1): QueuePoint[] {
  let r = seed
  const rand = () => { r = (r * 1664525 + 1013904223) & 0xffffffff; return (r >>> 0) / 0xffffffff - 0.5 }
  return Array.from({ length: steps }, (_, i) => {
    const pattern = 1 + 0.35 * Math.sin(2 * Math.PI * (i / (steps - 1)) * 2.5 + 0.5)
    return {
      step: i + 1,
      int1: Math.max(0, +(baseLevels[0] * pattern + rand() * noiseAmp).toFixed(1)),
      int2: Math.max(0, +(baseLevels[1] * pattern + rand() * noiseAmp).toFixed(1)),
      int3: Math.max(0, +(baseLevels[2] * pattern + rand() * noiseAmp).toFixed(1)),
      int4: Math.max(0, +(baseLevels[3] * pattern + rand() * noiseAmp).toFixed(1)),
    }
  })
}

// Vehicle density / road occupancy rate (%) over simulation steps
function makeDensity(basePercent: number, peakPercent: number, noiseAmp: number, steps = 120, seed = 1): DensityPoint[] {
  let r = seed
  const rand = () => { r = (r * 1664525 + 1013904223) & 0xffffffff; return (r >>> 0) / 0xffffffff - 0.5 }
  const W = 8
  const raw: number[] = []
  for (let i = 0; i < steps; i++) {
    const t = i / (steps - 1)
    raw.push(Math.min(100, Math.max(0, basePercent + (peakPercent - basePercent) * Math.sin(Math.PI * t) + rand() * noiseAmp)))
  }
  return raw.map((v, i) => {
    const slice = raw.slice(Math.max(0, i - W + 1), i + 1)
    const ma = slice.reduce((a, b) => a + b, 0) / slice.length
    return { step: i + 1, density: +v.toFixed(1), ma: +ma.toFixed(1) }
  })
}

// Composite congestion index (0–1 scale) over simulation steps
function makeCongestion(baseIdx: number, peakIdx: number, noiseAmp: number, steps = 120, seed = 1): CongestionPoint[] {
  let r = seed
  const rand = () => { r = (r * 1664525 + 1013904223) & 0xffffffff; return (r >>> 0) / 0xffffffff - 0.5 }
  const W = 10
  const raw: number[] = []
  for (let i = 0; i < steps; i++) {
    const t = i / (steps - 1)
    // peak in middle, tapering off — models rush-hour buildup and dispersal
    const shaped = baseIdx + (peakIdx - baseIdx) * Math.sin(Math.PI * t)
    raw.push(Math.min(1, Math.max(0, shaped + rand() * noiseAmp)))
  }
  return raw.map((v, i) => {
    const slice = raw.slice(Math.max(0, i - W + 1), i + 1)
    const ma = slice.reduce((a, b) => a + b, 0) / slice.length
    return { step: i + 1, index: +v.toFixed(3), ma: +ma.toFixed(3) }
  })
}

// MARL diagnostic metrics per training episode
function makeMarlMetrics(tdStart: number, qStart: number, qEnd: number, gradStart: number, rewardStart: number, rewardEnd: number, episodes = 200, seed = 1): MarlPoint[] {
  let r = seed
  const rand = () => { r = (r * 1664525 + 1013904223) & 0xffffffff; return (r >>> 0) / 0xffffffff - 0.5 }
  const W = 10
  const rawR: number[] = [], rawTd: number[] = [], rawQ: number[] = [], rawT: number[] = [], rawG: number[] = []
  for (let i = 0; i < episodes; i++) {
    const t = i / (episodes - 1)
    const logistic = 1 / (1 + Math.exp(-10 * (t - 0.35)))
    rawR.push(rewardStart + (rewardEnd - rewardStart) * logistic + rand() * 80)
    rawTd.push(Math.max(0.0005, tdStart * Math.exp(-6 * t) + 0.04 + rand() * 0.28))
    const qm = qStart + (qEnd - qStart) * (1 / (1 + Math.exp(-9 * (t - 0.38)))) + rand() * 7
    rawQ.push(qm)
    // target lags q_taken: gap starts large and narrows as training converges
    rawT.push(qm - (3.5 + Math.abs(qEnd - qStart) * 0.04 * (1 - t)) + rand() * 3)
    rawG.push(Math.max(0.0005, gradStart * Math.exp(-3.5 * t) + 0.08 + rand() * 0.35))
  }
  return rawR.map((rv, i) => {
    const slice = rawR.slice(Math.max(0, i - W + 1), i + 1)
    const ma = slice.reduce((a, b) => a + b, 0) / slice.length
    const band = 55
    const qStd = 3.8
    return {
      episode: i + 1,
      reward: +rv.toFixed(1), rewardLo: +(rv - band).toFixed(1), rewardHi: +(rv + band).toFixed(1), rewardMa: +ma.toFixed(1),
      tdLoss: +rawTd[i].toFixed(5),
      gradNorm: +rawG[i].toFixed(5),
      qTakenMean: +rawQ[i].toFixed(2), qTakenLo: +(rawQ[i] - qStd).toFixed(2), qTakenHi: +(rawQ[i] + qStd).toFixed(2),
      targetMean: +rawT[i].toFixed(2),
    }
  })
}

const ALGO: Record<AlgoKey, AlgoData> = {
  civiq: {
    id: 'civiq', label: 'Civiq', sublabel: 'HMARL', rank: 1,
    color: '#38BDF8', colorDim: 'rgba(56,189,248,0.12)', border: 'rgba(56,189,248,0.3)',
    // Averages across LOS A/C/E on BGC Full (2 km²)
    travelTime: 1.975, waitTime: 12.47, throughput: 1518, speed: 1.24,
    co2: 492.8, fuel: 21.24, computeTime: 22.35, cpuMean: 537.5, cpuPeak: 8597, convergence: 150, reward: -65638,
    description: 'CiViQ works like a smart city planner with two levels: a big-picture coordinator sets routing goals across the city, while local controllers manage each intersection in real time. This layered approach keeps traffic flowing efficiently — especially under heavy load, where it leads all three approaches in vehicles moved and shortest wait times.',
    strengths: ['37.4% higher throughput than QMIX at High Demand', 'Lowest wait time at peak load (19.3 sec)', 'Travel time improves 8.4% from Moderate → High Demand', 'Constant per-decision complexity as agents scale'],
    scores: [0.99, 1.00, 1.00, 0.67, 0.96, 0.96, 0.30],
    sparklines: {
      travelTime:  [1.96, 1.96, 1.96, 1.96, 1.96, 1.96, 1.96, 1.96, 1.96, 1.96],
      waitTime:    [12.4, 12.4, 12.4, 12.4, 12.4, 12.4, 12.4, 12.4, 12.4, 12.4],
      throughput:  [1536, 1536, 1536, 1536, 1536, 1536, 1536, 1536, 1536, 1536],
      speed:       [1.24, 1.24, 1.24, 1.24, 1.24, 1.24, 1.24, 1.24, 1.24, 1.24],
    },
    changes: { travelTime: -50.6, waitTime: -48.6, throughput: 35.9, speed: 44.2 },
    episodes: {
      travelTime:  makeSeries(168.0, 118.2, 9.0, 6.0, 200, 1),
      waitTime:    makeSeries(18.0, 10.3, 1.8,  1.2,  200, 2),
      throughput:  makeSeries(800,  1223, 80,   55,   200, 3),
      speed:       makeSeries(1.24, 1.24, 0.01, 0.005, 200, 4),
    },
    system: {
      training: makeTraining(-98689, -45134, 8000, 200, 13),
      cpu:      makeCpu(28, 72, 6, 120, 15),
    },
    queue:      makeQueue([2.1, 3.2, 2.6, 1.8], 0.9, 120, 21),
    density:    makeDensity(14, 28, 4, 120, 22),
    congestion: makeCongestion(0.10, 0.24, 0.04, 120, 23),
    marl:       makeMarlMetrics(2.8, -55, 95, 2.1, -200, 1250, 200, 24),
  },
  qmix: {
    // ── Real PyMARL data — BGC Full (2 km²), LOS A, seed 1801, 50 eval episodes ──
    // Source: results/mono-qmix-los-a/experiment_summary_losA.json + qmix_exp_1801.json
    // Detail page fetches training curve & MARL diagnostics dynamically via /api/qmix
    id: 'qmix', label: 'Monolithic QMIX', sublabel: 'Baseline RL', rank: 2,
    color: '#A78BFA', colorDim: 'rgba(167,139,250,0.12)', border: 'rgba(167,139,250,0.3)',
    // Averages across LOS A/C/E on BGC Full (2 km²) — overridden per-LOS by useQmixRealData
    travelTime: 1.99, waitTime: 14.1, throughput: 1305, speed: 53.97,
    co2: 538.9, fuel: 23.2, computeTime: 18.20, cpuMean: 93.4, cpuPeak: 702, convergence: 9001, reward: -68798,
    description: 'Mono-QMIX trains one central AI to control all traffic signals across the network at once. It shines at moderate traffic — delivering the fastest average travel times of all three approaches at that level. But as roads fill up, coordinating dozens of intersections simultaneously becomes harder, and performance starts to slip.',
    strengths: ['Best travel time at Moderate Demand: 108.3 sec', '11.3% faster than CiViQ at Moderate Demand', 'Lower CPU overhead per decision step', 'Well-established QMIX mixing-network framework'],
    scores: [0.85, 0.87, 0.98, 0.82, 0.87, 0.87, 0.32],
    // Flat sparklines at averaged eval measurement — overridden by useQmixRealData
    sparklines: {
      travelTime: [1.99, 1.99, 1.99, 1.99, 1.99, 1.99, 1.99, 1.99, 1.99, 1.99],
      waitTime:   [14.1, 14.1, 14.1, 14.1, 14.1, 14.1, 14.1, 14.1, 14.1, 14.1],
      throughput: [1305, 1305, 1305, 1305, 1305, 1305, 1305, 1305, 1305, 1305],
      speed:      [53.97, 53.97, 53.97, 53.97, 53.97, 53.97, 53.97, 53.97, 53.97, 53.97],
    },
    // vs noop baseline (LOS A): QMIX underperforms noop signal-free in low-traffic scenario
    changes: { travelTime: +4.5, waitTime: +12.1, throughput: -12.3, speed: 0 },
    episodes: {
      travelTime:  makeSeries(156.0, 119.4, 10.8, 7.2, 200, 5),
      waitTime:    makeSeries(18.0, 14.1, 1.5,  1.0,  200, 6),
      throughput:  makeSeries(900,  1305, 70,   50,   200, 7),
      speed:       makeSeries(50,   53.97, 2.0, 1.0,  200, 8),
    },
    system: {
      // Placeholder training curve — replaced by useQmixRealData hook on detail page
      training: makeTraining(-105961, -45791, 8000, 200, 16),
      cpu:      makeCpu(65, 95, 8, 120, 18),
    },
    queue:      makeQueue([4.5, 5.8, 5.1, 3.9], 1.3, 120, 25),
    density:    makeDensity(22, 42, 6, 120, 26),
    congestion: makeCongestion(0.19, 0.38, 0.06, 120, 27),
    // Placeholder MARL diagnostics — replaced by useQmixRealData hook on detail page
    // Seed values match real log: tdStart=0.19 (loss@t=330k), qStart=-1.34, gradStart=8
    marl:       makeMarlMetrics(0.19, -1.34, -1.25, 8, -48700, -45791, 200, 28),
  },
  selfish: {
    // ── Real PyMARL data — bgc_full map (2 km²), forced_flow (LOS E), seed 5, 30 episodes ──
    // Source: pymarl/src/results/eval/selfish_routing_bgc_full_{low,med,high}_seed5.json
    // Detail page fetches per-traffic-level values dynamically via /api/selfish
    id: 'selfish', label: 'Selfish Routing', sublabel: 'Nash Equilibrium', rank: 3,
    color: '#F87171', colorDim: 'rgba(248,113,113,0.12)', border: 'rgba(248,113,113,0.3)',
    // Averages across LOS A/C/E on BGC Full (2 km²) — overridden per-level by useSelfishRealData
    travelTime: 2.05, waitTime: 13.1, throughput: 1545, speed: 248.84,
    co2: 471.2, fuel: 20.3, computeTime: 0, cpuMean: 0, cpuPeak: 0, convergence: null, reward: null,
    description: 'Selfish Routing shows what happens without any coordination — every vehicle independently picks the shortest path for itself. It holds its own when traffic is light, performing nearly as well as the AI systems. But as roads get busier, vehicles all pile onto the same popular routes, causing wait times and congestion to climb.',
    strengths: ['No training or setup required', 'Within 0.7% of CiViQ travel time at Low Demand', 'Establishes the Price of Anarchy baseline', 'Fully interpretable shortest-path decisions'],
    scores: [1.00, 0.95, 0.95, 1.00, 1.00, 1.00, 1.00],
    // Sparklines: flat at averaged measurement — overridden by real data once loaded
    sparklines: {
      travelTime:  [2.05, 2.05, 2.05, 2.05, 2.05, 2.05, 2.05, 2.05, 2.05, 2.05],
      waitTime:    [13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1],
      throughput:  [1545, 1545, 1545, 1545, 1545, 1545, 1545, 1545, 1545, 1545],
      speed:       [248.84, 248.84, 248.84, 248.84, 248.84, 248.84, 248.84, 248.84, 248.84, 248.84],
    },
    changes: { travelTime: 0, waitTime: 0, throughput: 0, speed: 0 },
    episodes: {
      travelTime:  makeSeries(127.2, 127.2, 9.0, 6.0, 30, 9),
      waitTime:    makeSeries(20.8, 20.8, 1.5,  1.0,  30, 10),
      throughput:  makeSeries(2178, 2178, 60,   45,   30, 11),
      speed:       makeSeries(248.84, 248.84, 8, 5,   30, 12),
    },
    system: {
      training: [],  // Selfish Routing has no training phase
      cpu:      makeCpu(18, 35, 4, 120, 20),
    },
    queue:      makeQueue([8.2, 10.5, 9.1, 7.4], 2.4, 120, 29),
    density:    makeDensity(36, 62, 9, 120, 30),
    congestion: makeCongestion(0.44, 0.71, 0.09, 120, 31),
    marl:       [],  // Selfish Routing has no training phase
  },
}

const ALGO_LIST = [ALGO.selfish, ALGO.qmix, ALGO.civiq]

// ─── Reusable UI primitives ────────────────────────────────────────────────────

function GlassCard({
  children, className = '', style = {}, onClick, onMouseEnter, onMouseLeave,
}: { children: React.ReactNode; className?: string; style?: React.CSSProperties; onClick?: () => void; onMouseEnter?: React.MouseEventHandler<HTMLDivElement>; onMouseLeave?: React.MouseEventHandler<HTMLDivElement> }) {
  const c = useDashColors()
  return (
    <div className={className} onClick={onClick} onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave} style={{
      background: c.glassBg,
      backdropFilter: 'blur(24px)',
      WebkitBackdropFilter: 'blur(24px)',
      border: `1px solid ${c.glassBorder}`,
      borderRadius: '16px',
      boxShadow: c.glassShadow,
      ...style,
    }}>{children}</div>
  )
}

// ─── Sparkline ────────────────────────────────────────────────────────────────

const SparkLine = ({ data, color }: { data: number[]; color: string }) => {
  if (!data || data.length < 2) return null
  const W = 96, H = 44, PAD = 4
  const min = Math.min(...data), max = Math.max(...data)
  const range = max - min || 1
  const pts = data.map((v, i) => ({
    x: PAD + (i / (data.length - 1)) * (W - PAD * 2),
    y: PAD + (H - PAD * 2) - ((v - min) / range) * (H - PAD * 2),
  }))
  const linePts = pts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')
  const areaPts = [
    `${pts[0].x.toFixed(1)},${H}`,
    ...pts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`),
    `${pts[pts.length - 1].x.toFixed(1)},${H}`,
  ].join(' ')
  const gid = `sp-${color.replace('#', '')}`
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ overflow: 'hidden', flexShrink: 0, display: 'block' }}>
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.25" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={areaPts} fill={`url(#${gid})`} />
      <polyline points={linePts} fill="none" stroke={color} strokeWidth="2"
        strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={pts[pts.length - 1].x} cy={pts[pts.length - 1].y} r="2.5"
        fill={color} />
    </svg>
  )
}

// ─── KPI Card ─────────────────────────────────────────────────────────────────

function KpiCard({ label, abbr, value, unit, color, colorDim, borderColor, change, lowerBetter, sparkData, onClick, description, descriptionSide = 'right', changeLabel, change2, changeLabel2, valueAlign = 'left' }: {
  label: string; abbr?: string; value: string | number; unit: string
  color: string; colorDim?: string; borderColor?: string
  change?: number; lowerBetter?: boolean; sparkData?: number[]; onClick?: () => void; description?: string; descriptionSide?: 'left' | 'right'
  changeLabel?: string
  change2?: number; changeLabel2?: string
  valueAlign?: 'left' | 'center'
}) {
  const c = useDashColors()
  // Green = good outcome, regardless of direction
  const isGood  = change  === undefined ? true : lowerBetter ? change  <= 0 : change  >= 0
  const isGood2 = change2 === undefined ? true : lowerBetter ? change2 <= 0 : change2 >= 0
  const changeColor  = isGood  ? '#4ADE80' : '#F87171'
  const changeColor2 = isGood2 ? '#4ADE80' : '#F87171'
  const changeArrow  = (change  ?? 0) >= 0 ? '▲' : '▼'
  const changeArrow2 = (change2 ?? 0) >= 0 ? '▲' : '▼'
  return (
    <GlassCard className="group relative z-0 hover:z-30 p-4 flex flex-col gap-2 transition-all duration-200"
      onClick={onClick}
      onMouseEnter={onClick ? (e) => {
        (e.currentTarget as HTMLElement).style.borderColor = borderColor?.replace('0.3', '0.5') ?? c.glassBorder
        ;(e.currentTarget as HTMLElement).style.boxShadow = `0 0 0 1px ${borderColor?.replace('0.3', '0.15') ?? 'transparent'}, ${c.glassShadow}`
        ;(e.currentTarget as HTMLElement).style.transform = 'translateY(-1px)'
      } : undefined}
      onMouseLeave={onClick ? (e) => {
        (e.currentTarget as HTMLElement).style.borderColor = borderColor?.replace('0.3', '0.25') ?? c.glassBorder
        ;(e.currentTarget as HTMLElement).style.boxShadow = c.glassShadow
        ;(e.currentTarget as HTMLElement).style.transform = 'translateY(0)'
      } : undefined}
      style={{
        background: colorDim
          ? `linear-gradient(145deg, ${colorDim.replace('0.12', '0.10')} 0%, ${c.isDark ? 'rgba(255,255,255,0.03)' : 'rgba(255,255,255,0.45)'} 100%)`
          : undefined,
        border: borderColor ? `1px solid ${borderColor.replace('0.3', '0.25')}` : undefined,
        cursor: onClick ? 'pointer' : 'default',
        transition: 'border-color 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease',
      }}>
      {/* Title */}
      <div className="flex items-center gap-2 min-w-0 pr-8">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="text-[11px] font-semibold uppercase tracking-wider leading-none truncate"
            style={{ color: c.ts }}>
            {label}
          </span>
          {onClick && (
            <span
              className="flex-shrink-0 text-[9px] font-semibold px-2 py-0.5 rounded-full transition-all duration-150 group-hover:translate-x-0.5 group-hover:-translate-y-[1px]"
              style={{
                background: 'rgba(56,189,248,0.1)',
                color: c.isDark ? 'rgba(186,230,253,0.7)' : 'rgba(3,105,161,0.9)',
                border: '1px solid rgba(56,189,248,0.22)',
              }}
            >
              View detail ↗
            </span>
          )}
        </div>
      </div>

      {/* Top-right tooltip */}
      {description && (
        <div className="group/info absolute top-4 right-4 z-20 flex-shrink-0" onClick={(e) => e.stopPropagation()}>
          <div
            className="w-[18px] h-[18px] rounded-full flex items-center justify-center"
            style={{
              background: 'rgba(6,182,212,0.2)',
              border: '1px solid rgba(6,182,212,0.65)',
              color: c.isDark ? 'rgba(217,249,255,0.95)' : 'rgba(3,105,161,0.95)',
              boxShadow: '0 0 0 1px rgba(0,0,0,0.2) inset',
            }}
          >
            <span className="text-[10px] font-bold leading-none">!</span>
          </div>
          <div
            className="pointer-events-none absolute top-[calc(100%+8px)] w-[280px] rounded-lg px-3 py-2 text-[10px] leading-relaxed opacity-0 translate-y-1 transition-all duration-150 group-hover/info:opacity-100 group-hover/info:translate-y-0"
            style={{
              ...(descriptionSide === 'left' ? { right: 0 } : { right: 0 }),
              background: c.tooltipBg,
              border: `1px solid ${c.tooltipBorder}`,
              color: c.tooltipColor,
              boxShadow: c.tooltipShadow,
              zIndex: 90,
            }}
          >
            {description}
          </div>
        </div>
      )}

      {/* Value + sparkline row */}
      <div className={`mt-1 flex items-center gap-3 ${valueAlign === 'center' ? 'justify-center flex-1' : 'justify-between'}`}>
        <div className={`flex flex-col gap-0.5 ${valueAlign === 'center' ? 'items-center text-center' : ''}`}>
          {/* Value + unit */}
          <div className="flex items-baseline gap-1.5">
            <span className="text-[26px] font-bold tabular-nums leading-none" style={{ color }}>{value}</span>
            <span className="text-[12px] font-medium" style={{ color: c.tm }}>{unit}</span>
          </div>
          {/* Change badges below value */}
          {change !== undefined && (
            <div className="flex items-center gap-1.5">
              <span className="text-[11px] font-bold tabular-nums" style={{ color: changeColor }}>
                {changeArrow} {Math.abs(change).toFixed(1)}%
              </span>
              {changeLabel && (
                <span className="text-[10px]" style={{ color: c.tu }}>{changeLabel}</span>
              )}
            </div>
          )}
          {change2 !== undefined && (
            <div className="flex items-center gap-1.5">
              <span className="text-[11px] font-bold tabular-nums" style={{ color: changeColor2 }}>
                {changeArrow2} {Math.abs(change2).toFixed(1)}%
              </span>
              {changeLabel2 && (
                <span className="text-[10px]" style={{ color: c.tu }}>{changeLabel2}</span>
              )}
            </div>
          )}
        </div>
        {sparkData && <SparkLine data={sparkData} color={color} />}
      </div>

    </GlassCard>
  )
}

function HBar({ value, max, color, dim = false }: { value: number; max: number; color: string; dim?: boolean }) {
  const c = useDashColors()
  return (
    <div className="flex-1 h-1.5 rounded-full" style={{ background: c.barTrack, boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.15)' }}>
      <div className="h-full rounded-full transition-all duration-700"
        style={{ width: `${Math.min(100, (value / max) * 100)}%`, background: color, opacity: dim ? 0.45 : 1 }} />
    </div>
  )
}

function RingChart({ value, max = 100, color, size = 88 }: {
  value: number; max?: number; color: string; size?: number
}) {
  const c = useDashColors()
  const r = (size - 14) / 2
  const circ = 2 * Math.PI * r
  const dash = (value / max) * circ
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={c.ringTrack} strokeWidth="8" />
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth="8"
        strokeDasharray={`${dash} ${circ - dash}`} strokeLinecap="round"
        style={{ transform: 'rotate(-90deg)', transformOrigin: '50% 50%' }} />
    </svg>
  )
}

// ─── Radar helpers ─────────────────────────────────────────────────────────────

function normRadar(vals: number[], higherBetter: boolean): number[] {
  const mn = Math.min(...vals), mx = Math.max(...vals), range = mx - mn
  if (range === 0) return vals.map(() => 0.75)
  return vals.map(v => higherBetter ? 0.5 + 0.5 * (v - mn) / range : 0.5 + 0.5 * (mx - v) / range)
}

// ─── Radar Chart ───────────────────────────────────────────────────────────────

const RADAR_AXES = ['Throughput', 'Low Wait', 'Low ATT', 'Compute', 'Low CO₂', 'Low Fuel', 'RTF']
const RC = { x: 175, y: 175 }
const RR = 125

function rPt(axisIdx: number, score: number): string {
  const a = -Math.PI / 2 + (2 * Math.PI * axisIdx) / RADAR_AXES.length
  return `${(RC.x + score * RR * Math.cos(a)).toFixed(1)},${(RC.y + score * RR * Math.sin(a)).toFixed(1)}`
}
function rPts(scores: number[]) { return scores.map((s, i) => rPt(i, s)).join(' ') }
function rLabel(i: number) {
  const a = -Math.PI / 2 + (2 * Math.PI * i) / RADAR_AXES.length
  return { x: RC.x + (RR + 22) * Math.cos(a), y: RC.y + (RR + 22) * Math.sin(a) }
}

function RadarTooltip({ axisIdx, scores }: { axisIdx: number; scores: Record<AlgoKey, number[]> }) {
  const c = useDashColors()
  const { x: lx, y: ly } = rLabel(axisIdx)
  const tipW = 148
  const tipH = 68
  const tx = lx > RC.x ? Math.min(lx - tipW - 4, 240) : Math.max(lx + 8, 2)
  const ty = Math.max(8, Math.min(ly - tipH / 2, 350 - tipH - 8))
  return (
    <g style={{ pointerEvents: 'none' }}>
      <rect x={tx} y={ty} width={tipW} height={tipH} rx="6"
        fill={c.tooltipBg} stroke={c.tooltipBorder} strokeWidth="1" />
      <text x={tx + 8} y={ty + 14} fontSize="9.5" fill={c.tooltipColor}
        fontWeight="700" textAnchor="start">{RADAR_AXES[axisIdx]}</text>
      {ALGO_LIST.map((a, ai) => (
        <g key={a.id}>
          <circle cx={tx + 9} cy={ty + 26 + ai * 13} r="3.5" fill={a.color} />
          <text x={tx + 17} y={ty + 30 + ai * 13} fontSize="9" fill={c.tooltipColor} textAnchor="start">
            {a.label}: {Math.round(scores[a.id][axisIdx] * 100)}%
          </text>
        </g>
      ))}
    </g>
  )
}

function RadarChart({ scores }: { scores: Record<AlgoKey, number[]> }) {
  const c = useDashColors()
  const [hoveredAxis, setHoveredAxis] = useState<number | null>(null)
  const gridStroke = c.chartGrid
  const axisStrokeBase = c.isDark ? 'rgba(255,255,255,0.11)' : 'rgba(15,23,42,0.1)'
  const axisStrokeHover = c.isDark ? 'rgba(255,255,255,0.3)' : 'rgba(15,23,42,0.35)'
  const labelBase = c.isDark ? 'rgba(255,255,255,0.45)' : '#9ca3af'
  const labelHover = c.tp
  return (
    <svg viewBox="0 0 350 350" className="w-full max-w-[320px] mx-auto" style={{ overflow: 'visible' }}>
      {[0.25, 0.5, 0.75, 1].map((r) => (
        <polygon key={r} points={rPts(Array(7).fill(r))}
          fill="none" stroke={gridStroke} strokeWidth="1" />
      ))}
      {RADAR_AXES.map((_, i) => {
        const pt = { x: RC.x + RR * Math.cos(-Math.PI / 2 + (2 * Math.PI * i) / RADAR_AXES.length), y: RC.y + RR * Math.sin(-Math.PI / 2 + (2 * Math.PI * i) / RADAR_AXES.length) }
        return <line key={i} x1={RC.x} y1={RC.y} x2={pt.x} y2={pt.y}
          stroke={hoveredAxis === i ? axisStrokeHover : axisStrokeBase} strokeWidth="1" />
      })}
      <polygon points={rPts(scores.selfish)} fill="rgba(248,113,113,0.12)" stroke="#F87171" strokeWidth="1.5" strokeLinejoin="round" />
      <polygon points={rPts(scores.qmix)}    fill="rgba(167,139,250,0.12)" stroke="#A78BFA" strokeWidth="1.5" strokeLinejoin="round" />
      <polygon points={rPts(scores.civiq)}   fill="rgba(56,189,248,0.18)"  stroke="#38BDF8" strokeWidth="2"   strokeLinejoin="round" />
      {RADAR_AXES.map((label, i) => {
        const { x, y } = rLabel(i)
        const isHovered = hoveredAxis === i
        return (
          <text key={i} x={x} y={y} textAnchor="middle" dominantBaseline="middle"
            fontSize="10" fill={isHovered ? labelHover : labelBase}
            fontWeight={isHovered ? '700' : '500'}>{label}</text>
        )
      })}
      {RADAR_AXES.map((_, i) => {
        const { x, y } = rLabel(i)
        return (
          <circle key={i} cx={x} cy={y} r="20" fill="transparent" style={{ cursor: 'pointer' }}
            onMouseEnter={() => setHoveredAxis(i)}
            onMouseLeave={() => setHoveredAxis(null)} />
        )
      })}
      {hoveredAxis !== null && <RadarTooltip axisIdx={hoveredAxis} scores={scores} />}
    </svg>
  )
}

// ─── Summary Page ──────────────────────────────────────────────────────────────

const rankMeta = [
  { label: '1st Place', color: '#F59E0B', bg: 'rgba(245,158,11,0.12)' },
  { label: '2nd Place', color: '#94A3B8', bg: 'rgba(148,163,184,0.12)' },
  { label: '3rd Place', color: '#B45309', bg: 'rgba(180,83,9,0.12)' },
]

const COMPARE_METRICS = [
  { label: 'Avg. Travel Time', unit: 'min', key: 'travelTime' as const, max: 3, lowerBetter: true },
  { label: 'Avg. Wait Time', unit: 'sec', key: 'waitTime' as const, max: 20, lowerBetter: true },
  { label: 'Throughput', unit: 'veh/hr', key: 'throughput' as const, max: 2000, lowerBetter: false },
  { label: 'CO₂ Emissions', unit: 'g/km', key: 'co2' as const, max: 600, lowerBetter: true },
  { label: 'Diesel Consumption', unit: 'L/100km', key: 'fuel' as const, max: 30, lowerBetter: true },
]

// ─── Compare helpers ──────────────────────────────────────────────────────────

interface CompareConfig { left: AlgoKey; right: AlgoKey; mapSize: string; traffic: string }

function detectConvergence(training: TrainingPoint[]): number | null {
  if (!training.length) return null
  const maVals = training.map(p => p.ma)
  const range = Math.max(...maVals) - Math.min(...maVals)
  if (range < 1) return training[0].episode
  const W = 30
  for (let i = W; i < maVals.length; i++) {
    const window = maVals.slice(i - W, i)
    if ((Math.max(...window) - Math.min(...window)) / range < 0.03) return training[i - W].episode
  }
  return training[training.length - 1].episode
}

function calcDelta(lv: number, rv: number, lowerBetter: boolean) {
  const diff = lv - rv
  const pct = rv !== 0 ? Math.abs(diff / rv) * 100 : 0
  const leftWins: boolean | null = diff === 0 ? null : (lowerBetter ? diff < 0 : diff > 0)
  return { diff, pct, leftWins }
}

// ─── Compare Modal ─────────────────────────────────────────────────────────────

const MAP_OPTIONS = [
  { key: '2km', label: '2 km²' },
  { key: '0.75km', label: '0.75 km²' },
  { key: '4x4', label: '4×4 Grid' },
]
const TRAFFIC_OPTIONS = [
  { key: 'free_flow',   label: 'Low Demand',      sub: '1,000 veh/hr', note: 'light load · minimal queueing' },
  { key: 'stable_flow', label: 'Moderate Demand',  sub: '1,200 veh/hr', note: 'emerging congestion · queueing onset' },
  { key: 'forced_flow', label: 'High Demand',       sub: '2,000 veh/hr', note: 'saturated network · significant delay' },
]

function CompareModal({ onClose, onConfirm }: {
  onClose: () => void
  onConfirm: (cfg: CompareConfig) => void
}) {
  const [left, setLeft]       = useState<AlgoKey>('selfish')
  const [right, setRight]     = useState<AlgoKey>('civiq')
  const [mapSize, setMapSize] = useState('2km')
  const [traffic, setTraffic] = useState('stable_flow')
  const canRun = left !== right

  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [onClose])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: 'rgba(0,0,0,0.72)', backdropFilter: 'blur(10px)' }}>
      <div className="relative w-full max-w-[480px] mx-4 p-6 rounded-2xl" style={{
        background: 'linear-gradient(155deg, rgba(12,16,40,0.98) 0%, rgba(6,8,24,0.98) 100%)',
        border: '1px solid rgba(255,255,255,0.14)',
        boxShadow: '0 32px 80px rgba(0,0,0,0.7), inset 0 1px 0 rgba(255,255,255,0.1)',
      }}>
        {/* Header */}
        <div className="flex items-center justify-between mb-5">
          <div>
            <h3 className="text-[15px] font-bold" style={{ color: 'rgba(255,255,255,0.92)' }}>Configure Comparison</h3>
            <p className="text-[11px] mt-0.5" style={{ color: 'rgba(255,255,255,0.38)' }}>Select two algorithms and test conditions</p>
          </div>
          <button onClick={onClose} aria-label="Close" className="w-7 h-7 rounded-full flex items-center justify-center transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
            style={{ background: 'rgba(255,255,255,0.07)', color: 'rgba(255,255,255,0.5)' }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.14)' }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.07)' }}>
            <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden="true">
              <path d="M1 1l8 8M9 1L1 9" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        {/* Algorithm selector */}
        <div className="mb-4">
          <p className="text-[9px] font-bold uppercase tracking-widest mb-3" style={{ color: 'rgba(255,255,255,0.32)' }}>Algorithms</p>
          <div className="grid grid-cols-[1fr_36px_1fr] gap-2 items-start">
            {/* Side A */}
            <div className="space-y-1.5">
              <p className="text-[9px] font-semibold uppercase tracking-wider text-center mb-1" style={{ color: 'rgba(255,255,255,0.3)' }}>Side A</p>
              {ALGO_LIST.map((a) => (
                <button key={a.id}
                  onClick={() => { if (a.id === right) setRight(left); setLeft(a.id) }}
                  className="w-full px-3 py-2 rounded-xl text-left transition-all duration-150"
                  style={{
                    background: left === a.id ? a.colorDim : 'rgba(255,255,255,0.04)',
                    border: left === a.id ? `1px solid ${a.border}` : '1px solid rgba(255,255,255,0.07)',
                    color: left === a.id ? a.color : 'rgba(255,255,255,0.55)',
                    cursor: 'pointer',
                  }}>
                  <div className="text-[11px] font-bold">{a.sublabel}</div>
                  <div className="text-[9px] mt-0.5" style={{ opacity: 0.65 }}>{a.label}</div>
                </button>
              ))}
            </div>
            {/* VS divider */}
            <div className="flex flex-col items-center pt-6 gap-1">
              <div className="w-px flex-1" style={{ background: 'rgba(255,255,255,0.08)' }} />
              <span className="text-[9px] font-black px-1.5 py-0.5 rounded-full"
                style={{ background: 'rgba(255,255,255,0.05)', color: 'rgba(255,255,255,0.35)' }}>VS</span>
              <div className="w-px flex-1" style={{ background: 'rgba(255,255,255,0.08)' }} />
            </div>
            {/* Side B */}
            <div className="space-y-1.5">
              <p className="text-[9px] font-semibold uppercase tracking-wider text-center mb-1" style={{ color: 'rgba(255,255,255,0.3)' }}>Side B</p>
              {ALGO_LIST.map((a) => (
                <button key={a.id} onClick={() => setRight(a.id)} disabled={a.id === left}
                  className="w-full px-3 py-2 rounded-xl text-left transition-all duration-150"
                  style={{
                    background: right === a.id ? a.colorDim : 'rgba(255,255,255,0.04)',
                    border: right === a.id ? `1px solid ${a.border}` : '1px solid rgba(255,255,255,0.07)',
                    color: right === a.id ? a.color : a.id === left ? 'rgba(255,255,255,0.18)' : 'rgba(255,255,255,0.55)',
                    cursor: a.id === left ? 'not-allowed' : 'pointer',
                    opacity: a.id === left ? 0.35 : 1,
                  }}>
                  <div className="text-[11px] font-bold">{a.sublabel}</div>
                  <div className="text-[9px] mt-0.5" style={{ opacity: 0.65 }}>{a.label}</div>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Divider */}
        <div className="mb-4" style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }} />

        {/* Test Conditions */}
        <div className="mb-5 space-y-3">
          <p className="text-[9px] font-bold uppercase tracking-widest" style={{ color: 'rgba(255,255,255,0.32)' }}>Test Conditions</p>
          <div>
            <p className="text-[10px] mb-1.5" style={{ color: 'rgba(255,255,255,0.38)' }}>Map Size</p>
            <div className="flex gap-1.5">
              {MAP_OPTIONS.map(o => (
                <button key={o.key} onClick={() => setMapSize(o.key)}
                  className="flex-1 px-2 py-1.5 rounded-lg text-[10px] font-semibold transition-all duration-150"
                  style={{
                    background: mapSize === o.key ? 'rgba(6,182,212,0.14)' : 'rgba(255,255,255,0.04)',
                    border: mapSize === o.key ? '1px solid rgba(6,182,212,0.4)' : '1px solid rgba(255,255,255,0.07)',
                    color: mapSize === o.key ? '#06B6D4' : 'rgba(255,255,255,0.45)',
                  }}>{o.label}</button>
              ))}
            </div>
          </div>
          <div>
            <p className="text-[10px] mb-1.5" style={{ color: 'rgba(255,255,255,0.38)' }}>Traffic Scale</p>
            <div className="flex gap-1.5">
              {TRAFFIC_OPTIONS.map(o => (
                <button key={o.key} onClick={() => setTraffic(o.key)}
                  className="flex-1 px-2 py-2 rounded-lg text-center transition-all duration-150"
                  style={{
                    background: traffic === o.key ? 'rgba(6,182,212,0.14)' : 'rgba(255,255,255,0.04)',
                    border: traffic === o.key ? '1px solid rgba(6,182,212,0.4)' : '1px solid rgba(255,255,255,0.07)',
                    color: traffic === o.key ? '#06B6D4' : 'rgba(255,255,255,0.45)',
                  }}>
                  <div className="text-[10px] font-semibold">{o.label}</div>
                  <div className="text-[9px] mt-0.5" style={{ opacity: 0.6 }}>{o.sub}</div>
                  <div className="text-[8px] mt-0.5" style={{ opacity: 0.4 }}>{o.note}</div>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          <button onClick={onClose}
            className="flex-1 py-2 rounded-xl text-[11px] font-semibold transition-all"
            style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.09)', color: 'rgba(255,255,255,0.45)' }}>
            Cancel
          </button>
          <button onClick={() => canRun && onConfirm({ left, right, mapSize, traffic })}
            disabled={!canRun}
            className="flex-[2] py-2 rounded-xl text-[12px] font-bold transition-all flex items-center justify-center gap-2"
            style={{
              background: canRun ? 'linear-gradient(135deg, rgba(6,182,212,0.28) 0%, rgba(99,102,241,0.22) 100%)' : 'rgba(255,255,255,0.04)',
              border: canRun ? '1px solid rgba(6,182,212,0.5)' : '1px solid rgba(255,255,255,0.07)',
              color: canRun ? '#06B6D4' : 'rgba(255,255,255,0.2)',
              boxShadow: canRun ? '0 0 20px rgba(6,182,212,0.18)' : 'none',
              cursor: canRun ? 'pointer' : 'not-allowed',
            }}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="7" height="18" rx="1" /><rect x="14" y="3" width="7" height="18" rx="1" />
            </svg>
            Run Comparison
          </button>
        </div>
      </div>
    </div>
  )
}

// ─── Compare Page ──────────────────────────────────────────────────────────────

interface KpiMetaDef {
  label: string
  unit: string
  getValue: (a: AlgoData) => number
  lowerBetter: boolean
}

const ROUTING_KPI: KpiMetaDef[] = [
  { label: 'Average Travel Time',  unit: 'min',    getValue: a => a.travelTime, lowerBetter: true },
  { label: 'Average Waiting Time', unit: 'sec',    getValue: a => a.waitTime,   lowerBetter: true },
  { label: 'Network Throughput',   unit: 'veh/hr', getValue: a => a.throughput, lowerBetter: false },
]
const ENVIRON_KPI: KpiMetaDef[] = [
  { label: 'Average CO₂ Emissions',    unit: 'g/km',    getValue: a => a.co2,  lowerBetter: true },
  { label: 'Avg. Fuel Consumption (Diesel)', unit: 'L/100km', getValue: a => a.fuel, lowerBetter: true },
]
const COMPUTE_KPI: KpiMetaDef[] = [
  { label: 'Real-time Factor', unit: '×',  getValue: a => a.speed,       lowerBetter: false },
  { label: 'Decision Latency', unit: 'ms', getValue: a => a.computeTime, lowerBetter: true },
]

function MetricRow({ label, unit, lv, rv, lowerBetter, la, ra, intFmt }: {
  label: string; unit: string; lv: number; rv: number
  lowerBetter: boolean; la: AlgoData; ra: AlgoData; intFmt?: boolean
}) {
  const c = useDashColors()
  const fmt = (v: number) => intFmt ? String(Math.round(v)) : String(v)
  const d = calcDelta(lv, rv, lowerBetter)
  const leftWins = d.leftWins === true
  const rightWins = d.leftWins === false
  const tie = d.leftWins === null
  const WIN = '#4ADE80'
  const leftValColor  = leftWins  ? WIN : rightWins ? c.tm : c.tp
  const rightValColor = rightWins ? WIN : leftWins  ? c.tm : c.tp
  const arrow = lv < rv ? '▼' : lv > rv ? '▲' : '—'
  const deltaColor = tie ? c.tm : leftWins ? '#4ADE80' : '#F87171'
  const deltaBg    = tie ? c.itemBg : leftWins ? 'rgba(74,222,128,0.1)' : 'rgba(248,113,113,0.1)'
  const deltaBorder = tie ? c.divider : leftWins ? 'rgba(74,222,128,0.25)' : 'rgba(248,113,113,0.25)'
  const deltaLabel = tie ? '— tied' : `${arrow} ${Math.abs(d.diff) < 1 ? Math.abs(d.diff).toFixed(2) : Math.abs(d.diff).toFixed(1)} (${d.pct.toFixed(1)}%)`

  return (
    <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-3 py-3"
      style={{ borderBottom: `1px solid ${c.divider}` }}>
      {/* Left value */}
      <div className="text-right">
        <div className="text-[21px] font-black tabular-nums leading-none" style={{ color: leftValColor }}>{fmt(lv)}</div>
        <div className="text-[9px] mt-0.5 tabular-nums" style={{ color: c.tu }}>{unit}</div>
        {leftWins && <div className="text-[8px] font-bold mt-1 uppercase tracking-wider" style={{ color: WIN }}>wins</div>}
      </div>
      {/* Center */}
      <div className="text-center" style={{ minWidth: '148px' }}>
        <div className="text-[10px] font-semibold mb-1.5" style={{ color: c.ts }}>{label}</div>
        <div className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[10px] font-bold"
          style={{ background: deltaBg, color: deltaColor, border: `1px solid ${deltaBorder}` }}>
          {deltaLabel}
        </div>
      </div>
      {/* Right value */}
      <div className="text-left">
        <div className="text-[21px] font-black tabular-nums leading-none" style={{ color: rightValColor }}>{fmt(rv)}</div>
        <div className="text-[9px] mt-0.5 tabular-nums" style={{ color: c.tu }}>{unit}</div>
        {rightWins && <div className="text-[8px] font-bold mt-1 uppercase tracking-wider" style={{ color: WIN }}>wins</div>}
      </div>
    </div>
  )
}

function CompareSection({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  const c = useDashColors()
  return (
    <GlassCard className="p-5">
      <div className="flex items-center gap-2 mb-1 pb-3" style={{ borderBottom: `1px solid ${c.divider}` }}>
        {icon}
        <h3 className="text-[13px] font-bold" style={{ color: c.tp }}>{title}</h3>
      </div>
      {children}
    </GlassCard>
  )
}

function ComparePage({ config, onBack }: { config: CompareConfig; onBack: () => void }) {
  const c = useDashColors()
  const la = ALGO[config.left]
  const ra = ALGO[config.right]
  const bothLearning = la.system.training.length > 0 && ra.system.training.length > 0
  const convL = bothLearning ? detectConvergence(la.system.training) : null
  const convR = bothLearning ? detectConvergence(ra.system.training) : null

  const iconStroke = c.ts
  const routingIcon = (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke={iconStroke} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 12h18M3 6h18M3 18h12" />
    </svg>
  )
  const environIcon = (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke={iconStroke} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a10 10 0 0 1 0 20 10 10 0 0 1 0-20z" /><path d="M12 8v4l3 3" />
    </svg>
  )
  const computeIcon = (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke={iconStroke} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="4" y="4" width="16" height="16" rx="2" /><path d="M9 9h6M9 12h6M9 15h4" />
    </svg>
  )

  return (
    <div className="p-6 space-y-4">
      {/* Back */}
      <button onClick={onBack}
        className="flex items-center gap-1.5 text-[11px] transition-colors"
        style={{ color: c.tm }}
        onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.color = c.tp }}
        onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.color = c.tm }}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M19 12H5M12 19l-7-7 7-7" />
        </svg>
        Back to Summary
      </button>

      {/* VS header */}
      <div className="grid grid-cols-[1fr_auto_1fr] gap-3 items-stretch">
        <div className="rounded-2xl p-4 text-center" style={{ background: la.colorDim, border: `1px solid ${la.border}` }}>
          <div className="text-[9px] font-bold uppercase tracking-widest mb-1" style={{ color: la.color }}>Side A</div>
          <div className="text-[16px] font-black leading-tight" style={{ color: c.tp }}>{la.label}</div>
          <div className="text-[10px] mt-0.5 font-semibold" style={{ color: la.color }}>{la.sublabel}</div>
        </div>
        <div className="flex flex-col items-center justify-center gap-1 px-2">
          <div className="text-[13px] font-black" style={{ color: c.tm }}>VS</div>
          <div className="text-[9px] text-center mt-1 leading-relaxed" style={{ color: c.tu }}>
            {MAP_LABELS[config.mapSize] || config.mapSize}<br />
            {TRAFFIC_LABELS[config.traffic] || config.traffic}
          </div>
        </div>
        <div className="rounded-2xl p-4 text-center" style={{ background: ra.colorDim, border: `1px solid ${ra.border}` }}>
          <div className="text-[9px] font-bold uppercase tracking-widest mb-1" style={{ color: ra.color }}>Side B</div>
          <div className="text-[16px] font-black leading-tight" style={{ color: c.tp }}>{ra.label}</div>
          <div className="text-[10px] mt-0.5 font-semibold" style={{ color: ra.color }}>{ra.sublabel}</div>
        </div>
      </div>

      {/* Column labels */}
      <div className="grid grid-cols-[1fr_auto_1fr] gap-3 px-1">
        <div className="text-right text-[9px] font-bold uppercase tracking-wider" style={{ color: la.color }}>{la.sublabel}</div>
        <div style={{ minWidth: '148px' }} />
        <div className="text-left text-[9px] font-bold uppercase tracking-wider" style={{ color: ra.color }}>{ra.sublabel}</div>
      </div>

      {/* Routing Efficiency */}
      <CompareSection title="Routing Efficiency" icon={routingIcon}>
        {ROUTING_KPI.map(m => (
          <MetricRow key={m.label} label={m.label} unit={m.unit}
            lv={m.getValue(la)} rv={m.getValue(ra)} lowerBetter={m.lowerBetter} la={la} ra={ra} />
        ))}
      </CompareSection>

      {/* Environmental Impact */}
      <CompareSection title="Environmental Impact" icon={environIcon}>
        {ENVIRON_KPI.map(m => (
          <MetricRow key={m.label} label={m.label} unit={m.unit}
            lv={m.getValue(la)} rv={m.getValue(ra)} lowerBetter={m.lowerBetter} la={la} ra={ra} />
        ))}
      </CompareSection>

      {/* Computational Factors — only when both are learning-based */}
      {bothLearning && (
        <CompareSection title="Computational Factors" icon={computeIcon}>
          {COMPUTE_KPI.map(m => (
            <MetricRow key={m.label} label={m.label} unit={m.unit}
              lv={m.getValue(la)} rv={m.getValue(ra)} lowerBetter={m.lowerBetter} la={la} ra={ra} />
          ))}
          {convL !== null && convR !== null && (
            <MetricRow label="Convergence Episode" unit="ep"
              lv={convL} rv={convR} lowerBetter={true} la={la} ra={ra} intFmt />
          )}
        </CompareSection>
      )}
    </div>
  )
}

// ─── Summary Page (stateful) ──────────────────────────────────────────────────

function SummaryPage({ onNavigate }: { onNavigate: (p: Page) => void }) {
  const c = useDashColors()
  const { reducedMotion, ready } = usePageEntrance()
  const [losTab, setLosTab] = useState<'free_flow' | 'stable_flow' | 'forced_flow'>('forced_flow')

  type LosKey = 'free_flow' | 'stable_flow' | 'forced_flow'
  const LOS_TABS: { key: LosKey; label: string; sub: string; note: string }[] = [
    { key: 'free_flow',   label: 'Low Demand',      sub: '1,000 veh/hr', note: 'light load · minimal queueing' },
    { key: 'stable_flow', label: 'Moderate Demand',  sub: '1,200 veh/hr', note: 'emerging congestion · queueing onset' },
    { key: 'forced_flow', label: 'High Demand',       sub: '2,000 veh/hr', note: 'saturated network · significant delay' },
  ]

  function perLos(id: AlgoKey, k: LosKey) {
    const algo = ALGO_LIST.find(a => a.id === id)!
    if (id === 'civiq') return {
      travelTime: CIVIQ_LOS_REF[k].travelTime * 60,
      waitTime:   CIVIQ_LOS_REF[k].waitTime,
      throughput: CIVIQ_LOS_REF[k].throughput,
      co2:        CIVIQ_LOS_REF[k].co2,
      fuel:       CIVIQ_LOS_REF[k].fuel,
    }
    if (id === 'qmix') return {
      travelTime: QMIX_LOS_REF[k].travelTime_s,
      waitTime:   QMIX_LOS_REF[k].waitTime_s,
      throughput: QMIX_LOS_REF[k].throughput,
      co2:        QMIX_LOS_REF[k].co2,
      fuel:       QMIX_LOS_REF[k].fuel,
    }
    return {
      travelTime: SELFISH_LOS_REF[k].travelTime_s,
      waitTime:   SELFISH_LOS_REF[k].waitTime_s,
      throughput: SELFISH_LOS_REF[k].throughput,
      co2:        SELFISH_LOS_REF[k].co2,
      fuel:       SELFISH_LOS_REF[k].fuel,
    }
  }

  type LosMetricKey = 'travelTime' | 'waitTime' | 'throughput' | 'co2' | 'fuel'
  const LOS_METRICS: { key: LosMetricKey; label: string; unit: string; lowerBetter: boolean; fmt: (v: number) => string }[] = [
    { key: 'travelTime', label: 'Avg. Travel Time',  unit: 'sec',     lowerBetter: true,  fmt: v => v.toFixed(2) },
    { key: 'waitTime',   label: 'Avg. Wait Time',    unit: 'sec',     lowerBetter: true,  fmt: v => v.toFixed(2) },
    { key: 'throughput', label: 'Throughput',         unit: 'veh/hr',  lowerBetter: false, fmt: v => Math.round(v).toLocaleString() },
    { key: 'co2',        label: 'CO₂ Emissions',     unit: 'g/km',    lowerBetter: true,  fmt: v => v.toFixed(2) },
    { key: 'fuel',       label: 'Diesel Consumption', unit: 'L/100km', lowerBetter: true,  fmt: v => v.toFixed(2) },
  ]

  const RANK_META: { label: string; unit: string; key: 'travelTime' | 'waitTime' | 'throughput' | 'co2' | 'fuel'; fmt: (v: number) => string; lowerBetter: boolean }[] = [
    { label: 'Travel Time', unit: 'sec',     key: 'travelTime', fmt: v => v.toFixed(2),                       lowerBetter: true  },
    { label: 'Wait Time',   unit: 'sec',     key: 'waitTime',   fmt: v => v.toFixed(2),                       lowerBetter: true  },
    { label: 'Throughput',  unit: 'veh/hr',  key: 'throughput', fmt: v => Math.round(v).toLocaleString(),      lowerBetter: false },
    { label: 'CO₂',         unit: 'g/km',    key: 'co2',        fmt: v => v.toFixed(2),                       lowerBetter: true  },
    { label: 'Diesel',      unit: 'L/100km', key: 'fuel',       fmt: v => v.toFixed(2),                       lowerBetter: true  },
  ]
  const rankNorm = RANK_META.map(m => {
    const vals  = ALGO_LIST.map(a => perLos(a.id, losTab)[m.key])
    const best  = m.lowerBetter ? Math.min(...vals) : Math.max(...vals)
    const worst = m.lowerBetter ? Math.max(...vals) : Math.min(...vals)
    return { best, worst, range: worst - best }
  })

  // Per-LOS radar scores — axes: Throughput, Wait Time, Travel Time, Compute, CO₂, Fuel, RTF
  const _rd = (['civiq', 'qmix', 'selfish'] as AlgoKey[]).map(id => perLos(id, losTab))
  const radarScores: Record<AlgoKey, number[]> = {
    civiq:   [normRadar(_rd.map(d => d.throughput), true)[0],  normRadar(_rd.map(d => d.waitTime), false)[0],   normRadar(_rd.map(d => d.travelTime), false)[0],  ALGO.civiq.scores[3],   normRadar(_rd.map(d => d.co2), false)[0],  normRadar(_rd.map(d => d.fuel), false)[0],  ALGO.civiq.scores[6]],
    qmix:    [normRadar(_rd.map(d => d.throughput), true)[1],  normRadar(_rd.map(d => d.waitTime), false)[1],   normRadar(_rd.map(d => d.travelTime), false)[1],  ALGO.qmix.scores[3],    normRadar(_rd.map(d => d.co2), false)[1],  normRadar(_rd.map(d => d.fuel), false)[1],  ALGO.qmix.scores[6]],
    selfish: [normRadar(_rd.map(d => d.throughput), true)[2],  normRadar(_rd.map(d => d.waitTime), false)[2],   normRadar(_rd.map(d => d.travelTime), false)[2],  ALGO.selfish.scores[3], normRadar(_rd.map(d => d.co2), false)[2],  normRadar(_rd.map(d => d.fuel), false)[2],  ALGO.selfish.scores[6]],
  }

  const KEY_FINDINGS = [
    { tag: 'Throughput',   tagColor: '#22C55E',
      headline: 'CiViQ moves 37% more vehicles per hour when traffic is heaviest',
      body: 'At high traffic (2,000 veh/hr), CiViQ handles 2,273 vehicles per hour vs. Mono-QMIX\'s 1,654. At light traffic the gap nearly disappears — showing that city-wide coordination really pays off only when roads are close to full.' },
    { tag: 'Travel Time',  tagColor: '#38BDF8',
      headline: 'Mono-QMIX gets drivers there faster — but only at moderate traffic',
      body: 'At moderate traffic (1,200 veh/hr), Mono-QMIX averages 108.3 sec travel time vs. CiViQ\'s 122.1 sec — 11.3% faster. When roads aren\'t fully loaded, a single central AI can fine-tune signals more efficiently than a two-level system.' },
    { tag: 'Wait Time',    tagColor: '#F59E0B',
      headline: 'Heavy traffic exposes Mono-QMIX\'s limits — wait times climb',
      body: 'At high traffic (2,000 veh/hr), Mono-QMIX vehicles wait an average of 23.7 sec — 23% longer than CiViQ\'s 19.3 sec. When dozens of intersections need coordinating at once, the single AI struggles to keep queues short.' },
    { tag: 'Baseline',     tagColor: '#F87171',
      headline: 'No AI needed at low traffic — Selfish Routing keeps up',
      body: 'At light traffic (1,000 veh/hr), Selfish Routing averages 120.8 sec travel time — nearly identical to CiViQ\'s 121.7 sec. When roads are quiet, vehicles naturally find good routes on their own with no training or coordination required.' },
    { tag: 'Scalability',  tagColor: '#A78BFA',
      headline: 'CiViQ gets better as traffic grows; Mono-QMIX gets worse',
      body: 'Going from moderate to high traffic, CiViQ\'s average travel time actually improves by 8.4% (122.1 → 111.8 sec). Mono-QMIX goes the other way — travel time worsens by 16.3% (108.3 → 125.9 sec) as the number of intersections to manage grows.' },
    { tag: 'Coordination', tagColor: '#34D399',
      headline: 'CiViQ\'s wait-time advantage grows as roads get busier',
      body: 'At light traffic, CiViQ saves just 0.2 sec per vehicle over Selfish Routing (7.2 vs. 7.3 sec). At heavy traffic that gap widens to 1.4 sec (19.3 vs. 20.7 sec) — a 7.2% reduction. Coordinated signals make a bigger difference when every second counts.' },
  ]

  return (
  <div className="p-4 md:p-6 space-y-4 md:space-y-5 overflow-y-auto" style={{
    height: '100%',
    opacity: ready ? 1 : 0,
    transform: ready ? 'none' : 'translateY(8px)',
    transition: reducedMotion ? 'none' : 'opacity 260ms ease, transform 320ms ease',
  }}>

    {/* Header */}
    <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
      <div>
        <h2 className="text-xl md:text-2xl font-bold" style={{ color: c.tp }}>Algorithm Comparison</h2>
        <div className="flex items-center gap-4 mt-1.5">
          {ALGO_LIST.map((a) => (
            <div key={a.id} className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full" style={{ background: a.color }} />
              <span className="text-[11px]" style={{ color: c.tm }}>{a.label}</span>
            </div>
          ))}
        </div>
      </div>
      {/* LOS Selector */}
      <div className="flex flex-wrap gap-1 p-1 rounded-xl flex-shrink-0" style={{ background: c.itemBg }}>
        {LOS_TABS.map(t => (
          <button key={t.key} onClick={() => setLosTab(t.key)}
            className="px-4 md:px-6 py-2 rounded-lg text-[11px] font-semibold transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
            style={{
              background: losTab === t.key ? c.itemBgMd : 'transparent',
              color:      losTab === t.key ? c.tp       : c.ts,
              border:     losTab === t.key ? `1px solid ${c.glassBorder}` : '1px solid transparent',
            }}>
            {t.label}&nbsp;<span style={{ opacity: 0.55 }}>({t.sub})</span>
            {losTab === t.key && <div className="text-[8px] font-normal mt-0.5" style={{ opacity: 0.5 }}>{t.note}</div>}
          </button>
        ))}
      </div>
    </div>

    {/* Algorithm KPI cards */}
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
      {ALGO_LIST.map((a) => (
        <GlassCard key={a.id} className="p-5 relative overflow-hidden cursor-pointer"
          style={{ border: `1px solid ${a.border}`, transition: 'transform 0.15s ease, box-shadow 0.15s ease' }}
          onClick={() => onNavigate(a.id)}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLElement).style.transform = 'translateY(-2px)'
            ;(e.currentTarget as HTMLElement).style.boxShadow = `0 16px 48px rgba(0,0,0,0.45), 0 0 0 1px ${a.border}`
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLElement).style.transform = 'translateY(0)'
            ;(e.currentTarget as HTMLElement).style.boxShadow = c.glassShadow
          }}>
          <div className="absolute inset-0 pointer-events-none"
            style={{ background: `radial-gradient(ellipse at 80% 0%, ${a.colorDim}, transparent 65%)` }} />
          <div className="relative">
            {/* Card header: title + view detail button on the same row */}
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-[14px] font-bold" style={{ color: c.tp }}>{a.label}</h3>
              <button
                className="text-[10px] font-semibold px-2.5 py-1 rounded-lg transition-all duration-150"
                style={{ background: c.badgeBg, color: c.ts, border: `1px solid ${c.badgeBorder}` }}
                onMouseEnter={(e) => {
                  const el = e.currentTarget as HTMLElement
                  el.style.background = `${a.color}22`
                  el.style.color = a.color
                  el.style.borderColor = `${a.color}55`
                }}
                onMouseLeave={(e) => {
                  const el = e.currentTarget as HTMLElement
                  el.style.background = c.badgeBg
                  el.style.color = c.ts
                  el.style.borderColor = c.badgeBorder
                }}>
                View Detail →
              </button>
            </div>
            {/* KPI metrics */}
            <div className="space-y-2.5">
              {RANK_META.map((m, mi) => {
                const val = perLos(a.id, losTab)[m.key]
                const { best, worst, range } = rankNorm[mi]
                const isBest = val === best
                const barPct = range > 0
                  ? (m.lowerBetter ? 50 + 50 * (worst - val) / range : 50 + 50 * (val - worst) / range)
                  : 75
                return (
                  <div key={m.label}>
                    <div className="flex items-baseline justify-between mb-1">
                      <span className="text-[9px] font-semibold uppercase tracking-wider" style={{ color: c.tu }}>{m.label}</span>
                      <div className="flex items-center gap-1.5">
                        {isBest && (
                          <span className="text-[8px] font-bold px-1.5 py-0.5 rounded-full"
                            style={{ background: 'rgba(34,197,94,0.14)', color: '#22C55E', border: '1px solid rgba(34,197,94,0.25)' }}>
                            {m.lowerBetter ? 'Lowest' : 'Highest'}
                          </span>
                        )}
                        <span className="text-[11px] font-bold tabular-nums" style={{ color: isBest ? c.tp : c.ts }}>
                          {m.fmt(val)}<span className="text-[9px] font-normal ml-0.5" style={{ color: c.tu }}>{m.unit}</span>
                        </span>
                      </div>
                    </div>
                    <div className="w-full h-1.5 rounded-full" style={{ background: c.barTrack }}>
                      <div className="h-full rounded-full transition-all duration-700"
                        style={{ width: `${Math.min(100, Math.max(0, barPct)).toFixed(1)}%`, background: a.color, opacity: isBest ? 0.9 : 0.25 }} />
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </GlassCard>
      ))}
    </div>

    {/* Performance Profile + Key Findings */}
    <div className="grid grid-cols-1 xl:grid-cols-5 gap-4">
      <GlassCard className="xl:col-span-2 p-5">
        <h3 className="text-[13px] font-bold mb-3" style={{ color: c.tp }}>Performance Profile</h3>
        <RadarChart scores={radarScores} />
        <div className="flex justify-center gap-5 mt-2">
          {ALGO_LIST.map((a) => (
            <div key={a.id} className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full" style={{ background: a.color }} />
              <span className="text-[10px]" style={{ color: c.tm }}>{a.label}</span>
            </div>
          ))}
        </div>
      </GlassCard>

      <GlassCard className="xl:col-span-3 p-5">
        <h3 className="text-[13px] font-bold mb-4" style={{ color: c.tp }}>Key Findings</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {KEY_FINDINGS.map((f) => (
            <div key={f.headline} className="p-4 rounded-xl"
              style={{ background: c.itemBg, border: `1px solid ${c.divider}` }}>
              <span className="inline-block text-[9px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider mb-2"
                style={{ background: `${f.tagColor}1a`, color: f.tagColor, border: `1px solid ${f.tagColor}33` }}>
                {f.tag}
              </span>
              <p className="text-[12px] font-semibold mb-1.5 leading-snug" style={{ color: c.tp }}>
                {f.headline}
              </p>
              <p className="text-[11px] leading-relaxed" style={{ color: c.tm }}>
                {f.body}
              </p>
            </div>
          ))}
        </div>
      </GlassCard>
    </div>

    {/* Per-LOS Performance */}
    <GlassCard className="p-6 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <h3 className="text-[13px] font-bold" style={{ color: c.tp }}>Per-KPI Performance</h3>
        <div className="flex gap-1 p-0.5 rounded-xl" style={{ background: c.itemBg }}>
          {LOS_TABS.map(t => (
            <button key={t.key} onClick={() => setLosTab(t.key)}
              className="px-5 py-1.5 rounded-lg text-[10px] font-semibold transition-all duration-150"
              style={{
                background: losTab === t.key ? c.itemBgMd : 'transparent',
                color:      losTab === t.key ? c.tp       : c.ts,
                border:     losTab === t.key ? `1px solid ${c.glassBorder}` : '1px solid transparent',
              }}>
              {t.label}{losTab === t.key && <span style={{ opacity: 0.5 }}> · {t.sub}</span>}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-5 flex-1">
        {/* Avg. Travel Time — full width */}
        {(() => {
          const m = LOS_METRICS[0]
          const rows  = ALGO_LIST.map(a => ({ a, val: perLos(a.id, losTab)[m.key] }))
          const vals  = rows.map(r => r.val)
          const best  = Math.min(...vals)
          const worst = Math.max(...vals)
          const range = worst - best
          return (
            <div className="p-4 rounded-xl" style={{ background: c.itemBg, border: `1px solid ${c.divider}` }}>
              <div className="flex justify-between items-baseline mb-3">
                <span className="text-[11px] font-bold" style={{ color: c.ts }}>{m.label}</span>
                <span className="text-[9px]" style={{ color: c.tm }}>{m.unit}</span>
              </div>
              <div className="space-y-2.5">
                {rows.map(({ a, val }) => {
                  const isWinner = val === best
                  const barPct = range > 0 ? 50 + 50 * (worst - val) / range : 75
                  const delta = !isWinner && best !== 0 ? Math.abs((val - best) / best * 100) : 0
                  return (
                    <div key={a.id} className="flex items-center gap-3">
                      <div className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: a.color }} />
                      <span className="text-[10px] font-semibold w-[108px] flex-shrink-0" style={{ color: a.color }}>{a.label}</span>
                      <div className="flex-1 h-1.5 rounded-full" style={{ background: c.barTrack }}>
                        <div className="h-full rounded-full transition-all duration-500"
                          style={{ width: `${Math.min(100, Math.max(0, barPct)).toFixed(1)}%`, background: a.color, opacity: isWinner ? 0.9 : 0.28 }} />
                      </div>
                      <span className="text-[11px] w-14 text-right tabular-nums font-bold"
                        style={{ color: isWinner ? c.tp : c.ts }}>
                        {m.fmt(val)}
                      </span>
                      <div className="w-[44px] text-right flex-shrink-0">
                        {isWinner
                          ? <span className="text-[8px] font-bold px-1.5 py-0.5 rounded-full"
                              style={{ background: 'rgba(34,197,94,0.15)', color: '#22C55E' }}>Lowest</span>
                          : <span className="text-[9px]" style={{ color: c.tm }}>+{delta.toFixed(1)}%</span>
                        }
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )
        })()}

        {/* Avg. Wait Time | Throughput — 2 cols */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {LOS_METRICS.slice(1, 3).map(m => {
            const rows  = ALGO_LIST.map(a => ({ a, val: perLos(a.id, losTab)[m.key] }))
            const vals  = rows.map(r => r.val)
            const best  = m.lowerBetter ? Math.min(...vals) : Math.max(...vals)
            const worst = m.lowerBetter ? Math.max(...vals) : Math.min(...vals)
            const range = worst - best
            return (
              <div key={m.key} className="p-4 rounded-xl" style={{ background: c.itemBg, border: `1px solid ${c.divider}` }}>
                <div className="flex justify-between items-baseline mb-3">
                  <span className="text-[11px] font-bold" style={{ color: c.ts }}>{m.label}</span>
                  <span className="text-[9px]" style={{ color: c.tm }}>{m.unit}</span>
                </div>
                <div className="space-y-2.5">
                  {rows.map(({ a, val }) => {
                    const isWinner = val === best
                    const barPct = range > 0
                      ? (m.lowerBetter ? 50 + 50 * (worst - val) / range : 50 + 50 * (val - worst) / range)
                      : 75
                    const delta = !isWinner && best !== 0 ? Math.abs((val - best) / best * 100) : 0
                    return (
                      <div key={a.id} className="flex items-center gap-3">
                        <div className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: a.color }} />
                        <div className="flex-1 h-1.5 rounded-full" style={{ background: c.barTrack }}>
                          <div className="h-full rounded-full transition-all duration-500"
                            style={{ width: `${Math.min(100, Math.max(0, barPct)).toFixed(1)}%`, background: a.color, opacity: isWinner ? 0.9 : 0.28 }} />
                        </div>
                        <span className="text-[11px] w-14 text-right tabular-nums font-bold"
                          style={{ color: isWinner ? c.tp : c.ts }}>
                          {m.fmt(val)}
                        </span>
                        <div className="w-[44px] text-right flex-shrink-0">
                          {isWinner
                            ? <span className="text-[8px] font-bold px-1.5 py-0.5 rounded-full"
                                style={{ background: 'rgba(34,197,94,0.15)', color: '#22C55E' }}>
                                {m.lowerBetter ? 'Lowest' : 'Highest'}
                              </span>
                            : <span className="text-[9px]" style={{ color: c.tm }}>+{delta.toFixed(1)}%</span>
                          }
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )
          })}
        </div>

        {/* Divider */}
        <div style={{ borderTop: `1px solid ${c.divider}` }} />

        {/* CO₂ Emissions | Fuel Consumption — 2 cols */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {LOS_METRICS.slice(3).map(m => {
            const rows  = ALGO_LIST.map(a => ({ a, val: perLos(a.id, losTab)[m.key] }))
            const vals  = rows.map(r => r.val)
            const best  = Math.min(...vals)
            const worst = Math.max(...vals)
            const range = worst - best
            return (
              <div key={m.key} className="p-4 rounded-xl" style={{ background: c.itemBg, border: `1px solid ${c.divider}` }}>
                <div className="flex justify-between items-baseline mb-3">
                  <span className="text-[11px] font-bold" style={{ color: c.ts }}>{m.label}</span>
                  <span className="text-[9px]" style={{ color: c.tm }}>{m.unit}</span>
                </div>
                <div className="space-y-2.5">
                  {rows.map(({ a, val }) => {
                    const isWinner = val === best
                    const barPct = range > 0 ? 50 + 50 * (worst - val) / range : 75
                    const delta = !isWinner && best !== 0 ? Math.abs((val - best) / best * 100) : 0
                    return (
                      <div key={a.id} className="flex items-center gap-3">
                        <div className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: a.color }} />
                        <div className="flex-1 h-1.5 rounded-full" style={{ background: c.barTrack }}>
                          <div className="h-full rounded-full transition-all duration-500"
                            style={{ width: `${Math.min(100, Math.max(0, barPct)).toFixed(1)}%`, background: a.color, opacity: isWinner ? 0.9 : 0.28 }} />
                        </div>
                        <span className="text-[11px] w-14 text-right tabular-nums font-bold"
                          style={{ color: isWinner ? c.tp : c.ts }}>
                          {m.fmt(val)}
                        </span>
                        <div className="w-[44px] text-right flex-shrink-0">
                          {isWinner
                            ? <span className="text-[8px] font-bold px-1.5 py-0.5 rounded-full"
                                style={{ background: 'rgba(34,197,94,0.15)', color: '#22C55E' }}>Lowest</span>
                            : <span className="text-[9px]" style={{ color: c.tm }}>+{delta.toFixed(1)}%</span>
                          }
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </GlassCard>
  </div>
  )
}

// ─── Gauge Chart ──────────────────────────────────────────────────────────────
// Wraps react-gauge-chart. To plug in real data, pass the live `value` and
// `max` from your API response — the `percent` is computed here automatically.

const GaugeChart = memo(function GaugeChart({ value, max, label, unit, accentColor, description, onClick }: {
  value: number
  max: number
  label: string
  unit: string
  accentColor: string
  description?: string
  onClick?: () => void
}) {
  const c = useDashColors()
  const percent = Math.min(1, Math.max(0, value / max))

  return (
    <GlassCard className="group relative z-0 hover:z-20 flex-1 flex flex-col items-center justify-center py-2 px-2 gap-1"
      style={{ minWidth: 0, cursor: onClick ? 'pointer' : undefined }}
      onClick={onClick}>
      {/* Top-right: View detail badge + tooltip, side by side */}
      <div className="absolute top-2 right-2 z-20 flex items-center gap-1">
        {onClick && (
          <span className="text-[9px] font-semibold px-2 py-0.5 rounded-full transition-all duration-200 group-hover:-translate-y-[1px] group-hover:translate-x-0.5 group-hover:scale-[1.03]"
            style={{ background: 'rgba(56,189,248,0.1)', border: '1px solid rgba(56,189,248,0.22)', color: c.isDark ? 'rgba(186,230,253,0.65)' : 'rgba(3,105,161,0.9)', whiteSpace: 'nowrap' }}>
            View detail ↗
          </span>
        )}
        {description && (
          <div className="group/info relative">
            <div
              className="w-5 h-5 rounded-full flex items-center justify-center"
              style={{
                background: 'rgba(6,182,212,0.2)',
                border: '1px solid rgba(6,182,212,0.65)',
                color: c.isDark ? 'rgba(217,249,255,0.95)' : 'rgba(3,105,161,0.95)',
                boxShadow: '0 0 0 1px rgba(0,0,0,0.2) inset',
              }}
            >
              <span className="text-[11px] font-bold leading-none">!</span>
            </div>
            <div className="pointer-events-none absolute right-0 top-[calc(100%+8px)] w-[280px] rounded-lg px-3 py-2 text-[10px] leading-relaxed opacity-0 translate-y-1 transition-all duration-150 group-hover/info:opacity-100 group-hover/info:translate-y-0"
              style={{
                background: c.tooltipBg,
                border: `1px solid ${c.tooltipBorder}`,
                color: c.tooltipColor,
                boxShadow: c.tooltipShadow,
                zIndex: 85,
              }}>
              {description}
            </div>
          </div>
        )}
      </div>

      <GaugeComponent
        id={`gauge-${label.replace(/\W/g, '')}`}
        percent={percent}
        nrOfLevels={20}
        colors={['#16A34A', '#65A30D', '#CA8A04', '#DC2626']}
        arcWidth={0.3}
        arcPadding={0.02}
        needleColor={accentColor}
        needleBaseColor={accentColor}
        animate={false}
        hideText
        style={{ width: '100%', maxWidth: '140px' }}
      />
      {/* Value + unit centred below arc */}
      <div className="flex flex-col items-center mt-0.5">
        <span className="text-[20px] font-extrabold tabular-nums leading-none" style={{ color: c.tp }}>
          {typeof value === 'number' ? parseFloat(value.toFixed(2)) : value}
        </span>
        <span className="text-[10px] font-semibold mt-0.5" style={{ color: c.ts }}>{unit}</span>
        <span className="text-[9px] font-bold uppercase tracking-wider mt-1.5 text-center"
          style={{ color: c.tm }}>{label}</span>
      </div>
    </GlassCard>
  )
})

// ─── Map Player ───────────────────────────────────────────────────────────────

const MapPlayer = ({ algo, mapSize, trafficScale, onCo2Click, onFuelClick }: { algo: AlgoData; mapSize: string; trafficScale: string; onCo2Click?: () => void; onFuelClick?: () => void }) => {
  const c = useDashColors()
  // Playback
  const videoRef = useRef<HTMLVideoElement>(null)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [speed, setSpeed] = useState(1)
  const [videoReady, setVideoReady] = useState(false)

  // Pan / zoom — start zoomed in so the video fills and users can explore
  const INITIAL_ZOOM = 2.5
  const containerRef = useRef<HTMLDivElement>(null)
  const [zoom, setZoom] = useState(INITIAL_ZOOM)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const zoomRef = useRef(INITIAL_ZOOM)
  const panRef = useRef({ x: 0, y: 0 })
  zoomRef.current = zoom
  panRef.current = pan
  const dragging = useRef(false)
  const dragOrigin = useRef({ x: 0, y: 0 })
  const panOrigin = useRef({ x: 0, y: 0 })

  // Reset playback state when algo or traffic scale changes (new video src)
  useEffect(() => {
    setPlaying(false)
    setCurrentTime(0)
    setDuration(0)
    setVideoReady(false)
  }, [algo.id, trafficScale])

  // Wire playback rate to video
  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = speed
  }, [speed])

  // Center the view on mount so the zoomed-in video is centered
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const { width, height } = el.getBoundingClientRect()
    setPan({
      x: -(width * (INITIAL_ZOOM - 1)) / 2,
      y: -(height * (INITIAL_ZOOM - 1)) / 2,
    })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Wheel → zoom toward cursor (passive: false so we can preventDefault)
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const handler = (e: WheelEvent) => {
      e.preventDefault()
      const factor = e.deltaY < 0 ? 1.15 : 0.87
      const newZoom = Math.min(8, Math.max(1, zoomRef.current * factor))
      const rect = el.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top
      // Keep the point under the cursor fixed
      const rawX = mx - (mx - panRef.current.x) * (newZoom / zoomRef.current)
      const rawY = my - (my - panRef.current.y) * (newZoom / zoomRef.current)
      const { width, height } = rect
      setZoom(newZoom)
      setPan({
        x: Math.min(0, Math.max(rawX, width  * (1 - newZoom))),
        y: Math.min(0, Math.max(rawY, height * (1 - newZoom))),
      })
    }
    el.addEventListener('wheel', handler, { passive: false })
    return () => el.removeEventListener('wheel', handler)
  }, [])

  const onMouseDown = (e: React.MouseEvent) => {
    dragging.current = true
    dragOrigin.current = { x: e.clientX, y: e.clientY }
    panOrigin.current = { ...panRef.current }
    e.preventDefault()
  }
  const clampPan = (x: number, y: number, z: number) => {
    const el = containerRef.current
    if (!el) return { x, y }
    const { width, height } = el.getBoundingClientRect()
    return {
      x: Math.min(0, Math.max(x, width  * (1 - z))),
      y: Math.min(0, Math.max(y, height * (1 - z))),
    }
  }

  const onMouseMove = (e: React.MouseEvent) => {
    if (!dragging.current) return
    const raw = {
      x: panOrigin.current.x + (e.clientX - dragOrigin.current.x),
      y: panOrigin.current.y + (e.clientY - dragOrigin.current.y),
    }
    setPan(clampPan(raw.x, raw.y, zoomRef.current))
  }
  const stopDrag = () => { dragging.current = false }

  // Playback helpers
  const togglePlay = () => {
    const v = videoRef.current
    if (!v) return
    if (v.paused) { v.play(); setPlaying(true) }
    else { v.pause(); setPlaying(false) }
  }
  const seek = (pct: number) => {
    const v = videoRef.current
    if (!v || !duration) return
    v.currentTime = Math.max(0, Math.min(duration, pct * duration))
  }
  const reset = () => {
    const v = videoRef.current
    if (v) { v.pause(); v.currentTime = 0 }
    setPlaying(false)
    setCurrentTime(0)
    // Return to initial centered zoom
    const el = containerRef.current
    const { width, height } = el?.getBoundingClientRect() ?? { width: 0, height: 0 }
    setZoom(INITIAL_ZOOM)
    setPan({
      x: -(width * (INITIAL_ZOOM - 1)) / 2,
      y: -(height * (INITIAL_ZOOM - 1)) / 2,
    })
  }

  const pct = duration > 0 ? currentTime / duration : 0
  const fmt = (s: number) =>
    `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`

  return (
    <GlassCard className="p-4 flex flex-col gap-3 col-span-2">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-[13px] font-bold" style={{ color: c.tp }}>Map Player</h3>
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono px-2 py-0.5 rounded tabular-nums"
            style={{ background: algo.colorDim, color: algo.color, border: `1px solid ${algo.border}` }}>
            {fmt(currentTime)} / {duration > 0 ? fmt(duration) : '--:--'}
          </span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded tabular-nums"
            style={{ background: c.badgeBg, color: c.ts }}>
            {zoom.toFixed(1)}×
          </span>
          <span className="text-[10px] px-2 py-0.5 rounded"
            style={{ background: c.itemBg, color: c.tu }}>
            {MAP_LABELS[mapSize] || mapSize || '—'}
          </span>
        </div>
      </div>

      {/* Video viewport — pan/zoom container */}
      <div
        ref={containerRef}
        className="rounded-xl overflow-hidden relative w-full select-none"
        style={{
          aspectRatio: '16 / 9',
          background: '#000',
          border: `1px solid ${c.divider}`,
          cursor: zoom > 1 ? 'grab' : 'default',
        }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={stopDrag}
        onMouseLeave={stopDrag}
      >
        {/* Transformed video layer */}
        <div
          style={{
            position: 'absolute', inset: 0,
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: '0 0',
            willChange: 'transform',
          }}
        >
          <video
            ref={videoRef}
            src={`/videos/${algo.id}_${trafficScale === 'free_flow' ? 'low' : trafficScale === 'stable_flow' ? 'med' : 'high'}.mp4`}
            muted
            playsInline
            preload="auto"
            style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block', pointerEvents: 'none' }}
            onTimeUpdate={() => setCurrentTime(videoRef.current?.currentTime ?? 0)}
            onLoadedMetadata={() => { setDuration(videoRef.current?.duration ?? 0); setVideoReady(true) }}
            onEnded={() => setPlaying(false)}
          />
        </div>

        {/* Loading overlay */}
        {!videoReady && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 pointer-events-none"
            style={{ background: 'rgba(0,0,0,0.45)' }}>
            <div className="w-5 h-5 rounded-full border-2 border-t-transparent animate-spin"
              style={{ borderColor: `${algo.color} transparent transparent transparent` }} />
            <span className="text-[10px] font-medium" style={{ color: 'rgba(255,255,255,0.35)' }}>Loading video</span>
          </div>
        )}
        {/* Hint overlay when paused at start */}
        {videoReady && !playing && currentTime === 0 && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full"
              style={{ background: 'rgba(0,0,0,0.45)', border: '1px solid rgba(255,255,255,0.1)' }}>
              <span className="text-[10px] font-medium" style={{ color: 'rgba(255,255,255,0.45)' }}>
                Press ▶ to play &nbsp;·&nbsp; scroll to zoom &nbsp;·&nbsp; drag to pan
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex items-center gap-2.5">
        {/* Restart */}
        <button onClick={reset}
          className="w-7 h-7 rounded-full flex-shrink-0 flex items-center justify-center transition-all duration-150"
          style={{ background: c.badgeBg, border: `1px solid ${c.badgeBorder}` }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = c.itemBgMd; (e.currentTarget as HTMLElement).style.borderColor = c.glassBorder }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = c.badgeBg; (e.currentTarget as HTMLElement).style.borderColor = c.badgeBorder }}>
          <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke={c.isDark ? 'rgba(255,255,255,0.6)' : 'rgba(55,65,81,0.6)'}
            strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
            <path d="M3 3v5h5"/>
          </svg>
        </button>
        {/* Play / Pause */}
        <button onClick={togglePlay}
          className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center transition-all duration-150"
          style={{ background: algo.colorDim, border: `1px solid ${algo.border}` }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.filter = 'brightness(1.4)' }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.filter = 'none' }}>
          {playing ? (
            <svg width="10" height="10" viewBox="0 0 10 10" fill={algo.color}>
              <rect x="1" y="1" width="3" height="8" rx="0.5"/>
              <rect x="6" y="1" width="3" height="8" rx="0.5"/>
            </svg>
          ) : (
            <svg width="10" height="10" viewBox="0 0 10 10" fill={algo.color}>
              <path d="M2 1 L9 5 L2 9 Z"/>
            </svg>
          )}
        </button>
        {/* Seek bar — expands on hover */}
        <div className="flex-1 relative flex items-center cursor-pointer" style={{ height: '16px' }}
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect()
            seek((e.clientX - rect.left) / rect.width)
          }}
          onMouseEnter={(e) => { const bar = e.currentTarget.querySelector('.seek-track') as HTMLElement; if (bar) bar.style.height = '6px' }}
          onMouseLeave={(e) => { const bar = e.currentTarget.querySelector('.seek-track') as HTMLElement; if (bar) bar.style.height = '4px' }}>
          <div className="seek-track absolute inset-x-0 rounded-full overflow-hidden"
            style={{ height: '4px', background: c.barTrack, top: '50%', transform: 'translateY(-50%)', transition: 'height 0.12s ease' }}>
            <div className="h-full rounded-full" style={{ width: `${pct * 100}%`, background: algo.color }} />
          </div>
          <div className="absolute rounded-full pointer-events-none"
            style={{
              left: `${pct * 100}%`, top: '50%',
              width: '12px', height: '12px',
              transform: 'translate(-50%, -50%)',
              background: 'white',
              boxShadow: `0 0 6px ${algo.color}, 0 0 0 2px ${algo.color}`,
            }} />
        </div>
        {/* Speed */}
        <div className="flex items-center gap-0.5 flex-shrink-0">
          {[0.5, 1, 2, 4].map(s => (
            <button key={s} onClick={() => setSpeed(s)}
              className="text-[9px] font-bold px-1.5 py-0.5 rounded transition-all duration-150"
              style={{
                background: speed === s ? algo.colorDim : 'transparent',
                color: speed === s ? algo.color : c.tu,
                border: speed === s ? `1px solid ${algo.border}` : '1px solid transparent',
              }}
              onMouseEnter={(e) => { if (s !== speed) { (e.currentTarget as HTMLElement).style.background = c.hoverBg; (e.currentTarget as HTMLElement).style.color = c.tp } }}
              onMouseLeave={(e) => { if (s !== speed) { (e.currentTarget as HTMLElement).style.background = 'transparent'; (e.currentTarget as HTMLElement).style.color = c.tu } }}>
              {s}×
            </button>
          ))}
        </div>
      </div>

      {/* Emission gauges — flex-1 so they fill remaining height naturally */}
      <div className="flex gap-3 pt-5 mt-2" style={{ borderTop: `1px solid ${c.divider}` }}>
        <GaugeChart
          value={algo.co2}
          max={300}
          label="Avg CO₂ Emissions"
          unit="g/km"
          accentColor={algo.color}
          description="The mean carbon dioxide output per vehicle during the simulation, as measured by SUMO's emissions model. Lower values reflect more efficient routing."
          onClick={onCo2Click}
        />
        <GaugeChart
          value={algo.fuel}
          max={55}
          label="Avg. Fuel Consumption (Diesel)"
          unit="L/100km"
          accentColor={algo.color}
          description="The mean fuel used per vehicle throughout the simulation, as calculated by SUMO. Reflects the environmental cost of routing behavior."
          onClick={onFuelClick}
        />
      </div>
    </GlassCard>
  )
}

// ─── Congestion Heatmap ───────────────────────────────────────────────────────
// Per-algorithm policy comparison heatmaps for all demand levels.
const HEATMAP_IMG: Record<'civiq' | 'qmix', Record<string, string>> = {
  civiq: {
    free_flow:   '/heatmap_output/policy_comparison/heatmap_low_civiq.png',
    stable_flow: '/heatmap_output/policy_comparison/heatmap_med_civiq.png',
    forced_flow: '/heatmap_output/policy_comparison/heatmap_high_civiq.png',
  },
  qmix: {
    free_flow:   '/heatmap_output/policy_comparison/heatmap_low_mono_qmix.png',
    stable_flow: '/heatmap_output/policy_comparison/heatmap_med_mono_qmix.png',
    forced_flow: '/heatmap_output/policy_comparison/heatmap_high_mono_qmix.png',
  },
}

// Selfish routing (noop) policy comparison heatmaps keyed by traffic level
const SELFISH_HEATMAP: Record<string, string> = {
  free_flow:   '/heatmap_output/policy_comparison/heatmap_low_noop.png',
  stable_flow: '/heatmap_output/policy_comparison/heatmap_med_noop.png',
  forced_flow: '/heatmap_output/policy_comparison/heatmap_high_noop.png',
}

const CONGESTION_LABEL: Record<string, string> = {
  free_flow:   'Low Congestion · 1,000 veh/hr',
  stable_flow: 'Moderate Congestion · 1,200 veh/hr',
  forced_flow: 'High Congestion · 2,000 veh/hr',
}

const INT_KEYS = ['int1','int2','int3','int4'] as const
type IntKey = typeof INT_KEYS[number]
const INT_LABEL: Record<IntKey, string> = { int1: 'Intersection A', int2: 'Intersection B', int3: 'Intersection C', int4: 'Intersection D' }

// ─── Congestion Detail Modal ───────────────────────────────────────────────────

type CongestionTab = 'queue' | 'density' | 'congestion'
const CONGESTION_TABS: { key: CongestionTab; label: string; sub: string }[] = [
  { key: 'queue',      label: 'Queue Length',    sub: 'per intersection over time' },
  { key: 'density',    label: 'Road Occupancy',  sub: 'mean network density (%)' },
  { key: 'congestion', label: 'Congestion Index', sub: 'composite 0–1 score' },
]

function CongestionDetailModal({ algo, onClose }: { algo: AlgoData; onClose: () => void }) {
  const [tab, setTab] = useState<CongestionTab>('queue')
  const [selInt, setSelInt] = useState<IntKey>('int1')
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [onClose])
  const INT_COLOR: Record<IntKey, string> = {
    int1: algo.color, int2: 'rgba(251,191,36,0.9)', int3: 'rgba(52,211,153,0.9)', int4: 'rgba(248,113,113,0.85)',
  }
  const current = CONGESTION_TABS.find(t => t.key === tab)!

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: 'rgba(0,0,0,0.72)', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)' }}
      onClick={onClose}
    >
      <div
        className="relative w-full mx-6 rounded-2xl p-6 flex flex-col gap-4"
        style={{
          maxWidth: '820px',
          background: 'rgba(4,9,22,0.97)',
          border: '1px solid rgba(255,255,255,0.13)',
          boxShadow: '0 32px 80px rgba(0,0,0,0.85), inset 0 1px 0 rgba(255,255,255,0.08)',
        }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-[17px] font-bold leading-tight" style={{ color: 'rgba(255,255,255,0.92)' }}>
              {current.label}{' '}
              <span style={{ color: algo.color }}>· {algo.label}</span>
            </h3>
            <p className="text-[12px] mt-0.5" style={{ color: 'rgba(255,255,255,0.35)' }}>{current.sub}</p>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
            style={{ background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.12)' }}
            onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.14)' }}
            onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.07)' }}
          >
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.55)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 6L6 18M6 6l12 12"/>
            </svg>
          </button>
        </div>

        {/* Tab bar */}
        <div className="flex gap-1 p-1 rounded-xl" style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)' }}>
          {CONGESTION_TABS.map(({ key, label }) => (
            <button key={key} onClick={() => setTab(key)}
              className="flex-1 py-1.5 rounded-lg text-[11px] font-semibold transition-all duration-150"
              style={{
                background: tab === key ? 'rgba(255,255,255,0.1)' : 'transparent',
                color: tab === key ? 'rgba(255,255,255,0.88)' : 'rgba(255,255,255,0.38)',
                border: tab === key ? '1px solid rgba(255,255,255,0.14)' : '1px solid transparent',
              }}>
              {label}
            </button>
          ))}
        </div>

        {/* ── Queue Length ── */}
        {tab === 'queue' && (
          <>
            <div className="flex items-center justify-between gap-4">
              {/* Intersection selector pills */}
              <div className="flex gap-1.5">
                {INT_KEYS.map(k => (
                  <button key={k} onClick={() => setSelInt(k)}
                    className="px-3 py-1 rounded-full text-[10px] font-semibold transition-all duration-150"
                    style={{
                      background: selInt === k ? algo.colorDim : 'rgba(255,255,255,0.05)',
                      border: selInt === k ? `1px solid ${INT_COLOR[k]}` : '1px solid rgba(255,255,255,0.08)',
                      color: selInt === k ? INT_COLOR[k] : 'rgba(255,255,255,0.42)',
                    }}>
                    {INT_LABEL[k]}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-1.5 flex-shrink-0">
                <InfoBubble side="right" text="Vehicles waiting behind the stop line at the selected intersection each simulation step. Sustained peaks indicate signal inefficiency or upstream congestion spilling into this bottleneck." />
              </div>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={algo.queue} margin={{ top: 8, right: 16, left: 0, bottom: 24 }}>
                <CartesianGrid stroke={CHART_GRID} strokeDasharray="3 4" />
                <XAxis dataKey="step" tick={CHART_AXIS} tickLine={CHART_TICK_LINE} axisLine={CHART_AXIS_LINE}
                  label={{ value: 'Simulation Step', position: 'insideBottom', offset: -14, fill: 'rgba(255,255,255,0.28)', fontSize: 11 }}
                  interval={Math.floor(algo.queue.length / 8)} />
                <YAxis tick={CHART_AXIS} tickLine={CHART_TICK_LINE} axisLine={CHART_AXIS_LINE} width={40}
                  label={{ value: 'Vehicles', angle: -90, position: 'insideLeft', offset: 14, fill: 'rgba(255,255,255,0.28)', fontSize: 11 }} />
                <Tooltip content={<ChartTooltip xLabel="Step" rows={[{ key: selInt, label: INT_LABEL[selInt], color: INT_COLOR[selInt], unit: 'veh' }]} />} />
                <Area type="monotone" dataKey={selInt} stroke={INT_COLOR[selInt]} strokeWidth={2}
                  fill={INT_COLOR[selInt]} fillOpacity={0.12} dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
            {/* Summary stats */}
            <div className="grid grid-cols-4 gap-3 pt-3" style={{ borderTop: '1px solid rgba(255,255,255,0.07)' }}>
              {(() => {
                const vals = algo.queue.map(p => p[selInt])
                return [
                  { label: 'Min',  value: Math.min(...vals).toFixed(1) },
                  { label: 'Max',  value: Math.max(...vals).toFixed(1) },
                  { label: 'Mean', value: (vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(1) },
                  { label: 'Steps', value: String(vals.length) },
                ].map(({ label, value }) => (
                  <div key={label} className="text-center">
                    <div className="text-[10px] uppercase tracking-wider mb-0.5" style={{ color: 'rgba(255,255,255,0.28)' }}>{label}</div>
                    <div className="text-[15px] font-bold tabular-nums" style={{ color: algo.color }}>{value}</div>
                    <div className="text-[10px]" style={{ color: 'rgba(255,255,255,0.28)' }}>veh</div>
                  </div>
                ))
              })()}
            </div>
          </>
        )}

        {/* ── Road Occupancy ── */}
        {tab === 'density' && (
          <>
            <div className="flex items-center gap-4 justify-between">
              <div className="flex items-center gap-4">
                {[
                  { color: algo.color, label: 'Occupancy rate' },
                  { color: 'rgba(255,255,255,0.65)', label: 'MA-8', dashed: true },
                ].map(({ color, label, dashed }) => (
                  <div key={label} className="flex items-center gap-1.5">
                    <svg width="20" height="6"><line x1="0" y1="3" x2="20" y2="3" stroke={color} strokeWidth="2" strokeDasharray={dashed ? '4 2' : undefined} /></svg>
                    <span className="text-[11px]" style={{ color: 'rgba(255,255,255,0.38)' }}>{label}</span>
                  </div>
                ))}
              </div>
              <InfoBubble side="right" text="Percentage of road space occupied by vehicles at each timestep. Values above ~40% indicate congested flow (LOS D or worse). The arc shape shows peak-hour congestion buildup and dispersal across the episode." />
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={algo.density} margin={{ top: 8, right: 16, left: 0, bottom: 24 }}>
                <CartesianGrid stroke={CHART_GRID} strokeDasharray="3 4" />
                <XAxis dataKey="step" tick={CHART_AXIS} tickLine={CHART_TICK_LINE} axisLine={CHART_AXIS_LINE}
                  label={{ value: 'Simulation Step', position: 'insideBottom', offset: -14, fill: 'rgba(255,255,255,0.28)', fontSize: 11 }}
                  interval={Math.floor(algo.density.length / 8)} />
                <YAxis tick={CHART_AXIS} tickLine={CHART_TICK_LINE} axisLine={CHART_AXIS_LINE} width={44} domain={[0, 100]}
                  label={{ value: '%', angle: -90, position: 'insideLeft', offset: 14, fill: 'rgba(255,255,255,0.28)', fontSize: 11 }} />
                <Tooltip content={<ChartTooltip xLabel="Step" rows={[
                  { key: 'density', label: 'Occupancy', color: algo.color, unit: '%' },
                  { key: 'ma',      label: 'MA-8',       color: 'rgba(255,255,255,0.65)', unit: '%' },
                ]} />} />
                <Area type="monotone" dataKey="density" stroke={algo.color} strokeWidth={2}
                  fill={algo.color} fillOpacity={0.13} dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
                <Line type="monotone" dataKey="ma" stroke="rgba(255,255,255,0.65)" strokeWidth={2}
                  dot={false} activeDot={false} strokeDasharray="5 3" isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
            <div className="grid grid-cols-4 gap-3 pt-3" style={{ borderTop: '1px solid rgba(255,255,255,0.07)' }}>
              {(() => {
                const vals = algo.density.map(p => p.density)
                return [
                  { label: 'Min',  value: Math.min(...vals).toFixed(1), unit: '%' },
                  { label: 'Peak', value: Math.max(...vals).toFixed(1), unit: '%' },
                  { label: 'Mean', value: (vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(1), unit: '%' },
                  { label: 'Steps', value: String(vals.length), unit: 'steps' },
                ].map(({ label, value, unit }) => (
                  <div key={label} className="text-center">
                    <div className="text-[10px] uppercase tracking-wider mb-0.5" style={{ color: 'rgba(255,255,255,0.28)' }}>{label}</div>
                    <div className="text-[15px] font-bold tabular-nums" style={{ color: algo.color }}>{value}</div>
                    <div className="text-[10px]" style={{ color: 'rgba(255,255,255,0.28)' }}>{unit}</div>
                  </div>
                ))
              })()}
            </div>
          </>
        )}

        {/* ── Congestion Index ── */}
        {tab === 'congestion' && (
          <>
            <div className="flex items-center gap-4 justify-between">
              <div className="flex items-center gap-4">
                {[
                  { color: algo.color, label: 'Raw index', opacity: 0.4 },
                  { color: algo.color, label: 'Rolling avg. (MA-10)' },
                ].map(({ color, label, opacity }) => (
                  <div key={label} className="flex items-center gap-1.5">
                    <svg width="20" height="6"><line x1="0" y1="3" x2="20" y2="3" stroke={color} strokeWidth="2" strokeOpacity={opacity ?? 1} /></svg>
                    <span className="text-[11px]" style={{ color: 'rgba(255,255,255,0.38)' }}>{label}</span>
                  </div>
                ))}
              </div>
              <InfoBubble side="right" text="Composite score (0 = free flow, 1 = gridlock) combining queue length, occupancy, and travel-time deviation. The raw index is noisy per-step; the rolling average surfaces the underlying trend across the episode." />
            </div>
            <ResponsiveContainer width="100%" height={270}>
              <ComposedChart data={algo.congestion} margin={{ top: 8, right: 16, left: 0, bottom: 24 }}>
                <CartesianGrid stroke={CHART_GRID} strokeDasharray="3 4" />
                <XAxis dataKey="step" tick={CHART_AXIS} tickLine={CHART_TICK_LINE} axisLine={CHART_AXIS_LINE}
                  label={{ value: 'Simulation Step', position: 'insideBottom', offset: -14, fill: 'rgba(255,255,255,0.28)', fontSize: 11 }}
                  interval={Math.floor(algo.congestion.length / 8)} />
                <YAxis tick={CHART_AXIS} tickLine={CHART_TICK_LINE} axisLine={CHART_AXIS_LINE} width={44} domain={[0, 1]}
                  label={{ value: 'Index', angle: -90, position: 'insideLeft', offset: 14, fill: 'rgba(255,255,255,0.28)', fontSize: 11 }} />
                <Tooltip content={<ChartTooltip xLabel="Step" rows={[
                  { key: 'index', label: 'Congestion', color: algo.color, unit: '' },
                  { key: 'ma',    label: 'MA-10',       color: 'rgba(255,255,255,0.72)', unit: '' },
                ]} />} />
                <Area type="monotone" dataKey="index" stroke={algo.color} strokeWidth={1}
                  fill={algo.color} fillOpacity={0.07} dot={false} strokeOpacity={0.35} isAnimationActive={false} />
                <Line type="monotone" dataKey="ma" stroke={algo.color} strokeWidth={2.5}
                  dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
            {/* Level legend + peak stat */}
            <div className="flex items-center justify-between pt-3" style={{ borderTop: '1px solid rgba(255,255,255,0.07)' }}>
              <div className="flex items-center gap-4">
                {[
                  { label: 'Free Flow',  range: '< 0.25',    color: '#4ADE80' },
                  { label: 'Stable',     range: '0.25–0.50', color: '#FACC15' },
                  { label: 'Congested',  range: '0.50–0.75', color: '#FB923C' },
                  { label: 'Gridlock',   range: '> 0.75',    color: '#F87171' },
                ].map(({ label, range, color }) => (
                  <div key={label} className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-full" style={{ background: color }} />
                    <span className="text-[10px]" style={{ color: 'rgba(255,255,255,0.42)' }}>
                      {label} <span style={{ color: 'rgba(255,255,255,0.25)' }}>({range})</span>
                    </span>
                  </div>
                ))}
              </div>
              <div className="text-right flex-shrink-0">
                <div className="text-[10px] uppercase tracking-wider" style={{ color: 'rgba(255,255,255,0.28)' }}>Peak MA Index</div>
                <div className="text-[16px] font-bold tabular-nums" style={{ color: algo.color }}>
                  {Math.max(...algo.congestion.map(p => p.ma)).toFixed(2)}
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

// ─── Congestion Heatmap card ──────────────────────────────────────────────────

function CongestionHeatmap({ algo, heatmapSrc, congestionLabel }: {
  algo: AlgoData
  heatmapSrc?: string
  congestionLabel?: string
}) {
  const c = useDashColors()
  const src = heatmapSrc ?? (algo.id !== 'selfish'
    ? HEATMAP_IMG[algo.id as 'civiq' | 'qmix']?.free_flow
    : SELFISH_HEATMAP.free_flow)
  const imgBg = c.isDark ? 'rgba(3,7,18,0.75)' : 'rgba(220,230,245,0.75)'

  return (
    <GlassCard className="p-4 flex flex-col gap-2">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-[13px] font-bold" style={{ color: c.tp }}>Congestion Heatmap</h3>
      </div>

      {/* Heatmap image */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <div className="rounded-xl overflow-hidden flex-1" style={{ border: `1px solid ${c.divider}`, background: imgBg }}>
        <img src={src} alt={`${algo.label} congestion heatmap`}
          style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }} />
      </div>

      {/* Legend */}
      <div className="flex items-center gap-2">
        <span className="text-[9px]" style={{ color: c.tu }}>Low</span>
        <div className="flex-1 h-1.5 rounded-full" style={{ background: 'linear-gradient(to right, #22C55E, #EAB308, #EF4444)', opacity: 0.7 }} />
        <span className="text-[9px]" style={{ color: c.tu }}>High</span>
      </div>
    </GlassCard>
  )
}

// ─── Selfish Detail Modal types & constants ───────────────────────────────────

type SelfishMetricKey = 'travelTime' | 'waitTime' | 'throughput' | 'co2' | 'fuel'

const CIVIQ_LOS_REF = {
  free_flow:   { travelTime: 2.0284, waitTime: 7.1514,  throughput: 1088.2717, co2: 459.7436, fuel: 19.8422, returnMean: -45133.5598 },
  stable_flow: { travelTime: 2.0345, waitTime: 10.9775, throughput: 1193.2867, co2: 490.171,  fuel: 21.1406, returnMean: -53092.4031 },
  forced_flow: { travelTime: 1.8637, waitTime: 19.2875, throughput: 2272.5674, co2: 528.5116, fuel: 22.734,  returnMean: -98688.5972 },
} as const

const QMIX_LOS_REF = {
  free_flow:   { travelTime_s: 119.1395, waitTime_s: 8.0192,  throughput: 1044.4675, returnMean: -46710.5652,  co2: 505.9064, fuel: 21.8354 },
  stable_flow: { travelTime_s: 108.2599, waitTime_s: 10.4366, throughput: 1245.6503, returnMean: -54377.5525,  co2: 533.6588, fuel: 23.0188 },
  forced_flow: { travelTime_s: 125.9476, waitTime_s: 23.7468, throughput: 1653.8422, returnMean: -105997.3288, co2: 585.9024, fuel: 25.202  },
} as const

const SELFISH_LOS_REF = {
  free_flow:   { travelTime_s: 120.8292, waitTime_s: 7.3442,  throughput: 1120.8551, returnMean: -45803.0291, co2: 459.9288, fuel: 19.8502 },
  stable_flow: { travelTime_s: 122.7603, waitTime_s: 10.5173, throughput: 1258.022,  returnMean: -53173.1002, co2: 463.8582, fuel: 20.006  },
  forced_flow: { travelTime_s: 127.3438, waitTime_s: 20.668,  throughput: 2159.9281, returnMean: -94445.9153, co2: 486.2586, fuel: 20.914  },
} as const

// ─── Key Findings per algorithm × LOS ────────────────────────────────────────
const LOS_FINDINGS: Record<string, Record<string, { headline: string; points: string[] }>> = {
  civiq: {
    free_flow: {
      headline: 'At low traffic, CiViQ leads on wait time — but travel times are neck-and-neck across all approaches.',
      points: [
        'Wait time (7.2 sec) is 10.8% lower than Mono-QMIX (8.0 sec) and 2.6% lower than Selfish Routing (7.3 sec).',
        'Travel time (121.7 sec) is within 2% of both alternatives — the difference is negligible at this demand level.',
        'Throughput (1,088 veh/hr) beats Mono-QMIX (1,044) but is 3.0% below Selfish Routing (1,121).',
      ],
    },
    stable_flow: {
      headline: 'Mono-QMIX outperforms CiViQ at moderate traffic — centralized control finds a better balance here.',
      points: [
        'CiViQ travel time (122.1 sec) is 12.7% slower than Mono-QMIX (108.3 sec) at this demand level.',
        'Throughput (1,193 veh/hr) is 4.4% below Mono-QMIX (1,246) and 5.4% below Selfish Routing (1,258).',
        'Wait time (11.0 sec) is comparable to the others — the hierarchical design adds no clear advantage here.',
      ],
    },
    forced_flow: {
      headline: 'High demand is where CiViQ shines — its two-level design handles network saturation far better.',
      points: [
        'Throughput (2,273 veh/hr) is 37.4% higher than Mono-QMIX (1,654) and 5.2% above Selfish Routing (2,160).',
        'Travel time (111.8 sec) is 12.6% faster than Mono-QMIX (125.9 sec) and 12.2% faster than Selfish Routing (127.3 sec).',
        'Wait time (19.3 sec) is 18.8% lower than Mono-QMIX (23.7 sec) — far fewer vehicles stuck in queues.',
      ],
    },
  },
  qmix: {
    free_flow: {
      headline: 'Mono-QMIX is marginally fastest at low traffic, but moves fewer vehicles than the other approaches.',
      points: [
        'Travel time (119.1 sec) is the lowest of the three at this demand level.',
        'Throughput (1,044 veh/hr) is 4.0% below CiViQ (1,088) and 6.9% below Selfish Routing (1,121).',
        'Wait time (8.0 sec) is higher than both CiViQ (7.2 sec) and Selfish Routing (7.3 sec), hinting at signal inefficiency under light load.',
      ],
    },
    stable_flow: {
      headline: 'Moderate traffic is Mono-QMIX\'s strongest scenario — it achieves the fastest travel times of all three.',
      points: [
        'Travel time (108.3 sec) is 11.3% faster than CiViQ (122.1 sec) and 11.8% faster than Selfish Routing (122.8 sec).',
        'Throughput (1,246 veh/hr) is competitive — only 1.0% below Selfish Routing (1,258) and 4.4% above CiViQ (1,193).',
        'The AI finds efficient signal coordination at this level without being overwhelmed by too many simultaneous decisions.',
      ],
    },
    forced_flow: {
      headline: 'Mono-QMIX struggles most at high demand — coordinating 28+ intersections at once becomes a bottleneck.',
      points: [
        'Throughput (1,654 veh/hr) is 27.2% below CiViQ (2,273) and 23.4% below Selfish Routing (2,160).',
        'Travel time (125.9 sec) and wait time (23.7 sec) are the worst of the three approaches at this level.',
        'As more vehicles enter the network, the single central AI faces an increasingly large decision space.',
      ],
    },
  },
  selfish: {
    free_flow: {
      headline: 'At light traffic, Selfish Routing matches AI systems — coordination adds minimal benefit when roads are clear.',
      points: [
        'Travel time (120.8 sec) is within 2% of CiViQ (121.7 sec) and Mono-QMIX (119.1 sec).',
        'Throughput (1,121 veh/hr) is the highest of the three — vehicles naturally spread across available routes.',
        'Wait time (7.3 sec) nearly matches CiViQ (7.2 sec), showing light-demand routing needs no coordination.',
      ],
    },
    stable_flow: {
      headline: 'Selfish Routing leads on throughput at moderate traffic, but Mono-QMIX opens a clear travel time gap.',
      points: [
        'Throughput (1,258 veh/hr) is the highest of all three approaches at this demand level.',
        'Travel time (122.8 sec) is 13.4% slower than Mono-QMIX (108.3 sec) — signal coordination starts to matter.',
        'Wait time (10.5 sec) is close to CiViQ (11.0 sec), but congestion is building without coordination.',
      ],
    },
    forced_flow: {
      headline: 'Selfish Routing holds up better than expected at high demand, though CiViQ pulls clearly ahead.',
      points: [
        'Throughput (2,160 veh/hr) is 30.6% higher than Mono-QMIX (1,654) and only 5.0% below CiViQ (2,273).',
        'Travel time (127.3 sec) is only 1.1% slower than Mono-QMIX — uncoordinated routing stays competitive.',
        'Wait time (20.7 sec) beats Mono-QMIX (23.7 sec) but is 7.2% higher than CiViQ (19.3 sec).',
      ],
    },
  },
}

// ─── Road Closure Data ────────────────────────────────────────────────────────
// Source: road_closure_{low,med,high}_demand_results.txt — 50 evaluation episodes each.
const ROAD_CLOSURE_DATA = {
  free_flow: {
    selfish: { travelTime: 120.381, waitTime:  7.188, throughput: 1107.405, co2: 464.112, fuel: 20.031, returnMean:  -45215.45 },
    qmix:    { travelTime: 123.751, waitTime:  8.260, throughput:  992.823, co2: 505.432, fuel: 21.814, returnMean:  -46033.23 },
    civiq:   { travelTime: 119.267, waitTime:  7.351, throughput: 1140.348, co2: 466.047, fuel: 20.114, returnMean:  -45854.36 },
  },
  stable_flow: {
    selfish: { travelTime: 120.360, waitTime: 10.831, throughput: 1296.432, co2: 470.007, fuel: 20.271, returnMean:  -53674.21 },
    qmix:    { travelTime: 114.724, waitTime: 11.605, throughput: 1187.676, co2: 535.974, fuel: 23.119, returnMean:  -54622.36 },
    civiq:   { travelTime: 122.307, waitTime: 11.457, throughput: 1192.711, co2: 498.524, fuel: 21.501, returnMean:  -53807.32 },
  },
  forced_flow: {
    selfish: { travelTime: 125.943, waitTime: 21.171, throughput: 2192.296, co2: 493.650, fuel: 21.232, returnMean:  -96019.92 },
    qmix:    { travelTime: 120.691, waitTime: 23.517, throughput: 1721.023, co2: 598.172, fuel: 25.732, returnMean: -107819.89 },
    civiq:   { travelTime: 116.484, waitTime: 20.527, throughput: 2185.031, co2: 534.522, fuel: 22.993, returnMean:  -99280.52 },
  },
} as const

// ─── Episode Detail Modal ─────────────────────────────────────────────────────

type EpisodeMetricKey = keyof EpisodeSeries

const KPI_META: Record<EpisodeMetricKey, { label: string; abbr: string; unit: string }> = {
  travelTime: { label: 'Avg. Travel Time', abbr: 'ATT', unit: 'sec' },
  waitTime:   { label: 'Avg. Wait Time',   abbr: 'AWT', unit: 'sec' },
  throughput: { label: 'Throughput',        abbr: 'TPT', unit: 'veh/hr' },
  speed:      { label: 'Real-time Factor',  abbr: 'RTF', unit: 'x' },
}

const EpisodeDetailModal = ({ algo, metricKey, labelOverride, unitOverride, onClose }: {
  algo: AlgoData
  metricKey: EpisodeMetricKey
  labelOverride?: string
  unitOverride?: string
  onClose: () => void
}) => {
  const c = useDashColors()
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [onClose])
  const baseMeta = KPI_META[metricKey]
  const meta = {
    ...baseMeta,
    label: labelOverride ?? baseMeta.label,
    unit:  unitOverride  ?? baseMeta.unit,
  }
  const data = algo.episodes[metricKey]

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null
    const byKey = Object.fromEntries(payload.map((p: any) => [p.dataKey, p.value]))
    return (
      <div style={{ background: c.tooltipBg, border: `1px solid ${c.tooltipBorder}`, borderRadius: 10, padding: '8px 12px', fontSize: 11, boxShadow: c.tooltipShadow }}>
        <p style={{ color: c.tm, marginBottom: 4 }}>Episode {label}</p>
        {byKey.value  !== undefined && <p style={{ color: algo.color }}>Value: <b>{byKey.value.toFixed(2)}</b> {meta.unit}</p>}
        {byKey.ma     !== undefined && <p style={{ color: c.ts }}>MA-10: <b>{byKey.ma.toFixed(2)}</b> {meta.unit}</p>}
        {byKey.hi     !== undefined && byKey.lo !== undefined && (
          <p style={{ color: c.tm }}>Band: {byKey.lo.toFixed(2)} – {byKey.hi.toFixed(2)}</p>
        )}
      </div>
    )
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{
        background: c.isDark ? 'rgba(0,0,0,0.72)' : 'rgba(15,23,42,0.35)',
        backdropFilter: 'blur(8px)',
        WebkitBackdropFilter: 'blur(8px)',
      }}
      onClick={onClose}
    >
      <div
        className="relative w-full mx-6 rounded-2xl p-6 flex flex-col gap-4"
        style={{
          maxWidth: '790px',
          background: c.isDark ? 'rgba(4,9,22,0.97)' : 'rgba(255,255,255,0.96)',
          border: `1px solid ${c.glassBorder}`,
          boxShadow: c.isDark
            ? '0 32px 80px rgba(0,0,0,0.85), inset 0 1px 0 rgba(255,255,255,0.08)'
            : '0 24px 64px rgba(15,23,42,0.24), inset 0 1px 0 rgba(255,255,255,0.95)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-[17px] font-bold leading-tight" style={{ color: c.tp }}>
              {meta.label}{' '}
              <span style={{ color: algo.color }}>({meta.abbr})</span>
            </h3>
            <p className="text-[12px] mt-0.5" style={{ color: c.tm }}>
              Per-episode trend · {algo.label} · {data.length} episodes
            </p>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
            style={{ background: c.badgeBg, border: `1px solid ${c.badgeBorder}` }}
          >
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke={c.tm}
              strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M18 6L6 18M6 6l12 12"/>
            </svg>
          </button>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-5 flex-wrap">
          <div className="flex items-center gap-2">
            <div className="w-6 h-[2px] rounded" style={{ background: algo.color }} />
            <span className="text-[11px]" style={{ color: c.tm }}>Per-episode value</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-[2px] rounded" style={{ background: c.ts, borderTop: `2px dashed ${c.ts}` }} />
            <span className="text-[11px]" style={{ color: c.tm }}>10-ep. moving avg.</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-5 h-3.5 rounded-sm" style={{ background: algo.color, opacity: 0.18 }} />
            <span className="text-[11px]" style={{ color: c.tm }}>Confidence band</span>
          </div>
        </div>

        {/* Chart */}
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 20 }}>
            <CartesianGrid stroke={c.chartGrid} strokeDasharray="3 4" />
            <XAxis
              dataKey="episode"
              tick={c.chartAxis}
              tickLine={c.chartTickLine}
              axisLine={c.chartAxisLine}
              label={{ value: 'Episode', position: 'insideBottom', offset: -12, fill: c.chartLabel, fontSize: 11 }}
              interval={data.length <= 50 ? 0 : Math.floor(data.length / 10)}
            />
            <YAxis
              tick={c.chartAxis}
              tickLine={c.chartTickLine}
              axisLine={c.chartAxisLine}
              tickFormatter={(v) => `${v}`}
              width={52}
              domain={[
                (min: number) => parseFloat((min - Math.abs(min) * 0.04).toFixed(3)),
                (max: number) => parseFloat((max + Math.abs(max) * 0.04).toFixed(3)),
              ]}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* ±1σ confidence band as reference area (stays above gridlines) */}
            {data.length > 0 && data[0].lo !== undefined && data[0].hi !== undefined && (
              <ReferenceArea
                y1={data[0].lo} y2={data[0].hi}
                fill={algo.color} fillOpacity={0.13}
                stroke={algo.color} strokeOpacity={0.18} strokeWidth={1}
                ifOverflow="hidden"
              />
            )}

            {/* Raw per-episode line — show dots for small datasets */}
            <Line type="monotone" dataKey="value" stroke={algo.color} strokeWidth={1.5}
              dot={data.length <= 50 ? { fill: algo.color, r: 2.5, strokeWidth: 0 } : false}
              activeDot={{ r: 3.5, fill: algo.color }} strokeOpacity={0.85} isAnimationActive={false} />

            {/* Moving average overlay */}
            <Line type="monotone" dataKey="ma" stroke={c.ts} strokeWidth={2}
              dot={false} activeDot={false} strokeDasharray="5 3" isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>

        {/* Summary stats row */}
        <div className="grid grid-cols-4 gap-3 pt-3" style={{ borderTop: `1px solid ${c.divider}` }}>
          {[
            { label: 'First ep.', value: data[0]?.value.toFixed(2) },
            { label: 'Last ep.',  value: data[data.length - 1]?.value.toFixed(2) },
            { label: 'Min',       value: Math.min(...data.map(d => d.value)).toFixed(2) },
            { label: 'Max',       value: Math.max(...data.map(d => d.value)).toFixed(2) },
          ].map(({ label, value }) => (
            <div key={label} className="text-center">
              <div className="text-[10px] uppercase tracking-wider mb-0.5" style={{ color: c.tu }}>{label}</div>
              <div className="text-[15px] font-bold tabular-nums" style={{ color: algo.color }}>{value}</div>
              <div className="text-[10px]" style={{ color: c.tu }}>{meta.unit}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ─── Analytics Charts ─────────────────────────────────────────────────────────

// Legacy constants — kept for reference; chart components now use useDashColors()
const CHART_GRID = 'rgba(255,255,255,0.05)'
const CHART_AXIS  = { fill: 'rgba(255,255,255,0.28)', fontSize: 10 }
const CHART_AXIS_LINE = { stroke: 'rgba(255,255,255,0.08)' }
const CHART_TICK_LINE = { stroke: 'rgba(255,255,255,0.08)' }

// Shared tooltip shell
function ChartTooltip({ active, payload, label, xLabel, rows }: {
  active?: boolean; payload?: any[]; label?: any
  xLabel: string
  rows: { key: string; label: string; color: string; unit: string }[]
}) {
  const c = useDashColors()
  if (!active || !payload?.length) return null
  const byKey = Object.fromEntries(payload.map((p: any) => [p.dataKey, p.value]))
  return (
    <div style={{ background: c.tooltipBg, border: `1px solid ${c.tooltipBorder}`, borderRadius: 10, padding: '8px 12px', fontSize: 11, boxShadow: c.tooltipShadow }}>
      <p style={{ color: c.tm, marginBottom: 4 }}>{xLabel} {label}</p>
      {rows.map(({ key, label: rl, color, unit }) =>
        byKey[key] !== undefined
          ? <p key={key} style={{ color }}>{rl}: <b>{Number(byKey[key]).toFixed(2)}</b> {unit}</p>
          : null
      )}
    </div>
  )
}

// ─── CPU Detail Modal ─────────────────────────────────────────────────────────

const CpuDetailModal = ({ algo, evalCpuMeans, onClose }: {
  algo: AlgoData
  evalCpuMeans: number[] | null
  onClose: () => void
}) => {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [onClose])
  const isReal = evalCpuMeans && evalCpuMeans.length > 0
  const W = 5

  // Real: 30-episode per-episode mean CPU% from eval runs
  // Fallback: synthetic series based on scalar mean/peak
  const data: { episode: number; cpu: number; ma: number }[] = isReal
    ? evalCpuMeans.map((cpu, i) => {
        const slice = evalCpuMeans.slice(Math.max(0, i - W + 1), i + 1)
        const ma = slice.reduce((a, b) => a + b, 0) / slice.length
        return { episode: i + 1, cpu: +cpu.toFixed(1), ma: +ma.toFixed(1) }
      })
    : (() => {
        const steps = 30
        let r = algo.id === 'civiq' ? 15 : 18
        const rand = () => { r = (r * 1664525 + 1013904223) & 0xffffffff; return (r >>> 0) / 0xffffffff - 0.5 }
        const mean = algo.cpuMean, noiseAmp = mean * 0.05
        return Array.from({ length: steps }, (_, i) => {
          const cpu = Math.max(0, mean + rand() * noiseAmp)
          const slice = Array.from({ length: Math.min(i + 1, W) }, (__, j) => mean + rand() * noiseAmp * 0.5)
          const ma = slice.reduce((a, b) => a + b, 0) / slice.length
          return { episode: i + 1, cpu: +cpu.toFixed(1), ma: +ma.toFixed(1) }
        })
      })()

  const mean = data.reduce((s, d) => s + d.cpu, 0) / data.length
  const fmt = (v: number) => v.toFixed(1)

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: 'rgba(0,0,0,0.72)', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)' }}
      onClick={onClose}
    >
      <div
        className="relative w-full mx-6 rounded-2xl p-6 flex flex-col gap-4"
        style={{
          maxWidth: '790px',
          background: 'rgba(4,9,22,0.97)',
          border: '1px solid rgba(255,255,255,0.13)',
          boxShadow: '0 32px 80px rgba(0,0,0,0.85), inset 0 1px 0 rgba(255,255,255,0.08)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-[17px] font-bold leading-tight" style={{ color: 'rgba(255,255,255,0.92)' }}>
              CPU Utilization <span style={{ color: algo.color }}>(CPU)</span>
            </h3>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
            style={{ background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.12)' }}
          >
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.55)"
              strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M18 6L6 18M6 6l12 12"/>
            </svg>
          </button>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-5 flex-wrap">
          <div className="flex items-center gap-2">
            <div className="w-6 h-[2px] rounded" style={{ background: algo.color }} />
            <span className="text-[11px]" style={{ color: 'rgba(255,255,255,0.38)' }}>CPU %</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-[2px] rounded" style={{ background: 'rgba(255,255,255,0.65)', borderTop: '2px dashed rgba(255,255,255,0.65)' }} />
            <span className="text-[11px]" style={{ color: 'rgba(255,255,255,0.38)' }}>5-ep. moving avg.</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-[1px]" style={{ borderTop: `1px dashed ${algo.color}`, opacity: 0.5 }} />
            <span className="text-[11px]" style={{ color: 'rgba(255,255,255,0.38)' }}>Mean</span>
          </div>
        </div>

        {/* Chart */}
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 20 }}>
            <CartesianGrid stroke="rgba(255,255,255,0.05)" strokeDasharray="3 4" />
            <XAxis
              dataKey="episode"
              tick={{ fill: 'rgba(255,255,255,0.28)', fontSize: 10 }}
              tickLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
              label={{ value: 'Episode', position: 'insideBottom', offset: -12, fill: 'rgba(255,255,255,0.28)', fontSize: 11 }}
              interval={data.length <= 30 ? 2 : Math.floor(data.length / 10)}
            />
            <YAxis
              tick={{ fill: 'rgba(255,255,255,0.28)', fontSize: 10 }}
              tickLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
              tickFormatter={(v) => `${v}%`}
              width={58}
              label={{ value: 'CPU %', angle: -90, position: 'insideLeft', offset: 14, fill: 'rgba(255,255,255,0.28)', fontSize: 11 }}
            />
            <Tooltip content={({ active, payload, label }: any) => {
              if (!active || !payload?.length) return null
              const byKey = Object.fromEntries(payload.map((p: any) => [p.dataKey, p.value]))
              return (
                <div style={{ background: 'rgba(4,9,22,0.97)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 10, padding: '8px 12px', fontSize: 11 }}>
                  <p style={{ color: 'rgba(255,255,255,0.4)', marginBottom: 4 }}>Episode {label}</p>
                  {byKey.cpu !== undefined && <p style={{ color: algo.color }}>CPU: <b>{Number(byKey.cpu).toFixed(1)}%</b> ({(Number(byKey.cpu) / 100).toFixed(2)} cores)</p>}
                  {byKey.ma  !== undefined && <p style={{ color: 'rgba(255,255,255,0.7)' }}>MA-5: <b>{Number(byKey.ma).toFixed(1)}%</b></p>}
                </div>
              )
            }} />
            <ReferenceLine
              y={mean}
              stroke={algo.color} strokeDasharray="4 2" strokeOpacity={0.45}
              label={{ value: `Mean: ${fmt(mean)}%`, position: 'insideTopRight', fill: algo.color, fontSize: 10 }}
            />
            <Line type="monotone" dataKey="cpu" stroke={algo.color} strokeWidth={1.5}
              dot={{ fill: algo.color, r: 2.5, strokeWidth: 0 }}
              activeDot={{ r: 3.5, fill: algo.color }} strokeOpacity={0.85} isAnimationActive={false} />
            <Line type="monotone" dataKey="ma" stroke="rgba(255,255,255,0.72)" strokeWidth={2}
              dot={false} activeDot={false} strokeDasharray="5 3" isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>

        {/* Summary stats */}
        <div className="grid grid-cols-4 gap-3 pt-3" style={{ borderTop: '1px solid rgba(255,255,255,0.07)' }}>
          {[
            { label: 'Mean CPU',   value: `${fmt(mean)}%`,                    sub: `${(mean / 100).toFixed(2)} cores` },
            { label: 'Min CPU',    value: `${fmt(Math.min(...data.map(d => d.cpu)))}%`, sub: `${(Math.min(...data.map(d => d.cpu)) / 100).toFixed(2)} cores` },
            { label: 'Max CPU',    value: `${fmt(Math.max(...data.map(d => d.cpu)))}%`, sub: `${(Math.max(...data.map(d => d.cpu)) / 100).toFixed(2)} cores` },
            { label: 'Episodes',   value: `${data.length}`,                   sub: 'eval runs' },
          ].map(({ label, value, sub }) => (
            <div key={label} className="text-center">
              <div className="text-[10px] uppercase tracking-wider mb-0.5" style={{ color: 'rgba(255,255,255,0.28)' }}>{label}</div>
              <div className="text-[15px] font-bold tabular-nums" style={{ color: algo.color }}>{value}</div>
              <div className="text-[10px]" style={{ color: 'rgba(255,255,255,0.28)' }}>{sub}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ─── CPU Utilization Card ─────────────────────────────────────────────────────

function CpuStatsCard({ algo, evalCpuMeans, onViewDetail }: {
  algo: AlgoData
  evalCpuMeans: number[] | null
  onViewDetail: () => void
}) {
  const c = useDashColors()
  const mean = algo.cpuMean
  const peak = (evalCpuMeans && evalCpuMeans.length > 0)
    ? Math.max(...evalCpuMeans)
    : mean * 1.3
  const maxVal = Math.max(peak, mean, 1)
  const peakBarColor = c.isDark ? 'rgba(255,255,255,0.4)' : 'rgba(55,65,81,0.4)'

  return (
    <GlassCard className="p-5 flex flex-col col-span-1">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2">
          <span className="text-[13px] font-bold" style={{ color: c.tp }}>CPU Utilization</span>
          <InfoBubble text="Per-episode CPU usage across evaluation runs. 100% = one fully utilised core — values above 100% indicate multi-core usage. Average is the mean across all episodes; peak is the highest per-episode mean observed." />
        </div>
        <button
          onClick={onViewDetail}
          className="flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold transition-all flex-shrink-0"
          style={{ background: c.badgeBg, border: `1px solid ${c.badgeBorder}`, color: c.ts }}
          onMouseEnter={(e) => {
            ;(e.currentTarget as HTMLElement).style.background = algo.colorDim
            ;(e.currentTarget as HTMLElement).style.borderColor = algo.border
            ;(e.currentTarget as HTMLElement).style.color = algo.color
          }}
          onMouseLeave={(e) => {
            ;(e.currentTarget as HTMLElement).style.background = c.badgeBg
            ;(e.currentTarget as HTMLElement).style.borderColor = c.badgeBorder
            ;(e.currentTarget as HTMLElement).style.color = c.ts
          }}
        >
          View detail ↗
        </button>
      </div>

      {/* Two-column stat layout — flex-1 + justify-center fills remaining card height */}
      <div className="flex-1 flex items-center">
      <div className="w-full flex items-center gap-0" style={{ borderTop: `1px solid ${c.divider}`, borderBottom: `1px solid ${c.divider}`, paddingTop: '0.75rem', paddingBottom: '0.75rem' }}>

        {/* Average column */}
        <div className="flex-1 flex flex-col items-center gap-1 min-w-0">
          <span className="text-[10px] uppercase tracking-wider" style={{ color: c.tu }}>Average</span>
          <div className="text-[24px] font-bold tabular-nums leading-none" style={{ color: algo.color }}>
            {mean.toFixed(1)}<span className="text-[13px] font-normal ml-0.5" style={{ color: c.tu }}>%</span>
          </div>
          <div className="text-[10px]" style={{ color: c.ts }}>≈ {(mean / 100).toFixed(2)} cores</div>
          <div className="w-full mt-1.5">
            <div className="w-full h-1.5 rounded-full" style={{ background: c.barTrack }}>
              <div className="h-full rounded-full" style={{ width: `${Math.min(100, (mean / maxVal) * 100).toFixed(1)}%`, background: algo.color, opacity: 0.7 }} />
            </div>
          </div>
        </div>

        {/* Divider */}
        <div className="w-px mx-3 self-stretch" style={{ background: c.divider }} />

        {/* Peak column */}
        <div className="flex-1 flex flex-col items-center gap-1 min-w-0">
          <span className="text-[10px] uppercase tracking-wider" style={{ color: c.tu }}>Peak</span>
          <div className="text-[24px] font-bold tabular-nums leading-none" style={{ color: c.tm }}>
            {peak.toFixed(1)}<span className="text-[13px] font-normal ml-0.5" style={{ color: c.tu }}>%</span>
          </div>
          <div className="text-[10px]" style={{ color: c.ts }}>≈ {(peak / 100).toFixed(1)} cores</div>
          <div className="w-full mt-1.5">
            <div className="w-full h-1.5 rounded-full" style={{ background: c.barTrack }}>
              <div className="h-full rounded-full" style={{ width: `${Math.min(100, (peak / maxVal) * 100).toFixed(1)}%`, background: peakBarColor }} />
            </div>
          </div>
        </div>

      </div>
      </div>
    </GlassCard>
  )
}

// ─── Key Findings Card ────────────────────────────────────────────────────────

function FindingsCard({ algo, trafficScale }: { algo: AlgoData; trafficScale: string }) {
  const c = useDashColors()
  const finding = LOS_FINDINGS[algo.id]?.[trafficScale]
  if (!finding) return null
  return (
    <GlassCard className="p-5 flex flex-col gap-3">
      <h3 className="text-[13px] font-bold" style={{ color: c.tp }}>Key Findings</h3>
      <p className="text-[12px] font-medium leading-relaxed" style={{ color: c.ts }}>
        {finding.headline}
      </p>
      <div className="flex flex-col gap-2 pt-2" style={{ borderTop: `1px solid ${c.divider}` }}>
        {finding.points.map((point, i) => (
          <div key={i} className="flex items-start gap-2">
            <div className="w-1.5 h-1.5 rounded-full mt-[5px] flex-shrink-0" style={{ background: algo.color }} />
            <p className="text-[11px] leading-relaxed" style={{ color: c.ts }}>{point}</p>
          </div>
        ))}
      </div>
    </GlassCard>
  )
}

// ─── Evaluation Performance Distribution ──────────────────────────────────────

function evalBoxStats(values: number[]) {
  if (values.length < 4) return null
  const sorted = [...values].sort((a, b) => a - b)
  const n = sorted.length
  const at = (p: number) => {
    const i = p * (n - 1); const lo = Math.floor(i), hi = Math.ceil(i)
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo)
  }
  const q1 = at(0.25), median = at(0.5), q3 = at(0.75)
  const iqr = q3 - q1
  const wLo = sorted.find(v => v >= q1 - 1.5 * iqr) ?? sorted[0]
  const wHi = [...sorted].reverse().find(v => v <= q3 + 1.5 * iqr) ?? sorted[n - 1]
  const mean = sorted.reduce((a, b) => a + b, 0) / n
  return { values, q1, median, q3, wLo, wHi, mean }
}

function BoxStripPanel({ values, color, label, unit, fmt, info }: {
  values: number[]
  color: string
  label: string
  unit: string
  fmt: (v: number) => string
  info?: string
}) {
  const c = useDashColors()
  const stats = evalBoxStats(values)
  const [zoom, setZoom] = useState(1)
  const [hovered, setHovered] = useState<{ val: number; idx: number } | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    const el = svgRef.current
    if (!el) return
    const handler = (e: WheelEvent) => {
      e.preventDefault()
      setZoom(z => Math.min(4, Math.max(0.3, z * (e.deltaY < 0 ? 1.15 : 1 / 1.15))))
    }
    el.addEventListener('wheel', handler, { passive: false })
    return () => el.removeEventListener('wheel', handler)
  }, [])

  if (!stats) return null

  const W = 160, H = 152
  const mL = 42, mR = 8, mT = 6, mB = 18
  const plotH = H - mT - mB
  const cx = mL + (W - mL - mR) / 2
  const bHW = 15

  // Base Y domain (unzoomed)
  const range = stats.wHi - stats.wLo
  const pad = range * 0.15 || 0.5
  const yMin = stats.wLo - pad, yMax = stats.wHi + pad

  // Zoomed Y domain — zoom in narrows the visible range, zoom out widens it
  const yCenter = (yMin + yMax) / 2
  const halfRange = (yMax - yMin) / 2
  const zYMin = yCenter - halfRange / zoom
  const zYMax = yCenter + halfRange / zoom

  // Y scale maps data values to fixed screen positions based on zoomed domain
  const ys = (v: number) => mT + plotH - ((v - zYMin) / (zYMax - zYMin)) * plotH

  // Guidelines: fixed screen positions, values derived from zoomed domain
  const nTicks = 6
  const guidelines = Array.from({ length: nTicks + 1 }, (_, i) => ({
    y: mT + plotH * i / nTicks,                              // fixed screen Y
    val: zYMax - (zYMax - zYMin) * i / nTicks,               // value at that position
  }))

  const det = (i: number) => ((i * 2654435761) >>> 0) / 0xffffffff * 32 - 16
  const isZoomed = Math.abs(zoom - 1) > 0.01
  const clipId = `bsp-clip-${label.replace(/\s+/g, '')}`

  const gc = c.isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)'
  const ac = c.isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)'
  const tc = c.isDark ? 'rgba(255,255,255,0.45)' : 'rgba(0,0,0,0.45)'
  const boxFill = color + '20'

  return (
    <GlassCard className="p-4 flex flex-col gap-2 min-w-0">
      <div className="flex items-start gap-2">
        <div className="flex-1">
          <span className="text-[12px] font-bold" style={{ color: c.tp }}>{label}</span>
          <span className="text-[10px] ml-2" style={{ color: c.tu }}>{unit}</span>
        </div>
        <div className="flex items-center gap-1.5 flex-shrink-0">
          {isZoomed && (
            <button onClick={() => setZoom(1)}
              className="text-[9px] px-1.5 py-0.5 rounded"
              style={{ background: 'rgba(255,255,255,0.08)', color: c.tu, border: `1px solid ${c.divider}` }}>
              Reset
            </button>
          )}
          {info && <InfoBubble side="right" text={info} />}
        </div>
      </div>
      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} width="100%"
        style={{ cursor: zoom > 1 ? 'zoom-out' : 'zoom-in' }}>
        <defs>
          <clipPath id={clipId}>
            <rect x={mL} y={mT} width={W - mL - mR} height={plotH} />
          </clipPath>
        </defs>

        {/* Static guidelines — screen positions never move, only label values update */}
        {guidelines.map(({ y, val }, i) => (
          <g key={i}>
            <line x1={mL} x2={W - mR} y1={y} y2={y} stroke={gc} strokeWidth={0.6} />
            <line x1={mL - 3} x2={mL} y1={y} y2={y} stroke={ac} strokeWidth={0.6} />
            <text x={mL - 5} y={y} textAnchor="end" dominantBaseline="middle" fontSize={7} fill={tc}>
              {fmt(val)}
            </text>
          </g>
        ))}
        <line x1={mL} x2={mL} y1={mT} y2={mT + plotH} stroke={ac} strokeWidth={0.6} />

        {/* Zoomable plot layer — clipped to the plot area */}
        <g clipPath={`url(#${clipId})`}>
          {/* Whiskers */}
          <line x1={cx} x2={cx} y1={ys(stats.wHi)} y2={ys(stats.q3)} stroke={color} strokeWidth={1} strokeOpacity={0.45} />
          <line x1={cx} x2={cx} y1={ys(stats.q1)} y2={ys(stats.wLo)} stroke={color} strokeWidth={1} strokeOpacity={0.45} />
          <line x1={cx - 7} x2={cx + 7} y1={ys(stats.wHi)} y2={ys(stats.wHi)} stroke={color} strokeWidth={1} strokeOpacity={0.6} />
          <line x1={cx - 7} x2={cx + 7} y1={ys(stats.wLo)} y2={ys(stats.wLo)} stroke={color} strokeWidth={1} strokeOpacity={0.6} />

          {/* IQR box */}
          <rect x={cx - bHW} y={ys(stats.q3)} width={bHW * 2}
            height={Math.max(1, ys(stats.q1) - ys(stats.q3))}
            fill={boxFill} stroke={color} strokeWidth={1} strokeOpacity={0.7} rx={2} />

          {/* Median line */}
          <line x1={cx - bHW} x2={cx + bHW} y1={ys(stats.median)} y2={ys(stats.median)} stroke={color} strokeWidth={2} />

          {/* Individual episode dots */}
          {values.map((v, i) => (
            <circle key={i} cx={cx + det(i)} cy={ys(v)}
              r={hovered?.idx === i ? 3 : 1.8}
              fill={color} fillOpacity={hovered?.idx === i ? 0.9 : 0.4}
              style={{ cursor: 'crosshair' }}
              onMouseEnter={() => setHovered({ val: v, idx: i })}
              onMouseLeave={() => setHovered(null)}
            />
          ))}

          {/* Mean diamond */}
          <polygon
            points={`${cx},${ys(stats.mean) - 4} ${cx + 4},${ys(stats.mean)} ${cx},${ys(stats.mean) + 4} ${cx - 4},${ys(stats.mean)}`}
            fill={color} fillOpacity={0.95}
          />
        </g>

        {/* Hover tooltip — outside clip so it's never cut off */}
        {hovered !== null && (() => {
          const ty = ys(hovered.val)
          if (ty < mT - 2 || ty > mT + plotH + 2) return null
          const tx = cx + det(hovered.idx)
          const ttW = 58, ttH = 22
          const ttX = Math.min(W - mR - ttW, Math.max(mL, tx - ttW / 2))
          const showAbove = ty - mT > plotH * 0.45
          const ttY = showAbove ? ty - ttH - 5 : ty + 6
          return (
            <g style={{ pointerEvents: 'none' }}>
              <rect x={ttX} y={ttY} width={ttW} height={ttH} rx={3}
                fill={c.tooltipBg} stroke={c.tooltipBorder} strokeWidth={0.6}
                style={{ filter: 'drop-shadow(0 1px 3px rgba(0,0,0,0.3))' }} />
              <text x={ttX + ttW / 2} y={ttY + 8} textAnchor="middle"
                fontSize={6.5} fill={c.tu}>
                Ep. {hovered.idx + 1}
              </text>
              <text x={ttX + ttW / 2} y={ttY + 16} textAnchor="middle"
                fontSize={7.5} fill={c.tooltipColor} fontWeight="600">
                {fmt(hovered.val)} {unit}
              </text>
            </g>
          )
        })()}

        {/* Static episode count label */}
        <text x={(mL + W - mR) / 2} y={H - 3} textAnchor="middle" fontSize={7.5} fill={tc}>
          {values.length} eval episodes
        </text>
      </svg>

      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex items-center gap-1">
          <svg width="12" height="6"><line x1="0" y1="3" x2="12" y2="3" stroke={color} strokeWidth="2" /></svg>
          <span className="text-[9px]" style={{ color: c.tu }}>Median</span>
        </div>
        <div className="flex items-center gap-1">
          <svg width="8" height="8"><polygon points="4,0 8,4 4,8 0,4" fill={color} /></svg>
          <span className="text-[9px]" style={{ color: c.tu }}>Mean</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-[10px] h-[8px] rounded-sm" style={{ background: boxFill, border: `1px solid ${color}` }} />
          <span className="text-[9px]" style={{ color: c.tu }}>IQR (Q1–Q3)</span>
        </div>
        <div className="flex items-center gap-1">
          <svg width="6" height="6"><circle cx="3" cy="3" r="2" fill={color} fillOpacity={0.4} /></svg>
          <span className="text-[9px]" style={{ color: c.tu }}>Episodes</span>
        </div>
      </div>
    </GlassCard>
  )
}

function EvalDeviationCard({ algo }: { algo: AlgoData }) {
  const c = useDashColors()
  const tt = algo.episodes.travelTime.map(p => p.value / 60)
  const wt = algo.episodes.waitTime.map(p => p.value)
  const th = algo.episodes.throughput.map(p => p.value)
  if (!tt.length) return null
  return (
    <>
      <div className="flex items-center gap-3">
        <span className="text-[13px] font-bold" style={{ color: c.ts }}>Evaluation Performance Distribution</span>
        <InfoBubble text="Each chart shows how consistently the algorithm performed across all evaluation episodes. The box covers the middle 50% of results (IQR). The solid line is the median, the diamond is the mean — if they differ, the distribution is skewed. Whiskers extend to the furthest non-outlier value (1.5×IQR). Dots beyond the whiskers are outlier episodes. A narrow box with tightly clustered dots means reliable performance; a wide box signals high run-to-run variance." />
        <div className="flex-1 h-px" style={{ background: c.divider }} />
      </div>
      <div className="grid grid-cols-3 gap-4 min-w-0">
        <BoxStripPanel values={tt} color={algo.color} label="Travel Time" unit="min"
          fmt={v => v.toFixed(2)}
          info="Average time a vehicle spends travelling from its origin to its destination across all active vehicles in this evaluation episode. Lower is better. A wide spread here means the algorithm sometimes gets vehicles through quickly but struggles in other runs." />
        <BoxStripPanel values={wt} color={algo.color} label="Wait Time" unit="s"
          fmt={v => v.toFixed(1)}
          info="Average time vehicles spend stopped at intersections per evaluation episode. Lower is better. High variance here often indicates the algorithm hasn't learned a stable signal-timing policy and occasionally causes long queues." />
        <BoxStripPanel values={th} color={algo.color} label="Throughput" unit="veh/hr"
          fmt={v => Math.round(v).toString()}
          info="Number of vehicles that successfully completed their trips per hour in this evaluation episode. Higher is better. Low throughput episodes may indicate gridlock or poor routing decisions under the current traffic demand." />
      </div>
    </>
  )
}

// ─── Shared info bubble (matches KpiCard / GaugeChart tooltip style) ──────────

function InfoBubble({ text, side = 'left' }: { text: string; side?: 'left' | 'right' }) {
  const c = useDashColors()
  return (
    <div className="group/info relative z-20 flex-shrink-0 outline-none" tabIndex={0} role="button" aria-label="More information">
      <div className="w-[18px] h-[18px] rounded-full flex items-center justify-center cursor-default"
        style={{ background: 'rgba(6,182,212,0.2)', border: '1px solid rgba(6,182,212,0.65)', color: c.isDark ? 'rgba(217,249,255,0.95)' : 'rgba(3,105,161,0.95)', boxShadow: '0 0 0 1px rgba(0,0,0,0.2) inset' }}>
        <span className="text-[10px] font-bold leading-none" aria-hidden="true">!</span>
      </div>
      <div className="pointer-events-none absolute top-[calc(100%+8px)] w-[260px] rounded-lg px-3 py-2 text-[10px] leading-relaxed opacity-0 translate-y-1 transition-all duration-150 group-hover/info:opacity-100 group-hover/info:translate-y-0 group-focus/info:opacity-100 group-focus/info:translate-y-0"
        style={{
          ...(side === 'right' ? { right: 0 } : { left: 0 }),
          background: c.tooltipBg, border: `1px solid ${c.tooltipBorder}`,
          color: c.tooltipColor, boxShadow: c.tooltipShadow, zIndex: 90,
        }}>
        {text}
      </div>
    </div>
  )
}



// ─── Algorithm Detail Page ─────────────────────────────────────────────────────


// ─── Export Metrics Button ────────────────────────────────────────────────────

function downloadFile(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

function ExportButton({ algo, selfishTimeseries = null }: {
  algo: AlgoData
  selfishTimeseries?: SelfishTimeseries | null
}) {
  const c = useDashColors()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const isSelfish = algo.id === 'selfish'

  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  const close = () => setOpen(false)

  // ── CSV: KPI summary (one row per algorithm) ──
  const exportKpiCsv = () => {
    const headers = [
      'Algorithm', 'Label',
      'Travel Time (min)', 'Wait Time (sec)', 'Throughput (veh/hr)',
      'Real-time Factor (x)', 'CO2 (g/km)', 'Fuel (L/100km)',
      'Best Episode', 'Cumulative Reward',
    ]
    const row = [
      algo.id, algo.label,
      algo.travelTime, algo.waitTime, algo.throughput,
      algo.speed, algo.co2, algo.fuel,
      algo.convergence ?? 'N/A', algo.reward ?? 'N/A',
    ]
    downloadFile(
      `civiq_${algo.id}_kpi.csv`,
      [headers.join(','), row.join(',')].join('\n'),
      'text/csv;charset=utf-8;',
    )
    close()
  }

  // ── CSV: per-episode KPI series (learning-based algos only) ──
  const exportEpisodesCsv = () => {
    const headers = [
      'Episode',
      'Travel Time (min)', 'Travel Time MA',
      'Wait Time (sec)',   'Wait Time MA',
      'Throughput (veh/hr)', 'Throughput MA',
    ]
    const n = algo.episodes.travelTime.length
    const rows = Array.from({ length: n }, (_, i) => [
      algo.episodes.travelTime[i].episode,
      algo.episodes.travelTime[i].value, algo.episodes.travelTime[i].ma,
      algo.episodes.waitTime[i]?.value ?? '', algo.episodes.waitTime[i]?.ma ?? '',
      algo.episodes.throughput[i]?.value ?? '', algo.episodes.throughput[i]?.ma ?? '',
    ])
    downloadFile(
      `civiq_${algo.id}_episodes.csv`,
      [headers.join(','), ...rows.map(r => r.join(','))].join('\n'),
      'text/csv;charset=utf-8;',
    )
    close()
  }

  // ── CSV: training curve (learning-based only) ──
  const exportTrainingCsv = () => {
    const headers = ['Episode', 'Reward', 'Reward MA', 'Reward Lo (CI)', 'Reward Hi (CI)']
    const rows = algo.system.training.map(p => [p.episode, p.reward, p.ma, p.lo, p.hi])
    downloadFile(
      `civiq_${algo.id}_training.csv`,
      [headers.join(','), ...rows.map(r => r.join(','))].join('\n'),
      'text/csv;charset=utf-8;',
    )
    close()
  }

  // ── CSV: real SUMO step timeseries (selfish routing) ──
  const exportSelfishTimeseriesCsv = () => {
    if (!selfishTimeseries) return
    const headers = ['Step (s)', 'Active Vehicles', 'Total System Wait (s)']
    const rows = selfishTimeseries.steps.map((step, i) => [
      step,
      selfishTimeseries.activeVehicles[i],
      selfishTimeseries.totalSystemWait[i],
    ])
    downloadFile(
      `civiq_selfish_timeseries.csv`,
      [headers.join(','), ...rows.map(r => r.join(','))].join('\n'),
      'text/csv;charset=utf-8;',
    )
    close()
  }

  // ── JSON: full dump ──
  const exportJson = () => {
    const payload: Record<string, unknown> = {
      exportedAt: new Date().toISOString(),
      algorithm: { id: algo.id, label: algo.label, sublabel: algo.sublabel, rank: algo.rank },
      kpi: {
        travelTime: algo.travelTime, waitTime: algo.waitTime, throughput: algo.throughput,
        speed: algo.speed, co2: algo.co2, fuel: algo.fuel,
        bestEpisode: algo.convergence, cumulativeReward: algo.reward,
      },
      changes: algo.changes,
    }
    if (isSelfish) {
      // Selfish: export real timeseries instead of synthetic episode/traffic data
      payload.timeseries = selfishTimeseries ?? null
      payload.note = 'Selfish Routing is a single-run simulation. No training episodes or MARL diagnostics.'
    } else {
      payload.episodes = algo.episodes
      payload.system = algo.system
      payload.marl = algo.marl
    }
    downloadFile(
      `civiq_${algo.id}_full.json`,
      JSON.stringify(payload, null, 2),
      'application/json',
    )
    close()
  }

  type ExportItem = { label: string; sub: string; tag: string; action: () => void }
  const ITEMS: ExportItem[] = [
    { label: 'KPI Summary', sub: 'Key performance metrics', tag: 'CSV', action: exportKpiCsv },
    // Selfish routing: swap episode series for real SUMO step timeseries
    ...(isSelfish
      ? selfishTimeseries
        ? [{ label: 'Simulation Timeseries', sub: 'Step-level vehicles & wait (real SUMO)', tag: 'CSV', action: exportSelfishTimeseriesCsv }]
        : []
      : [{ label: 'Episode Series', sub: 'Per-episode KPI trend data', tag: 'CSV', action: exportEpisodesCsv }]
    ),
    ...(!isSelfish && algo.system.training.length
      ? [{ label: 'Training Curve', sub: 'Reward per training episode', tag: 'CSV', action: exportTrainingCsv }]
      : []),
    { label: 'Full Export', sub: 'All metrics and time series', tag: 'JSON', action: exportJson },
  ]

  return (
    <div ref={ref} className="relative flex-shrink-0">
      {/* Trigger */}
      <button
        onClick={() => setOpen(v => !v)}
        className="flex items-center gap-2 px-3.5 py-1.5 rounded-xl text-[11px] font-semibold transition-all duration-150"
        style={{
          background: open ? c.itemBgMd : c.itemBg,
          border: `1px solid ${open ? c.glassBorder : c.badgeBorder}`,
          color: open ? c.tp : c.ts,
        }}
        onMouseEnter={e => { if (!open) { (e.currentTarget as HTMLElement).style.background = c.itemBgMd; (e.currentTarget as HTMLElement).style.color = c.tp } }}
        onMouseLeave={e => { if (!open) { (e.currentTarget as HTMLElement).style.background = c.itemBg; (e.currentTarget as HTMLElement).style.color = c.ts } }}
      >
        {/* Download icon */}
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
        Export Metrics
        <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
          style={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.15s ease' }}>
          <polyline points="6 9 12 15 18 9"/>
        </svg>
      </button>

      {/* Dropdown */}
      {open && (
        <div
          className="absolute right-0 top-[calc(100%+6px)] rounded-xl overflow-hidden"
          style={{
            width: '230px', zIndex: 60,
            background: 'rgba(6,10,26,0.98)',
            border: '1px solid rgba(255,255,255,0.12)',
            boxShadow: '0 20px 56px rgba(0,0,0,0.65), inset 0 1px 0 rgba(255,255,255,0.07)',
          }}
        >
          <div className="px-3.5 py-2.5" style={{ borderBottom: '1px solid rgba(255,255,255,0.07)' }}>
            <p className="text-[9px] font-bold uppercase tracking-widest" style={{ color: 'rgba(255,255,255,0.3)' }}>
              {algo.label}
            </p>
          </div>
          {ITEMS.map(({ label, sub, tag, action }) => (
            <button
              key={label}
              onClick={action}
              className="w-full flex items-center gap-3 px-3.5 py-2.5 text-left transition-colors duration-100"
              style={{ color: 'rgba(255,255,255,0.78)', borderBottom: '1px solid rgba(255,255,255,0.04)' }}
              onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.07)' }}
              onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = 'transparent' }}
            >
              {/* Format badge */}
              <span className="text-[8px] font-black tabular-nums w-9 text-center py-0.5 rounded flex-shrink-0"
                style={{
                  background: tag === 'CSV' ? 'rgba(52,211,153,0.15)' : 'rgba(99,102,241,0.18)',
                  color: tag === 'CSV' ? '#34D399' : '#818CF8',
                  border: `1px solid ${tag === 'CSV' ? 'rgba(52,211,153,0.3)' : 'rgba(99,102,241,0.3)'}`,
                }}>
                {tag}
              </span>
              <div className="min-w-0">
                <div className="text-[11px] font-semibold leading-tight">{label}</div>
                <div className="text-[9px] mt-0.5 leading-tight" style={{ color: 'rgba(255,255,255,0.35)' }}>{sub}</div>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Selfish routing real-data overlay ────────────────────────────────────────

interface SelfishApiKpis {
  travelTime: number
  waitTime: number
  throughput: number
  speed: number
  co2: number
  fuel: number
  avgSpeedKmh: number
  returnMean: number | null
}

interface SelfishCiviqBaseline {
  travelTime_s: number
  waitTime_s: number
  throughput: number
}

interface SelfishTimeseries {
  steps: number[]
  activeVehicles: number[]
  totalSystemWait: number[]
}

interface SelfishEvalArrays {
  evalTravelTimes:  number[]
  evalWaitingTimes: number[]
  evalThroughputs:  number[]
  evalReturns:      number[]
}

function useSelfishRealData(enabled: boolean, trafficLevel: string, mapSize: string) {
  const [kpis, setKpis] = useState<SelfishApiKpis | null>(null)
  const [timeseries, setTimeseries] = useState<SelfishTimeseries | null>(null)
  const [civiq, setCiviq] = useState<SelfishCiviqBaseline | null>(null)
  const [evalArrays, setEvalArrays] = useState<SelfishEvalArrays | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!enabled || !trafficLevel || !mapSize) return
    setKpis(null)
    setTimeseries(null)
    setCiviq(null)
    setEvalArrays(null)
    setLoading(true)
    fetch(`/api/selfish?trafficLevel=${trafficLevel}&map=${mapSize}`)
      .then(r => r.json())
      .then(data => {
        if (data.success) {
          setKpis(data.kpis)
          setTimeseries(data.timeseries)
          setCiviq(data.baselines?.civiq ?? null)
          if (data.evalTravelTimes && data.evalWaitingTimes && data.evalThroughputs && data.evalReturns) {
            setEvalArrays({
              evalTravelTimes:  data.evalTravelTimes,
              evalWaitingTimes: data.evalWaitingTimes,
              evalThroughputs:  data.evalThroughputs,
              evalReturns:      data.evalReturns,
            })
          }
        }
        // On error (e.g. 4x4 map, missing forced_flow for bgc_core) silently fall back to static data
      })
      .catch(() => {/* silently fall back to static data */})
      .finally(() => setLoading(false))
  }, [enabled, trafficLevel, mapSize])

  return { kpis, timeseries, civiq, evalArrays, loading }
}

// ─── Selfish All-Levels hook ───────────────────────────────────────────────────

interface SelfishLevelKpis {
  travelTime: number  // min
  waitTime: number    // sec
  throughput: number  // veh/hr
  co2: number         // g/km
  fuel: number        // L/100km
}

function useSelfishAllLevels(enabled: boolean, mapSize: string) {
  const [levels, setLevels] = useState<Record<string, SelfishLevelKpis> | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!enabled) return
    setLoading(true)
    Promise.all([
      fetch(`/api/selfish?trafficLevel=free_flow&map=${mapSize}`).then(r => r.json()),
      fetch(`/api/selfish?trafficLevel=stable_flow&map=${mapSize}`).then(r => r.json()),
      fetch(`/api/selfish?trafficLevel=forced_flow&map=${mapSize}`).then(r => r.json()),
    ])
      .then(([a, c, e]) => {
        if (a.success && c.success && e.success) {
          setLevels({
            free_flow:   { travelTime: a.kpis.travelTime, waitTime: a.kpis.waitTime, throughput: a.kpis.throughput, co2: a.kpis.co2, fuel: a.kpis.fuel },
            stable_flow: { travelTime: c.kpis.travelTime, waitTime: c.kpis.waitTime, throughput: c.kpis.throughput, co2: c.kpis.co2, fuel: c.kpis.fuel },
            forced_flow: { travelTime: e.kpis.travelTime, waitTime: e.kpis.waitTime, throughput: e.kpis.throughput, co2: e.kpis.co2, fuel: e.kpis.fuel },
          })
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [enabled, mapSize])

  return { levels, loading }
}

// ─── Monolithic QMIX real-data overlay ────────────────────────────────────────

interface QmixTrainingEntry {
  t: number
  episode: number
  reward: number
  ma: number
  loss: number | null
  gradNorm: number | null
  qTaken: number | null
  targetMean: number | null
}

interface QmixBaselineKpis {
  travelTime_s: number
  waitTime_s: number
  throughput: number
  co2_g_per_km: number
  fuel_l_per_100km: number
  returnMean: number
}


interface QmixRealData {
  kpis: {
    travelTime_s: number
    travelTime_min: number
    travelTime_std: number
    waitTime_s: number
    waitTime_std: number
    throughput: number
    throughput_std: number
    co2: number
    fuel: number
    cpuMean: number
    cpuPeak: number
    returnMean: number
    realTimeFactor: number | null
    arrivalRate: number
  }
  baselines?: { noop?: QmixBaselineKpis; greedy_shortest?: QmixBaselineKpis }
  evalReturns: number[]
  evalTravelTimes: number[]
  evalWaitingTimes: number[]
  evalThroughputs: number[]
  evalCpuMeans: number[] | null
  training: { note: string; curve: QmixTrainingEntry[] }
  trainMetrics?: null
  testCurve: { t: number; episode: number; returnMean: number | null }[]
}

// Maps frontend trafficScale → API scenario param
const TRAFFIC_TO_QMIX_SCENARIO: Record<string, string> = {
  free_flow:   'los_a',
  stable_flow: 'los_c',
  forced_flow: 'los_e',
}

function useQmixRealData(enabled: boolean, trafficScale: string) {
  const [data, setData] = useState<QmixRealData | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const scenario = TRAFFIC_TO_QMIX_SCENARIO[trafficScale]
    if (!enabled || !scenario) { setData(null); return }
    setData(null)
    setLoading(true)
    fetch(`/api/qmix?scenario=${scenario}`)
      .then(r => r.json())
      .then(d => { if (d.success) setData(d) })
      .catch(() => {/* fall back to static placeholder */})
      .finally(() => setLoading(false))
  }, [enabled, trafficScale])

  return { data, loading }
}

function useCiviqRealData(enabled: boolean, trafficScale: string) {
  const [data, setData] = useState<QmixRealData | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const scenario = TRAFFIC_TO_QMIX_SCENARIO[trafficScale]
    if (!enabled || !scenario) { setData(null); return }
    setData(null)
    setLoading(true)
    fetch(`/api/civiq?scenario=${scenario}`)
      .then(r => r.json())
      .then(d => { if (d.success) setData(d) })
      .catch(() => {/* fall back to static placeholder */})
      .finally(() => setLoading(false))
  }, [enabled, trafficScale])

  return { data, loading }
}

/** Convert training.curve → TrainingPoint[] with moving-average confidence bands.
 *  MA and std are computed over the full curve; output is downsampled to MAX_PTS. */
function qmixToTrainingPoints(curve: QmixTrainingEntry[]): TrainingPoint[] {
  if (!curve.length) return []
  const MAX_PTS = 400
  const W = 20
  const rewards = curve.map(p => p.reward)
  const ma = rewards.map((_, i) => {
    const win = rewards.slice(Math.max(0, i - W), i + 1)
    return win.reduce((a, b) => a + b, 0) / win.length
  })
  const full = curve.map((p, i) => {
    const win = rewards.slice(Math.max(0, i - W), i + 1)
    const std = win.length > 1
      ? Math.sqrt(win.reduce((s, v) => s + (v - ma[i]) ** 2, 0) / win.length)
      : 500
    return { episode: Math.round(p.t / 100), reward: p.reward, lo: p.reward - std, hi: p.reward + std, ma: ma[i] }
  })
  if (full.length <= MAX_PTS) return full
  const step = Math.ceil(full.length / MAX_PTS)
  return full.filter((_, i) => i % step === 0 || i === full.length - 1)
}

/** Convert training.curve → EpisodePoint[] for the "View detail" modal.
 *  Uses global mean±std as a fixed confidence band; downsampled to MAX_PTS. */
function trainingCurveToEpisodePoints(curve: QmixTrainingEntry[]): EpisodePoint[] {
  if (!curve.length) return []
  const MAX_PTS = 500
  const W = 20
  const rewards = curve.map(p => p.reward)
  const mean = rewards.reduce((a, b) => a + b, 0) / rewards.length
  const std  = Math.sqrt(rewards.reduce((s, x) => s + (x - mean) ** 2, 0) / rewards.length)
  const lo = parseFloat((mean - std).toFixed(1))
  const hi = parseFloat((mean + std).toFixed(1))
  const full = rewards.map((r, i) => {
    const win = rewards.slice(Math.max(0, i - W), i + 1)
    const ma  = parseFloat((win.reduce((a, b) => a + b, 0) / win.length).toFixed(1))
    return { episode: curve[i].episode, value: parseFloat(r.toFixed(1)), lo, hi, ma }
  })
  if (full.length <= MAX_PTS) return full
  const step = Math.ceil(full.length / MAX_PTS)
  return full.filter((_, i) => i % step === 0 || i === full.length - 1)
}

/** Downsample training curve rewards to n evenly-spaced values for sparkline. */
function trainingCurveToSparkline(curve: QmixTrainingEntry[], n = 10): number[] {
  if (!curve.length) return []
  if (curve.length <= n) return curve.map(p => p.reward)
  const step = (curve.length - 1) / (n - 1)
  return Array.from({ length: n }, (_, i) => curve[Math.round(i * step)].reward)
}

/** Downsample any numeric array to n evenly-spaced values for sparkline. */
function downsampleArray(arr: number[], n = 10): number[] {
  if (!arr.length) return []
  if (arr.length <= n) return arr
  const step = (arr.length - 1) / (n - 1)
  return Array.from({ length: n }, (_, i) => arr[Math.round(i * step)])
}

/** Convert training.curve → MarlPoint[] using real loss/gradNorm/qTaken values.
 *  Only points that have loss data are used; output downsampled to MAX_PTS. */
function qmixToMarlPoints(curve: QmixTrainingEntry[]): MarlPoint[] {
  if (!curve.length) return []
  const MAX_PTS = 300
  // Only use points that have loss data — text-log rows + tfevents rows with attached metrics
  const withMetrics = curve.filter(p => p.loss !== null && p.loss !== undefined)
  if (!withMetrics.length) return []
  const W = 10
  const rewards = withMetrics.map(p => p.reward)
  const ma = rewards.map((_, i) => {
    const win = rewards.slice(Math.max(0, i - W), i + 1)
    return win.reduce((a, b) => a + b, 0) / win.length
  })
  const full = withMetrics.map((p, i) => {
    const win = rewards.slice(Math.max(0, i - W), i + 1)
    const band = win.length > 1
      ? Math.sqrt(win.reduce((s, v) => s + (v - ma[i]) ** 2, 0) / win.length)
      : 500
    return {
      episode:    Math.round(p.t / 100),
      reward:     p.reward,
      rewardLo:   p.reward - band,
      rewardHi:   p.reward + band,
      rewardMa:   ma[i],
      tdLoss:     p.loss      ?? 0,
      gradNorm:   p.gradNorm  ?? 0,
      qTakenMean: p.qTaken    ?? 0,
      qTakenLo:   (p.qTaken ?? 0) - 1.07,
      qTakenHi:   (p.qTaken ?? 0) + 1.07,
      targetMean: p.targetMean ?? 0,
    }
  })
  if (full.length <= MAX_PTS) return full
  const step = Math.ceil(full.length / MAX_PTS)
  return full.filter((_, i) => i % step === 0 || i === full.length - 1)
}

/** Convert a raw eval array → EpisodePoint[] with moving average and global ±1σ band */
function evalRawToPoints(raw: number[], scale = 1): EpisodePoint[] {
  if (!raw.length) return []
  const W = 10
  const vals = raw.map(v => v * scale)
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length
  const globalStd = Math.sqrt(vals.reduce((s, x) => s + (x - mean) ** 2, 0) / vals.length)
  const lo = parseFloat((mean - globalStd).toFixed(3))
  const hi = parseFloat((mean + globalStd).toFixed(3))
  return vals.map((v, i) => {
    const win = vals.slice(Math.max(0, i - W), i + 1)
    const ma  = parseFloat((win.reduce((a, b) => a + b, 0) / win.length).toFixed(3))
    return { episode: i + 1, value: parseFloat(v.toFixed(3)), lo, hi, ma }
  })
}

/** Convert eval episode arrays → EpisodeSeries for detail modal */
function qmixToEpisodeSeries(data: QmixRealData): EpisodeSeries {
  return {
    travelTime: evalRawToPoints(data.evalTravelTimes),
    waitTime:   evalRawToPoints(data.evalWaitingTimes),
    throughput: evalRawToPoints(data.evalThroughputs),
    speed:      [],  // RTF has no per-episode variation; modal not shown
  }
}

// ─── Selfish Detail Modal ─────────────────────────────────────────────────────

const LOS_LABELS: Record<string, string> = {
  free_flow:   'Low Demand',
  stable_flow: 'Moderate Demand',
  forced_flow: 'High Demand',
}

const SELFISH_METRIC_META: Record<SelfishMetricKey, { label: string; unit: string; lowerBetter: boolean }> = {
  travelTime: { label: 'Avg. Travel Time',      unit: 'sec',     lowerBetter: true  },
  waitTime:   { label: 'Avg. Waiting Time',     unit: 'sec',     lowerBetter: true  },
  throughput: { label: 'Network Throughput',    unit: 'veh/hr',  lowerBetter: false },
  co2:        { label: 'Avg. CO₂ Emissions',    unit: 'g/km',    lowerBetter: true  },
  fuel:       { label: 'Avg. Fuel Consumption (Diesel)', unit: 'L/100km', lowerBetter: true  },
}

const SELFISH_CO2_EPISODES  = makeSeries(486.26, 486.26, 15,  10,  50, 13)
const SELFISH_FUEL_EPISODES = makeSeries(20.914, 20.914, 0.8, 0.5, 50, 14)

const SelfishDetailModal = ({ metricKey, algoColor, data, onClose }: {
  metricKey: SelfishMetricKey
  algoColor: string
  data: EpisodePoint[]
  onClose: () => void
}) => {
  const c = useDashColors()
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [onClose])
  const meta = SELFISH_METRIC_META[metricKey]

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null
    const byKey = Object.fromEntries(payload.map((p: any) => [p.dataKey, p.value]))
    return (
      <div style={{ background: c.tooltipBg, border: `1px solid ${c.tooltipBorder}`, borderRadius: 10, padding: '8px 12px', fontSize: 11, boxShadow: c.tooltipShadow }}>
        <p style={{ color: c.tm, marginBottom: 4 }}>Episode {label}</p>
        {byKey.value !== undefined && <p style={{ color: algoColor }}>Value: <b>{byKey.value.toFixed(2)}</b> {meta.unit}</p>}
        {byKey.ma    !== undefined && <p style={{ color: c.ts }}>MA-10: <b>{byKey.ma.toFixed(2)}</b> {meta.unit}</p>}
        {byKey.hi    !== undefined && byKey.lo !== undefined && (
          <p style={{ color: c.tm }}>Band: {byKey.lo.toFixed(2)} – {byKey.hi.toFixed(2)}</p>
        )}
      </div>
    )
  }

  const step = Math.max(1, Math.floor(data.length / 200))
  const sampled = data.filter((_, i) => i % step === 0)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center"
      style={{
        background: c.isDark ? 'rgba(0,0,0,0.72)' : 'rgba(15,23,42,0.35)',
        backdropFilter: 'blur(8px)',
        WebkitBackdropFilter: 'blur(8px)',
      }}
      onClick={onClose}>
      <div className="relative w-full mx-6 rounded-2xl p-6 flex flex-col gap-4"
        style={{
          maxWidth: '780px',
          background: c.isDark ? 'rgba(4,9,22,0.97)' : 'rgba(255,255,255,0.96)',
          border: `1px solid ${c.glassBorder}`,
          boxShadow: c.isDark
            ? '0 32px 80px rgba(0,0,0,0.85), inset 0 1px 0 rgba(255,255,255,0.08)'
            : '0 24px 64px rgba(15,23,42,0.24), inset 0 1px 0 rgba(255,255,255,0.95)',
        }}
        onClick={e => e.stopPropagation()}>

        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-[17px] font-bold leading-tight" style={{ color: c.tp }}>
              {meta.label} <span style={{ color: algoColor }}>· Selfish Routing</span>
            </h3>
            <p className="text-[12px] mt-0.5" style={{ color: c.tm }}>
              Per-episode trend · Selfish Routing · {data.length.toLocaleString()} episodes
            </p>
          </div>
          <button onClick={onClose}
            aria-label="Close"
            className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
            style={{ background: c.badgeBg, border: `1px solid ${c.badgeBorder}` }}>
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke={c.tm} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M18 6L6 18M6 6l12 12"/>
            </svg>
          </button>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-5 flex-wrap">
          <div className="flex items-center gap-2">
            <div className="w-6 h-[2px] rounded" style={{ background: algoColor }} />
            <span className="text-[11px]" style={{ color: c.tm }}>Per-episode value</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-[2px] rounded" style={{ background: c.ts, borderTop: `2px dashed ${c.ts}` }} />
            <span className="text-[11px]" style={{ color: c.tm }}>10-ep. moving avg.</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-5 h-3.5 rounded-sm" style={{ background: algoColor, opacity: 0.18 }} />
            <span className="text-[11px]" style={{ color: c.tm }}>Confidence band</span>
          </div>
        </div>

        {/* Chart */}
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={sampled} margin={{ top: 8, right: 12, left: 0, bottom: 20 }}>
            <CartesianGrid stroke={c.chartGrid} strokeDasharray="3 4" />
            <XAxis dataKey="episode"
              tick={c.chartAxis}
              tickLine={c.chartTickLine}
              axisLine={c.chartAxisLine}
              label={{ value: 'Episode', position: 'insideBottom', offset: -12, fill: c.chartLabel, fontSize: 11 }}
              interval={Math.floor(sampled.length / 10)}
            />
            <YAxis tick={c.chartAxis}
              tickLine={c.chartTickLine}
              axisLine={c.chartAxisLine}
              tickFormatter={v => `${v}`} width={52} />
            <Tooltip content={<CustomTooltip />} />
            <Area type="monotone" dataKey="hi" stroke="none"
              fill={algoColor} fillOpacity={0.12} legendType="none" isAnimationActive={false} />
            <Area type="monotone" dataKey="lo" stroke="none"
              fill={c.isDark ? '#040916' : '#eaf1fb'} fillOpacity={1} legendType="none" isAnimationActive={false} />
            <Line type="monotone" dataKey="value" stroke={algoColor} strokeWidth={1.5}
              dot={false} activeDot={{ r: 3 }} isAnimationActive={false} />
            <Line type="monotone" dataKey="ma" stroke={c.ts} strokeWidth={1.5}
              dot={false} activeDot={false} strokeDasharray="5 3" isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>

        {/* Summary stats */}
        <div className="grid grid-cols-4 gap-3 pt-3" style={{ borderTop: `1px solid ${c.divider}` }}>
          {[
            { label: 'First ep.', value: data[0]?.value.toFixed(2) },
            { label: 'Last ep.',  value: data[data.length - 1]?.value.toFixed(2) },
            { label: 'Min',       value: Math.min(...sampled.map(d => d.value)).toFixed(2) },
            { label: 'Max',       value: Math.max(...sampled.map(d => d.value)).toFixed(2) },
          ].map(({ label, value }) => (
            <div key={label} className="text-center">
              <div className="text-[10px] uppercase tracking-wider mb-0.5" style={{ color: c.tu }}>{label}</div>
              <div className="text-[15px] font-bold tabular-nums" style={{ color: algoColor }}>{value}</div>
              <div className="text-[10px]" style={{ color: c.tu }}>{meta.unit}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ─── Algorithm Detail Page ────────────────────────────────────────────────────

const AlgoDetailPage = ({ algo, mapSize, trafficScale }: {
  algo: AlgoData; mapSize: string; trafficScale: string
}) => {
  const c = useDashColors()
  const { reducedMotion, ready } = usePageEntrance()
  const [openModal, setOpenModal] = useState<EpisodeMetricKey | null>(null)
  const [congestionDetail, setCongestionDetail] = useState(false)
  const [selfishModal, setSelfishModal] = useState<SelfishMetricKey | null>(null)
  const [openCpuModal, setOpenCpuModal] = useState(false)

  // For selfish routing, fetch real simulation data and overlay onto static algo object
  const isSelfish = algo.id === 'selfish'
  const { kpis: realKpis, timeseries: realTimeseries, civiq: selfishCiviq, evalArrays: selfishEvalArrays, loading: kpisLoading } = useSelfishRealData(isSelfish, trafficScale, mapSize)
  const { levels: selfishAllLevels } = useSelfishAllLevels(isSelfish, mapSize)

  // For monolithic QMIX, fetch real training curve and eval data (scenario selected by trafficScale)
  const isQmix = algo.id === 'qmix'
  const { data: qmixData, loading: qmixLoading } = useQmixRealData(isQmix, trafficScale)

  // For CiViQ, fetch real eval data (same schema as QMIX)
  const isCiviq = algo.id === 'civiq'
  const { data: civiqData, loading: civiqLoading } = useCiviqRealData(isCiviq, trafficScale)

  // Build a 10-point sparkline interpolated across all 3 LOS levels for selfish
  function selfishSparkline(key: SelfishMetricKey): number[] {
    if (!selfishAllLevels) return algo.sparklines[key as keyof typeof algo.sparklines] as number[]
    const a = selfishAllLevels.free_flow[key]
    const c = selfishAllLevels.stable_flow[key]
    const e = selfishAllLevels.forced_flow[key]
    const lerp = (x: number, y: number, t: number) => x + (y - x) * t
    return [
      a, a,
      lerp(a, c, 0.33), lerp(a, c, 0.67),
      c, c,
      lerp(c, e, 0.33), lerp(c, e, 0.67),
      e, e,
    ].map(v => parseFloat(v.toFixed(3)))
  }

  const displayAlgo: AlgoData = (() => {
    if (isSelfish && realKpis) {
      const changePct = (val: number, ref: number) => parseFloat(((val - ref) / ref * 100).toFixed(1))
      const travelTime_s = realKpis.travelTime * 60
      return {
        ...algo,
        travelTime: realKpis.travelTime,
        waitTime:   realKpis.waitTime,
        throughput: realKpis.throughput,
        speed:      realKpis.speed,
        co2:        realKpis.co2,
        fuel:       realKpis.fuel,
        reward:     realKpis.returnMean,
        sparklines: {
          travelTime: selfishSparkline('travelTime'),
          waitTime:   selfishSparkline('waitTime'),
          throughput: selfishSparkline('throughput'),
          speed:      algo.sparklines.speed.map(() => realKpis.speed),
        },
        changes: (() => {
          const civiqRef = CIVIQ_LOS_REF[trafficScale as keyof typeof CIVIQ_LOS_REF]
          return civiqRef ? {
            travelTime: changePct(travelTime_s,        civiqRef.travelTime * 60),
            waitTime:   changePct(realKpis.waitTime,   civiqRef.waitTime),
            throughput: changePct(realKpis.throughput, civiqRef.throughput),
            speed: 0,
          } : algo.changes
        })(),
        episodes: selfishEvalArrays
          ? {
            travelTime: evalRawToPoints(selfishEvalArrays.evalTravelTimes),
            waitTime:   evalRawToPoints(selfishEvalArrays.evalWaitingTimes),
            throughput: evalRawToPoints(selfishEvalArrays.evalThroughputs),
            speed:      [],
          }
          : {
            travelTime: makeSeries(travelTime_s, travelTime_s, 3.0, 2.0, 50, 9),
            waitTime:   makeSeries(realKpis.waitTime, realKpis.waitTime, 1.5, 1.0, 50, 10),
            throughput: makeSeries(realKpis.throughput, realKpis.throughput, 60, 45, 50, 11),
            speed:      makeSeries(realKpis.speed, realKpis.speed, 8, 5, 50, 12),
          },
      }
    }
    if (isQmix && qmixData) {
      const k = qmixData.kpis
      const changePct = (val: number, ref: number) => parseFloat(((val - ref) / ref * 100).toFixed(1))
      const civiqRef  = CIVIQ_LOS_REF[trafficScale as keyof typeof CIVIQ_LOS_REF]
      const trainCurve = qmixData.training?.curve ?? []
      const peakTrainEp = trainCurve.length > 0
        ? trainCurve.reduce((a: any, b: any) => b.reward > a.reward ? b : a).episode as number
        : null
      const qmixEvalCpuMeans = qmixData.evalCpuMeans ?? []
      const realisticQmixPeak = qmixEvalCpuMeans.length > 0 ? Math.max(...qmixEvalCpuMeans) : k.cpuMean * 1.3
      return {
        ...algo,
        convergence: peakTrainEp,
        travelTime: k.travelTime_s / 60,
        waitTime:   k.waitTime_s,
        throughput: k.throughput,
        speed:      k.realTimeFactor ?? algo.speed,
        co2:        k.co2,
        fuel:       k.fuel,
        reward:     k.returnMean,
        cpuMean:    k.cpuMean,
        cpuPeak:    realisticQmixPeak,
        sparklines: {
          travelTime: downsampleArray((qmixData.evalTravelTimes ?? []).map(t => t / 60), 10),
          waitTime:   downsampleArray(qmixData.evalWaitingTimes ?? [], 10),
          throughput: downsampleArray(qmixData.evalThroughputs  ?? [], 10),
          speed:      trainingCurveToSparkline(trainCurve, 10),
        },
        changes: civiqRef ? {
          travelTime: changePct(k.travelTime_s, civiqRef.travelTime * 60),
          waitTime:   changePct(k.waitTime_s,   civiqRef.waitTime),
          throughput: changePct(k.throughput,   civiqRef.throughput),
          speed: 0,
        } : algo.changes,
        system: {
          ...algo.system,
          training: qmixToTrainingPoints(trainCurve),
          cpu: makeCpu(k.cpuMean, Math.min(realisticQmixPeak, 150), 8, 120, 18),
        },
        marl:     qmixToMarlPoints(trainCurve),
        episodes: qmixToEpisodeSeries(qmixData),
      }
    }
    if (isCiviq && civiqData) {
      const k          = civiqData.kpis
      const changePct  = (val: number, ref: number) => parseFloat(((val - ref) / ref * 100).toFixed(1))
      const qmixRef    = QMIX_LOS_REF[trafficScale as keyof typeof QMIX_LOS_REF]
      const selfishRef = SELFISH_LOS_REF[trafficScale as keyof typeof SELFISH_LOS_REF]
      const trainCurve = civiqData.training?.curve ?? []
      const peakTrainEp = trainCurve.length > 0
        ? trainCurve.reduce((a: any, b: any) => b.reward > a.reward ? b : a).episode as number
        : null
      const civiqEvalCpuMeans = civiqData.evalCpuMeans ?? []
      const realisticCiviqPeak = civiqEvalCpuMeans.length > 0 ? Math.max(...civiqEvalCpuMeans) : k.cpuMean * 1.3
      return {
        ...algo,
        convergence: peakTrainEp,
        travelTime: k.travelTime_s / 60,
        waitTime:   k.waitTime_s,
        throughput: k.throughput,
        speed:      k.realTimeFactor ?? algo.speed,
        co2:        k.co2,
        fuel:       k.fuel,
        reward:     k.returnMean,
        cpuMean:    k.cpuMean,
        cpuPeak:    realisticCiviqPeak,
        sparklines: {
          travelTime: downsampleArray((civiqData.evalTravelTimes ?? []).map(t => t / 60), 10),
          waitTime:   downsampleArray(civiqData.evalWaitingTimes ?? [], 10),
          throughput: downsampleArray(civiqData.evalThroughputs  ?? [], 10),
          speed:      trainingCurveToSparkline(trainCurve, 10),
        },
        changes: qmixRef ? {
          travelTime: changePct(k.travelTime_s, qmixRef.travelTime_s),
          waitTime:   changePct(k.waitTime_s,   qmixRef.waitTime_s),
          throughput: changePct(k.throughput,   qmixRef.throughput),
          speed: 0,
        } : algo.changes,
        changes2: selfishRef ? {
          travelTime: changePct(k.travelTime_s, selfishRef.travelTime_s),
          waitTime:   changePct(k.waitTime_s,   selfishRef.waitTime_s),
          throughput: changePct(k.throughput,   selfishRef.throughput),
          speed: 0,
        } : undefined,
        system: {
          ...algo.system,
          training: trainCurve.length ? qmixToTrainingPoints(trainCurve) : algo.system.training,
          cpu: makeCpu(k.cpuMean, Math.min(realisticCiviqPeak, 150), 8, 120, 15),
        },
        marl: trainCurve.length ? qmixToMarlPoints(trainCurve) : algo.marl,
        episodes: qmixToEpisodeSeries(civiqData),
      }
    }
    return algo
  })()

  const isLoading = (isSelfish && kpisLoading) || (isQmix && qmixLoading) || (isCiviq && civiqLoading)

  return (
  <div className="p-4 md:p-6 space-y-4 md:space-y-5 overflow-y-auto" style={{
    height: '100%',
    opacity: ready ? 1 : 0,
    transform: ready ? 'none' : 'translateY(8px)',
    transition: reducedMotion ? 'none' : 'opacity 260ms ease, transform 320ms ease',
  }}>
    {/* Episode detail modal */}
    {openModal && (
      <EpisodeDetailModal
        algo={displayAlgo}
        metricKey={openModal}
        onClose={() => setOpenModal(null)}
      />
    )}
    {/* Selfish detail modal */}
    {isSelfish && selfishModal && (() => {
      const data = selfishModal === 'co2'  ? (realKpis ? makeSeries(realKpis.co2,  realKpis.co2,  15,  10,  50, 13) : SELFISH_CO2_EPISODES)
                 : selfishModal === 'fuel' ? (realKpis ? makeSeries(realKpis.fuel, realKpis.fuel, 0.8, 0.5, 50, 14) : SELFISH_FUEL_EPISODES)
                 : displayAlgo.episodes[selfishModal as EpisodeMetricKey]
      return (
        <SelfishDetailModal
          metricKey={selfishModal}
          algoColor={displayAlgo.color}
          data={data}
          onClose={() => setSelfishModal(null)}
        />
      )
    })()}
    {/* Congestion detail modal */}
    {congestionDetail && (
      <CongestionDetailModal algo={displayAlgo} onClose={() => setCongestionDetail(false)} />
    )}
    {/* CPU detail modal */}
    {openCpuModal && (
      <CpuDetailModal
        algo={displayAlgo}
        evalCpuMeans={isQmix ? (qmixData?.evalCpuMeans ?? null) : isCiviq ? (civiqData?.evalCpuMeans ?? null) : null}
        onClose={() => setOpenCpuModal(false)}
      />
    )}

    {/* Header */}
    <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
      <div>
        <h2 className="text-xl font-bold leading-tight" style={{ color: c.tp }}>{displayAlgo.label}</h2>
      </div>
      <div className="flex items-center gap-2 flex-1">
        {mapSize && (
          <span className="text-[11px] font-medium px-2.5 py-1 rounded-full"
            style={{ background: c.badgeBg, border: `1px solid ${c.badgeBorder}`, color: c.ts }}>
            {MAP_LABELS[mapSize] || mapSize}
          </span>
        )}
        {trafficScale && (
          <span className="text-[11px] font-medium px-2.5 py-1 rounded-full"
            style={{ background: c.badgeBg, border: `1px solid ${c.badgeBorder}`, color: c.ts }}>
            {TRAFFIC_LABELS[trafficScale] || trafficScale}
          </span>
        )}
        {/* Loading indicator */}
        {((isSelfish && kpisLoading) || (isQmix && qmixLoading) || (isCiviq && civiqLoading)) && (
          <span className="inline-flex items-center gap-1.5 text-[10px] font-semibold px-2.5 py-1 rounded-full flex-shrink-0"
            style={{ background: c.itemBg, border: `1px solid ${c.divider}`, color: c.tm }}>
            <span className="inline-block w-2.5 h-2.5 rounded-full border-[1.5px] border-current border-t-transparent animate-spin" aria-hidden="true" />
            Loading data…
          </span>
        )}
      </div>
      <ExportButton algo={displayAlgo} selfishTimeseries={isSelfish ? realTimeseries : null} />
    </div>

    {/* Loading gate — show spinner while real data is fetching */}
    {isLoading ? (
      <div className="flex flex-col items-center justify-center py-32 gap-5">
        <div
          className="w-12 h-12 rounded-full border-2 border-t-transparent animate-spin"
          style={{ borderColor: algo.color, borderTopColor: 'transparent' }}
          role="status"
          aria-label="Loading simulation data"
        />
        <p className="text-[13px] font-semibold" style={{ color: c.tm }}>
          Loading simulation data…
        </p>
        <p className="text-[11px]" style={{ color: c.tu }}>
          Fetching results for {TRAFFIC_LABELS[trafficScale] || trafficScale}
        </p>
      </div>
    ) : (<>

    {/* KPI Row */}
    <div className="grid grid-cols-1 sm:grid-cols-2 2xl:grid-cols-4 gap-4">
      <KpiCard label="Avg. Travel Time" abbr="ATT" value={parseFloat((displayAlgo.travelTime * 60).toFixed(2))} unit="sec"
        color={displayAlgo.color} colorDim={displayAlgo.colorDim} borderColor={displayAlgo.border}
        change={displayAlgo.changes.travelTime} lowerBetter sparkData={displayAlgo.sparklines.travelTime}
        changeLabel={isQmix ? 'vs. CiViQ' : isCiviq ? 'vs. QMIX' : (isSelfish && CIVIQ_LOS_REF[trafficScale as keyof typeof CIVIQ_LOS_REF] ? 'vs. CiViQ' : undefined)}
        change2={isCiviq ? displayAlgo.changes2?.travelTime : undefined}
        changeLabel2={isCiviq ? 'vs. Selfish' : undefined}
        onClick={isSelfish ? () => setSelfishModal('travelTime') : () => setOpenModal('travelTime')}
        description="The mean time it takes for a vehicle to complete its route from entry to exit, across all vehicles in the simulation." />
      <KpiCard label="Avg. Wait Time" abbr="AWT" value={parseFloat(displayAlgo.waitTime.toFixed(2))} unit="sec"
        color={displayAlgo.color} colorDim={displayAlgo.colorDim} borderColor={displayAlgo.border}
        change={displayAlgo.changes.waitTime} lowerBetter sparkData={displayAlgo.sparklines.waitTime}
        changeLabel={isQmix ? 'vs. CiViQ' : isCiviq ? 'vs. QMIX' : (isSelfish && CIVIQ_LOS_REF[trafficScale as keyof typeof CIVIQ_LOS_REF] ? 'vs. CiViQ' : undefined)}
        change2={isCiviq ? displayAlgo.changes2?.waitTime : undefined}
        changeLabel2={isCiviq ? 'vs. Selfish' : undefined}
        onClick={isSelfish ? () => setSelfishModal('waitTime') : () => setOpenModal('waitTime')}
        description="The mean time vehicles spent fully stopped in traffic. High values indicate congestion or poor routing decisions." />
      <KpiCard label="Throughput" abbr="TPT" value={Math.round(displayAlgo.throughput).toLocaleString()} unit="veh/hr"
        color={displayAlgo.color} colorDim={displayAlgo.colorDim} borderColor={displayAlgo.border}
        change={displayAlgo.changes.throughput} sparkData={displayAlgo.sparklines.throughput}
        changeLabel={isQmix ? 'vs. CiViQ' : isCiviq ? 'vs. QMIX' : (isSelfish && CIVIQ_LOS_REF[trafficScale as keyof typeof CIVIQ_LOS_REF] ? 'vs. CiViQ' : undefined)}
        change2={isCiviq ? displayAlgo.changes2?.throughput : undefined}
        changeLabel2={isCiviq ? 'vs. Selfish' : undefined}
        onClick={isSelfish ? () => setSelfishModal('throughput') : () => setOpenModal('throughput')}
        description="The number of vehicles that successfully completed their routes per minute. Higher values indicate better overall traffic flow." />
      <KpiCard label="Real-time Factor" abbr="RTF" value={displayAlgo.speed.toFixed(2)} unit="x"
        color={displayAlgo.color} colorDim={displayAlgo.colorDim} borderColor={displayAlgo.border}
        valueAlign="center"
        onClick={undefined}
        descriptionSide="left"
        description="The ratio of simulation time to actual wall-clock time. A value of 1.0 means the simulation runs in real time; higher values indicate faster-than-real-time execution." />
    </div>

    {/* Charts row: left column stack + Map Player */}
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
      {/* Left column: Algorithm Overview + Heatmap stacked */}
      <div className="flex flex-col gap-4 relative z-10">
        <GlassCard className="p-5 flex flex-col gap-3">
          <h3 className="text-[13px] font-bold" style={{ color: c.tp }}>Algorithm Overview</h3>
          <p className="text-[12px] leading-relaxed" style={{ color: c.ts }}>
            {displayAlgo.description}
          </p>
          {(displayAlgo.convergence !== null || displayAlgo.reward !== null) && (
            <div className="pt-3 grid grid-cols-2 gap-3" style={{ borderTop: `1px solid ${c.divider}` }}>
              {displayAlgo.convergence !== null && (
                <div>
                  <div className="text-[10px] uppercase tracking-wider mb-1" style={{ color: c.tu }}>Peak Episode Performance</div>
                  <div className="text-[17px] font-bold tabular-nums" style={{ color: displayAlgo.color }}>Ep. {displayAlgo.convergence}</div>
                </div>
              )}
              {displayAlgo.reward !== null && (
                <div className={displayAlgo.convergence === null ? 'col-span-2' : ''}>
                  <div className="flex items-center gap-1 mb-1">
                    <span className="text-[10px] uppercase tracking-wider" style={{ color: c.tu }}>Episode Return</span>
                    <div className="group/tip relative flex-shrink-0">
                      <div className="w-[14px] h-[14px] rounded-full flex items-center justify-center cursor-default"
                        style={{ background: 'rgba(6,182,212,0.18)', border: '1px solid rgba(6,182,212,0.5)', color: c.isDark ? 'rgba(217,249,255,0.9)' : 'rgba(3,105,161,0.9)' }}>
                        <span className="text-[9px] font-bold leading-none">!</span>
                      </div>
                      <div className="pointer-events-none absolute left-0 bottom-[calc(100%+6px)] w-[260px] rounded-lg px-3 py-2 text-[10px] leading-relaxed opacity-0 translate-y-1 transition-all duration-150 group-hover/tip:opacity-100 group-hover/tip:translate-y-0"
                        style={{ background: c.tooltipBg, border: `1px solid ${c.tooltipBorder}`, color: c.tooltipColor, boxShadow: c.tooltipShadow, zIndex: 90 }}>
                        {isSelfish
                          ? 'Mean cumulative reward per evaluation episode, derived from greedy shortest-path routing (Nash Equilibrium). Negative values reflect the penalty-based reward signal used across all SUMO-MARL experiments — lower (more negative) indicates worse overall traffic performance.'
                          : 'Mean cumulative reward per evaluation episode across all seeds. Negative values reflect the penalty-based reward signal — lower (more negative) indicates worse traffic performance.'}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-baseline gap-2 flex-wrap">
                    <div className="text-[17px] font-bold tabular-nums" style={{ color: displayAlgo.color }}>
                      {typeof displayAlgo.reward === 'number' ? displayAlgo.reward.toLocaleString(undefined, { maximumFractionDigits: 0 }) : displayAlgo.reward}
                    </div>
                    {(() => {
                      if (typeof displayAlgo.reward !== 'number') return null
                      const reward = displayAlgo.reward
                      const mkBadge = (pct: number, label: string) => {
                        const isBetter = pct > 0
                        const color = isBetter ? '#4ADE80' : '#F87171'
                        const arrow = isBetter ? '▲' : '▼'
                        return (
                          <div key={label} className="flex items-baseline gap-1.5 whitespace-nowrap">
                            <span className="text-[11px] font-bold tabular-nums leading-none" style={{ color }}>
                              {arrow} {Math.abs(pct).toFixed(1)}%
                            </span>
                            <span className="text-[9px]" style={{ color: c.tu }}>{label}</span>
                          </div>
                        )
                      }
                      if (isSelfish) {
                        const ref = CIVIQ_LOS_REF[trafficScale as keyof typeof CIVIQ_LOS_REF]
                        if (!ref) return null
                        const pct = (reward - ref.returnMean) / Math.abs(ref.returnMean) * 100
                        return mkBadge(pct, 'vs. CiViQ')
                      }
                      if (isQmix) {
                        const ref = CIVIQ_LOS_REF[trafficScale as keyof typeof CIVIQ_LOS_REF]
                        if (!ref) return null
                        const pct = (reward - ref.returnMean) / Math.abs(ref.returnMean) * 100
                        return mkBadge(pct, 'vs. CiViQ')
                      }
                      if (isCiviq) {
                        const qRef = QMIX_LOS_REF[trafficScale as keyof typeof QMIX_LOS_REF]
                        const sRef = SELFISH_LOS_REF[trafficScale as keyof typeof SELFISH_LOS_REF]
                        return (
                          <>
                            {qRef && mkBadge((reward - qRef.returnMean) / Math.abs(qRef.returnMean) * 100, 'vs. QMIX')}
                            {sRef && mkBadge((reward - sRef.returnMean) / Math.abs(sRef.returnMean) * 100, 'vs. Selfish')}
                          </>
                        )
                      }
                      return null
                    })()}
                  </div>
                </div>
              )}
            </div>
          )}
        </GlassCard>

        <CongestionHeatmap
          algo={displayAlgo}
          heatmapSrc={isSelfish
            ? SELFISH_HEATMAP[trafficScale]
            : HEATMAP_IMG[displayAlgo.id as 'civiq' | 'qmix']?.[trafficScale]}
          congestionLabel={CONGESTION_LABEL[trafficScale]}
        />
      </div>

      {/* Map Player (col 2–3) */}
      <MapPlayer algo={displayAlgo} mapSize={mapSize} trafficScale={trafficScale}
        onCo2Click={isSelfish ? () => setSelfishModal('co2') : undefined}
        onFuelClick={isSelfish ? () => setSelfishModal('fuel') : undefined}
      />
    </div>

    {/* CPU utilization + Key Findings */}
    {displayAlgo.id !== 'selfish' ? (
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <CpuStatsCard
          algo={displayAlgo}
          evalCpuMeans={isQmix ? (qmixData?.evalCpuMeans ?? null) : isCiviq ? (civiqData?.evalCpuMeans ?? null) : null}
          onViewDetail={() => setOpenCpuModal(true)}
        />
        <FindingsCard algo={displayAlgo} trafficScale={trafficScale} />
      </div>
    ) : (
      <FindingsCard algo={displayAlgo} trafficScale={trafficScale} />
    )}

    {/* Evaluation performance distribution */}
    <EvalDeviationCard algo={displayAlgo} />
    </>)}
  </div>
  )
}

// ─── Road Closures Page ────────────────────────────────────────────────────────

function RoadClosuresPage() {
  const c = useDashColors()
  const { reducedMotion, ready } = usePageEntrance()
  const [demandTab, setDemandTab] = useState<'free_flow' | 'stable_flow' | 'forced_flow'>('free_flow')

  const DEMAND_TABS = [
    { key: 'free_flow'   as const, label: 'Low Demand',     sub: '1,000 veh/hr' },
    { key: 'stable_flow' as const, label: 'Moderate Demand', sub: '1,200 veh/hr' },
    { key: 'forced_flow' as const, label: 'High Demand',      sub: '2,000 veh/hr' },
  ]

  const hasPartialData = false
  const hasData = true

  const closureData = ROAD_CLOSURE_DATA[demandTab]

  const baseline = {
    selfish: {
      travelTime: SELFISH_LOS_REF[demandTab].travelTime_s,
      waitTime:   SELFISH_LOS_REF[demandTab].waitTime_s,
      throughput: SELFISH_LOS_REF[demandTab].throughput,
      co2:        SELFISH_LOS_REF[demandTab].co2,
      fuel:       SELFISH_LOS_REF[demandTab].fuel,
    },
    qmix: {
      travelTime: QMIX_LOS_REF[demandTab].travelTime_s,
      waitTime:   QMIX_LOS_REF[demandTab].waitTime_s,
      throughput: QMIX_LOS_REF[demandTab].throughput,
      co2:        QMIX_LOS_REF[demandTab].co2,
      fuel:       QMIX_LOS_REF[demandTab].fuel,
    },
    civiq: {
      travelTime: CIVIQ_LOS_REF[demandTab].travelTime * 60,
      waitTime:   CIVIQ_LOS_REF[demandTab].waitTime,
      throughput: CIVIQ_LOS_REF[demandTab].throughput,
      co2:        CIVIQ_LOS_REF[demandTab].co2,
      fuel:       CIVIQ_LOS_REF[demandTab].fuel,
    },
  }

  type AlgoEntry = { algo: AlgoData; closureKey: 'selfish' | 'qmix' | 'civiq' }
  const ENTRIES: AlgoEntry[] = [
    { algo: ALGO.civiq,   closureKey: 'civiq'   },
    { algo: ALGO.qmix,    closureKey: 'qmix'    },
    { algo: ALGO.selfish, closureKey: 'selfish' },
  ]

  type MetaDef = { label: string; unit: string; lowerBetter: boolean; fmt: (v: number) => string; baseKey: 'travelTime' | 'waitTime' | 'throughput' | 'co2' | 'fuel'; closureKey: keyof typeof closureData.selfish }
  const METRICS: MetaDef[] = [
    { label: 'Travel Time', unit: 'sec',     lowerBetter: true,  fmt: v => v.toFixed(2), baseKey: 'travelTime', closureKey: 'travelTime' },
    { label: 'Wait Time',   unit: 'sec',     lowerBetter: true,  fmt: v => v.toFixed(2), baseKey: 'waitTime',   closureKey: 'waitTime'   },
    { label: 'Throughput',  unit: 'veh/hr',  lowerBetter: false, fmt: v => Math.round(v).toLocaleString(), baseKey: 'throughput', closureKey: 'throughput' },
    { label: 'CO₂',         unit: 'g/km',    lowerBetter: true,  fmt: v => v.toFixed(2), baseKey: 'co2',        closureKey: 'co2'        },
    { label: 'Diesel',      unit: 'L/100km', lowerBetter: true,  fmt: v => v.toFixed(2), baseKey: 'fuel',       closureKey: 'fuel'       },
  ]

  return (
    <div className="p-4 md:p-6 space-y-4 md:space-y-5 overflow-y-auto" style={{
      height: '100%',
      opacity: ready ? 1 : 0,
      transform: ready ? 'none' : 'translateY(8px)',
      transition: reducedMotion ? 'none' : 'opacity 260ms ease, transform 320ms ease',
    }}>

      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5 mb-1">
            <div className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0"
              style={{ background: 'rgba(251,146,60,0.15)' }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#FB923C" strokeWidth="2.2"
                strokeLinecap="round" strokeLinejoin="round">
                <path d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 1 1-18 0 9 9 0 0 1 18 0z" />
              </svg>
            </div>
            <h2 className="text-xl md:text-2xl font-bold" style={{ color: c.tp }}>Road Closure Analysis</h2>
          </div>
          <p className="text-[12px]" style={{ color: c.tm }}>
            Evaluates algorithm robustness when a subset of road edges is forced closed. Baseline = normal operation.
          </p>
        </div>

        {/* Demand selector */}
        <div className="flex flex-wrap gap-1 p-1 rounded-xl flex-shrink-0" style={{ background: c.itemBg }}>
          {DEMAND_TABS.map(t => (
            <button key={t.key} onClick={() => setDemandTab(t.key)}
              className="px-4 md:px-5 py-2 rounded-lg text-[11px] font-semibold transition-all duration-150 focus-visible:outline-none"
              style={{
                background: demandTab === t.key ? c.itemBgMd : 'transparent',
                color:      demandTab === t.key ? c.tp       : c.ts,
                border:     demandTab === t.key ? `1px solid ${c.glassBorder}` : '1px solid transparent',
              }}>
              <span>{t.label}</span>
              <span className="block text-[9px] font-normal" style={{ color: c.tu }}>{t.sub}</span>
            </button>
          ))}
        </div>
      </div>


      {/* Per-algorithm cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {ENTRIES.map(({ algo, closureKey }) => {
          const rc  = closureData[closureKey]
          const bl  = baseline[closureKey]

          const allVals = METRICS.map(m => rc[m.closureKey] as number)
          const allZero = allVals.every(v => v === 0)

          return (
            <GlassCard key={algo.id} className="p-5 relative overflow-hidden cursor-default transition-all duration-300 hover:scale-[1.01]"
              style={{ border: `1px solid ${c.glassBorder}`, boxShadow: c.glassShadow }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.transform = 'translateY(-2px) scale(1.01)'
                ;(e.currentTarget as HTMLElement).style.boxShadow = `0 16px 48px rgba(0,0,0,0.45), 0 0 0 1px ${algo.border}`
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.transform = 'translateY(0) scale(1)'
                ;(e.currentTarget as HTMLElement).style.boxShadow = c.glassShadow
              }}>
              <div className="absolute inset-0 pointer-events-none"
                style={{ background: `radial-gradient(ellipse at 80% 0%, ${algo.colorDim}, transparent 65%)` }} />
              <div className="relative">
                {/* Card header */}
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-[14px] font-bold" style={{ color: c.tp }}>{algo.label}</h3>
                  <span className="text-[9px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider"
                    style={{ background: `${algo.color}18`, color: algo.color, border: `1px solid ${algo.color}40` }}>
                    {allZero ? 'No Data' : 'w/ Closure'}
                  </span>
                </div>

                {/* Metrics */}
                <div className="space-y-3">
                  {METRICS.map(m => {
                    const rcVal  = rc[m.closureKey] as number
                    const blVal  = bl[m.baseKey]
                    const isZero = rcVal === 0 && allZero

                    const pct = blVal > 0 && !isZero
                      ? parseFloat(((rcVal - blVal) / blVal * 100).toFixed(1))
                      : null
                    const better = pct !== null && (m.lowerBetter ? pct < 0 : pct > 0)
                    const worse  = pct !== null && (m.lowerBetter ? pct > 0 : pct < 0)
                    const pctColor = better ? '#4ADE80' : worse ? '#F87171' : c.ts

                    return (
                      <div key={m.label}>
                        <div className="flex items-baseline justify-between mb-1">
                          <span className="text-[9px] font-semibold uppercase tracking-wider" style={{ color: c.tu }}>{m.label}</span>
                          <div className="flex items-center gap-1.5">
                            {pct !== null && (
                              <span className="text-[9px] font-bold tabular-nums" style={{ color: pctColor }}>
                                {pct > 0 ? '+' : ''}{pct.toFixed(1)}%
                              </span>
                            )}
                            <span className="text-[11px] font-bold tabular-nums" style={{ color: isZero ? c.tu : c.tp }}>
                              {isZero ? '—' : m.fmt(rcVal)}
                              <span className="text-[9px] font-normal ml-0.5" style={{ color: c.tu }}>{m.unit}</span>
                            </span>
                          </div>
                        </div>
                        {/* Baseline vs closure bar */}
                        <div className="w-full h-1.5 rounded-full relative" style={{ background: c.barTrack }}>
                          {/* Baseline marker */}
                          {!isZero && blVal > 0 && (
                            <div className="absolute top-0 w-0.5 h-full rounded-full opacity-40"
                              style={{
                                left: '50%',
                                background: c.ts,
                              }} />
                          )}
                          {/* Closure bar */}
                          <div className="h-full rounded-full transition-all duration-700"
                            style={{
                              width: isZero ? '0%' : `${Math.min(100, Math.max(4, 50 + 50 * (
                                m.lowerBetter
                                  ? (blVal - rcVal) / (blVal * 0.5)
                                  : (rcVal - blVal) / (blVal * 0.5)
                              ))).toFixed(1)}%`,
                              background: algo.color,
                              opacity: isZero ? 0 : 0.7,
                            }} />
                        </div>
                        {/* Baseline ref */}
                        {!isZero && (
                          <div className="flex justify-between mt-0.5">
                            <span className="text-[8px]" style={{ color: c.tu }}>Baseline: {m.fmt(blVal)} {m.unit}</span>
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>

                {/* Return */}
                  <div className="mt-4 pt-3" style={{ borderTop: `1px solid ${c.divider}` }}>
                    <div className="flex justify-between items-baseline">
                      <span className="text-[9px] font-semibold uppercase tracking-wider" style={{ color: c.tu }}>Mean Return</span>
                      <span className="text-[12px] font-bold tabular-nums" style={{ color: algo.color }}>
                        {rc.returnMean.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                      </span>
                    </div>
                  </div>
              </div>
            </GlassCard>
          )
        })}
      </div>

      {/* Road Closure Heatmap + Info */}
      {(() => {
        const HEATMAP_SRCS: Record<typeof demandTab, string> = {
          free_flow:   '/heatmap_output/blockages/blockage_heatmap_civiq-los-a_best_low_seed60.png',
          stable_flow: '/heatmap_output/blockages/blockage_heatmap_civiq-los-c_best_med_seed61.png',
          forced_flow: '/heatmap_output/blockages/blockage_heatmap_civiq-los-e_best_high_seed62.png',
        }
        const demandMeta = DEMAND_TABS.find(d => d.key === demandTab)!
        return (
          <div className="grid grid-cols-1 xl:grid-cols-5 gap-4">

            {/* Heatmap — left */}
            <GlassCard className="xl:col-span-2 p-5" style={{ border: `1px solid ${c.glassBorder}` }}>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-[13px] font-bold" style={{ color: c.tp }}>Road Closure Heatmap</h3>
                <span className="text-[9px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider"
                  style={{ background: 'rgba(251,146,60,0.12)', color: '#FB923C', border: '1px solid rgba(251,146,60,0.3)' }}>
                  {demandMeta.label}
                </span>
              </div>
              <div className="rounded-xl overflow-hidden" style={{ border: `1px solid ${c.glassBorder}` }}>
                <img
                  src={HEATMAP_SRCS[demandTab]}
                  alt={`Road closure heatmap — ${demandMeta.label}`}
                  className="w-full h-auto object-contain"
                />
              </div>
            </GlassCard>

            {/* Right side — stacked cards */}
            <div className="xl:col-span-3 flex flex-col gap-4">

              {/* Top card: Blockage Configuration */}
              <GlassCard className="p-5" style={{ border: `1px solid ${c.glassBorder}` }}>
                <h3 className="text-[13px] font-bold mb-3" style={{ color: c.tp }}>Blockage Configuration</h3>
                <div className="grid grid-cols-2 gap-3">
                  {([
                    { label: 'Injection Probability', value: '10%',     sub: 'per simulation second'  },
                    { label: 'Max Concurrent Edges',  value: '3',       sub: 'blocked simultaneously' },
                    { label: 'Duration per Closure',  value: '30–60 s', sub: 'randomized per event'   },
                    { label: 'Episodes Evaluated',    value: '50',      sub: 'random seed per episode' },
                  ] as const).map(item => (
                    <div key={item.label} className="p-3 rounded-xl" style={{ background: c.itemBg, border: `1px solid ${c.divider}` }}>
                      <p className="text-[9px] font-semibold uppercase tracking-wider mb-1" style={{ color: c.tu }}>{item.label}</p>
                      <p className="text-[15px] font-bold tabular-nums" style={{ color: c.tp }}>{item.value}</p>
                      <p className="text-[9px]" style={{ color: c.tu }}>{item.sub}</p>
                    </div>
                  ))}
                </div>
                <p className="text-[10px] mt-3 leading-relaxed" style={{ color: c.tm }}>
                  Edges are randomly selected from the BGC Full network at each simulation step. The heatmap shows closure frequency across all evaluation episodes.
                </p>
              </GlassCard>

              {/* Bottom card: Findings */}
              <GlassCard className="p-5 flex-1" style={{ border: `1px solid ${c.glassBorder}` }}>
                <h3 className="text-[13px] font-bold mb-3" style={{ color: c.tp }}>Findings</h3>
                <div className="space-y-3">
                  {([
                    {
                      tag: 'Selfish Routing', tagColor: '#94A3B8',
                      text: 'Maintains the lowest travel times under closure at low and moderate demand, benefiting from fewer congested vehicles and faster equilibrium rerouting.',
                    },
                    {
                      tag: 'MARL Resilience', tagColor: '#06B6D4',
                      text: 'At high demand, Mono-QMIX and CiViQ achieve significantly higher throughput (2,221 vs. 1,231 veh/hr) despite higher individual wait times, demonstrating cooperative advantage under congestion.',
                    },
                    {
                      tag: 'CiViQ vs QMIX', tagColor: '#A78BFA',
                      text: 'Both algorithms produce identical traffic KPIs under closure across all demand levels, reflecting equivalent learned routing policies evaluated on the same blockage scenario.',
                    },
                  ] as const).map(f => (
                    <div key={f.tag} className="p-3.5 rounded-xl" style={{ background: c.itemBg, border: `1px solid ${c.divider}` }}>
                      <span className="inline-block text-[9px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider mb-1.5"
                        style={{ background: `${f.tagColor}1a`, color: f.tagColor, border: `1px solid ${f.tagColor}33` }}>
                        {f.tag}
                      </span>
                      <p className="text-[11px] leading-relaxed" style={{ color: c.tm }}>{f.text}</p>
                    </div>
                  ))}
                </div>
              </GlassCard>

            </div>
          </div>
        )
      })()}

    </div>
  )
}

// ─── Sidebar ───────────────────────────────────────────────────────────────────

const NAV: { id: Page; label: string; d: string; hidden?: boolean }[] = [
  {
    id: 'summary', label: 'Summary',
    d: 'M3 3h7v7H3V3zm11 0h7v7h-7V3zM3 14h7v7H3v-7zm14 3.5a3.5 3.5 0 1 1-7 0 3.5 3.5 0 0 1 7 0z',
  },
  {
    id: 'civiq', label: 'Civiq',
    d: 'M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5',
  },
  {
    id: 'qmix', label: 'Monolithic QMIX',
    d: 'M19 11H5m14 0a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-6a2 2 0 0 1 2-2m14 0V9a2 2 0 0 0-2-2M5 11V9a2 2 0 0 1 2-2m0 0V5a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v2M7 7h10',
  },
  {
    id: 'selfish', label: 'Selfish Routing',
    d: 'M9 20l-5.447-2.724A1 1 0 0 1 3 16.382V5.618a1 1 0 0 1 1.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0 0 21 18.382V7.618a1 1 0 0 0-1.447-.894L15 9m0 8V9m0 0L9 7',
  },
  {
    id: 'road_closures', label: 'Road Closures',
    d: 'M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 1 1-18 0 9 9 0 0 1 18 0z',
    hidden: true,
  },
]

const PAGE_COLOR: Record<Page, string> = {
  summary:       '#60A5FA',
  civiq:         ALGO.civiq.color,
  selfish:       ALGO.selfish.color,
  qmix:          ALGO.qmix.color,
  road_closures: '#FB923C',
}

interface SidebarProps {
  activePage: Page
  setActivePage: (p: Page) => void
  onRunAlgorithm: (algorithm: string) => void
  mapSize: string
  trafficScale: string
  algorithm1: string
  showRoadClosures: boolean
}

function Sidebar({ activePage, setActivePage, onRunAlgorithm, mapSize, trafficScale, algorithm1, showRoadClosures }: SidebarProps) {
  const c = useDashColors()
  const visibleNav = NAV.filter(item => !item.hidden || (item.id === 'road_closures' && showRoadClosures))
  return (
  <div className="flex flex-col flex-shrink-0 w-full xl:w-[248px]"
    style={{ background: c.sidebarBg, borderRight: `1px solid ${c.sidebarBorder}`, borderBottom: `1px solid ${c.sidebarBorder}` }}>

    {/* Analytics nav */}
    <div className="px-5 pt-6 pb-3 flex-shrink-0">
      <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: c.tu }}>
        Analytics
      </span>
    </div>

    <nav className="px-3 pb-2 xl:pb-0 flex xl:block gap-2 xl:space-y-1 overflow-x-auto flex-shrink-0">
      {visibleNav.map((item) => {
        const isActive = activePage === item.id
        const col = PAGE_COLOR[item.id]
        return (
          <button key={item.id} onClick={() => setActivePage(item.id)}
            className="w-full min-w-max xl:min-w-0 flex items-center gap-3 px-3 py-2.5 rounded-xl text-left transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
            style={{
              background: isActive ? `${col}18` : 'transparent',
              border: isActive ? `1px solid ${col}40` : '1px solid transparent',
              color: isActive ? col : c.ts,
            }}
            onMouseEnter={(e) => { if (!isActive) { (e.currentTarget as HTMLElement).style.background = c.hoverBg; (e.currentTarget as HTMLElement).style.color = c.hoverColor } }}
            onMouseLeave={(e) => { if (!isActive) { (e.currentTarget as HTMLElement).style.background = 'transparent'; (e.currentTarget as HTMLElement).style.color = c.ts } }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d={item.d} />
            </svg>
            <span className="text-[13px] font-medium">{item.label}</span>
          </button>
        )
      })}
    </nav>

    {/* Divider */}
    <div className="mx-4 my-4 flex-shrink-0" style={{ height: '1px', background: c.divider }} />

    {/* Simulation Controls section label */}
    <div className="px-5 pb-3 flex-shrink-0">
      <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: c.tu }}>
        Simulation Controls
      </span>
    </div>

    {/* Controls */}
    <div className="px-4 pb-4 xl:pb-0 flex-shrink-0">
      <SimulationControls
        darkMode={c.isDark}
        vertical
        hideHeader
        onRunAlgorithm={onRunAlgorithm}
        initialMapSize={mapSize}
        initialTrafficScale={trafficScale}
        initialAlgorithm={algorithm1 === 'hierarchical_qmix' ? 'hierarchical_qmix'
          : algorithm1 === 'monolithic_qmix' ? 'monolithic_qmix'
          : algorithm1 === 'selfish_routing' ? 'selfish_routing'
          : undefined}
      />
    </div>

  </div>
  )
}

// ─── Labels ────────────────────────────────────────────────────────────────────

const MAP_LABELS: Record<string, string> = {
  '2km': '2 km²', '0.75km': '0.75 km²', '4x4': '4×4 Grid (2.25 km²)',
}
const TRAFFIC_LABELS: Record<string, string> = {
  free_flow:   'Low Demand — 1,000 veh/hr',
  stable_flow: 'Moderate Demand — 1,200 veh/hr',
  forced_flow: 'High Demand — 2,000 veh/hr',
}

// ─── Main Export ───────────────────────────────────────────────────────────────

function SimulationDashboardContent() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const { theme } = useTheme()
  const c = buildDashColors(theme === 'dark')

  const mapSize = searchParams.get('mapSize') || ''
  const trafficScale = searchParams.get('trafficScale') || ''
  const view = searchParams.get('view') || ''
  const algorithm1 = searchParams.get('algorithm1') || ''
  const algorithm2 = searchParams.get('algorithm2') || ''

  const algoToPage = (algo: string): Page => {
    if (algo === 'hierarchical_qmix') return 'civiq'
    if (algo === 'monolithic_qmix') return 'qmix'
    if (algo === 'selfish_routing') return 'selfish'
    return 'summary'
  }
  const [activePage, setActivePage] = useState<Page>(() => algoToPage(algorithm1))
  const [roadClosuresVisible, setRoadClosuresVisible] = useState(false)

  // Keep active tab in sync when URL params change (e.g. Run from sidebar controls)
  useEffect(() => {
    setActivePage(algoToPage(algorithm1))
  }, [algorithm1])

  // Ctrl+Alt+K toggles the Road Closures tab visibility
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.altKey && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setRoadClosuresVisible(prev => {
          if (!prev) {
            setActivePage('road_closures')
          } else {
            setActivePage(p => p === 'road_closures' ? 'summary' : p)
          }
          return !prev
        })
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  useEffect(() => {
    if (!mapSize || !trafficScale || !view || !algorithm1) { router.push('/'); return }
    const validMapSizes = ['2km', '0.75km', '4x4']
    const validTrafficScales = ['free_flow', 'stable_flow', 'forced_flow']
    const validViews = ['focused', 'comparative']
    const validAlgorithms = ['selfish_routing', 'monolithic_qmix', 'hierarchical_qmix']
    if (
      !validMapSizes.includes(mapSize) ||
      !validTrafficScales.includes(trafficScale) ||
      !validViews.includes(view) ||
      !validAlgorithms.includes(algorithm1)
    ) { router.push('/'); return }
    if (view === 'comparative' && (!algorithm2 || algorithm1 === algorithm2 || !validAlgorithms.includes(algorithm2))) {
      router.push('/')
    }
  }, [mapSize, trafficScale, view, algorithm1, algorithm2, router])

  return (
    <DashColorsContext.Provider value={c}>
    <main className="w-full h-screen overflow-hidden" style={{ position: 'relative' }}>
      {/* Base gradient */}
      <div className="fixed inset-0 pointer-events-none"
        style={{ background: c.pageBgGrad, zIndex: -1 }} />
      <AnimatedBackground />

      {/* Outer wrapper */}
      <div className="flex items-stretch justify-center"
        style={{ height: '100vh', padding: 'clamp(8px, 1.2vw, 16px)', zIndex: 2, position: 'relative' }}>

        {/* OBU Bezel */}
        <div className="relative w-full flex flex-col" style={{
          maxWidth: '1640px',
          background: c.obuBg,
          backdropFilter: 'blur(32px)', WebkitBackdropFilter: 'blur(32px)',
          borderRadius: '24px', padding: '12px',
          border: `1px solid ${c.obuBorder}`,
          boxShadow: c.obuShadow,
        }}>
          {/* Side screws */}
          {['left-2', 'right-2'].map((side) => (
            <div key={side} className={`absolute ${side} top-1/2 -translate-y-1/2 flex flex-col gap-2.5 pointer-events-none`}>
              {[0, 1, 2].map((i) => (
                <div key={i} className="w-2 h-2 rounded-full"
                  style={{ background: c.screwBg, boxShadow: c.screwShadow }} />
              ))}
            </div>
          ))}

          {/* Screen */}
          <div className="relative flex flex-col overflow-hidden"
            style={{
              flex: 1,
              background: c.screenBg,
              backdropFilter: 'blur(20px)', WebkitBackdropFilter: 'blur(20px)',
              borderRadius: '14px', border: `1px solid ${c.screenBorder}`,
              boxShadow: c.screenShadow,
            }}>
            {/* Top gloss */}
            <div className="absolute inset-x-0 top-0 h-20 pointer-events-none rounded-t-[14px]"
              style={{ background: c.screenGloss }} />

            {/* Status bar */}
            <StatusBar />

            {/* Body: Sidebar + Content */}
            <div className="flex flex-col xl:flex-row overflow-hidden" style={{ flex: 1, minHeight: 0 }}>
              <Sidebar
                activePage={activePage}
                setActivePage={setActivePage}
                onRunAlgorithm={(algorithm) => setActivePage(algoToPage(algorithm))}
                mapSize={mapSize}
                trafficScale={trafficScale}
                algorithm1={algorithm1}
                showRoadClosures={roadClosuresVisible}
              />
              {/* Content pane: fixed blob layer + scrollable content on top */}
              <div className="flex-1 relative" style={{
                minHeight: 0,
                overflow: 'hidden',
                background: c.contentBg,
                backdropFilter: 'blur(20px)',
                WebkitBackdropFilter: 'blur(20px)',
                borderLeft: `1px solid ${c.contentBorder}`,
              }}>
                {/* Animated colour blobs — fixed behind content */}
                <div className="absolute inset-0 overflow-hidden pointer-events-none" style={{ zIndex: 0 }}>
                  <div className="absolute rounded-full" style={{
                    width: 500, height: 500, top: '-120px', left: '-80px',
                    background: c.blob1,
                    filter: 'blur(60px)', animation: 'blob1 16s ease-in-out infinite', animationDelay: '-3s',
                  }} />
                  <div className="absolute rounded-full" style={{
                    width: 460, height: 460, bottom: '-100px', right: '-60px',
                    background: c.blob2,
                    filter: 'blur(60px)', animation: 'blob2 20s ease-in-out infinite', animationDelay: '-8s',
                  }} />
                  <div className="absolute rounded-full" style={{
                    width: 360, height: 360, top: '35%', left: '50%',
                    background: c.blob3,
                    filter: 'blur(50px)', animation: 'blob3 18s ease-in-out infinite', animationDelay: '-12s',
                  }} />
                </div>

                {/* Scrollable page content */}
                <div className="absolute inset-0 overflow-y-auto" style={{ zIndex: 1 }}>
                  {activePage === 'summary'       && <SummaryPage onNavigate={setActivePage} />}
                  {activePage === 'civiq'          && <AlgoDetailPage algo={ALGO.civiq}   mapSize={mapSize} trafficScale={trafficScale} />}
                  {activePage === 'selfish'        && <AlgoDetailPage algo={ALGO.selfish} mapSize={mapSize} trafficScale={trafficScale} />}
                  {activePage === 'qmix'           && <AlgoDetailPage algo={ALGO.qmix}    mapSize={mapSize} trafficScale={trafficScale} />}
                  {activePage === 'road_closures'  && <RoadClosuresPage />}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
    </DashColorsContext.Provider>
  )
}

export default function SimulationDashboard() {
  return (
    <Suspense fallback={<main className="w-full h-screen" style={{ background: '#060112' }} />}>
      <SimulationDashboardContent />
    </Suspense>
  )
}

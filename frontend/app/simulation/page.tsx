'use client'

import { Suspense, useState, useEffect, useRef, memo } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import dynamic from 'next/dynamic'
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Bar, Cell,
} from 'recharts'
import { AnimatedBackground } from '@/components/AnimatedBackground'
import { SimulationControls } from '@/components/SimulationControls'

// react-gauge-chart uses D3 and must be client-only (no SSR)
const GaugeComponent = dynamic(() => import('react-gauge-chart'), { ssr: false })

// ─── Status Bar ───────────────────────────────────────────────────────────────

const StatusBar = () => {
  const router = useRouter()
  const NAV_LINKS = [
    { label: 'About',        href: '/#about' },
    { label: 'The Research', href: '/research' },
    { label: 'Contact Us',   href: '/#contact' },
  ]
  return (
    <div className="flex items-center justify-between px-7 py-2.5 flex-shrink-0"
      style={{ background: 'rgba(0,0,0,0.35)', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
      {/* Logo — click to go home */}
      <button
        onClick={() => router.push('/')}
        className="flex items-center gap-2.5 transition-opacity duration-150 hover:opacity-80"
      >
        <img src="/icons/civiq-logo.png" alt="Civiq" className="w-5 h-5 object-contain brightness-0 invert opacity-80" />
        <span className="font-bold text-[13px] tracking-widest" style={{ color: 'rgba(255,255,255,0.75)' }}>CIVIQ</span>
        <span className="text-[10px] font-medium" style={{ color: 'rgba(255,255,255,0.3)' }}>
          ·&nbsp; A Hierarchical Multi-Agent Coordination Framework
        </span>
      </button>
      {/* Nav buttons */}
      <div className="flex items-center gap-1">
        {NAV_LINKS.map(({ label, href }) => (
          <button
            key={label}
            onClick={() => router.push(href)}
            className="px-4 py-1.5 rounded-full text-[12px] font-medium transition-all duration-150"
            style={{ color: 'rgba(255,255,255,0.52)', background: 'transparent' }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.9)'
              ;(e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.07)'
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.52)'
              ;(e.currentTarget as HTMLElement).style.background = 'transparent'
            }}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  )
}

// ─── Data ─────────────────────────────────────────────────────────────────────

type AlgoKey = 'civiq' | 'qmix' | 'selfish'
type Page = 'summary' | AlgoKey

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
  convergence: number | null
  reward: number | null
  efficiency: number
  description: string
  strengths: string[]
  scores: number[]
  // Pluggable time-series — replace with real API data when backend is ready
  sparklines: KpiSeries
  changes: KpiChanges
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
    id: 'civiq', label: 'Hierarchical QMIX', sublabel: 'Civiq', rank: 1,
    color: '#38BDF8', colorDim: 'rgba(56,189,248,0.12)', border: 'rgba(56,189,248,0.3)',
    travelTime: 4.2, waitTime: 18.5, throughput: 1875, speed: 1.24,
    co2: 142, fuel: 23, computeTime: 22.35, convergence: 150, reward: 1250, efficiency: 88,
    description: 'Civiq employs a hierarchical two-tier coordination mechanism where a global orchestrator assigns zone-level routing goals, while local agents optimize intersection-level decisions using QMIX. This architecture enables scalable, cooperative traffic management that generalizes across varying network topologies and traffic densities.',
    strengths: ['Lowest travel time across all scenarios', 'Best emission reduction (37% vs baseline)', 'Superior network throughput coordination', 'Scalable to larger road networks'],
    scores: [0.90, 0.90, 0.88, 0.70, 0.88, 0.88, 0.82],
    sparklines: {
      travelTime:  [8.5, 7.8, 7.1, 6.4, 5.9, 5.3, 4.9, 4.6, 4.4, 4.2],
      waitTime:    [36, 32, 29, 26, 24, 22, 21, 20, 19, 18.5],
      throughput:  [1380, 1480, 1570, 1650, 1710, 1760, 1810, 1840, 1862, 1875],
      speed:       [0.86, 0.92, 0.97, 1.02, 1.08, 1.12, 1.16, 1.19, 1.22, 1.24],
    },
    changes: { travelTime: -50.6, waitTime: -48.6, throughput: 35.9, speed: 44.2 },
    episodes: {
      travelTime:  makeSeries(8.5, 4.2,   0.5, 0.35, 200, 1),
      waitTime:    makeSeries(36,  18.5,  1.8, 1.2,  200, 2),
      throughput:  makeSeries(1380,1875,  38,  28,   200, 3),
      speed:       makeSeries(0.86, 1.24,  0.06, 0.04, 200, 4),
    },
    system: {
      training: makeTraining(-200, 1250, 120, 200, 13),
      cpu:      makeCpu(28, 72, 6, 120, 15),
    },
    queue:      makeQueue([2.1, 3.2, 2.6, 1.8], 0.9, 120, 21),
    density:    makeDensity(14, 28, 4, 120, 22),
    congestion: makeCongestion(0.10, 0.24, 0.04, 120, 23),
    marl:       makeMarlMetrics(2.8, -55, 95, 2.1, -200, 1250, 200, 24),
  },
  qmix: {
    // ── Real PyMARL data — BGC Full (2 km²), LOS A, seed 1801, 30 eval episodes ──
    // Source: results/mono-qmix-los-a/experiment_summary_losA.json + qmix_exp_1801.json
    // Detail page fetches training curve & MARL diagnostics dynamically via /api/qmix
    id: 'qmix', label: 'Monolithic QMIX', sublabel: 'Baseline RL', rank: 2,
    color: '#A78BFA', colorDim: 'rgba(167,139,250,0.12)', border: 'rgba(167,139,250,0.3)',
    // LOS A eval defaults (seed 1801, 30 episodes) — overridden by useQmixRealData
    travelTime: 2.08, waitTime: 7.96, throughput: 981, speed: 53.97,
    co2: 498.2, fuel: 21.5, computeTime: 18.20, convergence: 9001, reward: -45791, efficiency: 71,
    description: 'Monolithic QMIX applies centralized multi-agent reinforcement learning where all agents share a joint action-value function. While effective at coordination, the monolithic architecture faces scalability limitations as network size grows, requiring more training episodes and compute to converge on larger topologies.',
    strengths: ['Fastest compute time per decision step', 'Good coordination at small scale', 'Solid baseline RL performance', 'Well-established QMIX framework'],
    scores: [0.80, 0.70, 0.70, 0.90, 0.72, 0.72, 0.75],
    // Flat sparklines at eval measurement (single post-training evaluation)
    sparklines: {
      travelTime: [2.08, 2.08, 2.08, 2.08, 2.08, 2.08, 2.08, 2.08, 2.08, 2.08],
      waitTime:   [7.96, 7.96, 7.96, 7.96, 7.96, 7.96, 7.96, 7.96, 7.96, 7.96],
      throughput: [981,  981,  981,  981,  981,  981,  981,  981,  981,  981],
      speed:      [53.97, 53.97, 53.97, 53.97, 53.97, 53.97, 53.97, 53.97, 53.97, 53.97],
    },
    // vs noop baseline (LOS A): QMIX underperforms noop signal-free in low-traffic scenario
    changes: { travelTime: +4.5, waitTime: +12.1, throughput: -12.3, speed: 0 },
    episodes: {
      travelTime:  makeSeries(2.66, 2.08, 0.20, 0.12, 200, 5),
      waitTime:    makeSeries(9.8,  7.96, 0.35, 0.22, 200, 6),
      throughput:  makeSeries(920,  981,  55,   35,   200, 7),
      speed:       makeSeries(50,   53.97, 2.5, 1.5,  200, 8),
    },
    system: {
      // Placeholder training curve — replaced by useQmixRealData hook on detail page
      training: makeTraining(-48700, -45791, 800, 200, 16),
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
    // forced_flow (LOS E) bgc_full defaults — overridden per-level by useSelfishRealData
    travelTime: 2.12, waitTime: 20.8, throughput: 2178, speed: 248.84,
    co2: 485.6, fuel: 20.89, computeTime: 0, convergence: null, reward: null, efficiency: 48,
    description: 'Selfish Routing models each vehicle independently optimizing its own route via shortest-path algorithms, representing the Nash Equilibrium state of the network. Without coordination, vehicles converge on popular routes causing Braess\'s Paradox — where adding road capacity can paradoxically worsen network-wide performance.',
    strengths: ['No training or setup required', 'Simple and fully interpretable', 'Establishes the Price of Anarchy baseline', 'Handles novel edge cases naturally'],
    scores: [0.58, 0.42, 0.40, 0.80, 0.42, 0.52, 0.70],
    // Sparklines: flat at the forced_flow measurement (overridden by real data once loaded)
    sparklines: {
      travelTime:  [2.12, 2.12, 2.12, 2.12, 2.12, 2.12, 2.12, 2.12, 2.12, 2.12],
      waitTime:    [20.8, 20.8, 20.8, 20.8, 20.8, 20.8, 20.8, 20.8, 20.8, 20.8],
      throughput:  [2178, 2178, 2178, 2178, 2178, 2178, 2178, 2178, 2178, 2178],
      speed:       [248.84, 248.84, 248.84, 248.84, 248.84, 248.84, 248.84, 248.84, 248.84, 248.84],
    },
    changes: { travelTime: 0, waitTime: 0, throughput: 0, speed: 0 },
    episodes: {
      travelTime:  makeSeries(2.12, 2.12, 0.15, 0.10, 10000, 9),
      waitTime:    makeSeries(20.8, 20.8, 1.5,  1.0,  10000, 10),
      throughput:  makeSeries(2178, 2178, 60,   45,   10000, 11),
      speed:       makeSeries(248.84, 248.84, 8, 5,   10000, 12),
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

const ALGO_LIST = [ALGO.civiq, ALGO.qmix, ALGO.selfish]

// ─── Reusable UI primitives ────────────────────────────────────────────────────

const GlassCard = ({
  children, className = '', style = {}, onClick, onMouseEnter, onMouseLeave,
}: { children: React.ReactNode; className?: string; style?: React.CSSProperties; onClick?: () => void; onMouseEnter?: React.MouseEventHandler<HTMLDivElement>; onMouseLeave?: React.MouseEventHandler<HTMLDivElement> }) => (
  <div className={className} onClick={onClick} onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave} style={{
    background: 'linear-gradient(155deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.04) 100%)',
    backdropFilter: 'blur(24px)',
    WebkitBackdropFilter: 'blur(24px)',
    border: '1px solid rgba(255,255,255,0.14)',
    borderRadius: '16px',
    boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.13), inset 0 -1px 0 rgba(0,0,0,0.15), 0 8px 32px rgba(0,0,0,0.32)',
    ...style,
  }}>{children}</div>
)

// ─── Sparkline ────────────────────────────────────────────────────────────────

const SparkLine = ({ data, color }: { data: number[]; color: string }) => {
  if (!data || data.length < 2) return null
  const W = 96, H = 44, PAD = 2
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
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ overflow: 'visible', flexShrink: 0 }}>
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.25" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={areaPts} fill={`url(#${gid})`} />
      <polyline points={linePts} fill="none" stroke={color} strokeWidth="2"
        strokeLinecap="round" strokeLinejoin="round" />
      {/* last-point dot */}
      <circle cx={pts[pts.length - 1].x} cy={pts[pts.length - 1].y} r="2.5"
        fill={color} />
    </svg>
  )
}

// ─── KPI Card ─────────────────────────────────────────────────────────────────

const KpiCard = ({ label, abbr, value, unit, color, colorDim, borderColor, change, lowerBetter, sparkData, onClick, description, descriptionSide = 'right', changeLabel }: {
  label: string; abbr?: string; value: string | number; unit: string
  color: string; colorDim?: string; borderColor?: string
  change?: number; lowerBetter?: boolean; sparkData?: number[]; onClick?: () => void; description?: string; descriptionSide?: 'left' | 'right'
  changeLabel?: string
}) => {
  // Green = good outcome, regardless of direction
  const isGood = change === undefined ? true : lowerBetter ? change <= 0 : change >= 0
  const changeColor = isGood ? '#4ADE80' : '#F87171'
  const changeArrow = (change ?? 0) >= 0 ? '▲' : '▼'
  return (
    <GlassCard className="group relative z-0 hover:z-30 p-4 flex flex-col gap-2 transition-all duration-200"
      onClick={onClick}
      onMouseEnter={onClick ? (e) => {
        (e.currentTarget as HTMLElement).style.borderColor = borderColor?.replace('0.3', '0.5') ?? 'rgba(255,255,255,0.25)'
        ;(e.currentTarget as HTMLElement).style.boxShadow = `0 0 0 1px ${borderColor?.replace('0.3', '0.15') ?? 'transparent'}, inset 0 1px 0 rgba(255,255,255,0.16), 0 12px 40px rgba(0,0,0,0.4)`
        ;(e.currentTarget as HTMLElement).style.transform = 'translateY(-1px)'
      } : undefined}
      onMouseLeave={onClick ? (e) => {
        (e.currentTarget as HTMLElement).style.borderColor = borderColor?.replace('0.3', '0.25') ?? 'rgba(255,255,255,0.14)'
        ;(e.currentTarget as HTMLElement).style.boxShadow = 'inset 0 1px 0 rgba(255,255,255,0.13), inset 0 -1px 0 rgba(0,0,0,0.15), 0 8px 32px rgba(0,0,0,0.32)'
        ;(e.currentTarget as HTMLElement).style.transform = 'translateY(0)'
      } : undefined}
      style={{
        background: colorDim
          ? `linear-gradient(145deg, ${colorDim.replace('0.12', '0.10')} 0%, rgba(255,255,255,0.03) 100%)`
          : undefined,
        border: borderColor ? `1px solid ${borderColor.replace('0.3', '0.25')}` : undefined,
        cursor: onClick ? 'pointer' : 'default',
        transition: 'border-color 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease',
      }}>
      {/* Title */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="text-[11px] font-semibold uppercase tracking-wider leading-none truncate"
            style={{ color: 'rgba(255,255,255,0.5)' }}>
            {label}
          </span>
          {description && (
            <div className="group/info relative z-20 flex-shrink-0" onClick={(e) => e.stopPropagation()}>
              <div
                className="w-[18px] h-[18px] rounded-full flex items-center justify-center"
                style={{
                  background: 'rgba(6,182,212,0.2)',
                  border: '1px solid rgba(6,182,212,0.65)',
                  color: 'rgba(217,249,255,0.95)',
                  boxShadow: '0 0 0 1px rgba(0,0,0,0.2) inset',
                }}
              >
                <span className="text-[10px] font-bold leading-none">!</span>
              </div>
              <div
                className="pointer-events-none absolute top-[calc(100%+8px)] w-[280px] rounded-lg px-3 py-2 text-[10px] leading-relaxed opacity-0 translate-y-1 transition-all duration-150 group-hover/info:opacity-100 group-hover/info:translate-y-0"
                style={{
                  ...(descriptionSide === 'left' ? { right: 0 } : { left: 0 }),
                  background: 'rgba(4,9,22,0.97)',
                  border: '1px solid rgba(255,255,255,0.12)',
                  color: 'rgba(255,255,255,0.88)',
                  boxShadow: '0 12px 24px rgba(0,0,0,0.45)',
                  zIndex: 90,
                }}
              >
                {description}
              </div>
            </div>
          )}
        </div>
        {onClick && (
          <span
            className="flex-shrink-0 text-[9px] font-semibold px-2 py-0.5 rounded-full"
            style={{
              background: 'rgba(56,189,248,0.1)',
              color: 'rgba(186,230,253,0.65)',
              border: '1px solid rgba(56,189,248,0.22)',
              transition: 'background 0.12s ease, color 0.12s ease, border-color 0.12s ease',
            }}
            onMouseEnter={(e) => {
              const el = e.currentTarget as HTMLElement
              el.style.background = 'rgba(56,189,248,0.2)'
              el.style.color = 'rgba(186,230,253,1)'
              el.style.borderColor = 'rgba(56,189,248,0.5)'
            }}
            onMouseLeave={(e) => {
              const el = e.currentTarget as HTMLElement
              el.style.background = 'rgba(56,189,248,0.1)'
              el.style.color = 'rgba(186,230,253,0.65)'
              el.style.borderColor = 'rgba(56,189,248,0.22)'
            }}
          >
            View detail ↗
          </span>
        )}
      </div>

      {/* Value + sparkline row */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex flex-col gap-0.5">
          {/* Value + unit */}
          <div className="flex items-baseline gap-1.5">
            <span className="text-[26px] font-bold tabular-nums leading-none" style={{ color }}>{value}</span>
            <span className="text-[12px] font-medium" style={{ color: 'rgba(255,255,255,0.38)' }}>{unit}</span>
          </div>
          {/* Change badge below value */}
          {change !== undefined && (
            <div className="flex items-center gap-1.5">
              <span className="text-[11px] font-bold tabular-nums"
                style={{ color: changeColor }}>
                {changeArrow} {Math.abs(change).toFixed(1)}%
              </span>
              {changeLabel && (
                <span className="text-[10px]" style={{ color: 'rgba(255,255,255,0.32)' }}>
                  {changeLabel}
                </span>
              )}
            </div>
          )}
        </div>
        {sparkData && <SparkLine data={sparkData} color={color} />}
      </div>

    </GlassCard>
  )
}

const HBar = ({ value, max, color, dim = false }: { value: number; max: number; color: string; dim?: boolean }) => (
  <div className="flex-1 h-1.5 rounded-full" style={{ background: 'rgba(255,255,255,0.1)', boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.25)' }}>
    <div className="h-full rounded-full transition-all duration-700"
      style={{ width: `${Math.min(100, (value / max) * 100)}%`, background: color, opacity: dim ? 0.45 : 1 }} />
  </div>
)

const RingChart = ({ value, max = 100, color, size = 88 }: {
  value: number; max?: number; color: string; size?: number
}) => {
  const r = (size - 14) / 2
  const circ = 2 * Math.PI * r
  const dash = (value / max) * circ
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="8" />
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth="8"
        strokeDasharray={`${dash} ${circ - dash}`} strokeLinecap="round"
        style={{ transform: 'rotate(-90deg)', transformOrigin: '50% 50%' }} />
    </svg>
  )
}

// ─── Radar Chart ───────────────────────────────────────────────────────────────

const RADAR_AXES = ['Throughput', 'Wait Time', 'Travel Time', 'Compute', 'CO₂', 'Fuel', 'RTF']
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

function RadarTooltip({ axisIdx }: { axisIdx: number }) {
  const { x: lx, y: ly } = rLabel(axisIdx)
  const tipW = 108
  const tipH = 62
  const tx = lx > RC.x ? Math.min(lx - tipW - 4, 240) : Math.max(lx + 8, 2)
  const ty = Math.max(8, Math.min(ly - tipH / 2, 350 - tipH - 8))
  return (
    <g style={{ pointerEvents: 'none' }}>
      <rect x={tx} y={ty} width={tipW} height={tipH} rx="6"
        fill="rgba(8,8,24,0.94)" stroke="rgba(255,255,255,0.14)" strokeWidth="1" />
      <text x={tx + 8} y={ty + 14} fontSize="9.5" fill="rgba(255,255,255,0.65)"
        fontWeight="700" textAnchor="start">{RADAR_AXES[axisIdx]}</text>
      {ALGO_LIST.map((a, ai) => (
        <g key={a.id}>
          <circle cx={tx + 9} cy={ty + 26 + ai * 13} r="3.5" fill={a.color} />
          <text x={tx + 17} y={ty + 30 + ai * 13} fontSize="9" fill="rgba(255,255,255,0.78)" textAnchor="start">
            {a.sublabel}: {Math.round(a.scores[axisIdx] * 100)}%
          </text>
        </g>
      ))}
    </g>
  )
}

const RadarChart = () => {
  const [hoveredAxis, setHoveredAxis] = useState<number | null>(null)
  return (
    <svg viewBox="0 0 350 350" className="w-full max-w-[320px] mx-auto" style={{ overflow: 'visible' }}>
      {[0.25, 0.5, 0.75, 1].map((r) => (
        <polygon key={r} points={rPts(Array(7).fill(r))}
          fill="none" stroke="rgba(255,255,255,0.11)" strokeWidth="1" />
      ))}
      {RADAR_AXES.map((_, i) => {
        const pt = { x: RC.x + RR * Math.cos(-Math.PI / 2 + (2 * Math.PI * i) / RADAR_AXES.length), y: RC.y + RR * Math.sin(-Math.PI / 2 + (2 * Math.PI * i) / RADAR_AXES.length) }
        return <line key={i} x1={RC.x} y1={RC.y} x2={pt.x} y2={pt.y}
          stroke={hoveredAxis === i ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.11)'} strokeWidth="1" />
      })}
      <polygon points={rPts(ALGO.selfish.scores)} fill="rgba(248,113,113,0.12)" stroke="#F87171" strokeWidth="1.5" strokeLinejoin="round" />
      <polygon points={rPts(ALGO.qmix.scores)} fill="rgba(167,139,250,0.12)" stroke="#A78BFA" strokeWidth="1.5" strokeLinejoin="round" />
      <polygon points={rPts(ALGO.civiq.scores)} fill="rgba(56,189,248,0.18)" stroke="#38BDF8" strokeWidth="2" strokeLinejoin="round" />
      {RADAR_AXES.map((label, i) => {
        const { x, y } = rLabel(i)
        const isHovered = hoveredAxis === i
        return (
          <text key={i} x={x} y={y} textAnchor="middle" dominantBaseline="middle"
            fontSize="10" fill={isHovered ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.45)'}
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
      {hoveredAxis !== null && <RadarTooltip axisIdx={hoveredAxis} />}
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
  { label: 'Avg. Travel Time', unit: 'min', key: 'travelTime' as const, max: 10, lowerBetter: true },
  { label: 'Avg. Wait Time', unit: 'sec', key: 'waitTime' as const, max: 40, lowerBetter: true },
  { label: 'Throughput', unit: 'veh/hr', key: 'throughput' as const, max: 2000, lowerBetter: false },
  { label: 'Real-time Factor', unit: 'x', key: 'speed' as const, max: 2, lowerBetter: false },
  { label: 'CO₂ Emissions', unit: 'g/km', key: 'co2' as const, max: 280, lowerBetter: true },
  { label: 'Compute Time', unit: 'ms', key: 'computeTime' as const, max: 30, lowerBetter: true },
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
  { key: 'free_flow',   label: 'Free Flow',   sub: 'LOS A' },
  { key: 'stable_flow', label: 'Stable Flow', sub: 'LOS C' },
  { key: 'forced_flow', label: 'Forced Flow', sub: 'LOS E' },
]

function CompareModal({ onClose, onConfirm }: {
  onClose: () => void
  onConfirm: (cfg: CompareConfig) => void
}) {
  const [left, setLeft]       = useState<AlgoKey>('civiq')
  const [right, setRight]     = useState<AlgoKey>('qmix')
  const [mapSize, setMapSize] = useState('2km')
  const [traffic, setTraffic] = useState('stable_flow')
  const canRun = left !== right

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
          <button onClick={onClose} className="w-7 h-7 rounded-full flex items-center justify-center transition-colors"
            style={{ background: 'rgba(255,255,255,0.07)', color: 'rgba(255,255,255,0.5)' }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.14)' }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.07)' }}>
            <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
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
  { label: 'Average Fuel Consumption', unit: 'l/100km', getValue: a => a.fuel, lowerBetter: true },
]
const COMPUTE_KPI: KpiMetaDef[] = [
  { label: 'Real-time Factor', unit: '×',  getValue: a => a.speed,       lowerBetter: false },
  { label: 'Decision Latency', unit: 'ms', getValue: a => a.computeTime, lowerBetter: true },
]

function MetricRow({ label, unit, lv, rv, lowerBetter, la, ra, intFmt }: {
  label: string; unit: string; lv: number; rv: number
  lowerBetter: boolean; la: AlgoData; ra: AlgoData; intFmt?: boolean
}) {
  const fmt = (v: number) => intFmt ? String(Math.round(v)) : String(v)
  const d = calcDelta(lv, rv, lowerBetter)
  const leftWins = d.leftWins === true
  const rightWins = d.leftWins === false
  const tie = d.leftWins === null
  const WIN = '#4ADE80'
  const leftValColor  = leftWins  ? WIN : rightWins ? 'rgba(255,255,255,0.38)' : 'rgba(255,255,255,0.72)'
  const rightValColor = rightWins ? WIN : leftWins  ? 'rgba(255,255,255,0.38)' : 'rgba(255,255,255,0.72)'
  const arrow = lv < rv ? '▼' : lv > rv ? '▲' : '—'
  const deltaColor = tie ? 'rgba(255,255,255,0.35)' : leftWins ? '#4ADE80' : '#F87171'
  const deltaBg    = tie ? 'rgba(255,255,255,0.05)' : leftWins ? 'rgba(74,222,128,0.1)' : 'rgba(248,113,113,0.1)'
  const deltaBorder = tie ? 'rgba(255,255,255,0.1)' : leftWins ? 'rgba(74,222,128,0.25)' : 'rgba(248,113,113,0.25)'
  const deltaLabel = tie ? '— tied' : `${arrow} ${Math.abs(d.diff) < 1 ? Math.abs(d.diff).toFixed(2) : Math.abs(d.diff).toFixed(1)} (${d.pct.toFixed(1)}%)`

  return (
    <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-3 py-3"
      style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
      {/* Left value */}
      <div className="text-right">
        <div className="text-[21px] font-black tabular-nums leading-none" style={{ color: leftValColor }}>{fmt(lv)}</div>
        <div className="text-[9px] mt-0.5 tabular-nums" style={{ color: 'rgba(255,255,255,0.28)' }}>{unit}</div>
        {leftWins && <div className="text-[8px] font-bold mt-1 uppercase tracking-wider" style={{ color: WIN }}>wins</div>}
      </div>
      {/* Center */}
      <div className="text-center" style={{ minWidth: '148px' }}>
        <div className="text-[10px] font-semibold mb-1.5" style={{ color: 'rgba(255,255,255,0.52)' }}>{label}</div>
        <div className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[10px] font-bold"
          style={{ background: deltaBg, color: deltaColor, border: `1px solid ${deltaBorder}` }}>
          {deltaLabel}
        </div>
      </div>
      {/* Right value */}
      <div className="text-left">
        <div className="text-[21px] font-black tabular-nums leading-none" style={{ color: rightValColor }}>{fmt(rv)}</div>
        <div className="text-[9px] mt-0.5 tabular-nums" style={{ color: 'rgba(255,255,255,0.28)' }}>{unit}</div>
        {rightWins && <div className="text-[8px] font-bold mt-1 uppercase tracking-wider" style={{ color: WIN }}>wins</div>}
      </div>
    </div>
  )
}

function CompareSection({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <GlassCard className="p-5">
      <div className="flex items-center gap-2 mb-1 pb-3" style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
        {icon}
        <h3 className="text-[13px] font-bold" style={{ color: 'rgba(255,255,255,0.78)' }}>{title}</h3>
      </div>
      {children}
    </GlassCard>
  )
}

function ComparePage({ config, onBack }: { config: CompareConfig; onBack: () => void }) {
  const la = ALGO[config.left]
  const ra = ALGO[config.right]
  const bothLearning = la.system.training.length > 0 && ra.system.training.length > 0
  const convL = bothLearning ? detectConvergence(la.system.training) : null
  const convR = bothLearning ? detectConvergence(ra.system.training) : null

  const routingIcon = (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.5)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 12h18M3 6h18M3 18h12" />
    </svg>
  )
  const environIcon = (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.5)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a10 10 0 0 1 0 20 10 10 0 0 1 0-20z" /><path d="M12 8v4l3 3" />
    </svg>
  )
  const computeIcon = (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.5)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="4" y="4" width="16" height="16" rx="2" /><path d="M9 9h6M9 12h6M9 15h4" />
    </svg>
  )

  return (
    <div className="p-6 space-y-4">
      {/* Back */}
      <button onClick={onBack}
        className="flex items-center gap-1.5 text-[11px] transition-colors"
        style={{ color: 'rgba(255,255,255,0.38)' }}
        onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.78)' }}
        onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.38)' }}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M19 12H5M12 19l-7-7 7-7" />
        </svg>
        Back to Summary
      </button>

      {/* VS header */}
      <div className="grid grid-cols-[1fr_auto_1fr] gap-3 items-stretch">
        <div className="rounded-2xl p-4 text-center" style={{ background: la.colorDim, border: `1px solid ${la.border}` }}>
          <div className="text-[9px] font-bold uppercase tracking-widest mb-1" style={{ color: la.color }}>Side A</div>
          <div className="text-[16px] font-black leading-tight" style={{ color: 'rgba(255,255,255,0.92)' }}>{la.label}</div>
          <div className="text-[10px] mt-0.5 font-semibold" style={{ color: la.color }}>{la.sublabel}</div>
        </div>
        <div className="flex flex-col items-center justify-center gap-1 px-2">
          <div className="text-[13px] font-black" style={{ color: 'rgba(255,255,255,0.28)' }}>VS</div>
          <div className="text-[9px] text-center mt-1 leading-relaxed" style={{ color: 'rgba(255,255,255,0.22)' }}>
            {MAP_LABELS[config.mapSize] || config.mapSize}<br />
            {TRAFFIC_LABELS[config.traffic] || config.traffic}
          </div>
        </div>
        <div className="rounded-2xl p-4 text-center" style={{ background: ra.colorDim, border: `1px solid ${ra.border}` }}>
          <div className="text-[9px] font-bold uppercase tracking-widest mb-1" style={{ color: ra.color }}>Side B</div>
          <div className="text-[16px] font-black leading-tight" style={{ color: 'rgba(255,255,255,0.92)' }}>{ra.label}</div>
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
  const [compareOpen, setCompareOpen]         = useState(false)
  const [compareConfig, setCompareConfig]     = useState<CompareConfig | null>(null)

  if (compareConfig) {
    return <ComparePage config={compareConfig} onBack={() => setCompareConfig(null)} />
  }

  return (
  <div className="p-6 space-y-5 overflow-y-auto" style={{ height: '100%' }}>
    {compareOpen && (
      <CompareModal
        onClose={() => setCompareOpen(false)}
        onConfirm={(cfg) => { setCompareOpen(false); setCompareConfig(cfg) }}
      />
    )}
    {/* Header */}
    <div className="flex items-center justify-between">
      <div>
        <h2 className="text-xl font-bold" style={{ color: 'rgba(255,255,255,0.9)' }}>Algorithm Comparison</h2>
        <p className="text-[12px] mt-0.5" style={{ color: 'rgba(255,255,255,0.38)' }}>
          Aggregate performance across Civiq, Monolithic QMIX, and Selfish Routing
        </p>
        <div className="flex items-center gap-4 mt-2">
          {ALGO_LIST.map((a) => (
            <div key={a.id} className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full" style={{ background: a.color }} />
              <span className="text-[11px]" style={{ color: 'rgba(255,255,255,0.45)' }}>{a.sublabel}</span>
            </div>
          ))}
        </div>
      </div>
      <button
        onClick={() => setCompareOpen(true)}
        className="flex items-center gap-2 px-4 py-2 rounded-xl text-[12px] font-bold transition-all duration-200"
        style={{
          background: 'linear-gradient(135deg, rgba(6,182,212,0.22) 0%, rgba(99,102,241,0.18) 100%)',
          border: '1px solid rgba(6,182,212,0.45)',
          color: '#06B6D4',
          boxShadow: '0 0 18px rgba(6,182,212,0.18)',
        }}
        onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.boxShadow = '0 0 28px rgba(6,182,212,0.35)' }}
        onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.boxShadow = '0 0 18px rgba(6,182,212,0.18)' }}
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="3" width="7" height="18" rx="1" /><rect x="14" y="3" width="7" height="18" rx="1" />
        </svg>
        Compare Side-by-Side
      </button>
    </div>

    {/* Rank cards */}
    <div className="grid grid-cols-3 gap-4">
      {ALGO_LIST.map((a, i) => (
        <GlassCard key={a.id} className="p-5 relative overflow-hidden"
          style={{ border: `1px solid ${a.border}`, transition: 'transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease' }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.transform = 'translateY(-2px)'; (e.currentTarget as HTMLElement).style.boxShadow = `0 16px 48px rgba(0,0,0,0.45), 0 0 0 1px ${a.border}` }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.transform = 'translateY(0)'; (e.currentTarget as HTMLElement).style.boxShadow = 'inset 0 1px 0 rgba(255,255,255,0.13), inset 0 -1px 0 rgba(0,0,0,0.15), 0 8px 32px rgba(0,0,0,0.32)' }}>
          <div className="absolute inset-0 pointer-events-none"
            style={{ background: `radial-gradient(ellipse at 80% 0%, ${a.colorDim}, transparent 65%)` }} />
          <div className="relative">
            <div className="flex items-center justify-between mb-3">
              <span className="text-[11px] font-bold px-2.5 py-1 rounded-full"
                style={{ background: rankMeta[i].bg, color: rankMeta[i].color }}>
                {rankMeta[i].label}
              </span>
              <button
                onClick={() => onNavigate(a.id)}
                className="text-[10px] font-semibold px-2.5 py-1 rounded-full transition-all duration-150"
                style={{ color: a.color, background: a.colorDim, border: `1px solid ${a.border}` }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.filter = 'brightness(1.3)'; (e.currentTarget as HTMLElement).style.transform = 'scale(1.04)' }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.filter = 'none'; (e.currentTarget as HTMLElement).style.transform = 'scale(1)' }}
              >
                View Detail →
              </button>
            </div>
            <div className="flex items-center gap-2 mb-3">
              <h3 className="text-[15px] font-bold" style={{ color: 'rgba(255,255,255,0.9)' }}>{a.label}</h3>
              <span className="text-[9px] font-bold px-2 py-0.5 rounded-full"
                style={{ background: a.colorDim, color: a.color, border: `1px solid ${a.border}` }}>
                {a.sublabel}
              </span>
            </div>
            <div className="space-y-2 text-[12px]">
              {[
                { k: 'Travel Time', v: `${a.travelTime} min` },
                { k: 'Throughput', v: `${a.throughput.toLocaleString()} veh/hr` },
                { k: 'CO₂ Emissions', v: `${a.co2} g/km` },
              ].map(({ k, v }) => (
                <div key={k} className="flex justify-between">
                  <span style={{ color: 'rgba(255,255,255,0.42)' }}>{k}</span>
                  <span className="font-semibold" style={{ color: 'rgba(255,255,255,0.82)' }}>{v}</span>
                </div>
              ))}
              <div className="flex justify-between">
                <span style={{ color: 'rgba(255,255,255,0.42)' }}>Efficiency</span>
                <span className="font-bold" style={{ color: a.color }}>{a.efficiency}%</span>
              </div>
            </div>
            <div className="mt-3 flex items-center gap-2">
              <HBar value={a.efficiency} max={100} color={a.color} />
              <span className="text-[11px] font-bold w-8 text-right tabular-nums" style={{ color: a.color }}>{a.efficiency}%</span>
            </div>
          </div>
        </GlassCard>
      ))}
    </div>

    {/* Radar + Breakdown */}
    <div className="grid grid-cols-5 gap-4">
      <GlassCard className="col-span-2 p-5">
        <h3 className="text-[13px] font-bold mb-3" style={{ color: 'rgba(255,255,255,0.78)' }}>Performance Profile</h3>
        <RadarChart />
        <div className="flex justify-center gap-5 mt-2">
          {ALGO_LIST.map((a) => (
            <div key={a.id} className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full" style={{ background: a.color }} />
              <span className="text-[10px]" style={{ color: 'rgba(255,255,255,0.45)' }}>{a.sublabel}</span>
            </div>
          ))}
        </div>
      </GlassCard>

      <GlassCard className="col-span-3 p-5">
        <h3 className="text-[13px] font-bold mb-4" style={{ color: 'rgba(255,255,255,0.78)' }}>Metrics Breakdown</h3>
        <div className="space-y-4">
          {COMPARE_METRICS.map((m) => {
            const vals = ALGO_LIST.map((a) => a[m.key] as number)
            const best = m.lowerBetter ? Math.min(...vals) : Math.max(...vals)
            return (
              <div key={m.key}>
                <div className="flex justify-between mb-1.5">
                  <span className="text-[11px] font-medium" style={{ color: 'rgba(255,255,255,0.48)' }}>{m.label}</span>
                  <span className="text-[10px]" style={{ color: 'rgba(255,255,255,0.28)' }}>{m.unit}</span>
                </div>
                <div className="space-y-1.5">
                  {ALGO_LIST.map((a) => {
                    const val = a[m.key] as number
                    const isBest = val === best
                    const barW = m.lowerBetter
                      ? ((m.max - val) / m.max) * 100
                      : (val / m.max) * 100
                    return (
                      <div key={a.id} className="flex items-center gap-2.5">
                        <span className="text-[10px] w-10 text-right tabular-nums" style={{ color: 'rgba(255,255,255,0.4)' }}>{val}</span>
                        <div className="flex-1 h-1.5 rounded-full" style={{ background: 'rgba(255,255,255,0.1)', boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.25)' }}>
                          <div className="h-full rounded-full transition-all duration-700"
                            style={{ width: `${barW}%`, background: a.color, opacity: isBest ? 1 : 0.42 }} />
                        </div>
                        <div className="w-3.5 flex items-center justify-center">
                          {isBest && <svg width="8" height="8" viewBox="0 0 8 8"><circle cx="4" cy="4" r="4" fill="#22C55E" /></svg>}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )
          })}
        </div>
      </GlassCard>
    </div>

    {/* Key Findings */}
    <GlassCard className="p-5">
      <h3 className="text-[13px] font-bold mb-3" style={{ color: 'rgba(255,255,255,0.78)' }}>Key Findings</h3>
      <div className="grid grid-cols-3 gap-5 text-[12px]" style={{ color: 'rgba(255,255,255,0.55)' }}>
        {ALGO_LIST.map((a) => (
          <div key={a.id}>
            <p className="font-semibold mb-1.5" style={{ color: a.color }}>{a.label}</p>
            <p className="leading-relaxed">{a.description}</p>
          </div>
        ))}
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
  const percent = Math.min(1, Math.max(0, value / max))

  return (
    <GlassCard className="group relative z-0 hover:z-20 flex-1 flex flex-col items-center justify-center py-2 px-2 gap-1"
      style={{ background: 'rgba(255,255,255,0.045)', minWidth: 0, cursor: onClick ? 'pointer' : undefined }}
      onClick={onClick}>
      {description && (
        <div className="group/info absolute top-2 right-2 z-20">
          <div
            className="w-5 h-5 rounded-full flex items-center justify-center"
            style={{
              background: 'rgba(6,182,212,0.2)',
              border: '1px solid rgba(6,182,212,0.65)',
              color: 'rgba(217,249,255,0.95)',
              boxShadow: '0 0 0 1px rgba(0,0,0,0.2) inset',
            }}
          >
            <span className="text-[11px] font-bold leading-none">!</span>
          </div>
          <div className="pointer-events-none absolute right-0 top-[calc(100%+8px)] w-[280px] rounded-lg px-3 py-2 text-[10px] leading-relaxed opacity-0 translate-y-1 transition-all duration-150 group-hover/info:opacity-100 group-hover/info:translate-y-0"
            style={{
              background: 'rgba(4,9,22,0.97)',
              border: '1px solid rgba(255,255,255,0.12)',
              color: 'rgba(255,255,255,0.88)',
              boxShadow: '0 12px 24px rgba(0,0,0,0.45)',
              zIndex: 85,
            }}>
            {description}
          </div>
        </div>
      )}

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
        <span className="text-[20px] font-extrabold tabular-nums leading-none" style={{ color: '#ffffff' }}>
          {value}
        </span>
        <span className="text-[10px] font-semibold mt-0.5" style={{ color: 'rgba(255,255,255,0.75)' }}>{unit}</span>
        <span className="text-[9px] font-bold uppercase tracking-wider mt-1.5 text-center"
          style={{ color: 'rgba(255,255,255,0.65)' }}>{label}</span>
      </div>
      {/* View detail badge — only when onClick provided */}
      {onClick && (
        <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-150 absolute bottom-2 left-1/2 -translate-x-1/2 pointer-events-none"
          style={{ whiteSpace: 'nowrap' }}>
          <span className="text-[9px] font-semibold px-2 py-0.5 rounded-full"
            style={{ background: 'rgba(255,255,255,0.1)', border: '1px solid rgba(255,255,255,0.18)', color: 'rgba(255,255,255,0.7)' }}>
            View detail ↗
          </span>
        </div>
      )}
    </GlassCard>
  )
})

// ─── Map Player ───────────────────────────────────────────────────────────────

const MapPlayer = ({ algo, mapSize, onCo2Click, onFuelClick }: { algo: AlgoData; mapSize: string; onCo2Click?: () => void; onFuelClick?: () => void }) => {
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
        <h3 className="text-[13px] font-bold" style={{ color: 'rgba(255,255,255,0.78)' }}>Map Player</h3>
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono px-2 py-0.5 rounded tabular-nums"
            style={{ background: algo.colorDim, color: algo.color, border: `1px solid ${algo.border}` }}>
            {fmt(currentTime)} / {duration > 0 ? fmt(duration) : '--:--'}
          </span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded tabular-nums"
            style={{ background: 'rgba(255,255,255,0.07)', color: 'rgba(255,255,255,0.4)' }}>
            {zoom.toFixed(1)}×
          </span>
          <span className="text-[10px] px-2 py-0.5 rounded"
            style={{ background: 'rgba(255,255,255,0.05)', color: 'rgba(255,255,255,0.35)' }}>
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
          border: '1px solid rgba(255,255,255,0.07)',
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
            src="/simulation.mp4"
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
          style={{ background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.12)' }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.14)'; (e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.22)' }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.07)'; (e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.12)' }}>
          <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.6)"
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
            style={{ height: '4px', background: 'rgba(255,255,255,0.1)', top: '50%', transform: 'translateY(-50%)', transition: 'height 0.12s ease' }}>
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
                color: speed === s ? algo.color : 'rgba(255,255,255,0.32)',
                border: speed === s ? `1px solid ${algo.border}` : '1px solid transparent',
              }}
              onMouseEnter={(e) => { if (s !== speed) { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.08)'; (e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.65)' } }}
              onMouseLeave={(e) => { if (s !== speed) { (e.currentTarget as HTMLElement).style.background = 'transparent'; (e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.32)' } }}>
              {s}×
            </button>
          ))}
        </div>
      </div>

      {/* Emission gauges — flex-1 so they fill remaining height naturally */}
      <div className="flex gap-3 pt-5 mt-2" style={{ borderTop: '1px solid rgba(255,255,255,0.07)' }}>
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
          label="Avg Fuel Consumption"
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
// Maps learning-based algos to their representative heatmap image.
const HEATMAP_IMG: Record<'civiq' | 'qmix', string> = {
  civiq: '/heatmap_output/bgc_full_intersection_based/heatmap_low.png',
  qmix:  '/heatmap_output/bgc_full_intersection_based/heatmap_low.png',
}

// Real selfish routing heatmaps keyed by traffic level
const SELFISH_HEATMAP: Record<string, string> = {
  free_flow:   '/heatmap_output/bgc_full_actual/heatmap_low.png',
  stable_flow: '/heatmap_output/bgc_full_actual/heatmap_med.png',
  forced_flow: '/heatmap_output/bgc_full_actual/heatmap_high.png',
}

const SELFISH_CONGESTION_LABEL: Record<string, string> = {
  free_flow:   'Low Congestion',
  stable_flow: 'Moderate Congestion',
  forced_flow: 'High Congestion',
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
            className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0"
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
                  {Math.max(...algo.congestion.map(p => p.ma)).toFixed(3)}
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

// ─── Congestion Heatmap card (simple — View Detail opens the modal) ────────────

function CongestionHeatmap({ algo, onViewDetail, heatmapSrc, congestionLabel }: {
  algo: AlgoData
  onViewDetail: () => void
  heatmapSrc?: string
  congestionLabel?: string
}) {
  const src = heatmapSrc ?? (algo.id !== 'selfish' ? HEATMAP_IMG[algo.id as 'civiq' | 'qmix'] : SELFISH_HEATMAP.forced_flow)
  const label = congestionLabel ?? (algo.id === 'selfish' ? 'High Congestion' : 'Low Congestion')
  const labelColor = label === 'Low Congestion' ? '#4ADE80' : label === 'Moderate Congestion' ? '#FACC15' : '#F87171'

  return (
    <GlassCard className="p-4 flex flex-col gap-2">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-baseline gap-2">
          <h3 className="text-[13px] font-bold" style={{ color: 'rgba(255,255,255,0.78)' }}>Congestion Heatmap</h3>
          <span className="text-[11px] font-semibold" style={{ color: labelColor }}>{label}</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={onViewDetail}
            className="text-[9px] font-semibold px-2.5 py-1 rounded-full transition-all duration-150"
            style={{ color: algo.color, background: algo.colorDim, border: `1px solid ${algo.border}` }}
            onMouseEnter={e => { (e.currentTarget as HTMLElement).style.filter = 'brightness(1.3)'; (e.currentTarget as HTMLElement).style.transform = 'scale(1.04)' }}
            onMouseLeave={e => { (e.currentTarget as HTMLElement).style.filter = 'none'; (e.currentTarget as HTMLElement).style.transform = 'scale(1)' }}
          >
            View Detail ↗
          </button>
        </div>
      </div>

      {/* Heatmap image */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <div className="rounded-xl overflow-hidden flex-1" style={{ border: '1px solid rgba(255,255,255,0.07)', background: 'rgba(3,7,18,0.75)' }}>
        <img src={src} alt={`${algo.label} congestion heatmap`}
          style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }} />
      </div>

      {/* Legend */}
      <div className="flex items-center gap-2">
        <span className="text-[9px]" style={{ color: 'rgba(255,255,255,0.3)' }}>Low</span>
        <div className="flex-1 h-1.5 rounded-full" style={{ background: 'linear-gradient(to right, #22C55E, #EAB308, #EF4444)', opacity: 0.7 }} />
        <span className="text-[9px]" style={{ color: 'rgba(255,255,255,0.3)' }}>High</span>
      </div>
    </GlassCard>
  )
}

// ─── Selfish Detail Modal types & constants ───────────────────────────────────

type SelfishMetricKey = 'travelTime' | 'waitTime' | 'throughput' | 'co2' | 'fuel'

const CIVIQ_LOS_REF = {
  free_flow:   { travelTime: 2.013, waitTime: 7.4794,  throughput: 1140.18, co2: 461.511, fuel: 19.917 },
  stable_flow: { travelTime: 1.965, waitTime: 10.2765, throughput: 1223.017, co2: 488.249, fuel: 21.057 },
  forced_flow: { travelTime: 1.885, waitTime: 19.3433, throughput: 2243.68,  co2: 525.807, fuel: 22.617 },
} as const

// ─── Episode Detail Modal ─────────────────────────────────────────────────────

type EpisodeMetricKey = keyof EpisodeSeries

const KPI_META: Record<EpisodeMetricKey, { label: string; abbr: string; unit: string }> = {
  travelTime: { label: 'Avg. Travel Time', abbr: 'ATT', unit: 'min' },
  waitTime:   { label: 'Avg. Wait Time',   abbr: 'AWT', unit: 'sec' },
  throughput: { label: 'Throughput',        abbr: 'TPT', unit: 'veh/hr' },
  speed:      { label: 'Real-time Factor',  abbr: 'RTF', unit: 'x' },
}

const EpisodeDetailModal = ({ algo, metricKey, onClose }: {
  algo: AlgoData
  metricKey: EpisodeMetricKey
  onClose: () => void
}) => {
  const meta = KPI_META[metricKey]
  const data = algo.episodes[metricKey]

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null
    const byKey = Object.fromEntries(payload.map((p: any) => [p.dataKey, p.value]))
    return (
      <div style={{ background: 'rgba(4,9,22,0.97)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 10, padding: '8px 12px', fontSize: 11 }}>
        <p style={{ color: 'rgba(255,255,255,0.4)', marginBottom: 4 }}>Episode {label}</p>
        {byKey.value  !== undefined && <p style={{ color: algo.color }}>Value: <b>{byKey.value.toFixed(2)}</b> {meta.unit}</p>}
        {byKey.ma     !== undefined && <p style={{ color: 'rgba(255,255,255,0.7)' }}>MA-10: <b>{byKey.ma.toFixed(2)}</b> {meta.unit}</p>}
        {byKey.hi     !== undefined && byKey.lo !== undefined && (
          <p style={{ color: 'rgba(255,255,255,0.35)' }}>Band: {byKey.lo.toFixed(2)} – {byKey.hi.toFixed(2)}</p>
        )}
      </div>
    )
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: 'rgba(0,0,0,0.72)', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)' }}
      onClick={onClose}
    >
      <div
        className="relative w-full mx-6 rounded-2xl p-6 flex flex-col gap-4"
        style={{
          maxWidth: '780px',
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
              {meta.label}{' '}
              <span style={{ color: algo.color }}>({meta.abbr})</span>
            </h3>
            <p className="text-[12px] mt-0.5" style={{ color: 'rgba(255,255,255,0.35)' }}>
              Per-episode trend · {algo.label} · {data.length} episodes
            </p>
          </div>
          <button
            onClick={onClose}
            className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0"
            style={{ background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.12)' }}
          >
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.55)"
              strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 6L6 18M6 6l12 12"/>
            </svg>
          </button>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-5 flex-wrap">
          <div className="flex items-center gap-2">
            <div className="w-6 h-[2px] rounded" style={{ background: algo.color }} />
            <span className="text-[11px]" style={{ color: 'rgba(255,255,255,0.38)' }}>Per-episode value</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-[2px] rounded" style={{ background: 'rgba(255,255,255,0.65)', borderTop: '2px dashed rgba(255,255,255,0.65)' }} />
            <span className="text-[11px]" style={{ color: 'rgba(255,255,255,0.38)' }}>10-ep. moving avg.</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-5 h-3.5 rounded-sm" style={{ background: algo.color, opacity: 0.18 }} />
            <span className="text-[11px]" style={{ color: 'rgba(255,255,255,0.38)' }}>Confidence band</span>
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
              interval={Math.floor(data.length / 10)}
            />
            <YAxis
              tick={{ fill: 'rgba(255,255,255,0.28)', fontSize: 10 }}
              tickLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
              tickFormatter={(v) => `${v}`}
              width={52}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Confidence band: hi fills to baseline, lo paints over with bg colour */}
            <Area type="monotone" dataKey="hi" stroke="none"
              fill={algo.color} fillOpacity={0.14} dot={false} activeDot={false} legendType="none" isAnimationActive={false} />
            <Area type="monotone" dataKey="lo" stroke="none"
              fill="#040916" fillOpacity={1} dot={false} activeDot={false} legendType="none" isAnimationActive={false} />

            {/* Raw per-episode line */}
            <Line type="monotone" dataKey="value" stroke={algo.color} strokeWidth={1.5}
              dot={false} activeDot={{ r: 3, fill: algo.color }} strokeOpacity={0.85} isAnimationActive={false} />

            {/* Moving average overlay */}
            <Line type="monotone" dataKey="ma" stroke="rgba(255,255,255,0.68)" strokeWidth={2}
              dot={false} activeDot={false} strokeDasharray="5 3" isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>

        {/* Summary stats row */}
        <div className="grid grid-cols-4 gap-3 pt-3" style={{ borderTop: '1px solid rgba(255,255,255,0.07)' }}>
          {[
            { label: 'First ep.', value: data[0]?.value.toFixed(2) },
            { label: 'Last ep.',  value: data[data.length - 1]?.value.toFixed(2) },
            { label: 'Min',       value: Math.min(...data.map(d => d.value)).toFixed(2) },
            { label: 'Max',       value: Math.max(...data.map(d => d.value)).toFixed(2) },
          ].map(({ label, value }) => (
            <div key={label} className="text-center">
              <div className="text-[10px] uppercase tracking-wider mb-0.5" style={{ color: 'rgba(255,255,255,0.28)' }}>{label}</div>
              <div className="text-[15px] font-bold tabular-nums" style={{ color: algo.color }}>{value}</div>
              <div className="text-[10px]" style={{ color: 'rgba(255,255,255,0.28)' }}>{meta.unit}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ─── Analytics Charts ─────────────────────────────────────────────────────────

const CHART_GRID = 'rgba(255,255,255,0.05)'
const CHART_AXIS  = { fill: 'rgba(255,255,255,0.28)', fontSize: 10 }
const CHART_AXIS_LINE = { stroke: 'rgba(255,255,255,0.08)' }
const CHART_TICK_LINE = { stroke: 'rgba(255,255,255,0.08)' }

// Shared tooltip shell
const ChartTooltip = ({ active, payload, label, xLabel, rows }: {
  active?: boolean; payload?: any[]; label?: any
  xLabel: string
  rows: { key: string; label: string; color: string; unit: string }[]
}) => {
  if (!active || !payload?.length) return null
  const byKey = Object.fromEntries(payload.map((p: any) => [p.dataKey, p.value]))
  return (
    <div style={{ background: 'rgba(4,9,22,0.97)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 10, padding: '8px 12px', fontSize: 11 }}>
      <p style={{ color: 'rgba(255,255,255,0.38)', marginBottom: 4 }}>{xLabel} {label}</p>
      {rows.map(({ key, label: rl, color, unit }) =>
        byKey[key] !== undefined
          ? <p key={key} style={{ color }}>{rl}: <b>{Number(byKey[key]).toFixed(2)}</b> {unit}</p>
          : null
      )}
    </div>
  )
}

// 1 — Training Curve
const TrainingCurveChart = ({ algo }: { algo: AlgoData }) => {
  const data = algo.system.training
  if (!data.length) return (
    <GlassCard className="p-5 flex flex-col gap-2 col-span-3">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-[13px] font-bold" style={{ color: 'rgba(255,255,255,0.78)' }}>Training Curve</span>
        <span className="text-[10px] px-2 py-0.5 rounded" style={{ background: 'rgba(248,113,113,0.12)', color: '#F87171', border: '1px solid rgba(248,113,113,0.25)' }}>No training phase</span>
      </div>
      <p className="text-[12px]" style={{ color: 'rgba(255,255,255,0.35)' }}>
        {algo.id === 'selfish'
          ? 'Selfish Routing is a rule-based baseline — it requires no training and has no reward curve.'
          : 'Training curve data is not available for this scenario.'}
      </p>
    </GlassCard>
  )
  return (
    <GlassCard className="p-5 flex flex-col gap-3 col-span-3">
      <div className="flex items-center justify-between">
        <div>
          <span className="text-[13px] font-bold" style={{ color: 'rgba(255,255,255,0.78)' }}>Training Curve</span>
          <p className="text-[11px] mt-0.5" style={{ color: 'rgba(255,255,255,0.32)' }}>
            Cumulative reward per episode · shaded area = seed confidence band
          </p>
        </div>
        <div className="flex items-center gap-4">
          {[
            { color: algo.color, label: 'Mean reward' },
            { color: 'rgba(255,255,255,0.6)', label: 'Moving avg.', dashed: true },
          ].map(({ color, label, dashed }) => (
            <div key={label} className="flex items-center gap-1.5">
              <svg width="20" height="8"><line x1="0" y1="4" x2="20" y2="4" stroke={color} strokeWidth="2" strokeDasharray={dashed ? '4 2' : undefined} /></svg>
              <span className="text-[10px]" style={{ color: 'rgba(255,255,255,0.35)' }}>{label}</span>
            </div>
          ))}
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-3 rounded-sm" style={{ background: algo.color, opacity: 0.18 }} />
            <span className="text-[10px]" style={{ color: 'rgba(255,255,255,0.35)' }}>Conf. band</span>
          </div>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={data} margin={{ top: 4, right: 12, left: 0, bottom: 20 }}>
          <CartesianGrid stroke={CHART_GRID} strokeDasharray="3 4" />
          <XAxis dataKey="episode" tick={CHART_AXIS} tickLine={CHART_TICK_LINE} axisLine={CHART_AXIS_LINE}
            label={{ value: 'Episode', position: 'insideBottom', offset: -12, fill: 'rgba(255,255,255,0.28)', fontSize: 11 }}
            interval={Math.floor(data.length / 8)} />
          <YAxis tick={CHART_AXIS} tickLine={CHART_TICK_LINE} axisLine={CHART_AXIS_LINE} width={52}
            label={{ value: 'Reward', angle: -90, position: 'insideLeft', offset: 14, fill: 'rgba(255,255,255,0.28)', fontSize: 11 }} />
          <Tooltip content={<ChartTooltip xLabel="Ep." rows={[
            { key: 'reward', label: 'Reward', color: algo.color, unit: '' },
            { key: 'ma',     label: 'MA-10',  color: 'rgba(255,255,255,0.65)', unit: '' },
          ]} />} />
          <Area type="monotone" dataKey="hi" stroke="none" fill={algo.color} fillOpacity={0.14} dot={false} activeDot={false} legendType="none" isAnimationActive={false} />
          <Area type="monotone" dataKey="lo" stroke="none" fill="#040916" fillOpacity={1} dot={false} activeDot={false} legendType="none" isAnimationActive={false} />
          <Line type="monotone" dataKey="reward" stroke={algo.color} strokeWidth={1.5} dot={false} activeDot={{ r: 3 }} strokeOpacity={0.85} isAnimationActive={false} />
          <Line type="monotone" dataKey="ma" stroke="rgba(255,255,255,0.6)" strokeWidth={2} dot={false} activeDot={false} strokeDasharray="5 3" isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </GlassCard>
  )
}

// ─── Shared info bubble (matches KpiCard / GaugeChart tooltip style) ──────────

function InfoBubble({ text, side = 'left' }: { text: string; side?: 'left' | 'right' }) {
  return (
    <div className="group/info relative z-20 flex-shrink-0">
      <div className="w-[18px] h-[18px] rounded-full flex items-center justify-center cursor-default"
        style={{ background: 'rgba(6,182,212,0.2)', border: '1px solid rgba(6,182,212,0.65)', color: 'rgba(217,249,255,0.95)', boxShadow: '0 0 0 1px rgba(0,0,0,0.2) inset' }}>
        <span className="text-[10px] font-bold leading-none">!</span>
      </div>
      <div className="pointer-events-none absolute top-[calc(100%+8px)] w-[260px] rounded-lg px-3 py-2 text-[10px] leading-relaxed opacity-0 translate-y-1 transition-all duration-150 group-hover/info:opacity-100 group-hover/info:translate-y-0"
        style={{
          ...(side === 'right' ? { right: 0 } : { left: 0 }),
          background: 'rgba(4,9,22,0.97)', border: '1px solid rgba(255,255,255,0.12)',
          color: 'rgba(255,255,255,0.88)', boxShadow: '0 12px 24px rgba(0,0,0,0.45)', zIndex: 90,
        }}>
        {text}
      </div>
    </div>
  )
}

// ─── MARL Training Diagnostics ────────────────────────────────────────────────

function MarlMetricsSection({ algo }: { algo: AlgoData }) {
  const data = algo.marl
  if (!data.length) return null  // not rendered for selfish (no training)

  const xAxis = (
    <XAxis dataKey="episode" tick={{ ...CHART_AXIS, fontSize: 9 }} tickLine={CHART_TICK_LINE} axisLine={CHART_AXIS_LINE}
      label={{ value: 'Episode', position: 'insideBottom', offset: -10, fill: 'rgba(255,255,255,0.28)', fontSize: 10 }}
      interval={Math.floor(data.length / 5)} />
  )

  return (
    <>
      {/* Section header */}
      <div className="flex items-center gap-3">
        <span className="text-[13px] font-bold" style={{ color: 'rgba(255,255,255,0.55)' }}>{algo.id === 'civiq' ? 'HMARL' : 'MARL'} Training Diagnostics</span>
        <div className="flex-1 h-px" style={{ background: 'rgba(255,255,255,0.07)' }} />
        <span className="text-[10px] px-2 py-0.5 rounded"
          style={{ background: algo.colorDim, color: algo.color, border: `1px solid ${algo.border}` }}>
          {data.length} episodes
        </span>
      </div>

      {/* ── 1. Episode Cumulative Reward (full-width, prominent) ── */}
      <GlassCard className="p-5 flex flex-col gap-3">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[13px] font-bold" style={{ color: 'rgba(255,255,255,0.78)' }}>Episode Cumulative Reward</span>
              <InfoBubble text="Total reward earned by all agents during each training episode. A rising curve that plateaus signals policy convergence. The shaded band shows the inter-seed confidence interval; the dashed line is a 10-episode moving average." />
            </div>
            <p className="text-[11px] mt-0.5" style={{ color: 'rgba(255,255,255,0.32)' }}>
              Primary training health indicator · shaded = seed confidence band
            </p>
          </div>
          <div className="flex items-center gap-4 flex-shrink-0">
            {[
              { color: algo.color, label: 'Per-episode reward' },
              { color: 'rgba(255,255,255,0.65)', label: 'MA-10', dashed: true },
            ].map(({ color, label, dashed }) => (
              <div key={label} className="flex items-center gap-1.5">
                <svg width="18" height="6"><line x1="0" y1="3" x2="18" y2="3" stroke={color} strokeWidth="2" strokeDasharray={dashed ? '4 2' : undefined} /></svg>
                <span className="text-[10px]" style={{ color: 'rgba(255,255,255,0.35)' }}>{label}</span>
              </div>
            ))}
            <div className="flex items-center gap-1.5">
              <div className="w-4 h-3 rounded-sm" style={{ background: algo.color, opacity: 0.18 }} />
              <span className="text-[10px]" style={{ color: 'rgba(255,255,255,0.35)' }}>Conf. band</span>
            </div>
          </div>
        </div>
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={data} margin={{ top: 4, right: 12, left: 0, bottom: 20 }}>
            <CartesianGrid stroke={CHART_GRID} strokeDasharray="3 4" />
            {xAxis}
            <YAxis tick={CHART_AXIS} tickLine={CHART_TICK_LINE} axisLine={CHART_AXIS_LINE} width={52}
              label={{ value: 'Reward', angle: -90, position: 'insideLeft', offset: 14, fill: 'rgba(255,255,255,0.28)', fontSize: 11 }} />
            <Tooltip content={<ChartTooltip xLabel="Ep." rows={[
              { key: 'reward',    label: 'Reward', color: algo.color, unit: '' },
              { key: 'rewardMa',  label: 'MA-10',  color: 'rgba(255,255,255,0.65)', unit: '' },
            ]} />} />
            <Area type="monotone" dataKey="rewardHi" stroke="none" fill={algo.color} fillOpacity={0.14} dot={false} activeDot={false} legendType="none" isAnimationActive={false} />
            <Area type="monotone" dataKey="rewardLo" stroke="none" fill="#040916" fillOpacity={1} dot={false} activeDot={false} legendType="none" isAnimationActive={false} />
            <Line type="monotone" dataKey="reward" stroke={algo.color} strokeWidth={1.5} dot={false} activeDot={{ r: 3 }} strokeOpacity={0.85} isAnimationActive={false} />
            <Line type="monotone" dataKey="rewardMa" stroke="rgba(255,255,255,0.65)" strokeWidth={2} dot={false} activeDot={false} strokeDasharray="5 3" isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </GlassCard>

      {/* ── 2–4. Bottom row: TD Loss | Q-values | Gradient Norm ── */}
      <div className="grid grid-cols-3 gap-4">

        {/* TD Loss — log Y-axis */}
        <GlassCard className="p-4 flex flex-col gap-2">
          <div className="flex items-start gap-2">
            <div className="flex-1">
              <span className="text-[12px] font-bold" style={{ color: 'rgba(255,255,255,0.78)' }}>TD Loss</span>
              <p className="text-[11px] mt-0.5" style={{ color: 'rgba(255,255,255,0.32)' }}>Log scale · Q-net + mixing network</p>
            </div>
            <InfoBubble side="right" text="Temporal-difference error used to update the Q-network and QMIX mixing network. Loss drops sharply early then flattens — a log Y-axis keeps the full curve readable instead of a flat line after the initial drop." />
          </div>
          <ResponsiveContainer width="100%" height={170}>
            <ComposedChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 16 }}>
              <CartesianGrid stroke={CHART_GRID} strokeDasharray="3 4" />
              {xAxis}
              <YAxis tick={{ ...CHART_AXIS, fontSize: 9 }} tickLine={CHART_TICK_LINE} axisLine={CHART_AXIS_LINE}
                width={44} scale="log" domain={[0.0001, 'auto']}
                tickFormatter={(v) => v < 0.01 ? v.toExponential(0) : String(+v.toFixed(3))} />
              <Tooltip content={<ChartTooltip xLabel="Ep." rows={[{ key: 'tdLoss', label: 'TD Loss', color: '#FB923C', unit: '' }]} />} />
              <Area type="monotone" dataKey="tdLoss" stroke="#FB923C" strokeWidth={1.5}
                fill="#FB923C" fillOpacity={0.12} dot={false} activeDot={{ r: 3 }} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </GlassCard>

        {/* Q-Value Estimates — two series + std band */}
        <GlassCard className="p-4 flex flex-col gap-2">
          <div className="flex items-start gap-2">
            <div className="flex-1">
              <span className="text-[12px] font-bold" style={{ color: 'rgba(255,255,255,0.78)' }}>Q-Value Estimates</span>
              <p className="text-[11px] mt-0.5" style={{ color: 'rgba(255,255,255,0.32)' }}>q_taken vs target · ± std band</p>
            </div>
            <InfoBubble side="right" text="q_taken_mean (solid) is the Q-value for actions actually taken; target_mean (dashed) is the Bellman target. A persistent gap between them indicates overestimation bias. The shaded band is ±1 std of q_taken across agents." />
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            {[
              { color: '#A78BFA', label: 'q_taken_mean' },
              { color: 'rgba(167,139,250,0.45)', label: 'target_mean', dashed: true },
            ].map(({ color, label, dashed }) => (
              <div key={label} className="flex items-center gap-1">
                <svg width="14" height="5"><line x1="0" y1="2.5" x2="14" y2="2.5" stroke={color} strokeWidth="1.8" strokeDasharray={dashed ? '3 2' : undefined} /></svg>
                <span className="text-[9px]" style={{ color: 'rgba(255,255,255,0.32)' }}>{label}</span>
              </div>
            ))}
          </div>
          <ResponsiveContainer width="100%" height={150}>
            <ComposedChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 16 }}>
              <CartesianGrid stroke={CHART_GRID} strokeDasharray="3 4" />
              {xAxis}
              <YAxis tick={{ ...CHART_AXIS, fontSize: 9 }} tickLine={CHART_TICK_LINE} axisLine={CHART_AXIS_LINE} width={40} />
              <Tooltip content={<ChartTooltip xLabel="Ep." rows={[
                { key: 'qTakenMean', label: 'q_taken', color: '#A78BFA', unit: '' },
                { key: 'targetMean', label: 'target',  color: 'rgba(167,139,250,0.6)', unit: '' },
              ]} />} />
              {/* ±std band around q_taken */}
              <Area type="monotone" dataKey="qTakenHi" stroke="none" fill="#A78BFA" fillOpacity={0.14} dot={false} activeDot={false} legendType="none" isAnimationActive={false} />
              <Area type="monotone" dataKey="qTakenLo" stroke="none" fill="#040916" fillOpacity={1}  dot={false} activeDot={false} legendType="none" isAnimationActive={false} />
              <Line type="monotone" dataKey="qTakenMean" stroke="#A78BFA" strokeWidth={2} dot={false} activeDot={{ r: 3 }} isAnimationActive={false} />
              <Line type="monotone" dataKey="targetMean" stroke="rgba(167,139,250,0.5)" strokeWidth={1.5} dot={false} activeDot={false} strokeDasharray="4 2" isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </GlassCard>

        {/* Gradient Norm — log Y-axis + reference line */}
        <GlassCard className="p-4 flex flex-col gap-2">
          <div className="flex items-start gap-2">
            <div className="flex-1">
              <span className="text-[12px] font-bold" style={{ color: 'rgba(255,255,255,0.78)' }}>Gradient Norm</span>
              <p className="text-[11px] mt-0.5" style={{ color: 'rgba(255,255,255,0.32)' }}>Log scale · instability detector</p>
            </div>
            <InfoBubble side="right" text="L2 norm of the policy gradient per episode. Large early spikes are expected; values should stabilise as training converges. Persistent large norms indicate instability. The dashed line marks the target healthy threshold." />
          </div>
          <ResponsiveContainer width="100%" height={170}>
            <ComposedChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 16 }}>
              <CartesianGrid stroke={CHART_GRID} strokeDasharray="3 4" />
              {xAxis}
              <YAxis tick={{ ...CHART_AXIS, fontSize: 9 }} tickLine={CHART_TICK_LINE} axisLine={CHART_AXIS_LINE}
                width={44} scale="log" domain={[0.0001, 'auto']}
                tickFormatter={(v) => v < 0.01 ? v.toExponential(0) : String(+v.toFixed(3))} />
              <Tooltip content={<ChartTooltip xLabel="Ep." rows={[{ key: 'gradNorm', label: 'Grad norm', color: '#34D399', unit: '' }]} />} />
              <ReferenceLine y={0.5} stroke="rgba(255,255,255,0.2)" strokeDasharray="5 3"
                label={{ value: 'healthy ≤ 0.5', position: 'insideTopRight', fill: 'rgba(255,255,255,0.28)', fontSize: 8 }} />
              <Area type="monotone" dataKey="gradNorm" stroke="#34D399" strokeWidth={1.5}
                fill="#34D399" fillOpacity={0.11} dot={false} activeDot={{ r: 3 }} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </GlassCard>
      </div>
    </>
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
      'Decision Latency (ms)', 'Convergence Episode', 'Cumulative Reward', 'Efficiency (%)',
    ]
    const row = [
      algo.id, algo.label,
      algo.travelTime, algo.waitTime, algo.throughput,
      algo.speed, algo.co2, algo.fuel,
      algo.computeTime, algo.convergence ?? 'N/A', algo.reward ?? 'N/A', algo.efficiency,
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
      'Speed (x)', 'Speed MA',
    ]
    const n = algo.episodes.travelTime.length
    const rows = Array.from({ length: n }, (_, i) => [
      algo.episodes.travelTime[i].episode,
      algo.episodes.travelTime[i].value, algo.episodes.travelTime[i].ma,
      algo.episodes.waitTime[i]?.value ?? '', algo.episodes.waitTime[i]?.ma ?? '',
      algo.episodes.throughput[i]?.value ?? '', algo.episodes.throughput[i]?.ma ?? '',
      algo.episodes.speed[i]?.value ?? '', algo.episodes.speed[i]?.ma ?? '',
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

  // ── CSV: traffic analytics (queue / density / congestion) — learning-based algos ──
  const exportTrafficCsv = () => {
    const headers = [
      'Step',
      'Queue Int-A (veh)', 'Queue Int-B (veh)', 'Queue Int-C (veh)', 'Queue Int-D (veh)',
      'Occupancy (%)', 'Occupancy MA',
      'Congestion Index', 'Congestion MA',
    ]
    const n = algo.queue.length
    const rows = Array.from({ length: n }, (_, i) => [
      algo.queue[i].step,
      algo.queue[i].int1, algo.queue[i].int2, algo.queue[i].int3, algo.queue[i].int4,
      algo.density[i]?.density ?? '', algo.density[i]?.ma ?? '',
      algo.congestion[i]?.index ?? '', algo.congestion[i]?.ma ?? '',
    ])
    downloadFile(
      `civiq_${algo.id}_traffic.csv`,
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
        speed: algo.speed, co2: algo.co2, fuel: algo.fuel, computeTime: algo.computeTime,
        convergenceEpisode: algo.convergence, cumulativeReward: algo.reward, efficiency: algo.efficiency,
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
      payload.traffic = { queue: algo.queue, density: algo.density, congestion: algo.congestion }
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
    // Traffic data: only for learning-based algos (queue/density/congestion are synthetic for selfish)
    ...(!isSelfish
      ? [{ label: 'Traffic Data', sub: 'Queue, occupancy & congestion', tag: 'CSV', action: exportTrafficCsv }]
      : []
    ),
    ...(algo.system.training.length
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
          background: open ? 'rgba(255,255,255,0.11)' : 'rgba(255,255,255,0.07)',
          border: `1px solid ${open ? 'rgba(255,255,255,0.2)' : 'rgba(255,255,255,0.12)'}`,
          color: open ? 'rgba(255,255,255,0.88)' : 'rgba(255,255,255,0.62)',
        }}
        onMouseEnter={e => { if (!open) { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.1)'; (e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.82)' } }}
        onMouseLeave={e => { if (!open) { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.07)'; (e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.62)' } }}
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

function useSelfishRealData(enabled: boolean, trafficLevel: string, mapSize: string) {
  const [kpis, setKpis] = useState<SelfishApiKpis | null>(null)
  const [timeseries, setTimeseries] = useState<SelfishTimeseries | null>(null)
  const [civiq, setCiviq] = useState<SelfishCiviqBaseline | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!enabled || !trafficLevel || !mapSize) return
    setKpis(null)
    setTimeseries(null)
    setCiviq(null)
    setLoading(true)
    fetch(`/api/selfish?trafficLevel=${trafficLevel}&map=${mapSize}`)
      .then(r => r.json())
      .then(data => {
        if (data.success) {
          setKpis(data.kpis)
          setTimeseries(data.timeseries)
          setCiviq(data.baselines?.civiq ?? null)
        }
        // On error (e.g. 4x4 map, missing forced_flow for bgc_core) silently fall back to static data
      })
      .catch(() => {/* silently fall back to static data */})
      .finally(() => setLoading(false))
  }, [enabled, trafficLevel, mapSize])

  return { kpis, timeseries, civiq, loading }
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

interface QmixTrainPoint {
  t: number
  return: number
  travelTime_s: number | null
  waitTime_s: number | null
  arrivalRate: number | null
  speedMs: number | null
  totalStops: number | null
  loss: number | null
  gradNorm: number | null
  qTaken: number | null
  qTakenStd: number | null
  targetMean: number | null
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
  training: { note: string; curve: QmixTrainingEntry[] }
  trainMetrics?: { note: string; curve: QmixTrainPoint[] }
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

/** Convert trainMetrics curve → TrainingPoint[] with moving-average confidence bands */
function qmixToTrainingPoints(curve: QmixTrainPoint[]): TrainingPoint[] {
  if (!curve.length) return []
  const W = 20
  const rewards = curve.map(p => p.return)
  const ma = rewards.map((_, i) => {
    const win = rewards.slice(Math.max(0, i - W), i + 1)
    return win.reduce((a, b) => a + b, 0) / win.length
  })
  return curve.map((p, i) => {
    const win = rewards.slice(Math.max(0, i - W), i + 1)
    const std = win.length > 1
      ? Math.sqrt(win.reduce((s, v) => s + (v - ma[i]) ** 2, 0) / win.length)
      : 500
    return { episode: Math.round(p.t / 100), reward: p.return, lo: p.return - std, hi: p.return + std, ma: ma[i] }
  })
}

/** Convert trainMetrics curve → MarlPoint[] using real loss/gradNorm/qTaken values */
function qmixToMarlPoints(curve: QmixTrainPoint[]): MarlPoint[] {
  if (!curve.length) return []
  const W = 10
  const rewards = curve.map(p => p.return)
  const ma = rewards.map((_, i) => {
    const win = rewards.slice(Math.max(0, i - W), i + 1)
    return win.reduce((a, b) => a + b, 0) / win.length
  })
  return curve.map((p, i) => {
    const win = rewards.slice(Math.max(0, i - W), i + 1)
    const band = win.length > 1
      ? Math.sqrt(win.reduce((s, v) => s + (v - ma[i]) ** 2, 0) / win.length)
      : 500
    const qStd = p.qTakenStd ?? 1.07
    return {
      episode:    Math.round(p.t / 100),
      reward:     p.return,
      rewardLo:   p.return - band,
      rewardHi:   p.return + band,
      rewardMa:   ma[i],
      tdLoss:     p.loss      ?? 0,
      gradNorm:   p.gradNorm  ?? 0,
      qTakenMean: p.qTaken    ?? 0,
      qTakenLo:   (p.qTaken ?? 0) - qStd,
      qTakenHi:   (p.qTaken ?? 0) + qStd,
      targetMean: p.targetMean ?? 0,
    }
  })
}

/** Convert trainMetrics training-progress curve → EpisodeSeries for detail modal */
function qmixToEpisodeSeries(data: QmixRealData): EpisodeSeries {
  const curve = data.trainMetrics?.curve
  if (curve && curve.length > 0) {
    const W = 10
    const throughputScale = data.kpis.arrivalRate > 0
      ? data.kpis.throughput / data.kpis.arrivalRate
      : 1
    function trainPoints(vals: (number | null)[], scale = 1): EpisodePoint[] {
      const nonNull = vals.reduce<{ v: number; ep: number }[]>((acc, v, i) => {
        if (v !== null) acc.push({ v: v * scale, ep: Math.round(curve![i].t / 100) })
        return acc
      }, [])
      return nonNull.map(({ v, ep }, j) => {
        const win = nonNull.slice(Math.max(0, j - W), j + 1).map(x => x.v)
        const avg = win.reduce((a, b) => a + b, 0) / win.length
        const std = win.length > 1
          ? Math.sqrt(win.reduce((s, x) => s + (x - avg) ** 2, 0) / win.length)
          : 0
        return { episode: ep, value: parseFloat(v.toFixed(3)), lo: parseFloat((v - std).toFixed(3)), hi: parseFloat((v + std).toFixed(3)), ma: parseFloat(avg.toFixed(3)) }
      })
    }
    return {
      travelTime: trainPoints(curve.map(p => p.travelTime_s), 1 / 60),
      waitTime:   trainPoints(curve.map(p => p.waitTime_s)),
      throughput: trainPoints(curve.map(p => p.arrivalRate !== null ? p.arrivalRate * throughputScale : null)),
      speed:      trainPoints(curve.map(p => p.speedMs !== null ? p.speedMs * 3.6 : null)),
    }
  }
  // Fallback: 30 post-training eval episodes
  const W = 5
  function evalPoints(raw: number[], scale = 1): EpisodePoint[] {
    const vals = raw.map(v => v * scale)
    return vals.map((v, i) => {
      const win = vals.slice(Math.max(0, i - W), i + 1)
      const avg = win.reduce((a, b) => a + b, 0) / win.length
      const std = win.length > 1 ? Math.sqrt(win.reduce((s, x) => s + (x - avg) ** 2, 0) / win.length) : 0
      return { episode: i + 1, value: parseFloat(v.toFixed(3)), lo: parseFloat((v - std).toFixed(3)), hi: parseFloat((v + std).toFixed(3)), ma: parseFloat(avg.toFixed(3)) }
    })
  }
  return {
    travelTime: evalPoints(data.evalTravelTimes, 1 / 60),
    waitTime:   evalPoints(data.evalWaitingTimes),
    throughput: evalPoints(data.evalThroughputs),
    speed:      evalPoints(data.evalReturns.map(() => data.kpis.realTimeFactor ?? 53.97)),
  }
}

// ─── Selfish Detail Modal ─────────────────────────────────────────────────────

const LOS_LABELS: Record<string, string> = {
  free_flow: 'LOS A',
  stable_flow: 'LOS C',
  forced_flow: 'LOS E',
}

const SELFISH_METRIC_META: Record<SelfishMetricKey, { label: string; unit: string; lowerBetter: boolean }> = {
  travelTime: { label: 'Avg. Travel Time',      unit: 'min',     lowerBetter: true  },
  waitTime:   { label: 'Avg. Waiting Time',     unit: 'sec',     lowerBetter: true  },
  throughput: { label: 'Network Throughput',    unit: 'veh/hr',  lowerBetter: false },
  co2:        { label: 'Avg. CO₂ Emissions',    unit: 'g/km',    lowerBetter: true  },
  fuel:       { label: 'Avg. Fuel Consumption', unit: 'L/100km', lowerBetter: true  },
}

const SELFISH_CO2_EPISODES  = makeSeries(485.6, 485.6, 15,  10,  10000, 13)
const SELFISH_FUEL_EPISODES = makeSeries(20.89, 20.89, 0.8, 0.5, 10000, 14)

const SelfishDetailModal = ({ metricKey, algoColor, data, onClose }: {
  metricKey: SelfishMetricKey
  algoColor: string
  data: EpisodePoint[]
  onClose: () => void
}) => {
  const meta = SELFISH_METRIC_META[metricKey]

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null
    const byKey = Object.fromEntries(payload.map((p: any) => [p.dataKey, p.value]))
    return (
      <div style={{ background: 'rgba(4,9,22,0.97)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 10, padding: '8px 12px', fontSize: 11 }}>
        <p style={{ color: 'rgba(255,255,255,0.4)', marginBottom: 4 }}>Episode {label}</p>
        {byKey.value !== undefined && <p style={{ color: algoColor }}>Value: <b>{byKey.value.toFixed(2)}</b> {meta.unit}</p>}
        {byKey.ma    !== undefined && <p style={{ color: 'rgba(255,255,255,0.7)' }}>MA-10: <b>{byKey.ma.toFixed(2)}</b> {meta.unit}</p>}
        {byKey.hi    !== undefined && byKey.lo !== undefined && (
          <p style={{ color: 'rgba(255,255,255,0.35)' }}>Band: {byKey.lo.toFixed(2)} – {byKey.hi.toFixed(2)}</p>
        )}
      </div>
    )
  }

  const step = Math.max(1, Math.floor(data.length / 200))
  const sampled = data.filter((_, i) => i % step === 0)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: 'rgba(0,0,0,0.72)', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)' }}
      onClick={onClose}>
      <div className="relative w-full mx-6 rounded-2xl p-6 flex flex-col gap-4"
        style={{ maxWidth: '780px', background: 'rgba(4,9,22,0.97)', border: '1px solid rgba(255,255,255,0.13)', boxShadow: '0 32px 80px rgba(0,0,0,0.85), inset 0 1px 0 rgba(255,255,255,0.08)' }}
        onClick={e => e.stopPropagation()}>

        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-[17px] font-bold leading-tight" style={{ color: 'rgba(255,255,255,0.92)' }}>
              {meta.label} <span style={{ color: algoColor }}>· Selfish Routing</span>
            </h3>
            <p className="text-[12px] mt-0.5" style={{ color: 'rgba(255,255,255,0.35)' }}>
              Per-episode trend · Selfish Routing · {data.length.toLocaleString()} episodes
            </p>
          </div>
          <button onClick={onClose}
            className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0"
            style={{ background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.12)' }}>
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.55)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 6L6 18M6 6l12 12"/>
            </svg>
          </button>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-5 flex-wrap">
          <div className="flex items-center gap-2">
            <div className="w-6 h-[2px] rounded" style={{ background: algoColor }} />
            <span className="text-[11px]" style={{ color: 'rgba(255,255,255,0.38)' }}>Per-episode value</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-[2px] rounded" style={{ background: 'rgba(255,255,255,0.65)', borderTop: '2px dashed rgba(255,255,255,0.65)' }} />
            <span className="text-[11px]" style={{ color: 'rgba(255,255,255,0.38)' }}>10-ep. moving avg.</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-5 h-3.5 rounded-sm" style={{ background: algoColor, opacity: 0.18 }} />
            <span className="text-[11px]" style={{ color: 'rgba(255,255,255,0.38)' }}>Confidence band</span>
          </div>
        </div>

        {/* Chart */}
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={sampled} margin={{ top: 8, right: 12, left: 0, bottom: 20 }}>
            <CartesianGrid stroke="rgba(255,255,255,0.05)" strokeDasharray="3 4" />
            <XAxis dataKey="episode"
              tick={{ fill: 'rgba(255,255,255,0.28)', fontSize: 10 }}
              tickLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
              label={{ value: 'Episode', position: 'insideBottom', offset: -12, fill: 'rgba(255,255,255,0.28)', fontSize: 11 }}
              interval={Math.floor(sampled.length / 10)}
            />
            <YAxis tick={{ fill: 'rgba(255,255,255,0.28)', fontSize: 10 }}
              tickLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
              tickFormatter={v => `${v}`} width={52} />
            <Tooltip content={<CustomTooltip />} />
            <Area type="monotone" dataKey="hi" stroke="none"
              fill={algoColor} fillOpacity={0.12} legendType="none" isAnimationActive={false} />
            <Area type="monotone" dataKey="lo" stroke="none"
              fill="rgba(4,9,22,1)" fillOpacity={1} legendType="none" isAnimationActive={false} />
            <Line type="monotone" dataKey="value" stroke={algoColor} strokeWidth={1.5}
              dot={false} activeDot={{ r: 3 }} isAnimationActive={false} />
            <Line type="monotone" dataKey="ma" stroke="rgba(255,255,255,0.65)" strokeWidth={1.5}
              dot={false} activeDot={false} strokeDasharray="5 3" isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>

        {/* Summary stats */}
        <div className="grid grid-cols-4 gap-3 pt-3" style={{ borderTop: '1px solid rgba(255,255,255,0.07)' }}>
          {[
            { label: 'First ep.', value: data[0]?.value.toFixed(2) },
            { label: 'Last ep.',  value: data[data.length - 1]?.value.toFixed(2) },
            { label: 'Min',       value: Math.min(...sampled.map(d => d.value)).toFixed(2) },
            { label: 'Max',       value: Math.max(...sampled.map(d => d.value)).toFixed(2) },
          ].map(({ label, value }) => (
            <div key={label} className="text-center">
              <div className="text-[10px] uppercase tracking-wider mb-0.5" style={{ color: 'rgba(255,255,255,0.28)' }}>{label}</div>
              <div className="text-[15px] font-bold tabular-nums" style={{ color: algoColor }}>{value}</div>
              <div className="text-[10px]" style={{ color: 'rgba(255,255,255,0.28)' }}>{meta.unit}</div>
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
  const [openModal, setOpenModal] = useState<EpisodeMetricKey | null>(null)
  const [congestionDetail, setCongestionDetail] = useState(false)
  const [selfishModal, setSelfishModal] = useState<SelfishMetricKey | null>(null)

  // For selfish routing, fetch real simulation data and overlay onto static algo object
  const isSelfish = algo.id === 'selfish'
  const { kpis: realKpis, timeseries: realTimeseries, civiq: selfishCiviq, loading: kpisLoading } = useSelfishRealData(isSelfish, trafficScale, mapSize)
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
        sparklines: {
          travelTime: selfishSparkline('travelTime'),
          waitTime:   selfishSparkline('waitTime'),
          throughput: selfishSparkline('throughput'),
          speed:      algo.sparklines.speed.map(() => realKpis.speed),
        },
        changes: selfishCiviq ? {
          travelTime: changePct(travelTime_s,        selfishCiviq.travelTime_s),
          waitTime:   changePct(realKpis.waitTime,   selfishCiviq.waitTime_s),
          throughput: changePct(realKpis.throughput, selfishCiviq.throughput),
          speed: 0,
        } : algo.changes,
      }
    }
    if (isQmix && qmixData) {
      const k = qmixData.kpis
      const noop = qmixData.baselines?.noop
      const changePct = (val: number, ref: number) => parseFloat(((val - ref) / ref * 100).toFixed(1))
      return {
        ...algo,
        // Override KPIs with scenario-specific real values
        travelTime: parseFloat((k.travelTime_s / 60).toFixed(3)),
        waitTime:   k.waitTime_s,
        throughput: k.throughput,
        speed:      k.realTimeFactor ?? algo.speed,
        co2:        k.co2,
        fuel:       k.fuel,
        reward:     k.returnMean,
        sparklines: {
          travelTime: algo.sparklines.travelTime.map(() => parseFloat((k.travelTime_s / 60).toFixed(3))),
          waitTime:   algo.sparklines.waitTime.map(()   => k.waitTime_s),
          throughput: algo.sparklines.throughput.map(() => k.throughput),
          speed:      algo.sparklines.speed.map(()      => k.realTimeFactor ?? algo.speed),
        },
        changes: noop ? {
          travelTime: changePct(k.travelTime_s, noop.travelTime_s),
          waitTime:   changePct(k.waitTime_s,   noop.waitTime_s),
          throughput: changePct(k.throughput,   noop.throughput),
          speed: 0,
        } : algo.changes,
        system: {
          ...algo.system,
          training: qmixToTrainingPoints(qmixData.trainMetrics?.curve ?? []),
          cpu: makeCpu(k.cpuMean, Math.min(k.cpuPeak, 120), 8, 120, 18),
        },
        marl:     qmixToMarlPoints(qmixData.trainMetrics?.curve ?? []),
        episodes: qmixToEpisodeSeries(qmixData),
      }
    }
    if (isCiviq && civiqData) {
      const k = civiqData.kpis
      const noop = civiqData.baselines?.noop
      const changePct = (val: number, ref: number) => parseFloat(((val - ref) / ref * 100).toFixed(1))
      return {
        ...algo,
        travelTime: parseFloat((k.travelTime_s / 60).toFixed(3)),
        waitTime:   k.waitTime_s,
        throughput: k.throughput,
        speed:      k.realTimeFactor ?? algo.speed,
        co2:        k.co2,
        fuel:       k.fuel,
        reward:     k.returnMean,
        sparklines: {
          travelTime: algo.sparklines.travelTime.map(() => parseFloat((k.travelTime_s / 60).toFixed(3))),
          waitTime:   algo.sparklines.waitTime.map(()   => k.waitTime_s),
          throughput: algo.sparklines.throughput.map(() => k.throughput),
          speed:      algo.sparklines.speed.map(()      => k.realTimeFactor ?? algo.speed),
        },
        changes: noop ? {
          travelTime: changePct(k.travelTime_s, noop.travelTime_s),
          waitTime:   changePct(k.waitTime_s,   noop.waitTime_s),
          throughput: changePct(k.throughput,   noop.throughput),
          speed: 0,
        } : algo.changes,
        system: {
          ...algo.system,
          training: civiqData.trainMetrics?.curve?.length
            ? qmixToTrainingPoints(civiqData.trainMetrics.curve)
            : algo.system.training,
          cpu: makeCpu(k.cpuMean, Math.min(k.cpuPeak / 100, 120), 8, 120, 15),
        },
        marl:     civiqData.trainMetrics?.curve?.length
          ? qmixToMarlPoints(civiqData.trainMetrics.curve)
          : algo.marl,
        episodes: qmixToEpisodeSeries(civiqData),
      }
    }
    return algo
  })()

  return (
  <div className="p-6 space-y-5 overflow-y-auto" style={{ height: '100%' }}>
    {/* Episode detail modal */}
    {openModal && (
      <EpisodeDetailModal algo={displayAlgo} metricKey={openModal} onClose={() => setOpenModal(null)} />
    )}
    {/* Selfish detail modal */}
    {isSelfish && selfishModal && (() => {
      const data = selfishModal === 'co2'  ? SELFISH_CO2_EPISODES
                 : selfishModal === 'fuel' ? SELFISH_FUEL_EPISODES
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

    {/* Header */}
    <div className="flex items-center justify-between gap-4">
      <div>
        <h2 className="text-xl font-bold leading-tight" style={{ color: 'rgba(255,255,255,0.9)' }}>{displayAlgo.label}</h2>
      </div>
      <div className="flex items-center gap-2 flex-1">
        {mapSize && (
          <span className="text-[11px] font-medium px-2.5 py-1 rounded-full"
            style={{ background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.1)', color: 'rgba(255,255,255,0.55)' }}>
            {MAP_LABELS[mapSize] || mapSize}
          </span>
        )}
        {trafficScale && (
          <span className="text-[11px] font-medium px-2.5 py-1 rounded-full"
            style={{ background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.1)', color: 'rgba(255,255,255,0.55)' }}>
            {TRAFFIC_LABELS[trafficScale] || trafficScale}
          </span>
        )}
        {/* Real-data badge for selfish routing */}
        {isSelfish && (
          <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full flex-shrink-0"
            style={{
              background: kpisLoading ? 'rgba(255,255,255,0.06)' : realKpis ? 'rgba(52,211,153,0.12)' : 'rgba(255,255,255,0.06)',
              border: kpisLoading ? '1px solid rgba(255,255,255,0.1)' : realKpis ? '1px solid rgba(52,211,153,0.3)' : '1px solid rgba(255,255,255,0.1)',
              color: kpisLoading ? 'rgba(255,255,255,0.3)' : realKpis ? '#34D399' : 'rgba(255,255,255,0.3)',
            }}>
            {kpisLoading ? 'Loading data…' : realKpis ? 'Live SUMO Data' : 'Static Data'}
          </span>
        )}
        {/* Real-data badge for monolithic QMIX */}
        {isQmix && (
          <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full flex-shrink-0"
            style={{
              background: qmixLoading ? 'rgba(255,255,255,0.06)' : qmixData ? 'rgba(167,139,250,0.15)' : 'rgba(255,255,255,0.06)',
              border: qmixLoading ? '1px solid rgba(255,255,255,0.1)' : qmixData ? '1px solid rgba(167,139,250,0.4)' : '1px solid rgba(255,255,255,0.1)',
              color: qmixLoading ? 'rgba(255,255,255,0.3)' : qmixData ? '#A78BFA' : 'rgba(255,255,255,0.3)',
            }}>
            {qmixLoading ? 'Loading data…' : qmixData ? `Real PyMARL Data · ${trafficScale === 'stable_flow' ? 'LOS C' : trafficScale === 'forced_flow' ? 'LOS E' : 'LOS A'}` : 'Static Data'}
          </span>
        )}
        {/* Real-data badge for CiViQ */}
        {isCiviq && (
          <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full flex-shrink-0"
            style={{
              background: civiqLoading ? 'rgba(255,255,255,0.06)' : civiqData ? 'rgba(56,189,248,0.15)' : 'rgba(255,255,255,0.06)',
              border: civiqLoading ? '1px solid rgba(255,255,255,0.1)' : civiqData ? '1px solid rgba(56,189,248,0.4)' : '1px solid rgba(255,255,255,0.1)',
              color: civiqLoading ? 'rgba(255,255,255,0.3)' : civiqData ? '#38BDF8' : 'rgba(255,255,255,0.3)',
            }}>
            {civiqLoading ? 'Loading data…' : civiqData ? `Real PyMARL Data · ${trafficScale === 'stable_flow' ? 'LOS C' : trafficScale === 'forced_flow' ? 'LOS E' : 'LOS A'}` : 'Static Data'}
          </span>
        )}
      </div>
      <ExportButton algo={displayAlgo} selfishTimeseries={isSelfish ? realTimeseries : null} />
      <div className="px-4 py-1.5 rounded-full text-[12px] font-bold flex-shrink-0"
        style={{ background: displayAlgo.colorDim, color: displayAlgo.color, border: `1px solid ${displayAlgo.border}` }}>
        {displayAlgo.efficiency}% Efficiency Score
      </div>
    </div>

    {/* KPI Row */}
    <div className="grid grid-cols-4 gap-4">
      <KpiCard label="Avg. Travel Time" abbr="ATT" value={displayAlgo.travelTime} unit="min"
        color={displayAlgo.color} colorDim={displayAlgo.colorDim} borderColor={displayAlgo.border}
        change={displayAlgo.changes.travelTime} lowerBetter sparkData={displayAlgo.sparklines.travelTime}
        changeLabel={isSelfish && selfishCiviq ? 'vs. CiViQ' : undefined}
        onClick={isSelfish ? () => setSelfishModal('travelTime') : () => setOpenModal('travelTime')}
        description="The mean time it takes for a vehicle to complete its route from entry to exit, across all vehicles in the simulation." />
      <KpiCard label="Avg. Wait Time" abbr="AWT" value={displayAlgo.waitTime} unit="sec"
        color={displayAlgo.color} colorDim={displayAlgo.colorDim} borderColor={displayAlgo.border}
        change={displayAlgo.changes.waitTime} lowerBetter sparkData={displayAlgo.sparklines.waitTime}
        changeLabel={isSelfish && selfishCiviq ? 'vs. CiViQ' : undefined}
        onClick={isSelfish ? () => setSelfishModal('waitTime') : () => setOpenModal('waitTime')}
        description="The mean time vehicles spent fully stopped in traffic. High values indicate congestion or poor routing decisions." />
      <KpiCard label="Throughput" abbr="TPT" value={displayAlgo.throughput.toLocaleString()} unit="veh/hr"
        color={displayAlgo.color} colorDim={displayAlgo.colorDim} borderColor={displayAlgo.border}
        change={displayAlgo.changes.throughput} sparkData={displayAlgo.sparklines.throughput}
        changeLabel={isSelfish && selfishCiviq ? 'vs. CiViQ' : undefined}
        onClick={isSelfish ? () => setSelfishModal('throughput') : () => setOpenModal('throughput')}
        description="The number of vehicles that successfully completed their routes per minute. Higher values indicate better overall traffic flow." />
      <KpiCard label="Real-time Factor" abbr="RTF" value={displayAlgo.speed.toFixed(2)} unit="x"
        color={displayAlgo.color} colorDim={displayAlgo.colorDim} borderColor={displayAlgo.border}
        sparkData={displayAlgo.sparklines.speed}
        onClick={isSelfish ? undefined : () => setOpenModal('speed')}
        descriptionSide="left"
        description="The ratio of simulation time to actual wall-clock time. A value of 1.0 means the simulation runs in real time; higher values indicate faster-than-real-time execution." />
    </div>

    {/* Charts row: left column stack + Map Player */}
    <div className="grid grid-cols-3 gap-4">
      {/* Left column: Algorithm Overview + Heatmap stacked */}
      <div className="flex flex-col gap-4">
        <GlassCard className="p-5 flex flex-col gap-3">
          <h3 className="text-[13px] font-bold" style={{ color: 'rgba(255,255,255,0.78)' }}>Algorithm Overview</h3>
          <p className="text-[12px] leading-relaxed" style={{ color: 'rgba(255,255,255,0.52)' }}>
            {displayAlgo.description}
          </p>
          {displayAlgo.convergence !== null && (
            <div className="pt-3 grid grid-cols-2 gap-3" style={{ borderTop: '1px solid rgba(255,255,255,0.07)' }}>
              <div>
                <div className="text-[10px] uppercase tracking-wider mb-1" style={{ color: 'rgba(255,255,255,0.28)' }}>Convergence</div>
                <div className="text-[17px] font-bold tabular-nums" style={{ color: displayAlgo.color }}>Ep. {displayAlgo.convergence}</div>
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-wider mb-1" style={{ color: 'rgba(255,255,255,0.28)' }}>Cumulative Reward</div>
                <div className="text-[17px] font-bold tabular-nums" style={{ color: displayAlgo.color }}>{displayAlgo.reward}</div>
              </div>
            </div>
          )}
        </GlassCard>

        <CongestionHeatmap
          algo={displayAlgo}
          onViewDetail={() => setCongestionDetail(true)}
          heatmapSrc={isSelfish ? SELFISH_HEATMAP[trafficScale] : undefined}
          congestionLabel={isSelfish ? SELFISH_CONGESTION_LABEL[trafficScale] : undefined}
        />
      </div>

      {/* Map Player (col 2–3) */}
      <MapPlayer algo={displayAlgo} mapSize={mapSize}
        onCo2Click={isSelfish ? () => setSelfishModal('co2') : undefined}
        onFuelClick={isSelfish ? () => setSelfishModal('fuel') : undefined}
      />
    </div>

    {/* Analytics row */}
    {displayAlgo.id !== 'selfish' && (
      <TrainingCurveChart algo={displayAlgo} />
    )}

    {/* MARL training diagnostics (learning-based only) */}
    <MarlMetricsSection algo={displayAlgo} />
  </div>
  )
}

// ─── Sidebar ───────────────────────────────────────────────────────────────────

const NAV: { id: Page; label: string; d: string }[] = [
  {
    id: 'summary', label: 'Summary',
    d: 'M3 3h7v7H3V3zm11 0h7v7h-7V3zM3 14h7v7H3v-7zm14 3.5a3.5 3.5 0 1 1-7 0 3.5 3.5 0 0 1 7 0z',
  },
  {
    id: 'civiq', label: 'Civiq',
    d: 'M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5',
  },
  {
    id: 'selfish', label: 'Selfish Routing',
    d: 'M9 20l-5.447-2.724A1 1 0 0 1 3 16.382V5.618a1 1 0 0 1 1.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0 0 21 18.382V7.618a1 1 0 0 0-1.447-.894L15 9m0 8V9m0 0L9 7',
  },
  {
    id: 'qmix', label: 'Monolithic QMIX',
    d: 'M19 11H5m14 0a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-6a2 2 0 0 1 2-2m14 0V9a2 2 0 0 0-2-2M5 11V9a2 2 0 0 1 2-2m0 0V5a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v2M7 7h10',
  },
]

const PAGE_COLOR: Record<Page, string> = {
  summary: '#60A5FA',
  civiq: ALGO.civiq.color,
  selfish: ALGO.selfish.color,
  qmix: ALGO.qmix.color,
}

interface SidebarProps {
  activePage: Page
  setActivePage: (p: Page) => void
  mapSize: string
  trafficScale: string
  algorithm1: string
}

const Sidebar = ({ activePage, setActivePage, mapSize, trafficScale, algorithm1 }: SidebarProps) => (
  <div className="flex flex-col flex-shrink-0"
    style={{ width: '248px', background: 'rgba(0,0,0,0.2)', borderRight: '1px solid rgba(255,255,255,0.06)' }}>

    {/* Analytics nav */}
    <div className="px-5 pt-6 pb-3 flex-shrink-0">
      <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: 'rgba(255,255,255,0.28)' }}>
        Analytics
      </span>
    </div>

    <nav className="px-3 space-y-1 flex-shrink-0">
      {NAV.map((item) => {
        const isActive = activePage === item.id
        const col = PAGE_COLOR[item.id]
        return (
          <button key={item.id} onClick={() => setActivePage(item.id)}
            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-left transition-all duration-200"
            style={{
              background: isActive ? `${col}18` : 'transparent',
              border: isActive ? `1px solid ${col}40` : '1px solid transparent',
              color: isActive ? col : 'rgba(255,255,255,0.45)',
            }}
            onMouseEnter={(e) => { if (!isActive) { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.06)'; (e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.72)' } }}
            onMouseLeave={(e) => { if (!isActive) { (e.currentTarget as HTMLElement).style.background = 'transparent'; (e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.45)' } }}>
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
    <div className="mx-4 my-4 flex-shrink-0" style={{ height: '1px', background: 'rgba(255,255,255,0.07)' }} />

    {/* Simulation Controls section label */}
    <div className="px-5 pb-3 flex-shrink-0">
      <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: 'rgba(255,255,255,0.28)' }}>
        Simulation Controls
      </span>
    </div>

    {/* Controls */}
    <div className="px-4 flex-shrink-0">
      <SimulationControls
        darkMode
        vertical
        hideHeader
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

// ─── Labels ────────────────────────────────────────────────────────────────────

const MAP_LABELS: Record<string, string> = {
  '2km': '2 km²', '0.75km': '0.75 km²', '4x4': '4×4 Grid (2.25 km²)',
}
const TRAFFIC_LABELS: Record<string, string> = {
  free_flow: 'Free Flow (LOS A)', stable_flow: 'Stable Flow (LOS C)', forced_flow: 'Forced Flow (LOS E)',
}

// ─── Main Export ───────────────────────────────────────────────────────────────

function SimulationDashboardContent() {
  const searchParams = useSearchParams()
  const router = useRouter()
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

  // Keep active tab in sync when URL params change (e.g. Run from sidebar controls)
  useEffect(() => {
    setActivePage(algoToPage(algorithm1))
  }, [algorithm1])

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
    <main className="w-full h-screen overflow-hidden" style={{ position: 'relative' }}>
      {/* Base gradient */}
      <div className="fixed inset-0 pointer-events-none"
        style={{ background: 'linear-gradient(135deg, #060112 0%, #0b0320 40%, #040c1c 100%)', zIndex: -1 }} />
      <AnimatedBackground />

      {/* Outer wrapper */}
      <div className="flex items-stretch justify-center"
        style={{ height: '100vh', padding: 'clamp(10px, 1.2vw, 18px)', zIndex: 2, position: 'relative', overflowX: 'auto' }}>

        {/* OBU Bezel */}
        <div className="relative w-full flex flex-col" style={{
          minWidth: '1400px', maxWidth: '1640px',
          background: 'rgba(8, 14, 32, 0.48)',
          backdropFilter: 'blur(32px)', WebkitBackdropFilter: 'blur(32px)',
          borderRadius: '24px', padding: '12px',
          border: '1px solid rgba(255,255,255,0.11)',
          boxShadow: '0 32px 80px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.14), inset 0 -1px 0 rgba(0,0,0,0.3)',
        }}>
          {/* Side screws */}
          {['left-2', 'right-2'].map((side) => (
            <div key={side} className={`absolute ${side} top-1/2 -translate-y-1/2 flex flex-col gap-2.5 pointer-events-none`}>
              {[0, 1, 2].map((i) => (
                <div key={i} className="w-2 h-2 rounded-full"
                  style={{ background: 'linear-gradient(135deg,rgba(255,255,255,0.18),rgba(255,255,255,0.04))', boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.6)' }} />
              ))}
            </div>
          ))}

          {/* Screen */}
          <div className="relative flex flex-col overflow-hidden"
            style={{
              flex: 1,
              background: 'rgba(6, 11, 26, 0.45)',
              backdropFilter: 'blur(20px)', WebkitBackdropFilter: 'blur(20px)',
              borderRadius: '14px', border: '1px solid rgba(255,255,255,0.08)',
              boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.06)',
            }}>
            {/* Top gloss */}
            <div className="absolute inset-x-0 top-0 h-20 pointer-events-none rounded-t-[14px]"
              style={{ background: 'linear-gradient(180deg, rgba(255,255,255,0.05) 0%, transparent 100%)' }} />

            {/* Status bar */}
            <StatusBar />

            {/* Body: Sidebar + Content */}
            <div className="flex overflow-hidden" style={{ flex: 1, minHeight: 0 }}>
              <Sidebar
                activePage={activePage}
                setActivePage={setActivePage}
                mapSize={mapSize}
                trafficScale={trafficScale}
                algorithm1={algorithm1}
              />
              {/* Content pane: fixed blob layer + scrollable content on top */}
              <div className="flex-1 relative" style={{
                minHeight: 0,
                overflow: 'hidden',
                background: 'rgba(4, 9, 22, 0.82)',
                backdropFilter: 'blur(20px)',
                WebkitBackdropFilter: 'blur(20px)',
                borderLeft: '1px solid rgba(255,255,255,0.07)',
              }}>
                {/* Animated colour blobs — fixed behind content */}
                <div className="absolute inset-0 overflow-hidden pointer-events-none" style={{ zIndex: 0 }}>
                  <div className="absolute rounded-full" style={{
                    width: 500, height: 500, top: '-120px', left: '-80px',
                    background: 'radial-gradient(circle at 40% 40%, rgba(6,182,212,0.28) 0%, transparent 70%)',
                    filter: 'blur(60px)', animation: 'blob1 16s ease-in-out infinite', animationDelay: '-3s',
                  }} />
                  <div className="absolute rounded-full" style={{
                    width: 460, height: 460, bottom: '-100px', right: '-60px',
                    background: 'radial-gradient(circle at 60% 55%, rgba(139,92,246,0.24) 0%, transparent 70%)',
                    filter: 'blur(60px)', animation: 'blob2 20s ease-in-out infinite', animationDelay: '-8s',
                  }} />
                  <div className="absolute rounded-full" style={{
                    width: 360, height: 360, top: '35%', left: '50%',
                    background: 'radial-gradient(circle at 50% 50%, rgba(37,99,235,0.18) 0%, transparent 70%)',
                    filter: 'blur(50px)', animation: 'blob3 18s ease-in-out infinite', animationDelay: '-12s',
                  }} />
                </div>

                {/* Scrollable page content */}
                <div className="absolute inset-0 overflow-y-auto" style={{ zIndex: 1 }}>
                  {activePage === 'summary' && <SummaryPage onNavigate={setActivePage} />}
                  {activePage === 'civiq' && <AlgoDetailPage algo={ALGO.civiq} mapSize={mapSize} trafficScale={trafficScale} />}
                  {activePage === 'selfish' && <AlgoDetailPage algo={ALGO.selfish} mapSize={mapSize} trafficScale={trafficScale} />}
                  {activePage === 'qmix' && <AlgoDetailPage algo={ALGO.qmix} mapSize={mapSize} trafficScale={trafficScale} />}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}

export default function SimulationDashboard() {
  return (
    <Suspense fallback={<main className="w-full h-screen" style={{ background: '#060112' }} />}>
      <SimulationDashboardContent />
    </Suspense>
  )
}

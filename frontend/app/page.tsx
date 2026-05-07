'use client'

import { useRouter } from 'next/navigation'
import { AnimatedBackground } from '@/components/AnimatedBackground'
import { SimulationControls } from '@/components/SimulationControls'

const researchers = [
  { name: 'Kristian Bautista', role: 'Project Manager',  email: 'kristiandavidbautista@gmail.com',  avatar: 'http://localhost:3845/assets/f111a4d9e98c2f1849285d198126666303e67f65.png' },
  { name: 'Angel Letada',      role: 'SUMO Engineer',     email: 'angel.letada1205@gmail.com',        avatar: 'http://localhost:3845/assets/eaa320717b7e77fd08d1bdaf9802cc375eb36366.png' },
  { name: 'Michael Pascual',   role: 'SUMO Engineer',     email: 'michaelkevinpascual47@gmail.com',   avatar: 'http://localhost:3845/assets/5f8ea6b9caf08d167684ed154ad8a85f97b6913b.png' },
  { name: 'Marianne Santos',   role: 'Software Engineer', email: 'mariannesantos174@gmail.com',       avatar: 'http://localhost:3845/assets/e61b32a6b96823a8b0214ef17a3aac015a2ed382.png' },
]

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
      <div className="flex items-center gap-2.5">
        <img src="/icons/civiq-logo.png" alt="Civiq" className="w-5 h-5 object-contain brightness-0 invert opacity-80" />
        <span className="font-bold text-[13px] tracking-widest" style={{ color: 'rgba(255,255,255,0.75)' }}>CIVIQ</span>
        <span className="text-[10px] font-medium" style={{ color: 'rgba(255,255,255,0.3)' }}>
          ·&nbsp; A Hierarchical Multi-Agent Coordination Framework
        </span>
      </div>
      <div className="flex items-center gap-1">
        {NAV_LINKS.map(({ label, href }) => (
          <button key={label} onClick={() => router.push(href)}
            className="px-4 py-1.5 rounded-full text-[12px] font-medium transition-all duration-150"
            style={{ color: 'rgba(255,255,255,0.52)', background: 'transparent' }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.9)'; (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.07)' }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.52)'; (e.currentTarget as HTMLElement).style.background = 'transparent' }}>
            {label}
          </button>
        ))}
      </div>
    </div>
  )
}

function GlassCard({ children, className = '', style = {} }: { children: React.ReactNode; className?: string; style?: React.CSSProperties }) {
  return (
    <div className={className} style={{
      background: 'linear-gradient(155deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.04) 100%)',
      backdropFilter: 'blur(24px)', WebkitBackdropFilter: 'blur(24px)',
      border: '1px solid rgba(255,255,255,0.12)',
      borderRadius: '16px',
      boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.1), 0 8px 32px rgba(0,0,0,0.28)',
      ...style,
    }}>{children}</div>
  )
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full mb-4"
      style={{ background: 'rgba(6,182,212,0.1)', border: '1px solid rgba(6,182,212,0.25)' }}>
      <div className="w-1.5 h-1.5 rounded-full" style={{ background: '#06B6D4' }} />
      <span className="text-[10px] font-bold tracking-widest uppercase" style={{ color: '#06B6D4' }}>{children}</span>
    </div>
  )
}

export default function Home() {
  return (
    <main className="w-full" style={{ background: '#060112' }}>
      <div className="fixed inset-0 pointer-events-none" style={{ background: 'linear-gradient(135deg, #060112 0%, #0b0320 40%, #040c1c 100%)', zIndex: -1 }} />
      <AnimatedBackground />

      {/* ── HERO ── */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden"
        style={{ zIndex: 2, padding: 'clamp(16px, 3vw, 48px)' }}>
        <div className="relative w-full mx-auto" style={{
          maxWidth: 'min(1160px, 100%)',
          background: 'rgba(8,14,32,0.48)', backdropFilter: 'blur(32px)', WebkitBackdropFilter: 'blur(32px)',
          borderRadius: '24px', padding: '12px',
          border: '1px solid rgba(255,255,255,0.11)',
          boxShadow: '0 32px 80px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.14), inset 0 -1px 0 rgba(0,0,0,0.3)',
        }}>
          {['left-2','right-2'].map(side => (
            <div key={side} className={`absolute ${side} top-1/2 -translate-y-1/2 flex flex-col gap-2.5 pointer-events-none`}>
              {[0,1,2].map(i => <div key={i} className="w-2 h-2 rounded-full" style={{ background: 'linear-gradient(135deg,rgba(255,255,255,0.18),rgba(255,255,255,0.04))', boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.6)' }} />)}
            </div>
          ))}
          <div className="relative w-full flex flex-col overflow-hidden" style={{
            background: 'rgba(6,11,26,0.62)', backdropFilter: 'blur(20px)', WebkitBackdropFilter: 'blur(20px)',
            borderRadius: '14px', border: '1px solid rgba(255,255,255,0.08)',
            boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.06)',
          }}>
            <div className="absolute inset-x-0 top-0 h-24 pointer-events-none rounded-t-[14px]"
              style={{ background: 'linear-gradient(180deg, rgba(255,255,255,0.05) 0%, transparent 100%)' }} />
            <StatusBar />
            <div className="flex flex-col items-center gap-5"
              style={{ padding: 'clamp(20px, 3vw, 40px) clamp(16px, 4vw, 48px) clamp(24px, 3vw, 44px)' }}>
              <div className="text-center w-full" style={{ maxWidth: '780px' }}>
                <h1 className="font-extrabold leading-[1.1] mb-4 select-none" style={{
                  fontSize: 'clamp(22px, 3vw, 42px)',
                  backgroundImage: 'linear-gradient(140deg, #93C5FD 0%, #60A5FA 40%, #38BDF8 100%)',
                  WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
                }}>
                  Solving Urban Congestion Through Hierarchical Coordination.
                </h1>
                <p className="leading-[1.75] mb-5" style={{ fontSize: 'clamp(12px, 1.1vw, 14px)', color: 'rgba(255,255,255,0.52)' }}>
                  An undergraduate thesis applying{' '}
                  <span className="font-semibold" style={{ color: '#38BDF8' }}>Hierarchical Multi-Agent Reinforcement Learning</span>{' '}
                  to urban traffic management. Vehicles act as intelligent agents that cooperatively learn routing decisions — reducing congestion across a simulated road network.
                </p>
                <div className="flex gap-3 justify-center">
                  <a href="#about" className="font-semibold rounded-full transition-all duration-200 text-white"
                    style={{ padding: 'clamp(8px,1vw,10px) clamp(16px,2vw,24px)', fontSize: 'clamp(11px,1vw,13.5px)', background: 'linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%)', boxShadow: '0 4px 20px rgba(59,130,246,0.45)' }}
                    onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.boxShadow = '0 6px 28px rgba(59,130,246,0.65)'; (e.currentTarget as HTMLElement).style.transform = 'translateY(-1px)' }}
                    onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.boxShadow = '0 4px 20px rgba(59,130,246,0.45)'; (e.currentTarget as HTMLElement).style.transform = 'translateY(0)' }}>
                    Explore Research
                  </a>
                  <a href="#" className="font-semibold rounded-full transition-all duration-200"
                    style={{ padding: 'clamp(8px,1vw,10px) clamp(16px,2vw,24px)', fontSize: 'clamp(11px,1vw,13.5px)', background: 'rgba(56,189,248,0.08)', border: '1.5px solid rgba(56,189,248,0.4)', color: '#38BDF8' }}
                    onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(56,189,248,0.15)' }}
                    onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(56,189,248,0.08)' }}>
                    Thesis Paper
                  </a>
                </div>
              </div>
              <div className="w-full" style={{ height: '1px', background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.09), transparent)' }} />
              <div className="w-full" style={{
                maxWidth: '860px', background: 'rgba(255,255,255,0.04)', backdropFilter: 'blur(12px)',
                WebkitBackdropFilter: 'blur(12px)', borderRadius: '18px',
                border: '1px solid rgba(255,255,255,0.08)',
                boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.05)', padding: 'clamp(16px, 2vw, 24px)',
              }}>
                <SimulationControls darkMode />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── ABOUT ── */}
      <section id="about" className="relative w-full py-24" style={{ zIndex: 2 }}>
        <div className="max-w-6xl mx-auto px-8 space-y-20">

          {/* About Civiq */}
          <div>
            <SectionLabel>About Civiq</SectionLabel>
            <h2 className="text-[36px] font-extrabold mb-6 leading-tight" style={{
              backgroundImage: 'linear-gradient(140deg, #93C5FD 0%, #38BDF8 100%)',
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
            }}>
              Redefining Urban Traffic Management
            </h2>
            <p className="text-[15px] leading-relaxed max-w-3xl" style={{ color: 'rgba(255,255,255,0.55)' }}>
              Civiq is a hierarchical software framework designed to redefine urban traffic management through cooperative intelligence. Built around the QMIX algorithm, Civiq addresses the scalability challenges of traditional Multi-Agent Reinforcement Learning by utilizing a three-level architecture: individual vehicle agents, local coordination via Roadside Units, and a central server for global optimization. By integrating directly with the SUMO environment, Civiq transforms selfish routing behaviors into a synchronized network, significantly improving throughput and reducing urban congestion.
            </p>
          </div>

          {/* KPI stat cards */}
          <div>
            <SectionLabel>Smart Traffic Routing</SectionLabel>
            <h3 className="text-[28px] font-bold mb-8" style={{ color: 'rgba(255,255,255,0.88)' }}>
              Scalable by Design, Fast in Practice
            </h3>
            <div className="grid grid-cols-3 gap-5 mb-10">
              {[
                { value: '22.35', unit: 'ms',      label: 'Average Compute Time',  sub: 'Decision latency per step', color: '#38BDF8', colorDim: 'rgba(56,189,248,0.1)', border: 'rgba(56,189,248,0.25)',
                  icon: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> },
                { value: '4.2',   unit: 'min',     label: 'Average Travel Time',   sub: '50% faster vs. baseline',  color: '#A78BFA', colorDim: 'rgba(167,139,250,0.1)', border: 'rgba(167,139,250,0.25)',
                  icon: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg> },
                { value: '1,875', unit: 'veh/hr',  label: 'Network Throughput',    sub: '36% above selfish routing', color: '#4ADE80', colorDim: 'rgba(74,222,128,0.1)',  border: 'rgba(74,222,128,0.25)',
                  icon: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg> },
              ].map(({ value, unit, label, sub, color, colorDim, border, icon }) => (
                <GlassCard key={label} className="p-6" style={{ border: `1px solid ${border}`, background: colorDim }}>
                  <div className="flex items-center justify-between mb-4">
                    <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: `rgba(${color === '#38BDF8' ? '56,189,248' : color === '#A78BFA' ? '167,139,250' : '74,222,128'},0.15)`, color }}>
                      {icon}
                    </div>
                    <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full" style={{ background: `rgba(${color === '#38BDF8' ? '56,189,248' : color === '#A78BFA' ? '167,139,250' : '74,222,128'},0.12)`, color, border: `1px solid ${border}` }}>Civiq</span>
                  </div>
                  <div className="flex items-baseline gap-1.5 mb-1">
                    <span className="text-[34px] font-black tabular-nums leading-none" style={{ color }}>{value}</span>
                    <span className="text-[13px] font-semibold" style={{ color: 'rgba(255,255,255,0.45)' }}>{unit}</span>
                  </div>
                  <div className="text-[13px] font-semibold mb-0.5" style={{ color: 'rgba(255,255,255,0.8)' }}>{label}</div>
                  <div className="text-[11px]" style={{ color: 'rgba(255,255,255,0.38)' }}>{sub}</div>
                </GlassCard>
              ))}
            </div>
            <GlassCard className="p-7">
              <p className="text-[14px] leading-relaxed" style={{ color: 'rgba(255,255,255,0.55)' }}>
                Civiq transforms congested urban traffic into a coordinated and efficient network by enabling vehicles to operate cooperatively rather than competitively. By distributing processing tasks between intelligent roadside sensors and a centralized optimization system, Civiq maintains high speed and accuracy even as traffic volume increases. This scalable architecture allows the AI to perform complex routing decisions in{' '}
                <span className="font-semibold" style={{ color: '#38BDF8' }}>22.35 milliseconds</span>, ensuring rapid response to changing traffic conditions.
              </p>
            </GlassCard>
          </div>

          {/* Green Edge Computing */}
          <div>
            <SectionLabel>Environmental Impact</SectionLabel>
            <h3 className="text-[28px] font-bold mb-8" style={{ color: 'rgba(255,255,255,0.88)' }}>
              Green Edge Computing
            </h3>
            <div className="grid grid-cols-2 gap-8 items-start">
              <div className="space-y-4">
                <p className="text-[14px] leading-relaxed" style={{ color: 'rgba(255,255,255,0.55)' }}>
                  Civiq improves urban sustainability by applying Green Edge Computing to traffic management at the point where traffic occurs. Instead of transmitting all data to a centralized data center, the system uses intelligent edge devices at each intersection to make timely, energy-efficient decisions locally.
                </p>
                <p className="text-[14px] leading-relaxed" style={{ color: 'rgba(255,255,255,0.55)' }}>
                  This decentralized approach reduces unnecessary data processing, limits energy waste, and ensures that traffic signals contribute to environmental efficiency across the city. As a result, Civiq achieves an average{' '}
                  <span className="font-semibold" style={{ color: '#4ADE80' }}>15% reduction in CO₂ emissions</span>, maintaining a low footprint of <span className="font-semibold" style={{ color: '#4ADE80' }}>142 g/km</span> even during peak traffic conditions.
                </p>
              </div>
              <div className="grid grid-cols-2 gap-4">
                {[
                  { value: '142', unit: 'g/km', label: 'Avg. CO₂ Emissions', pct: 72, color: '#4ADE80' },
                  { value: '23',  unit: 'l/100km', label: 'Avg. Fuel Consumption', pct: 58, color: '#38BDF8' },
                ].map(({ value, unit, label, pct, color }) => {
                  const r = 54, circ = 2 * Math.PI * r
                  return (
                    <GlassCard key={label} className="p-5 flex flex-col items-center gap-3">
                      <div className="relative w-32 h-32">
                        <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
                          <circle cx="60" cy="60" r={r} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="10" />
                          <circle cx="60" cy="60" r={r} fill="none" stroke={color} strokeWidth="10"
                            strokeDasharray={circ} strokeDashoffset={circ * (1 - pct / 100)}
                            strokeLinecap="round" style={{ filter: `drop-shadow(0 0 6px ${color}60)` }} />
                        </svg>
                        <div className="absolute inset-0 flex flex-col items-center justify-center">
                          <span className="text-[24px] font-black tabular-nums" style={{ color }}>{value}</span>
                          <span className="text-[10px]" style={{ color: 'rgba(255,255,255,0.4)' }}>{unit}</span>
                        </div>
                      </div>
                      <span className="text-[11px] font-semibold text-center" style={{ color: 'rgba(255,255,255,0.6)' }}>{label}</span>
                    </GlassCard>
                  )
                })}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── RESEARCHERS ── */}
      <section id="researchers" className="relative w-full py-24" style={{ zIndex: 2, borderTop: '1px solid rgba(255,255,255,0.06)' }}>
        <div className="max-w-5xl mx-auto px-8">
          <div className="text-center mb-14">
            <SectionLabel>The Team</SectionLabel>
            <h2 className="text-[36px] font-extrabold" style={{
              backgroundImage: 'linear-gradient(140deg, #93C5FD 0%, #38BDF8 100%)',
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
            }}>The Researchers</h2>
          </div>
          <div className="grid grid-cols-4 gap-6">
            {researchers.map((r) => (
              <GlassCard key={r.name} className="p-5 flex flex-col items-center text-center gap-3 group transition-all duration-200">
                <div className="w-20 h-20 rounded-full overflow-hidden ring-2 ring-white/10 group-hover:ring-cyan-400/40 transition-all duration-200"
                  style={{ boxShadow: '0 4px 20px rgba(0,0,0,0.4)' }}>
                  <img src={r.avatar} alt={r.name} className="w-full h-full object-cover" />
                </div>
                <div>
                  <p className="text-[13px] font-bold mb-0.5" style={{ color: 'rgba(255,255,255,0.9)' }}>{r.name}</p>
                  <p className="text-[11px] italic mb-2" style={{ color: 'rgba(255,255,255,0.42)' }}>{r.role}</p>
                  <a href={`mailto:${r.email}`} className="text-[10px] transition-colors duration-150 break-all"
                    style={{ color: 'rgba(255,255,255,0.3)' }}
                    onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.color = '#38BDF8' }}
                    onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.3)' }}>
                    {r.email}
                  </a>
                </div>
              </GlassCard>
            ))}
          </div>
        </div>
      </section>

      {/* ── FOOTER ── */}
      <footer id="contact" className="relative w-full py-14" style={{ zIndex: 2, borderTop: '1px solid rgba(255,255,255,0.06)', background: 'rgba(0,0,0,0.25)' }}>
        <div className="max-w-5xl mx-auto px-8">
          <div className="flex items-start justify-between gap-12 mb-12">
            {/* Brand */}
            <div className="max-w-xs">
              <div className="flex items-center gap-2.5 mb-3">
                <img src="/icons/civiq-logo.png" alt="Civiq" className="w-6 h-6 object-contain brightness-0 invert opacity-75" />
                <span className="font-bold text-[14px] tracking-widest" style={{ color: 'rgba(255,255,255,0.75)' }}>CIVIQ</span>
              </div>
              <p className="text-[12px] leading-relaxed" style={{ color: 'rgba(255,255,255,0.35)' }}>
                A Hierarchical Multi-Agent Coordination Framework for urban traffic management.
              </p>
              <div className="flex gap-3 mt-4">
                {[
                  { label: 'GitHub', href: 'https://github.com', path: 'M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.6.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z' },
                  { label: 'LinkedIn', href: 'https://linkedin.com', path: 'M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.225 0z' },
                ].map(({ label, href, path }) => (
                  <a key={label} href={href} target="_blank" rel="noopener noreferrer" aria-label={label}
                    className="w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-150"
                    style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)', color: 'rgba(255,255,255,0.45)' }}
                    onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(6,182,212,0.12)'; (e.currentTarget as HTMLElement).style.color = '#06B6D4' }}
                    onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.06)'; (e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.45)' }}>
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d={path} /></svg>
                  </a>
                ))}
              </div>
            </div>
            {/* Links */}
            <div className="flex gap-16">
              {[
                { title: 'Explore', links: [{ label: 'Selfish Routing', href: '#' }, { label: 'Monolithic QMIX', href: '#' }, { label: 'Civiq', href: '#' }] },
                { title: 'Resources', links: [{ label: 'About Civiq', href: '/#about' }, { label: 'The Researchers', href: '/#researchers' }, { label: 'Research Paper', href: '#' }] },
              ].map(({ title, links }) => (
                <div key={title}>
                  <h4 className="text-[10px] font-bold uppercase tracking-widest mb-4" style={{ color: 'rgba(255,255,255,0.35)' }}>{title}</h4>
                  <ul className="space-y-2.5">
                    {links.map(({ label, href }) => (
                      <li key={label}>
                        <a href={href} className="text-[13px] transition-colors duration-150"
                          style={{ color: 'rgba(255,255,255,0.45)' }}
                          onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.85)' }}
                          onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.45)' }}>
                          {label}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
          <div style={{ borderTop: '1px solid rgba(255,255,255,0.07)', paddingTop: '24px' }}>
            <p className="text-[12px] text-center" style={{ color: 'rgba(255,255,255,0.25)' }}>
              © {new Date().getFullYear()} A Priori Group. All rights reserved.
            </p>
          </div>
        </div>
      </footer>
    </main>
  )
}

'use client'

import { useRouter } from 'next/navigation'
import { AnimatedBackground } from '@/components/AnimatedBackground'

export default function ResearchPage() {
  const router = useRouter()

  const sections = [
    {
      tag: 'Abstract',
      title: 'Research Overview',
      body: 'This section will contain the abstract of the research — a concise summary of the problem statement, methodology, key findings, and implications of the Civiq framework.',
    },
    {
      tag: 'Problem Statement',
      title: 'The Urban Traffic Problem',
      body: 'Urban traffic congestion is a growing challenge in dense metropolitan areas. Existing rule-based and centralised approaches struggle to generalise across dynamic road conditions. This research investigates whether hierarchical multi-agent reinforcement learning can provide scalable, real-time traffic coordination.',
    },
    {
      tag: 'Methodology',
      title: 'Civiq Framework',
      body: 'Civiq employs a two-tier QMIX architecture: a global orchestrator assigns zone-level routing objectives, while local agents optimise intersection-level control decisions. The system is evaluated against Monolithic QMIX and Selfish Routing baselines across multiple map configurations and traffic densities.',
    },
    {
      tag: 'Findings',
      title: 'Key Results',
      body: 'Preliminary results demonstrate that Civiq achieves up to 50% reduction in average travel time and 37% reduction in CO₂ emissions compared to Selfish Routing, with superior scalability characteristics versus Monolithic QMIX on larger road networks.',
    },
    {
      tag: 'Implications',
      title: 'Real-World Applicability',
      body: 'The framework is designed for deployment on On-Board Units (OBUs) in smart city infrastructure. Decision latency benchmarks confirm that Civiq operates within the bounds required by ISO 25010 Performance Efficiency criteria for real-time traffic management systems.',
    },
  ]

  return (
    <main className="w-full min-h-screen" style={{ position: 'relative', background: '#060112' }}>
      <AnimatedBackground />

      <div className="relative z-10 flex flex-col min-h-screen">
        {/* Header */}
        <div className="flex items-center justify-between px-8 py-4 flex-shrink-0"
          style={{ background: 'rgba(0,0,0,0.45)', borderBottom: '1px solid rgba(255,255,255,0.07)', backdropFilter: 'blur(20px)' }}>
          <button
            onClick={() => router.push('/')}
            className="flex items-center gap-2.5 transition-opacity duration-150 hover:opacity-80"
          >
            <img src="/icons/civiq-logo.png" alt="Civiq" className="w-5 h-5 object-contain brightness-0 invert opacity-80" />
            <span className="font-bold text-[13px] tracking-widest" style={{ color: 'rgba(255,255,255,0.75)' }}>CIVIQ</span>
          </button>

          <div className="flex items-center gap-1">
            {[
              { label: 'About', href: '/#about' },
              { label: 'The Research', href: '/research' },
              { label: 'Contact Us', href: '/#contact' },
            ].map(({ label, href }) => (
              <button
                key={label}
                onClick={() => router.push(href)}
                className="px-4 py-1.5 rounded-full text-[12px] font-medium transition-all duration-150"
                style={{
                  color: href === '/research' ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.52)',
                  background: href === '/research' ? 'rgba(6,182,212,0.12)' : 'transparent',
                  border: href === '/research' ? '1px solid rgba(6,182,212,0.3)' : '1px solid transparent',
                }}
                onMouseEnter={(e) => {
                  if (href !== '/research') {
                    (e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.9)'
                    ;(e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.07)'
                  }
                }}
                onMouseLeave={(e) => {
                  if (href !== '/research') {
                    (e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.52)'
                    ;(e.currentTarget as HTMLElement).style.background = 'transparent'
                  }
                }}
              >
                {label}
              </button>
            ))}
          </div>

          <button
            onClick={() => router.back()}
            className="flex items-center gap-1.5 text-[12px] transition-colors duration-150"
            style={{ color: 'rgba(255,255,255,0.35)' }}
            onMouseEnter={(e) => ((e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.72)')}
            onMouseLeave={(e) => ((e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.35)')}
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M19 12H5M12 19l-7-7 7-7" />
            </svg>
            Back
          </button>
        </div>

        {/* Hero */}
        <div className="text-center pt-20 pb-14 px-8">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full mb-6"
            style={{ background: 'rgba(6,182,212,0.12)', border: '1px solid rgba(6,182,212,0.3)' }}>
            <div className="w-1.5 h-1.5 rounded-full" style={{ background: '#06B6D4' }} />
            <span className="text-[11px] font-semibold tracking-wider uppercase" style={{ color: '#06B6D4' }}>
              Thesis Research · Placeholder
            </span>
          </div>
          <h1 className="text-[42px] font-extrabold leading-tight mb-4" style={{ color: 'rgba(255,255,255,0.92)' }}>
            The Research
          </h1>
          <p className="text-[16px] max-w-2xl mx-auto leading-relaxed" style={{ color: 'rgba(255,255,255,0.42)' }}>
            A summary of the Civiq thesis — problem context, methodology, findings, and implications.
            Full paper content will be added here upon completion.
          </p>
        </div>

        {/* Content sections */}
        <div className="max-w-4xl mx-auto w-full px-8 pb-24 space-y-5">
          {sections.map(({ tag, title, body }) => (
            <div
              key={tag}
              className="rounded-2xl p-7"
              style={{
                background: 'linear-gradient(155deg, rgba(255,255,255,0.07) 0%, rgba(255,255,255,0.03) 100%)',
                border: '1px solid rgba(255,255,255,0.1)',
                backdropFilter: 'blur(20px)',
              }}
            >
              <span className="inline-block text-[10px] font-bold uppercase tracking-widest px-2.5 py-1 rounded-full mb-3"
                style={{ background: 'rgba(6,182,212,0.12)', color: '#06B6D4', border: '1px solid rgba(6,182,212,0.25)' }}>
                {tag}
              </span>
              <h2 className="text-[18px] font-bold mb-3" style={{ color: 'rgba(255,255,255,0.88)' }}>{title}</h2>
              <p className="text-[14px] leading-relaxed" style={{ color: 'rgba(255,255,255,0.48)' }}>{body}</p>
            </div>
          ))}

          {/* Coming soon notice */}
          <div className="rounded-2xl p-6 text-center"
            style={{ background: 'rgba(6,182,212,0.06)', border: '1px dashed rgba(6,182,212,0.25)' }}>
            <p className="text-[13px] font-medium" style={{ color: 'rgba(6,182,212,0.7)' }}>
              Full research content — abstract, methodology details, results tables, and bibliography — will be populated here.
            </p>
          </div>
        </div>
      </div>
    </main>
  )
}

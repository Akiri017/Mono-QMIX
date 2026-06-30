'use client'

import { useEffect } from 'react'
import { ROLES, useRole, type RoleId } from '@/contexts/RoleContext'

export function RoleSelectModal() {
  const { showModal, setRole, role, closeModal } = useRole()

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && role) closeModal()
    }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [role, closeModal])

  if (!showModal) return null

  return (
    <div
      className="fixed inset-0 z-[9999] flex items-center justify-center px-4"
      style={{ background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(12px)' }}
    >
      <div
        className="w-full max-w-[520px] rounded-2xl p-7"
        style={{
          background: 'linear-gradient(155deg, rgba(10,24,56,0.98) 0%, rgba(6,14,38,0.98) 100%)',
          border: '1px solid rgba(255,255,255,0.14)',
          boxShadow: '0 32px 80px rgba(0,0,0,0.7), inset 0 1px 0 rgba(255,255,255,0.08)',
        }}
      >
        {/* Header */}
        <div className="mb-6 text-center">
          <h2 className="text-[22px] font-extrabold mb-2" style={{ color: 'rgba(255,255,255,0.95)' }}>
            What best describes your role?
          </h2>
          <p className="text-[13px]" style={{ color: 'rgba(255,255,255,0.45)' }}>
            We&apos;ll tailor the dashboard content and explanations to what matters most for you.
          </p>
        </div>

        {/* Role cards */}
        <div className="space-y-3">
          {ROLES.map((r) => (
            <button
              key={r.id}
              onClick={() => setRole(r.id as RoleId)}
              className="w-full flex items-start gap-4 p-4 rounded-xl text-left transition-all duration-150"
              style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.09)' }}
              onMouseEnter={e => {
                const el = e.currentTarget as HTMLElement
                el.style.background = `${r.color}14`
                el.style.borderColor = `${r.color}40`
              }}
              onMouseLeave={e => {
                const el = e.currentTarget as HTMLElement
                el.style.background = 'rgba(255,255,255,0.04)'
                el.style.borderColor = 'rgba(255,255,255,0.09)'
              }}
            >
              <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
                style={{ background: `${r.color}18`, color: r.color, border: `1px solid ${r.color}30` }}>
                {r.icon}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-[14px] font-bold mb-0.5" style={{ color: r.color }}>{r.label}</p>
                <p className="text-[12px] leading-relaxed" style={{ color: 'rgba(255,255,255,0.52)' }}>{r.description}</p>
              </div>
              <svg className="flex-shrink-0 mt-2" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'rgba(255,255,255,0.25)' }}>
                <path d="M9 18l6-6-6-6" />
              </svg>
            </button>
          ))}
        </div>

        {/* Keep current role (only shown when re-opening from badge) */}
        {role && (
          <button
            onClick={closeModal}
            className="w-full mt-4 text-[11px] transition-colors duration-150"
            style={{ color: 'rgba(255,255,255,0.28)' }}
            onMouseEnter={e => ((e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.55)')}
            onMouseLeave={e => ((e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.28)')}
          >
            Keep current role
          </button>
        )}
      </div>
    </div>
  )
}

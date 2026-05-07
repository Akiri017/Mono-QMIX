'use client'

import { useState, useRef, useEffect } from 'react'
import { createPortal } from 'react-dom'
import { useRouter } from 'next/navigation'
import { IconChevronDown } from './icons'

interface DropdownOption {
  label: string
  value: string
  description?: string
}

const mapSizeOptions: DropdownOption[] = [
  { label: '2 km²', value: '2km', description: 'Medium urban grid suitable for general simulation' },
  { label: '0.75 km²', value: '0.75km', description: 'Small dense urban area for focused analysis' },
  { label: '4x4 Grid (2.25 km²)', value: '4x4', description: 'Structured grid layout for controlled testing' },
]

const trafficScaleOptions: DropdownOption[] = [
  { label: 'Free Flow (LOS A)', value: 'free_flow', description: 'Minimal congestion — vehicles move freely at desired speeds' },
  { label: 'Stable Flow (LOS C)', value: 'stable_flow', description: 'Moderate traffic density with acceptable delays' },
  { label: 'Forced Flow (LOS E)', value: 'forced_flow', description: 'Near-capacity traffic with significant congestion' },
]

const algorithmOptions: DropdownOption[] = [
  { label: 'Selfish Routing', value: 'selfish_routing', description: 'Each vehicle independently optimizes its own route' },
  { label: 'Monolithic QMIX', value: 'monolithic_qmix', description: 'Centralized multi-agent reinforcement learning control' },
  { label: 'Hierarchical QMIX (Civiq)', value: 'hierarchical_qmix', description: "Civiq's hierarchical coordination framework for urban optimization" },
]

const DEFAULT_MAP_SIZE = '2km'
const DEFAULT_TRAFFIC_SCALE = 'stable_flow'
const DEFAULT_ALGORITHM = 'hierarchical_qmix'

interface GlassDropdownProps {
  label: string
  options: DropdownOption[]
  selected: string
  onSelect: (value: string) => void
  isOpen: boolean
  onToggle: () => void
  darkMode?: boolean
  openRight?: boolean
}

const GlassDropdown = ({ label, options, selected, onSelect, isOpen, onToggle, darkMode, openRight }: GlassDropdownProps) => {
  const selectedOption = options.find(opt => opt.value === selected)
  const triggerRef = useRef<HTMLDivElement>(null)
  const [fixedPos, setFixedPos] = useState({ top: 0, left: 0 })
  const [mounted, setMounted] = useState(false)

  useEffect(() => { setMounted(true) }, [])

  // Recompute fixed position whenever the panel opens (or window resizes while open)
  useEffect(() => {
    if (!isOpen || !openRight || !triggerRef.current) return
    const update = () => {
      const r = triggerRef.current!.getBoundingClientRect()
      // Estimate panel height: header (~44px) + each option (~64px)
      const estimatedH = options.length * 64 + 50
      const maxTop = window.innerHeight - estimatedH - 12
      setFixedPos({ top: Math.max(8, Math.min(r.top, maxTop)), left: r.right + 8 })
    }
    update()
    window.addEventListener('resize', update)
    window.addEventListener('scroll', update, true)
    return () => {
      window.removeEventListener('resize', update)
      window.removeEventListener('scroll', update, true)
    }
  }, [isOpen, openRight, options.length])

  const triggerBg = darkMode
    ? isOpen ? 'rgba(6,182,212,0.15)' : 'rgba(255,255,255,0.07)'
    : isOpen ? 'rgba(255,255,255,0.85)' : 'rgba(255,255,255,0.55)'

  const triggerBorder = darkMode
    ? isOpen ? '1px solid rgba(6,182,212,0.5)' : '1px solid rgba(255,255,255,0.12)'
    : isOpen ? '1.5px solid rgba(6,182,212,0.5)' : '1px solid rgba(255,255,255,0.7)'

  const textColor = darkMode
    ? selectedOption ? '#e2e8f0' : 'rgba(255,255,255,0.35)'
    : selectedOption ? '#1e293b' : '#64748b'

  const chevronColor = darkMode
    ? isOpen ? '#06B6D4' : 'rgba(255,255,255,0.35)'
    : isOpen ? '#06B6D4' : '#94a3b8'

  const chevronRotate = openRight
    ? isOpen ? 'rotate(270deg)' : 'rotate(-90deg)'
    : isOpen ? 'rotate(180deg)' : 'rotate(0deg)'

  const panelContent = (
    <div
      data-dropdown-portal
      style={{
        ...(openRight
          ? { position: 'fixed' as const, top: fixedPos.top, left: fixedPos.left, width: '224px', zIndex: 9999 }
          : { position: 'absolute' as const, top: 'calc(100% + 8px)', left: 0, right: 0, zIndex: 200 }),
        background: darkMode ? '#0d1a2d' : 'rgba(255,255,255,0.96)',
        backdropFilter: 'blur(24px)',
        WebkitBackdropFilter: 'blur(24px)',
        borderRadius: '14px',
        border: darkMode ? '1px solid rgba(255,255,255,0.1)' : '1px solid rgba(255,255,255,0.8)',
        boxShadow: darkMode ? '0 16px 48px rgba(0,0,0,0.7)' : '0 12px 40px rgba(0,0,0,0.12)',
        maxHeight: 'calc(100vh - 24px)',
        overflowY: 'auto' as const,
      }}
    >
      {options.map((option, index) => (
        <div
          key={option.value}
          onClick={(e) => { e.stopPropagation(); onSelect(option.value); onToggle() }}
          className="px-4 py-3 cursor-pointer transition-colors duration-150"
          style={{
            borderBottom: index < options.length - 1
              ? darkMode ? '1px solid rgba(255,255,255,0.06)' : '1px solid rgba(0,0,0,0.05)'
              : 'none',
            borderRadius: index === 0 ? '14px 14px 0 0' : index === options.length - 1 ? '0 0 14px 14px' : undefined,
          }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = darkMode ? 'rgba(6,182,212,0.1)' : 'rgba(6,182,212,0.08)' }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'transparent' }}
        >
          <p className="font-semibold text-[12px]" style={{ color: darkMode ? '#e2e8f0' : '#1e293b' }}>
            {option.label}
          </p>
          {option.description && (
            <p className="text-[11px] mt-0.5 leading-snug" style={{ color: darkMode ? 'rgba(255,255,255,0.4)' : '#94a3b8' }}>
              {option.description}
            </p>
          )}
        </div>
      ))}
    </div>
  )

  return (
    <div className="relative flex-1 min-w-0">
      <div
        ref={triggerRef}
        onClick={onToggle}
        className="w-full px-3 py-3 flex items-center justify-between cursor-pointer transition-all duration-200"
        style={{
          background: triggerBg,
          backdropFilter: 'blur(8px)',
          WebkitBackdropFilter: 'blur(8px)',
          borderRadius: '9999px',
          border: triggerBorder,
          boxShadow: isOpen ? '0 4px 20px rgba(6,182,212,0.15)' : 'none',
        }}
      >
        <span className="font-medium text-[12px] select-none truncate" style={{ color: textColor }}>
          {selectedOption?.label || label}
        </span>
        <div
          className="transition-transform duration-200 flex-shrink-0 ml-1.5"
          style={{ transform: chevronRotate, color: chevronColor }}
        >
          <IconChevronDown size={13} />
        </div>
      </div>

      {isOpen && (
        openRight && mounted
          ? createPortal(panelContent, document.body)
          : panelContent
      )}
    </div>
  )
}

interface SimulationControlsProps {
  darkMode?: boolean
  vertical?: boolean
  hideHeader?: boolean
  initialMapSize?: string
  initialTrafficScale?: string
  initialAlgorithm?: string
}

export const SimulationControls = ({
  darkMode = false,
  vertical = false,
  hideHeader = false,
  initialMapSize,
  initialTrafficScale,
  initialAlgorithm,
}: SimulationControlsProps) => {
  const router = useRouter()
  const containerRef = useRef<HTMLDivElement>(null)
  const [mapSize, setMapSize] = useState(initialMapSize || '')
  const [trafficScale, setTrafficScale] = useState(initialTrafficScale || '')
  const [algorithm, setAlgorithm] = useState(initialAlgorithm || '')
  const [openDropdown, setOpenDropdown] = useState<string | null>(null)

  const toggleDropdown = (name: string) => {
    setOpenDropdown(prev => (prev === name ? null : name))
  }

  const isFormValid = () => mapSize !== '' && trafficScale !== '' && algorithm !== ''

  const handleSelectDefault = () => {
    setMapSize(DEFAULT_MAP_SIZE)
    setTrafficScale(DEFAULT_TRAFFIC_SCALE)
    setAlgorithm(DEFAULT_ALGORITHM)
    setOpenDropdown(null)
  }

  const handleRun = () => {
    if (!isFormValid()) return
    const params = new URLSearchParams()
    params.set('mapSize', mapSize)
    params.set('trafficScale', trafficScale)
    params.set('view', 'focused')
    params.set('algorithm1', algorithm)
    router.push(`/simulation?${params.toString()}`)
  }

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Element
      // Don't close if the click is inside the container or inside a portal panel
      if (containerRef.current?.contains(target)) return
      if (target.closest('[data-dropdown-portal]')) return
      setOpenDropdown(null)
    }
    if (openDropdown) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [openDropdown])

  const headerColor = '#06B6D4'
  const iconBg = darkMode ? 'rgba(6,182,212,0.15)' : 'rgba(6,182,212,0.12)'

  return (
    <div
      ref={containerRef}
      className="w-full transition-all duration-200"
      style={{ paddingBottom: openDropdown && !vertical ? '220px' : '0px' }}
    >
      {/* Header */}
      {!hideHeader && (
        <div className="flex items-center gap-3 mb-5">
          <div className="w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: iconBg }}>
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke={headerColor} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="4"  y1="21" x2="4"  y2="14" />
              <line x1="4"  y1="10" x2="4"  y2="3"  />
              <line x1="12" y1="21" x2="12" y2="12" />
              <line x1="12" y1="8"  x2="12" y2="3"  />
              <line x1="20" y1="21" x2="20" y2="16" />
              <line x1="20" y1="12" x2="20" y2="3"  />
              <line x1="1"  y1="14" x2="7"  y2="14" />
              <line x1="9"  y1="8"  x2="15" y2="8"  />
              <line x1="17" y1="16" x2="23" y2="16" />
            </svg>
          </div>
          <h2 className="font-bold text-[17px] tracking-wide" style={{ color: headerColor }}>
            Simulation Controls
          </h2>
        </div>
      )}

      {/* Dropdowns */}
      <div className={`flex ${vertical ? 'flex-col' : 'flex-row'} gap-2 mb-3`}>
        <GlassDropdown label="Map Size" options={mapSizeOptions} selected={mapSize} onSelect={setMapSize}
          isOpen={openDropdown === 'mapSize'} onToggle={() => toggleDropdown('mapSize')}
          darkMode={darkMode} openRight={vertical} />
        <GlassDropdown label="Traffic Scale" options={trafficScaleOptions} selected={trafficScale} onSelect={setTrafficScale}
          isOpen={openDropdown === 'trafficScale'} onToggle={() => toggleDropdown('trafficScale')}
          darkMode={darkMode} openRight={vertical} />
        <GlassDropdown label="Algorithm" options={algorithmOptions} selected={algorithm} onSelect={setAlgorithm}
          isOpen={openDropdown === 'algorithm'} onToggle={() => toggleDropdown('algorithm')}
          darkMode={darkMode} openRight={vertical} />
      </div>

      {/* Buttons */}
      <div className={`flex ${vertical ? 'flex-col' : 'flex-row'} gap-2`}>
        <button
          onClick={handleSelectDefault}
          className="flex-1 py-2.5 font-semibold transition-all duration-200 rounded-full"
          style={{
            fontSize: vertical ? '11px' : '13px',
            background: 'transparent', border: `1.5px solid ${headerColor}`, color: headerColor,
          }}
          onMouseEnter={(e) => { ;(e.currentTarget as HTMLElement).style.background = 'rgba(6,182,212,0.1)' }}
          onMouseLeave={(e) => { ;(e.currentTarget as HTMLElement).style.background = 'transparent' }}
        >
          Use Default
        </button>
        <button
          onClick={handleRun}
          disabled={!isFormValid()}
          className="flex-1 py-2.5 font-bold text-white transition-all duration-200 rounded-full"
          style={{
            fontSize: vertical ? '12px' : '14px',
            background: isFormValid() ? 'linear-gradient(135deg, #06B6D4 0%, #0891B2 100%)' : darkMode ? 'rgba(255,255,255,0.08)' : 'rgba(148,163,184,0.4)',
            border: 'none',
            cursor: isFormValid() ? 'pointer' : 'not-allowed',
            boxShadow: isFormValid() ? '0 4px 20px rgba(6,182,212,0.45)' : 'none',
            color: isFormValid() ? 'white' : darkMode ? 'rgba(255,255,255,0.25)' : 'rgba(100,116,139,0.8)',
          }}
          onMouseEnter={(e) => {
            if (isFormValid()) {
              ;(e.currentTarget as HTMLElement).style.boxShadow = '0 6px 28px rgba(6,182,212,0.6)'
              ;(e.currentTarget as HTMLElement).style.transform = 'translateY(-1px)'
            }
          }}
          onMouseLeave={(e) => {
            if (isFormValid()) {
              ;(e.currentTarget as HTMLElement).style.boxShadow = '0 4px 20px rgba(6,182,212,0.45)'
              ;(e.currentTarget as HTMLElement).style.transform = 'translateY(0)'
            }
          }}
        >
          Run
        </button>
      </div>
    </div>
  )
}

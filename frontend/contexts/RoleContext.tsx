'use client'

import { createContext, useContext, useEffect, useState } from 'react'

export type RoleId = 'urban_planner' | 'traffic_engineer' | 'software_engineer'

export interface RoleDef {
  id: RoleId
  label: string
  color: string
  description: string
  icon: React.ReactNode
}

export const ROLES: RoleDef[] = [
  {
    id: 'urban_planner',
    label: 'Urban Planner',
    color: '#38BDF8',
    description: 'I plan and manage city infrastructure, transportation networks, and policies that shape how people move through urban areas.',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>
      </svg>
    ),
  },
  {
    id: 'traffic_engineer',
    label: 'Traffic Engineer',
    color: '#A78BFA',
    description: 'I design, analyze, and optimize traffic systems — from signal timing to road capacity — to keep vehicles moving safely and efficiently.',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
      </svg>
    ),
  },
  {
    id: 'software_engineer',
    label: 'IT / Software Engineer',
    color: '#4ADE80',
    description: 'I build and evaluate software systems, algorithms, and technical infrastructure — including AI-powered applications.',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>
      </svg>
    ),
  },
]

interface RoleCtx {
  role: RoleId | null
  roleDef: RoleDef | null
  setRole: (r: RoleId) => void
  clearRole: () => void
  showModal: boolean
  openModal: () => void
  closeModal: () => void
}

const RoleContext = createContext<RoleCtx>({
  role: null, roleDef: null,
  setRole: () => {}, clearRole: () => {},
  showModal: false, openModal: () => {}, closeModal: () => {},
})

export function RoleProvider({ children }: { children: React.ReactNode }) {
  const [role, setRoleState] = useState<RoleId | null>(null)
  const [showModal, setShowModal] = useState(false)
  const [hydrated, setHydrated] = useState(false)

  useEffect(() => {
    const saved = localStorage.getItem('civiq-role') as RoleId | null
    if (saved && ROLES.find(r => r.id === saved)) {
      setRoleState(saved)
    } else {
      setShowModal(true)
    }
    setHydrated(true)
  }, [])

  const setRole = (r: RoleId) => {
    localStorage.setItem('civiq-role', r)
    setRoleState(r)
    setShowModal(false)
  }

  const clearRole = () => {
    localStorage.removeItem('civiq-role')
    setRoleState(null)
    setShowModal(true)
  }

  const roleDef = role ? (ROLES.find(r => r.id === role) ?? null) : null

  if (!hydrated) return <>{children}</>

  return (
    <RoleContext.Provider value={{ role, roleDef, setRole, clearRole, showModal, openModal: () => setShowModal(true), closeModal: () => setShowModal(false) }}>
      {children}
    </RoleContext.Provider>
  )
}

export const useRole = () => useContext(RoleContext)

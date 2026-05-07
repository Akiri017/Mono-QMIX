'use client'

import { Button } from './Button'
import Image from 'next/image'

export const Header = () => {
  return (
    <header className="sticky top-0 z-50 bg-white border-b border-gray-200 shadow-sm">
      <div className="max-w-[1400px] mx-auto px-6">
        <div className="flex items-center justify-between h-[64px]">
          <a href="/" className="flex items-center gap-2.5 hover:opacity-80 transition-opacity">
            <div className="relative w-[36px] h-[36px]">
              <img
                alt="Civiq Logo"
                className="w-full h-full object-contain"
                src="/icons/civiq-logo.png"
              />
            </div>
            <p className="font-bold text-[20px] text-civiq-purple">Civiq</p>
          </a>

          <div className="flex items-center gap-6">
            <a href="/#about" className="font-medium text-[15px] text-civiq-dark hover:text-civiq-purple transition-colors">
              About Civiq
            </a>
            <Button variant="primary">Contact Us</Button>
          </div>
        </div>
      </div>
    </header>
  )
}

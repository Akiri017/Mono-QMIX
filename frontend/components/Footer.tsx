'use client'

import Image from 'next/image'

export const Footer = () => {
  return (
    <footer className="bg-civiq-purple text-white mt-16">
      <div className="max-w-7xl mx-auto px-6 sm:px-8">
        {/* Top section with branding and social icons */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-8 py-12 border-b border-white border-opacity-20">
          <div className="flex items-center gap-4">
            <Image
              src="/icons/civiq-logo.png"
              alt="Civiq Logo"
              width={55}
              height={55}
              className="flex-shrink-0"
            />
            <span className="text-sm text-white text-opacity-70">A Hierarchical Multi-Agent Coordination Framework</span>
          </div>
          <div className="flex gap-6">
            <a 
              href="https://github.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-white hover:text-opacity-70 transition-opacity" 
              aria-label="GitHub"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.6.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v 3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
            </a>
            <a 
              href="https://www.linkedin.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-white hover:text-opacity-70 transition-opacity"
              aria-label="LinkedIn"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.225 0z"/>
              </svg>
            </a>
          </div>
        </div>

        {/* Links section */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-8 sm:gap-16 py-12">
          {/* Explore */}
          <div>
            <h3 className="font-semibold text-white text-sm mb-4 uppercase tracking-wide">Explore</h3>
            <ul className="space-y-2">
              <li><a href="#" className="text-white text-opacity-80 hover:text-opacity-100 transition-opacity text-sm">Selfish Routing</a></li>
              <li><a href="#" className="text-white text-opacity-80 hover:text-opacity-100 transition-opacity text-sm">Monolithic QMIX</a></li>
              <li><a href="#" className="text-white text-opacity-80 hover:text-opacity-100 transition-opacity text-sm">Civiq</a></li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h3 className="font-semibold text-white text-sm mb-4 uppercase tracking-wide">Resources</h3>
            <ul className="space-y-2">
              <li><a href="/#about" className="text-white text-opacity-80 hover:text-opacity-100 transition-opacity text-sm">About Civiq</a></li>
              <li><a href="/#researchers" className="text-white text-opacity-80 hover:text-opacity-100 transition-opacity text-sm">The Researchers</a></li>
              <li><a href="#" className="text-white text-opacity-80 hover:text-opacity-100 transition-opacity text-sm">Research Paper</a></li>
            </ul>
          </div>
        </div>

        {/* Copyright */}
        <div className="border-t border-white border-opacity-20 py-8 text-center">
          <p className="text-white text-opacity-70 text-sm">
            © {new Date().getFullYear()} A Priori Group. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  )
}

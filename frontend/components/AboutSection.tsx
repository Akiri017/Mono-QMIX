'use client'

import { useState, useEffect } from 'react';
import { IconAlertCircle } from './icons'

// Simple Speedometer/Gauge Icon Component
const SpeedometerIcon = () => {
  return (
    <svg viewBox="0 0 48 48" className="w-12 h-12" fill="none">
      {/* Outer circle */}
      <circle cx="24" cy="24" r="18" stroke="#04CE00" strokeWidth="2.5" fill="none" />
      
      {/* Inner arc segments */}
      <path d="M 12 24 A 12 12 0 0 1 18 14" stroke="#04CE00" strokeWidth="2" fill="none" opacity="0.4" />
      <path d="M 18 14 A 12 12 0 0 1 30 14" stroke="#04CE00" strokeWidth="2" fill="none" opacity="0.6" />
      <path d="M 30 14 A 12 12 0 0 1 36 24" stroke="#04CE00" strokeWidth="2" fill="none" opacity="0.8" />
      
      {/* Tick marks */}
      <line x1="24" y1="8" x2="24" y2="12" stroke="#04CE00" strokeWidth="1.5" />
      <line x1="36" y1="16" x2="33" y2="18" stroke="#04CE00" strokeWidth="1.5" />
      <line x1="40" y1="24" x2="36" y2="24" stroke="#04CE00" strokeWidth="1.5" />
      <line x1="12" y1="16" x2="15" y2="18" stroke="#04CE00" strokeWidth="1.5" />
      <line x1="8" y1="24" x2="12" y2="24" stroke="#04CE00" strokeWidth="1.5" />
      
      {/* Needle pointing to fast/right */}
      <line x1="24" y1="24" x2="34" y2="18" stroke="#04CE00" strokeWidth="2.5" strokeLinecap="round" />
      
      {/* Center dot */}
      <circle cx="24" cy="24" r="2" fill="#04CE00" />
    </svg>
  );
};

// Simple Line Chart Component
const LineChart = ({ data = [2, 4, 3, 5, 4, 6, 5, 7] }: { data?: number[] }) => {
  const max = Math.max(...data);
  const points = data.map((d, i) => ({
    x: (i / (data.length - 1)) * 100,
    y: 100 - (d / max) * 80,
  }));
  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');

  return (
    <svg viewBox="0 0 100 100" className="w-14 h-14 transition-transform duration-300 hover:scale-110">
      <defs>
        <linearGradient id="lineGradient" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#04CE00" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#04CE00" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path 
        d={pathD + ' L 100 100 L 0 100 Z'} 
        fill="url(#lineGradient)" 
      />
      <path 
        d={pathD} 
        stroke="#04CE00" 
        strokeWidth="2.5" 
        fill="none" 
        vectorEffect="non-scaling-stroke"
        className="transition-all duration-300"
      />
    </svg>
  );
};

// Simple Circular Metric Component - cleaner design
const CircleMetric = ({ 
  value, 
  max = 100, 
  size = 'small'
}: { 
  value: number; 
  max?: number; 
  size?: 'small' | 'large' | 'xlarge' | 'xxlarge';
}) => {
  const percentage = Math.min((value / max) * 100, 100);
  const sizeConfig = {
    small: { radius: 35, viewBox: '0 0 120 120', center: 60, strokeWidth: 8, className: 'w-12 h-12' },
    large: { radius: 45, viewBox: '0 0 140 140', center: 70, strokeWidth: 10, className: 'w-20 h-20' },
    xlarge: { radius: 55, viewBox: '0 0 160 160', center: 80, strokeWidth: 12, className: 'w-32 h-32' },
    xxlarge: { radius: 65, viewBox: '0 0 180 180', center: 90, strokeWidth: 14, className: 'w-40 h-40' }
  };
  
  const config = sizeConfig[size || 'small'];
  const { radius, viewBox, center: centerX, strokeWidth, className: svgClassName } = config;
  const centerY = centerX;
  
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (percentage / 100) * circumference;

  return (
    <svg viewBox={viewBox} className={`${svgClassName} transition-transform duration-300 hover:scale-110`} style={{ transform: 'rotate(-90deg)' }}>
      {/* Background circle */}
      <circle cx={centerX} cy={centerY} r={radius} fill="none" stroke="#e8e8e8" strokeWidth={strokeWidth} />
      
      {/* Progress circle */}
      <circle
        cx={centerX}
        cy={centerY}
        r={radius}
        fill="none"
        stroke="#04CE00"
        strokeWidth={strokeWidth}
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        strokeLinecap="round"
        style={{ transition: 'stroke-dashoffset 0.8s ease-in-out' }}
      />
    </svg>
  );
};

export const AboutSection = () => {
  const [hoveredTooltip, setHoveredTooltip] = useState<string | null>(null);
  const [isVisible, setIsVisible] = useState(false);

  // Trigger animations on mount
  useEffect(() => {
    setTimeout(() => setIsVisible(true), 100);
  }, []);

  return (
    <div className="w-full space-y-16">
      {/* About Civiq */}
      <div className={`transition-all duration-700 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
        <h2 className="font-bold text-civiq-purple text-[42px] mb-6">About Civiq</h2>
        <p className="text-civiq-dark text-[18px] text-justify leading-relaxed">
          Civiq is a hierarchical software framework designed to redefine urban traffic management
          through cooperative intelligence. Built around the QMIX algorithm, Civiq addresses the
          scalability challenges of traditional Multi-Agent Reinforcement Learning (MARL) by utilizing
          a three-level architecture: individual vehicle agents, local coordination via Roadside Units
          (RSUs), and a central server for global optimization. By integrating directly with the SUMO
          (Simulation of Urban Mobility) environment, Civiq transforms selfish routing behaviors into
          a synchronized network, significantly improving throughput and reducing urban congestion.
        </p>
      </div>

      {/* Smart Traffic Routing */}
      <div className={`transition-all duration-700 delay-100 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
        {/* Metrics Row - Horizontal */}
        <div className="flex justify-center gap-8 mb-8">
          {/* ACT */}
          <div 
            className="bg-white rounded-[24px] shadow-md p-6 w-[240px] h-[200px] hover:shadow-xl hover:scale-105 transition-all duration-300 cursor-pointer flex flex-col items-center justify-between relative"
            onMouseEnter={() => setHoveredTooltip('act')}
            onMouseLeave={() => setHoveredTooltip(null)}
          >
            {/* Tooltip Icon - Top Right */}
            <div className="absolute top-4 right-4">
              <IconAlertCircle size={16} />
              {hoveredTooltip === 'act' && (
                <div className="absolute right-0 top-6 bg-civiq-dark text-white text-[12px] font-semibold px-3 py-2 rounded-lg whitespace-nowrap z-10 shadow-lg pointer-events-none">
                  Time taken to compute optimal routing decisions
                </div>
              )}
            </div>

            {/* Speedometer Icon */}
            <div className="mt-2">
              <SpeedometerIcon />
            </div>
            
            {/* Numerical Value */}
            <div className="flex items-baseline justify-center gap-1 mt-4">
              <p className="font-bold text-civiq-dark text-[36px] leading-none">22.35</p>
              <p className="text-civiq-dark text-[14px]">ms</p>
            </div>

            {/* Label at bottom */}
            <p className="text-civiq-blue text-[13px] font-medium text-center">Average Compute Time</p>
          </div>

          {/* ATT */}
          <div 
            className="bg-white rounded-[24px] shadow-md p-6 w-[240px] h-[200px] hover:shadow-xl hover:scale-105 transition-all duration-300 cursor-pointer flex flex-col items-center justify-between relative"
            onMouseEnter={() => setHoveredTooltip('att')}
            onMouseLeave={() => setHoveredTooltip(null)}
          >
            {/* Top Right: Tooltip Icon and Improvement */}
            <div className="absolute top-4 right-4 flex items-center gap-2">
              <div className="flex items-center gap-1">
                <img alt="Trending Up" src="/icons/trending_up.svg" className="w-3 h-3" style={{filter: 'invert(1) hue-rotate(130deg) brightness(1.2) saturate(2)'}} />
                <p className="text-civiq-green text-[11px] animate-pulse whitespace-nowrap">+12.34%</p>
              </div>
              <div className="relative">
                <IconAlertCircle size={16} />
                {hoveredTooltip === 'att' && (
                  <div className="absolute right-0 top-6 bg-civiq-dark text-white text-[12px] font-semibold px-3 py-2 rounded-lg whitespace-nowrap z-10 shadow-lg pointer-events-none">
                    Average time for vehicles to complete their journeys
                  </div>
                )}
              </div>
            </div>

            {/* Graph */}
            <div className="flex justify-center mt-2">
              <LineChart data={[2, 3, 2.5, 4, 3.5, 5, 4, 4.2]} />
            </div>
            
            {/* Numerical Value */}
            <div className="flex items-baseline justify-center gap-1 mt-4">
              <p className="font-bold text-civiq-dark text-[36px] leading-none">4.2</p>
              <p className="text-civiq-dark text-[14px]">min</p>
            </div>

            {/* Label at bottom */}
            <p className="text-civiq-blue text-[13px] font-medium text-center">Average Travel Time</p>
          </div>

          {/* Network Throughput */}
          <div 
            className="bg-white rounded-[24px] shadow-md p-6 w-[240px] h-[200px] hover:shadow-xl hover:scale-105 transition-all duration-300 cursor-pointer flex flex-col items-center justify-between relative"
            onMouseEnter={() => setHoveredTooltip('throughput')}
            onMouseLeave={() => setHoveredTooltip(null)}
          >
            {/* Top Right: Tooltip Icon and Improvement */}
            <div className="absolute top-4 right-4 flex items-center gap-2">
              <div className="flex items-center gap-1">
                <img alt="Trending Up" src="/icons/trending_up.svg" className="w-3 h-3" style={{filter: 'invert(1) hue-rotate(130deg) brightness(1.2) saturate(2)'}} />
                <p className="text-civiq-green text-[11px] animate-pulse whitespace-nowrap">+8.32%</p>
              </div>
              <div className="relative">
                <IconAlertCircle size={16} />
                {hoveredTooltip === 'throughput' && (
                  <div className="absolute right-0 top-6 bg-civiq-dark text-white text-[12px] font-semibold px-3 py-2 rounded-lg whitespace-nowrap z-10 shadow-lg pointer-events-none">
                    Number of vehicles processed through the network per hour
                  </div>
                )}
              </div>
            </div>

            {/* Graph */}
            <div className="flex justify-center mt-2">
              <LineChart data={[1200, 1400, 1350, 1600, 1500, 1750, 1800, 1875]} />
            </div>
            
            {/* Numerical Value */}
            <div className="flex items-baseline justify-center gap-1 mt-4">
              <p className="font-bold text-civiq-dark text-[36px] leading-none">1,875</p>
              <p className="text-civiq-dark text-[14px]">veh/hr</p>
            </div>

            {/* Label at bottom */}
            <p className="text-civiq-blue text-[13px] font-medium text-center overflow-hidden text-ellipsis">Network Throughput</p>
          </div>
        </div>

        {/* Description Below */}
        <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-[24px] shadow-md p-8 hover:shadow-xl transition-shadow duration-300 mt-12">
          <h3 className="font-bold text-civiq-blue text-[32px] mb-6 text-left">Smart Traffic Routing, Scalable</h3>
          <p className="text-civiq-dark text-[18px] text-justify leading-relaxed">
              Civiq transforms congested urban traffic into a coordinated and efficient network by
              enabling vehicles to operate cooperatively rather than competitively. By distributing
              processing tasks between intelligent roadside sensors and a centralized optimization
              system, Civiq maintains high speed and accuracy even as traffic volume increases.
              This scalable architecture allows the AI to perform complex routing decisions in 22.35
              milliseconds, ensuring rapid response to changing traffic conditions. By keeping vehicle
              flow smooth and consistent, Civiq enables effective real-time traffic management and
              significantly increases the number of vehicles that can move through the city with minimal
              delays.
            </p>
        </div>
      </div>

      {/* Green Edge Computing */}
      <div className={`transition-all duration-700 delay-200 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
        <h3 className="font-bold text-civiq-blue text-[32px] mb-8">Green Edge Computing</h3>
        
        <div className="grid grid-cols-2 gap-10">
          {/* Left: Description */}
          <div className="space-y-4">
            <p className="text-civiq-dark text-[18px] text-justify leading-relaxed">
              Civiq improves urban sustainability by applying Green Edge Computing to traffic management
              at the point where traffic occurs. Instead of transmitting all data to a centralized data
              center, the system uses intelligent edge devices installed at each intersection to make
              timely and energy-efficient decisions locally.
            </p>
            <p className="text-civiq-dark text-[18px] text-justify leading-relaxed">
              This decentralized approach reduces unnecessary data processing, limits energy waste, and
              ensures that traffic signals contribute to environmental efficiency across the city. As a
              result, Civiq achieves an average 15% reduction in CO₂ emissions, maintaining a low
              footprint of 142 g/km even during peak traffic conditions.
            </p>
            <p className="text-civiq-dark text-[18px] text-justify leading-relaxed italic">
              Through Civiq, smarter traffic control directly supports a cleaner, more sustainable urban
              environment.
            </p>
          </div>

          {/* Right: Circular Metrics - Horizontal */}
          <div className="flex gap-12 justify-center items-center">
            {/* Average CO2 Emissions */}
            <div className="text-center">
              <div className="relative w-[200px] h-[200px] mb-4 mx-auto">
                <svg className="w-full h-full -rotate-90">
                  <circle 
                    cx="50%" 
                    cy="50%" 
                    r="45%" 
                    fill="none" 
                    stroke="#e5e7eb" 
                    strokeWidth="16"
                  />
                  <circle 
                    cx="50%" 
                    cy="50%" 
                    r="45%" 
                    fill="none" 
                    stroke="#7FE47E" 
                    strokeWidth="16"
                    strokeDasharray="565.5"
                    strokeDashoffset="160"
                    strokeLinecap="round"
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <p className="text-5xl font-bold text-civiq-dark">142</p>
                  <p className="text-base text-civiq-dark">g/km</p>
                </div>
              </div>
              <p className="text-sm italic text-civiq-dark">Average CO2 Emissions per kilometer</p>
            </div>

            {/* Average Fuel Consumption */}
            <div className="text-center">
              <div className="relative w-[200px] h-[200px] mb-4 mx-auto">
                <svg className="w-full h-full -rotate-90">
                  <circle 
                    cx="50%" 
                    cy="50%" 
                    r="45%" 
                    fill="none" 
                    stroke="#e5e7eb" 
                    strokeWidth="16"
                  />
                  <circle 
                    cx="50%" 
                    cy="50%" 
                    r="45%" 
                    fill="none" 
                    stroke="#04CE00" 
                    strokeWidth="16"
                    strokeDasharray="565.5"
                    strokeDashoffset="70"
                    strokeLinecap="round"
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <p className="text-5xl font-bold text-civiq-dark">23</p>
                  <p className="text-base text-civiq-dark">g/km</p>
                </div>
              </div>
              <p className="text-sm italic text-civiq-dark">Average Fuel Consumption of a Vehicle</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

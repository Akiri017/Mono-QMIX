'use client'

import React from 'react'

interface ButtonProps {
  variant?: 'primary' | 'secondary'
  onPress?: () => void
  children: React.ReactNode
  className?: string
  fullWidth?: boolean
  disabled?: boolean
}

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  onPress,
  children,
  className = '',
  fullWidth = false,
  disabled = false,
}) => {
  const baseStyles = 'rounded-[30px] px-6 py-3 font-semibold text-[15px] transition-all duration-200 flex items-center justify-center'
  const variants = {
    primary: disabled 
      ? 'bg-gray-300 text-gray-500 shadow-sm'
      : 'bg-civiq-purple text-white hover:bg-opacity-90 shadow-md hover:shadow-lg',
    secondary: disabled
      ? 'bg-gray-100 text-gray-400 border-2 border-gray-300'
      : 'bg-white text-civiq-purple border-2 border-civiq-purple hover:bg-opacity-90',
  }
  const widthStyle = fullWidth ? 'w-full' : ''
  const cursorStyle = disabled ? 'cursor-not-allowed' : 'cursor-pointer'

  return (
    <button
      onClick={disabled ? undefined : onPress}
      disabled={disabled}
      className={`${baseStyles} ${variants[variant]} ${widthStyle} ${cursorStyle} ${className}`}
    >
      {children}
    </button>
  )
}

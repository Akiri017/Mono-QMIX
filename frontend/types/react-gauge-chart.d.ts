declare module 'react-gauge-chart' {
  import * as React from 'react'

  interface GaugeChartProps {
    id?: string
    percent?: number
    nrOfLevels?: number
    colors?: string[]
    arcWidth?: number
    arcPadding?: number
    cornerRadius?: number
    needleColor?: string
    needleBaseColor?: string
    hideText?: boolean
    animate?: boolean
    animDelay?: number
    animateDuration?: number
    formatTextValue?: (value: string) => string
    textColor?: string
    fontSize?: string
    style?: React.CSSProperties
    className?: string
    marginInPercent?: number
  }

  const GaugeChart: React.FC<GaugeChartProps>
  export default GaugeChart
}

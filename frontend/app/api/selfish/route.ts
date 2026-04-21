/**
 * GET /api/selfish?trafficLevel=free_flow|stable_flow|forced_flow&map=2km|0.75km|4x4
 *
 * Returns real KPIs and time-series data for the Selfish Routing algorithm.
 * Data availability:
 *   2km   (bgc_full)  — free_flow, stable_flow, forced_flow  ✓  (PyMARL pipeline, 30 eps)
 *   0.75km (bgc_core) — free_flow, stable_flow               ✓  (SUMO stats XML; forced_flow run incomplete)
 *   4x4               — free_flow, stable_flow, forced_flow  ✓  (PyMARL pipeline, 100 eps)
 */
import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

// data/ lives inside the frontend directory — self-contained for migration
const RESULTS_DIR = path.join(process.cwd(), 'data', 'selfish_routing')

type TrafficLevel = 'free_flow' | 'stable_flow' | 'forced_flow'
const VALID_LEVELS: TrafficLevel[] = ['free_flow', 'stable_flow', 'forced_flow']

// Map query param → metrics file
const MAP_METRICS: Record<string, string> = {
  '2km':    'metrics_bgc_full.json',
  '0.75km': 'metrics_bgc_core.json',
  '4x4':    'metrics_4x4.json',
}

function parseCSV(content: string): { step: number; active_vehicles: number; total_system_wait: number }[] {
  const lines = content.trim().split('\n')
  return lines.slice(1).map(line => {
    const [step, active_vehicles, total_system_wait] = line.split(',').map(Number)
    return { step, active_vehicles, total_system_wait }
  })
}

export async function GET(request: NextRequest) {
  try {
    const trafficLevel = (request.nextUrl.searchParams.get('trafficLevel') || 'forced_flow') as TrafficLevel
    const mapParam = request.nextUrl.searchParams.get('map') || '2km'

    if (!VALID_LEVELS.includes(trafficLevel)) {
      return NextResponse.json(
        { success: false, error: `Invalid trafficLevel. Must be one of: ${VALID_LEVELS.join(', ')}` },
        { status: 400 }
      )
    }

    const metricsFile = MAP_METRICS[mapParam]
    if (!metricsFile) {
      return NextResponse.json(
        { success: false, error: `Unsupported map: ${mapParam}` },
        { status: 400 }
      )
    }

    // Load compiled metrics
    const metricsPath = path.join(RESULTS_DIR, metricsFile)
    if (!fs.existsSync(metricsPath)) {
      return NextResponse.json(
        { success: false, error: `${metricsFile} not found.` },
        { status: 404 }
      )
    }

    const metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf-8'))
    const levelData = metrics.traffic_levels[trafficLevel]

    if (!levelData) {
      return NextResponse.json(
        { success: false, error: `No data for traffic level "${trafficLevel}" on map "${mapParam}". ${metrics.notes || ''}` },
        { status: 404 }
      )
    }

    const kpis = levelData.kpis
    const vehicles = levelData.vehicles

    // Load timeseries CSV (timeseries_file may be null for some maps)
    let timeseries: ReturnType<typeof parseCSV> = []
    if (levelData.timeseries_file) {
      const csvPath = path.join(RESULTS_DIR, levelData.timeseries_file)
      if (fs.existsSync(csvPath)) {
        timeseries = parseCSV(fs.readFileSync(csvPath, 'utf-8'))
      }
    }

    return NextResponse.json({
      success: true,
      algorithm: 'selfish_routing',
      trafficLevel,
      map: mapParam,
      vehicles,
      baselines: levelData.baselines ?? null,
      kpis: {
        // Primary display KPIs (matching AlgoData fields)
        travelTime: parseFloat(kpis.avg_travel_time_min.toFixed(2)),             // min
        waitTime:   parseFloat(kpis.avg_waiting_time_s.toFixed(1)),               // sec
        throughput: parseFloat(kpis.network_throughput_veh_h.toFixed(0)),         // veh/hr
        speed:      parseFloat(kpis.real_time_factor.toFixed(2)),                  // real-time factor ×
        co2:        kpis.avg_co2_g_per_km,                                         // g/km
        fuel:       kpis.avg_fuel_l_per_100km,                                     // L/100km
        avgSpeedKmh: kpis.avg_speed_kmh,
        avgRouteLengthM: kpis.avg_route_length_m,
        timeLossS: kpis.time_loss_s,
        returnMean: kpis.returnMean ?? null,
      },
      timeseries: {
        steps: timeseries.map(r => r.step),
        activeVehicles: timeseries.map(r => r.active_vehicles),
        totalSystemWait: timeseries.map(r => r.total_system_wait),
      },
      meta: {
        source_stats: levelData.source_stats,
        source_tripinfo: levelData.source_tripinfo,
        generated_at: metrics.generated_at,
        map: metrics.map,
        simulation_duration_s: levelData.simulation_duration_s ?? metrics.simulation_duration_s,
      },
    })
  } catch (error) {
    console.error('[/api/selfish] Error:', error)
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    )
  }
}

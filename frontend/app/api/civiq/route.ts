/**
 * GET /api/civiq?scenario=los_a|los_c|los_e
 *
 * Returns real KPIs and evaluation data for CiViQ (Hierarchical QMIX).
 * LOS A: results/civiq/civiq-los-a/ (seed 1804, 1M timesteps, 30 eval episodes)
 * LOS C: results/civiq/civiq-los-c/ (seed 1805, 1M timesteps, 30 eval episodes)
 * LOS E: results/civiq/civiq-los-e/ (seed 1806, 1M timesteps, 30 eval episodes)
 */
import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

const DATA_DIR = path.join(process.cwd(), 'data', 'civiq')

const SCENARIO_FILES: Record<string, string> = {
  los_a: 'metrics_los_a.json',
  los_c: 'metrics_los_c.json',
  los_e: 'metrics_los_e.json',
}

export async function GET(request: NextRequest) {
  try {
    const scenario = request.nextUrl.searchParams.get('scenario') || 'los_a'

    const file = SCENARIO_FILES[scenario]
    if (!file) {
      return NextResponse.json(
        { success: false, error: `Unknown scenario: ${scenario}. Available: ${Object.keys(SCENARIO_FILES).join(', ')}` },
        { status: 400 }
      )
    }

    const filePath = path.join(DATA_DIR, file)
    if (!fs.existsSync(filePath)) {
      return NextResponse.json(
        { success: false, error: `Data file not found: ${file}` },
        { status: 404 }
      )
    }

    const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'))
    const k = data.kpis

    return NextResponse.json({
      success: true,
      algorithm: 'civiq',
      scenario,
      map: data.map,
      seed: data.seed,
      nEvalEpisodes: data.n_eval_episodes,

      kpis: {
        travelTime_s:     k.travelTime_s,
        travelTime_min:   parseFloat((k.travelTime_s / 60).toFixed(3)),
        travelTime_std:   k.travelTime_std,
        waitTime_s:       k.waitTime_s,
        waitTime_std:     k.waitTime_std,
        throughput:       k.throughput,
        throughput_std:   k.throughput_std,
        co2:              k.co2_g_per_km,
        fuel:             k.fuel_l_per_100km,
        cpuMean:          k.cpuMean,
        cpuPeak:          k.cpuPeak,
        arrivalRate:      k.arrivalRate,
        avgRouteLength_m: k.avgRouteLength_m,
        totalStops:       k.totalStops,
        returnMean:       k.returnMean,
        realTimeFactor:   null,
      },

      baselines: data.baselines,

      // Per-episode evaluation arrays (30 episodes)
      evalReturns:      data.evalReturns,
      evalTravelTimes:  data.evalTravelTimes,
      evalWaitingTimes: data.evalWaitingTimes,
      evalThroughputs:  data.evalThroughputs,

      // No training curve available (tfevents not yet parsed)
      training: data.training,
      trainMetrics: data.trainMetrics ?? null,
      testCurve: data.testCurve,

      meta: {
        generated_at:  data.generated_at,
        t_max:         data.t_max,
        training_note: data.training.note,
        train_pts:     0,
      },
    })
  } catch (error) {
    console.error('[/api/civiq] Error:', error)
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    )
  }
}

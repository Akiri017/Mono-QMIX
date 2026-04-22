/**
 * GET /api/qmix?scenario=los_a|los_c|los_e
 *
 * Returns real KPIs and training curve data for Monolithic QMIX.
 * LOS A: results/mono-qmix-los-a/ (seed 1801, 1M timesteps, 30 eval episodes)
 * LOS C: results/mono-qmix/mono-qmix-los-c/ (seed 1802, 1M timesteps, 30 eval episodes)
 * LOS E: results/mono-qmix-los-e/ (seed 1803, 1M timesteps, 30 eval episodes)
 */
import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

const DATA_DIR = path.join(process.cwd(), 'data', 'mono_qmix')

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
      algorithm: 'monolithic_qmix',
      scenario,
      map: data.map,
      seed: data.seed,
      nEvalEpisodes: data.n_eval_episodes,

      kpis: {
        // travel time in seconds (raw) and minutes (for display)
        travelTime_s:      k.travelTime_s,
        travelTime_min:    parseFloat((k.travelTime_s / 60).toFixed(3)),
        travelTime_std:    k.travelTime_std,
        waitTime_s:        k.waitTime_s,
        waitTime_std:      k.waitTime_std,
        throughput:        k.throughput,
        throughput_std:    k.throughput_std,
        co2:               k.co2_g_per_km,
        fuel:              k.fuel_l_per_100km,
        cpuMean:           k.cpuMean,
        cpuPeak:           k.cpuPeak,
        arrivalRate:       k.arrivalRate,
        avgRouteLength_m:  k.avgRouteLength_m,
        totalStops:        k.totalStops,
        returnMean:        k.returnMean,
        realTimeFactor:    k.realTimeFactor ?? null,
      },

      baselines: data.baselines,

      // Per-episode evaluation arrays (30 episodes)
      evalReturns:      data.evalReturns,
      evalTravelTimes:  data.evalTravelTimes,
      evalWaitingTimes: data.evalWaitingTimes,
      evalThroughputs:  data.evalThroughputs,
      evalCpuMeans:     data.evalCpuMeans ?? null,

      // Training curve (points depend on log coverage — see meta.training_note)
      training: data.training,

      // Combined text-log + tfevents training metrics (travelTime, waitTime, arrivalRate, speedMs, loss, etc.)
      trainMetrics: data.trainMetrics ?? null,

      // Test/validation checkpoints during training
      testCurve: data.testCurve,

      meta: {
        generated_at:   data.generated_at,
        t_max:          data.t_max,
        training_note:  data.training.note,
        train_pts:      data.trainMetrics?.curve?.length ?? 0,
      },
    })
  } catch (error) {
    console.error('[/api/qmix] Error:', error)
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    )
  }
}

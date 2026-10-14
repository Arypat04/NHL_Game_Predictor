import { useEffect, useRef, useState } from "react"
import "./ProbScale.css"

/*
 * Simplified instrument scale: left=away (0%), right=home (100%), center=even.
 * - model reading: a filled pointer at the model's home-win probability
 * - market reference (when odds exist): a hollow ring at the implied probability
 * - the span between them is the edge — a cyan band; the side the pointer sits
 *   on relative to the ring is the value side.
 * No lean fill, no favor-coloring — the scale confirms the numbers beside it
 * rather than being the only source of them.
 */
export default function ProbScale({ modelHome, marketHome = null, edge = null, edgeNegative = false, homeCode = "HOME", awayCode = "AWAY" }) {
  const hasMarket = marketHome != null && !Number.isNaN(marketHome)
  const start = hasMarket ? marketHome : 0.5
  const [pos, setPos] = useState(start)
  const raf = useRef(0)

  useEffect(() => {
    raf.current = requestAnimationFrame(() => requestAnimationFrame(() => setPos(modelHome)))
    return () => cancelAnimationFrame(raf.current)
  }, [modelHome])

  const pct = (v) => `${(v * 100).toFixed(1)}%`
  const caliperLeft = hasMarket ? Math.min(pos, marketHome) : null
  const caliperW = hasMarket ? Math.abs(pos - marketHome) : 0
  const valueCode = hasMarket ? (modelHome > marketHome ? homeCode : awayCode) : null

  const label = hasMarket
    ? `Model ${pct(modelHome)} home win vs market ${pct(marketHome)} implied; edge ${edge} points toward ${valueCode}.`
    : `Model: ${awayCode} ${pct(1 - modelHome)}, ${homeCode} ${pct(modelHome)} to win.`

  return (
    <div className="scale" role="img" aria-label={label} data-neg={edgeNegative ? "true" : undefined}>
      <div className="scale-track">
        {[0, 25, 50, 75, 100].map((t) => (
          <span key={t} className={`tick ${t === 50 ? "tick-even" : ""}`} style={{ left: `${t}%` }} />
        ))}
        {hasMarket && <span className="caliper" style={{ left: pct(caliperLeft), width: pct(caliperW) }} />}
        {hasMarket && <span className="mark-market" style={{ left: pct(marketHome) }} aria-hidden="true" />}
        <span className="mark-model" style={{ left: pct(pos) }} aria-hidden="true" />
      </div>
      <div className="scale-base">
        <span className="scale-cap">{awayCode}</span>
        <span className="scale-even">EVEN</span>
        <span className="scale-cap scale-cap--r">{homeCode}</span>
      </div>
    </div>
  )
}

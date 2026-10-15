import { useState } from "react"

/* Persistent key — the board rests on decoding these marks and the pick/bet
   split. Inline on desktop; collapsible on mobile so it never buries the first
   game. */
export default function Legend() {
  const [open, setOpen] = useState(() => {
    try { return !window.matchMedia?.("(max-width: 720px)").matches } catch { return true }
  })

  return (
    <div className={`legend ${open ? "is-open" : ""}`} role="note" aria-label="How to read the board">
      <button className="legend-toggle" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <span className="legend-k">How to read</span>
        <span className="legend-chevron" aria-hidden="true" />
      </button>

      <div className="legend-body">
        <span className="legend-k legend-k--inline">How to read</span>
        <span className="legend-item">
          <span className="lg-ring" aria-hidden="true" />
          <span>Market implied</span>
        </span>
        <span className="legend-item">
          <span className="lg-pointer" aria-hidden="true" />
          <span>Model reading</span>
        </span>
        <span className="legend-item">
          <span className="lg-bar" aria-hidden="true" />
          <span><b>Edge</b> = model&nbsp;% − market&nbsp;% <span className="legend-mute">(in points)</span></span>
        </span>
        <span className="legend-sep" aria-hidden="true" />
        <span className="legend-item"><b>Pick</b> = model's winner</span>
        <span className="legend-item"><b>Bet</b> = best value (can differ from the pick)</span>
        <span className="legend-orient">left = away · right = home · times ET</span>
      </div>
    </div>
  )
}

import { IconHit, IconMiss } from "./Icons"
import { byTime, splitTime } from "./format"
import { ReadingSkeleton, surfaceStateFor } from "./States"

function ResultsTable({ results, loading, error }) {
  if (loading) return <ReadingSkeleton />
  const state = surfaceStateFor(error, {
    empty: !results || results.length === 0,
    emptyProps: { title: "No results yet", msg: "No completed games to grade for this date." },
  })
  if (state) return state

  const rows = [...results].sort(byTime)

  return (
    <div className="readings">
      {rows.map((g, i) => {
        const { t, ap } = splitTime(g.Time)
        const live = g.Status === "Live"
        const homeWon = g.Actual_Winner === g.Home
        return (
          <div className="reading" key={i}>
            <div className="r-time">
              {t}<span className="r-ampm"> {ap}</span>
              <div className={`r-status ${live ? "live" : ""}`}>
                {live && <span className="dot dot-live" />}
                {g.Status}
              </div>
            </div>

            <div className="r-match">
              <div className={`r-team ${!homeWon ? "is-pick" : ""}`}>
                <span className="r-code mono">{g.Away}</span>
              </div>
              <div className={`r-team ${homeWon ? "is-pick" : ""}`}>
                <span className="r-at">@</span>
                <span className="r-code mono">{g.Home}</span>
              </div>
            </div>

            <div className="r-scale" style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <div className="r-score mono">
                {g.Away_Score}<span className="r-sep">–</span>{g.Home_Score}
              </div>
              <div className="r-sub">pick&nbsp;<span className="mono" style={{ color: "var(--ink-dim)" }}>{g.Predicted_Winner}</span></div>
            </div>

            <div className="r-read">
              <div className={`r-verdict ${g.Correct ? "hit" : "miss"}`}>
                <span className="chip">{g.Correct ? <IconHit size={14} /> : <IconMiss size={13} />}</span>
                {g.Correct ? "HIT" : "MISS"}
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

export default ResultsTable

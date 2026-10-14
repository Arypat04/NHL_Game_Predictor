/* Renders the instrument status readouts as a fragment of .stat cells; App
   wraps them in the header strip (desktop) and a stacked strip (mobile). */
function StatsBar({ status, oddsLive }) {
  const acc = status ? `${(status.season_accuracy * 100).toFixed(1)}%` : "—"
  const n   = status ? status.total_predictions.toLocaleString() : "—"
  const trained = status
    ? new Date(status.model_last_trained).toLocaleDateString("en-US", { month: "short", day: "numeric" })
    : "—"
  const configured = !!(status && status.odds_api_configured)
  const live = oddsLive !== undefined ? oddsLive : configured
  const feedLabel = live ? "Live" : configured ? "Limited" : "Off"
  const feedDot = live ? "dot-live" : configured ? "dot-warn" : "dot-off"

  return (
    <>
      <div className="stat" title="Season accuracy, measured on games the model trained on — slightly optimistic vs. true cross-validation. Treat it as an upper bound.">
        <span className="stat-k">Accuracy</span>
        <span className="stat-v is-signal">{acc}</span>
      </div>
      <div className="stat" title="Number of past predictions this accuracy is measured over.">
        <span className="stat-k">Sample</span>
        <span className="stat-v">{n}</span>
      </div>
      <div className="stat">
        <span className="stat-k">Trained</span>
        <span className="stat-v">{trained}</span>
      </div>
      <div className="stat">
        <span className="stat-k">Odds feed</span>
        <span className="stat-v stat-live">
          <span className={`dot ${feedDot}`} />
          {feedLabel}
        </span>
      </div>
    </>
  )
}

export default StatsBar

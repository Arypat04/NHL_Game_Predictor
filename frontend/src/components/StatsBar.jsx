import './StatsBar.css'

function StatsBar({ status, loading, error }) {

  if (loading) {
    return <p>Loading...</p>
  }

  if (error) {
    return null 
  }

  if (!status) {
    return null
  }

return (
  <div className="stats-bar">
    <div className="stat-card">
      <div className="stat-label">Season Accuracy</div>
      <div className="stat-value positive">{(status.season_accuracy * 100).toFixed(1)}%</div>
    </div>
    <div className="stat-card">
      <div className="stat-label">Total Predictions</div>
      <div className="stat-value">{status.total_predictions}</div>
    </div>
    <div className="stat-card">
      <div className="stat-label">Last Trained</div>
      <div className="stat-value accent">{new Date(status.model_last_trained).toLocaleDateString("en-US", { month: "short", day: "numeric" })}</div>
    </div>
    <div className="stat-card">
      <div className="stat-label">Odds API</div>
      <div className="stat-value positive">{status.odds_api_configured ? "✓ Live" : "✗ Off"}</div>
    </div>
  </div>
)
}

export default StatsBar
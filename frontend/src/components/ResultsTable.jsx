import './Tables.css'
import TableSkeleton from "./TableSkeleton"


function ResultsTable({ results, loading, error }) {

  if (loading) {
    return (
      <div className="predictions-table">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Away</th>
              <th>Home</th>
              <th>Predicted Winner</th>
              <th>Home %</th>
              <th>Away %</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            <TableSkeleton rows={5} cols={7} />
          </tbody>
        </table>
      </div>
    )
  }
  if (error) {
    return <p>Error: {error}</p>
  }

  if (!results || results.length === 0) {
     return <div className="results-table"><p className="empty-state">No games scheduled for this date.</p></div>
  }

  return (
    <div className="results-table">
      <table>
        <thead>
          <tr>
            <th>Time</th>
            <th>Status</th>
            <th>Away</th>
            <th>Home</th>
            <th>Result</th>
            <th>Predicted Winner</th>
            <th>Correct?</th>
            <th>Score</th>
          </tr>
        </thead>
        <tbody>
          {results.map((game, index) => (
            <tr key={index}>
              <td>{game.Time}</td>
              <td style={{ color: game.Status === "Live" ? "var(--positive)" : "var(--text-secondary)" }}>
                {game.Status === "Live" ? "🔴 Live" : game.Status}</td>
              <td className={game.Actual_Winner === game.Away ? "winner" : ""}>
                {game.Away}
              </td>
              <td className={game.Actual_Winner === game.Home ? "winner" : ""}>
                {game.Home}
              </td>
              <td>{game.Actual_Winner}</td>
              <td>{game.Predicted_Winner}</td>
              <td className={game.Correct ? "correct" : "incorrect"}>
                {game.Correct ? "✓" : "✗"}
              </td>
              <td>{game.Away_Score} - {game.Home_Score}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default ResultsTable
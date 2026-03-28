import EdgeBadge from "./EdgeBadge"
import './Tables.css'
import TableSkeleton from "./TableSkeleton"

function PredictionsTable({ predictions, loading, error }) {

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

  if (!predictions || predictions.length === 0) {
     return <div className="predictions-table"><p className="empty-state">No games scheduled for this date.</p></div>
  }

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
          {predictions.map((game, index) => (
            <tr key={index}>
              <td>{game.Time}</td>
              <td className={game.Predicted_Winner === game.Away ? "winner" : ""}>
                {game.Away}
              </td>
              <td className={game.Predicted_Winner === game.Home ? "winner" : ""}>
                {game.Home}
              </td>
              <td>{game.Predicted_Winner}</td>
              <td>{(game.Home_Win_Prob * 100).toFixed(1)}%</td>
              <td>{(game.Away_Win_Prob * 100).toFixed(1)}%</td>
              <td><EdgeBadge value={game.Confidence} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default PredictionsTable
import EdgeBadge from "./EdgeBadge"
import './Tables.css'
import TableSkeleton from "./TableSkeleton"

function formatOdds(odds) {
  return odds > 0 ? `+${odds}` : `${odds}`
}

function formatEdge(edge) {
  const pct = (edge * 100).toFixed(1)
  return edge > 0 ? `+${pct}%` : `${pct}%`
}

function EdgesTable({ edges, loading, error }) {

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

  if (!edges || edges.length === 0) {
     return <div className="edges-table"><p className="empty-state">No games scheduled for this date.</p></div>
  }

    const sortedEdges = [...edges].sort((a, b) => {
  const parseTime = (timeStr) => {
    const [time, modifier] = timeStr.split(' ')
    let [hours, minutes] = time.split(':').map(Number)

    if (modifier === 'PM' && hours !== 12) hours += 12
    if (modifier === 'AM' && hours === 12) hours = 0

    return hours * 60 + minutes
  }

  return parseTime(a.Time) - parseTime(b.Time)
})


  return (
    <div className="edges-table">
      <table>
        <thead>
          <tr>
            <th>Time</th>
            <th>Away</th>
            <th>Home</th>
            <th>Best Bet</th>
            <th>Away Odds</th>
            <th>Home Odds</th>
            <th>Away Edge</th>
            <th>Home Edge</th>
            <th>Bookmaker</th>
          </tr>
        </thead>
        <tbody>
          {sortedEdges.map((game, index) => (
            <tr key={index}>
              <td>{game.Time}</td>
              <td className={game.Best_Bet === "Away" ? "best-bet" : ""}>
                {game.Away}
              </td>
              <td className={game.Best_Bet === "Home" ? "best-bet" : ""}>
                {game.Home}
              </td>
              <td>{game.Best_Bet === "Home" ? game.Home : game.Away}</td>
              <td>{formatOdds(game.Away_Odds)}</td>
              <td>{formatOdds(game.Home_Odds)}</td>
              <td style={{ color: game.Away_Edge > 0 ? "green" : "red" }}>
                {formatEdge(game.Away_Edge)}
              </td>
              <td style={{ color: game.Home_Edge > 0 ? "green" : "red" }}>
                {formatEdge(game.Home_Edge)}
              </td>
              <td>{game.Bookmaker}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default EdgesTable
function EdgeBadge({ value }) {
  const color = getBadgeColor(value)

  return (
    <span className="edge-badge badge" style={{ backgroundColor: color.toLowerCase() }}>
        {Math.round(value * 100)}%
    </span>
  )
}

function getBadgeColor(value) {
    const pct = value * 100
  if(pct >= 60) {
    return 'Green'
  } else if(pct >= 55) {
    return 'GoldenRod'
  } 
  return 'Red'
}

export default EdgeBadge
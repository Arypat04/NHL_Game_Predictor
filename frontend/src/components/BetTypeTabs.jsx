import './BetTypeTabs.css'

function BetTypeTabs({ selectedBetType, onBetTypeSelect }) {
  const betTypes = [
    { key: "moneyline", label: "Moneyline", active: true },
    { key: "player_props", label: "Player Props", active: false },
  ]

  return (
    <div className="bet-type-tabs">
      {betTypes.map((bet) => (
        <button
          key={bet.key}
          className={`bet-type-tab ${selectedBetType === bet.key ? "active" : ""} ${!bet.active ? "disabled" : ""}`}
          onClick={() => bet.active && onBetTypeSelect(bet.key)}
          disabled={!bet.active}
          title={!bet.active ? "Coming soon" : ""}
        >
          {bet.label}
          {!bet.active && <span className="coming-soon">Soon</span>}
        </button>
      ))}
    </div>
  )
}

export default BetTypeTabs
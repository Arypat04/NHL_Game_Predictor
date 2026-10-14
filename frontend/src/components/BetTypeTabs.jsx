import { IconLock } from "./Icons"
import { useRovingTablist } from "./roving"

const BET_TYPES = [
  { key: "moneyline", label: "Moneyline", active: true },
  { key: "player_props", label: "Player Props", active: false },
]

function BetTypeTabs({ selectedBetType, onBetTypeSelect }) {
  const { refs, onKeyDown } = useRovingTablist()
  return (
    <div className="seg seg--secondary" role="tablist" aria-label="Bet type">
      {BET_TYPES.map((bet, i) => (
        <button
          key={bet.key}
          ref={(el) => (refs.current[i] = el)}
          role="tab"
          aria-selected={selectedBetType === bet.key}
          aria-controls="linelab-panel"
          aria-disabled={!bet.active}
          disabled={!bet.active}
          tabIndex={selectedBetType === bet.key ? 0 : -1}
          className={`seg-btn ${selectedBetType === bet.key ? "is-active" : ""}`}
          onClick={() => bet.active && onBetTypeSelect(bet.key)}
          onKeyDown={(e) => onKeyDown(e, i, BET_TYPES.length)}
          title={!bet.active ? "Coming soon" : undefined}
        >
          {!bet.active && <IconLock size={12} />}
          {bet.label}
          {!bet.active && <span className="seg-soon">Soon</span>}
        </button>
      ))}
    </div>
  )
}

export default BetTypeTabs

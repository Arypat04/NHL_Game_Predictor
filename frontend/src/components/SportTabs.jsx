import { useRovingTablist } from "./roving"

const SPORTS = [
  { key: "nhl", label: "NHL" },
  { key: "mlb", label: "MLB" },
]

export default function SportTabs({ selectedSport, onSportSelect }) {
  const { refs, onKeyDown } = useRovingTablist()
  return (
    <div className="seg" role="tablist" aria-label="Sport">
      {SPORTS.map((s, i) => (
        <button
          key={s.key}
          ref={(el) => (refs.current[i] = el)}
          role="tab"
          aria-selected={selectedSport === s.key}
          aria-controls="linelab-panel"
          tabIndex={selectedSport === s.key ? 0 : -1}
          className={`seg-btn ${selectedSport === s.key ? "is-active" : ""}`}
          onClick={() => onSportSelect(s.key)}
          onKeyDown={(e) => onKeyDown(e, i, SPORTS.length)}
        >
          {s.label}
        </button>
      ))}
    </div>
  )
}

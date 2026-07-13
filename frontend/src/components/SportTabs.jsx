import "./SportTabs.css"

const SPORTS = [
  { key: "nhl", label: "NHL" },
  { key: "mlb", label: "MLB" },
]

export default function SportTabs({ selectedSport, onSportSelect }) {
  return (
    <div className="sport-tabs">
      {SPORTS.map((sport) => (
        <button
          key={sport.key}
          className={`sport-tab ${selectedSport === sport.key ? "sport-tab--active" : ""}`}
          onClick={() => onSportSelect(sport.key)}
        >
          {sport.label}
        </button>
      ))}
    </div>
  )
}
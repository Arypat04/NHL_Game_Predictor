import { toAPIDate } from "../services/api"
import { useRovingTablist } from "./roving"

const DAY = 86400000
const sub = (d) => d.toLocaleDateString("en-US", { month: "short", day: "numeric" }).toUpperCase()

function DateTabs({ selectedDate, onDateSelect, todayStr }) {
  const today = new Date(todayStr + "T12:00:00")
  const sel = new Date(selectedDate + "T12:00:00")

  // Labels are anchored to the REAL today, never the selected date. The window
  // spans yesterday..+7 and always stretches to include the selected date.
  const start = Math.min(today.getTime() - DAY, sel.getTime())
  const end = Math.max(today.getTime() + 7 * DAY, sel.getTime())
  const n = Math.round((end - start) / DAY)

  const tabs = []
  for (let i = 0; i <= n; i++) {
    const d = new Date(start + i * DAY)
    const diff = Math.round((d.getTime() - today.getTime()) / DAY)
    const label =
      diff === -1 ? "Yesterday" :
      diff === 0 ? "Today" :
      diff === 1 ? "Tomorrow" :
      d.toLocaleDateString("en-US", { weekday: "short" })
    tabs.push({ date: toAPIDate(d), label, sub: sub(d), isToday: diff === 0 })
  }

  const { refs, onKeyDown } = useRovingTablist()

  return (
    <div className="datestrip" role="tablist" aria-label="Date">
      {tabs.map((tab, i) => (
        <button
          key={tab.date}
          ref={(el) => (refs.current[i] = el)}
          role="tab"
          aria-selected={selectedDate === tab.date}
          aria-controls="linelab-panel"
          aria-current={tab.isToday ? "date" : undefined}
          tabIndex={selectedDate === tab.date ? 0 : -1}
          className={`date-btn ${selectedDate === tab.date ? "is-active" : ""} ${tab.isToday ? "is-today" : ""}`}
          onClick={() => onDateSelect(tab.date)}
          onKeyDown={(e) => onKeyDown(e, i, tabs.length)}
        >
          {tab.label}
          <span className="date-sub">{tab.sub}</span>
        </button>
      ))}
    </div>
  )
}

export default DateTabs

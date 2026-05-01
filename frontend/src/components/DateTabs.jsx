import './DateTabs.css'
import { toAPIDate } from '../services/api'

function DateTabs({ selectedDate, onDateSelect }) {
  // build the 9-tab strip centered on selectedDate:
  // yesterday (relative to selected) + selected as "Today" + 7 forward
  const selected = new Date(selectedDate + "T12:00:00")

  const tabs = []

  // yesterday relative to selected date
  const yesterday = new Date(selected)
  yesterday.setDate(yesterday.getDate() - 1)
  tabs.push({ label: "Yesterday", date: toAPIDate(yesterday) })

  // selected date is always "Today" in the strip
  tabs.push({ label: "Today", date: selectedDate })

  // 7 days forward from selected date
  for (let i = 1; i <= 7; i++) {
    const d = new Date(selected)
    d.setDate(d.getDate() + i)
    tabs.push({
      label: i === 1
        ? "Tomorrow"
        : d.toLocaleDateString("en-US", { weekday: "short", day: "numeric" }),
      date: toAPIDate(d),
    })
  }

  return (
    <div className="date-tabs">
      {tabs.map((tab) => (
        <button
          key={tab.date}
          className={`tab ${selectedDate === tab.date ? "active" : ""}`}
          onClick={() => onDateSelect(tab.date)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  )
}

export default DateTabs
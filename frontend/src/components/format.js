export function parseTime(timeStr) {
  if (!timeStr) return 0
  const [time, modifier] = timeStr.split(" ")
  let [h, m] = time.split(":").map(Number)
  if (modifier === "PM" && h !== 12) h += 12
  if (modifier === "AM" && h === 12) h = 0
  return h * 60 + (m || 0)
}

export function byTime(a, b) { return parseTime(a.Time) - parseTime(b.Time) }

/* "7:05 PM" -> { t: "7:05", ap: "PM" } for mono readout styling */
export function splitTime(timeStr) {
  if (!timeStr) return { t: "—", ap: "" }
  const [t, ap = ""] = timeStr.split(" ")
  return { t, ap }
}

export function americanToImplied(odds) {
  const o = Number(odds)
  if (!o) return null
  return o > 0 ? 100 / (o + 100) : Math.abs(o) / (Math.abs(o) + 100)
}

export function fmtOdds(odds) {
  const o = Number(odds)
  return o > 0 ? `+${o}` : `${o}`
}

export function fmtPct(v, digits = 1) {
  return `${(v * 100).toFixed(digits)}%`
}

/* edge is a probability gap; show it in points, signed */
export function fmtEdge(edge) {
  const pts = (edge * 100).toFixed(1)
  return `${edge > 0 ? "+" : ""}${pts}`
}

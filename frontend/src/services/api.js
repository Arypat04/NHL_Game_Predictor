const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000"

export function toAPIDate(date) {
  const y = date.getFullYear()
  const m = String(date.getMonth() + 1).padStart(2, "0")
  const d = String(date.getDate()).padStart(2, "0")
  return `${y}-${m}-${d}`
}

async function req(path) {
  if (import.meta.env.VITE_MOCK === "1") {
    const { mockReq } = await import("./mock")   // dev fixtures; tree-shaken in prod
    return mockReq(path)
  }
  let resp
  try {
    resp = await fetch(`${BASE_URL}${path}`)
  } catch {
    const err = new Error("Can't reach the server.")
    err.kind = "offline"
    throw err
  }
  if (!resp.ok) {
    let detail = resp.statusText
    try {
      const j = await resp.json()
      if (j && j.detail) detail = j.detail
    } catch { /* non-JSON error body */ }
    const err = new Error(detail)
    err.status = resp.status
    // 503 = model still warming; 401 = odds free-tier quota exhausted
    err.kind = resp.status === 503 ? "warming" : resp.status === 401 ? "quota" : "error"
    throw err
  }
  return resp.json()
}

export const getPredictions = (date, sport = "nhl") => req(`/predictions?date=${date}&sport=${sport}`)
export const getResults     = (date, sport = "nhl") => req(`/results?date=${date}&sport=${sport}`)
export const getEdges       = (date, sport = "nhl") => req(`/edges?date=${date}&sport=${sport}`)
export const getStatus      = (sport = "nhl")       => req(`/status?sport=${sport}`)

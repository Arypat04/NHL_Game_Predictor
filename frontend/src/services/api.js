const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000"

export function toAPIDate(date) {
  const y = date.getFullYear()
  const m = String(date.getMonth() + 1).padStart(2, "0")
  const d = String(date.getDate()).padStart(2, "0")
  return `${y}-${m}-${d}`
}

export async function getPredictions(date) {
  const resp = await fetch(`${BASE_URL}/predictions?date=${date}`)
  if (!resp.ok) throw new Error(`Error fetching predictions: ${resp.statusText}`)
  return resp.json()
}

export async function getResults(date) {
  const resp = await fetch(`${BASE_URL}/results?date=${date}`)
  if (!resp.ok) throw new Error(`Error fetching results: ${resp.statusText}`)
  return resp.json()
}

export async function getEdges(date) {
  const resp = await fetch(`${BASE_URL}/edges?date=${date}`)
  if (!resp.ok) throw new Error(`Error fetching edges: ${resp.statusText}`)
  return resp.json()
}

export async function getStatus() {
  const resp = await fetch(`${BASE_URL}/status`)
  if (!resp.ok) throw new Error(`Error fetching status: ${resp.statusText}`)
  return resp.json()
}
const BASE_URL =
  (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

// ensures no trailing slash → prevents //status bugs

export function toAPIDate(date) {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

// helper to safely build URLs
function buildUrl(path) {
  return `${BASE_URL}${path.startsWith("/") ? "" : "/"}${path}`;
}

async function safeFetch(url) {
  const resp = await fetch(url);

  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(
      `API Error ${resp.status}: ${resp.statusText} - ${text}`
    );
  }

  return resp.json();
}

export async function getPredictions(date) {
  return safeFetch(buildUrl(`/predictions?date=${date}`));
}

export async function getResults(date) {
  return safeFetch(buildUrl(`/results?date=${date}`));
}

export async function getEdges(date) {
  return safeFetch(buildUrl(`/edges?date=${date}`));
}

export async function getStatus() {
  return safeFetch(buildUrl(`/status`));
}
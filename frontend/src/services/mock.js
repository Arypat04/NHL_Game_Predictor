/* Dev-only fixtures for UI work (enabled with VITE_MOCK=1). Realistic shapes
   matching the live API, with a spread of cases worth seeing: a near-even game,
   strong picks, favorable and unfavorable edges, and a results day with a live
   game plus hits and misses. Never bundled unless VITE_MOCK is set at build. */

const predictions = [
  { Time: "7:00 PM", Away: "TOR", Home: "BOS", Predicted_Winner: "BOS", Home_Win_Prob: 0.631, Away_Win_Prob: 0.369, Confidence: 0.631 },
  { Time: "7:00 PM", Away: "MTL", Home: "TBL", Predicted_Winner: "TBL", Home_Win_Prob: 0.548, Away_Win_Prob: 0.452, Confidence: 0.548 },
  { Time: "7:30 PM", Away: "NYR", Home: "PIT", Predicted_Winner: "NYR", Home_Win_Prob: 0.421, Away_Win_Prob: 0.579, Confidence: 0.579 },
  { Time: "9:00 PM", Away: "EDM", Home: "CGY", Predicted_Winner: "EDM", Home_Win_Prob: 0.330, Away_Win_Prob: 0.670, Confidence: 0.670 },
  { Time: "10:00 PM", Away: "VGK", Home: "COL", Predicted_Winner: "COL", Home_Win_Prob: 0.513, Away_Win_Prob: 0.487, Confidence: 0.513 },
  { Time: "10:30 PM", Away: "SEA", Home: "LAK", Predicted_Winner: "LAK", Home_Win_Prob: 0.702, Away_Win_Prob: 0.298, Confidence: 0.702 },
]

// Edges are model prob − no-vig implied (the backend de-vigs the two best prices
// before comparing to the model). A clean consequence: home edge = −away edge,
// since model and fair probabilities each sum to 1. The "model % − market %"
// equation on each row resolves exactly to the edge shown.
const edges = [
  { Time: "7:00 PM", Away: "TOR", Home: "BOS", Best_Bet: "Home", Away_Odds: 145, Home_Odds: -170, Away_Edge: -0.024, Home_Edge: 0.024, Bookmaker: "DraftKings" },
  { Time: "7:30 PM", Away: "NYR", Home: "PIT", Best_Bet: "Away", Away_Odds: 120, Home_Odds: -140, Away_Edge: 0.141, Home_Edge: -0.141, Bookmaker: "FanDuel" },
  { Time: "9:00 PM", Away: "EDM", Home: "CGY", Best_Bet: "Away", Away_Odds: -125, Home_Odds: 105, Away_Edge: 0.138, Home_Edge: -0.138, Bookmaker: "BetMGM" },
  // a SPLIT: model's pick is LAK (home) to win, but the value is on SEA (away)
  { Time: "10:30 PM", Away: "SEA", Home: "LAK", Best_Bet: "Away", Away_Odds: 290, Home_Odds: -300, Away_Edge: 0.043, Home_Edge: -0.043, Bookmaker: "Caesars" },
]

const results = [
  { Time: "7:00 PM", Status: "Final", Away: "TOR", Home: "BOS", Actual_Winner: "BOS", Predicted_Winner: "BOS", Correct: true,  Away_Score: 2, Home_Score: 4 },
  { Time: "7:00 PM", Status: "Final", Away: "MTL", Home: "TBL", Actual_Winner: "MTL", Predicted_Winner: "TBL", Correct: false, Away_Score: 3, Home_Score: 1 },
  { Time: "8:00 PM", Status: "Live",  Away: "NYR", Home: "PIT", Actual_Winner: "PIT", Predicted_Winner: "NYR", Correct: false, Away_Score: 1, Home_Score: 2 },
  { Time: "8:30 PM", Status: "Live",  Away: "FLA", Home: "NYI", Actual_Winner: "FLA", Predicted_Winner: "NYI", Correct: false, Away_Score: 2, Home_Score: 2 },
  { Time: "9:30 PM", Status: "Live",  Away: "DAL", Home: "WPG", Actual_Winner: "WPG", Predicted_Winner: "WPG", Correct: true,  Away_Score: 1, Home_Score: 3 },
  { Time: "9:00 PM", Status: "Final", Away: "EDM", Home: "CGY", Actual_Winner: "EDM", Predicted_Winner: "EDM", Correct: true,  Away_Score: 5, Home_Score: 2 },
  { Time: "10:00 PM", Status: "Final", Away: "VGK", Home: "COL", Actual_Winner: "COL", Predicted_Winner: "COL", Correct: true,  Away_Score: 1, Home_Score: 3 },
]

const status = {
  season_accuracy: 0.583,
  total_predictions: 1312,
  data_updated: "2026-10-13T08:00:00Z",
  odds_api_configured: true,
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms))

function throwKind(kind, message, status) {
  const err = new Error(message)
  err.kind = kind
  err.status = status
  throw err
}

let warmStart = null
export async function mockReq(path) {
  await wait(400)
  // ?ms=warm|quota|empty lets us preview degraded states during UI review.
  // warm recovers after ~7s so the warming poll's recovery can be verified.
  const ms = new URLSearchParams(window.location.search).get("ms")
  if (ms === "warm" && !path.startsWith("/status")) {
    if (warmStart == null) warmStart = Date.now()
    if (Date.now() - warmStart < 7000)
      throwKind("warming", "Model is still warming up — try again shortly.", 503)
  }
  if (ms === "quota" && path.startsWith("/edges"))
    throwKind("quota", "Odds API quota exhausted.", 401)
  if (ms === "empty") {
    if (path.startsWith("/status")) return status
    return []
  }

  if (path.startsWith("/status")) return status
  if (path.startsWith("/predictions")) return structuredClone(predictions)
  if (path.startsWith("/edges")) return structuredClone(edges)
  if (path.startsWith("/results")) return structuredClone(results)
  return {}
}

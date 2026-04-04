import { useState, useEffect, useMemo } from "react"
import DateTabs from "./components/DateTabs"
import PredictionsTable from "./components/PredictionsTable"
import ResultsTable from "./components/ResultsTable"
import StatsBar from "./components/StatsBar"
import EdgesTable from "./components/EdgesTable"
import BetTypeTabs from "./components/BetTypeTabs"
import { getPredictions, getResults, getStatus, toAPIDate, getEdges } from "./services/api"
import './App.css'

function App() {
  const todayStr = useMemo(() => toAPIDate(new Date()), [])
  const yesterdayStr = useMemo(() => {
    const d = new Date()
    d.setDate(d.getDate() - 1)
    return toAPIDate(d)
  }, [])

  const [selectedDate, setSelectedDate] = useState(todayStr)
  const [theme, setTheme] = useState("dark")
  const [betType, setBetType] = useState("moneyline")

  const [predictions, setPredictions] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const [results, setResults] = useState([])
  const [resultsLoading, setResultsLoading] = useState(false)
  const [resultsError, setResultsError] = useState(null)

  const [edges, setEdges] = useState([])
  const [edgesLoading, setEdgesLoading] = useState(false)
  const [edgesError, setEdgesError] = useState(null)

  const [status, setStatus] = useState(null)
  const [statusLoading, setStatusLoading] = useState(false)
  const [statusError, setStatusError] = useState(null)

  const isYesterday = selectedDate === yesterdayStr
  const isToday     = selectedDate === todayStr
  const isFuture    = selectedDate > todayStr
  const [minConfidence, setMinConfidence] = useState(50)

  // apply theme to document
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme)
  }, [theme])

  // fetch predictions, results, edges when date changes
  useEffect(() => {
    let cancelled = false

    async function fetchData() {
      setLoading(true)
      setResultsLoading(true)
      setEdgesLoading(true)
      setError(null)
      setResultsError(null)
      setEdgesError(null)

      try {
        const data = await getPredictions(selectedDate)
        if (!cancelled) setPredictions(data)
      } catch (err) {
        if (!cancelled) setError(err.message)
      } finally {
        if (!cancelled) setLoading(false)
      }

      try {
        const data = await getResults(selectedDate)
        if (!cancelled) setResults(data)
      } catch (err) {
        if (!cancelled) setResultsError(err.message)
      } finally {
        if (!cancelled) setResultsLoading(false)
      }

      try {
        const data = await getEdges(selectedDate)
        if (!cancelled) setEdges(data)
      } catch (err) {
        if (!cancelled) setEdgesError(err.message)
      } finally {
        if (!cancelled) setEdgesLoading(false)
      }
    }

    fetchData()

    // cleanup — if selectedDate changes before fetch completes, ignore stale results
    return () => { cancelled = true }
  }, [selectedDate])

  // fetch status once on mount
  useEffect(() => {
    async function fetchStatus() {
      setStatusLoading(true)
      try {
        const data = await getStatus()
        setStatus(data)
      } catch (err) {
        setStatusError(err.message)
      } finally {
        setStatusLoading(false)
      }
    }
    fetchStatus()
  }, [])

  useEffect(() => {
  if (!isToday) return

  const interval = setInterval(async () => {
    try {
      const [resultsData, edgesData] = await Promise.all([
        getResults(todayStr),
        getEdges(todayStr),
      ])
      setResults(resultsData)
      setEdges(edgesData)
    } catch (err) {
      // silent fail — don't disrupt the UI for a background poll
    }
  }, 5 * 60 * 1000)

  return () => clearInterval(interval)
  }, [isToday, todayStr])


const filteredPredictions = predictions.filter(
  g => g.Confidence * 100 >= minConfidence
)

const filteredEdges = edges.filter(
  g => Math.max(g.Home_Win_Prob, g.Away_Win_Prob) * 100 >= minConfidence
)


  return (
    <div className="app">
      <header className="header">
        <div className="header-logo">Line <span>Lab</span></div>
        <button className="theme-toggle" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
          {theme === "dark" ? "☀ Light" : "☾ Dark"}
        </button>
      </header>
      <div className="content">
        <StatsBar status={status} loading={statusLoading} error={statusError} />
        <DateTabs selectedDate={selectedDate} onDateSelect={setSelectedDate} />
        <BetTypeTabs selectedBetType={betType} onBetTypeSelect={setBetType} />
        <div className="filter-bar">
          <label className="filter-label">
            Min Confidence: <span>{minConfidence}%</span>
          </label>
          <input
            type="range"
            min="50"
            max="75"
            step="1"
            value={minConfidence}
            onChange={(e) => setMinConfidence(Number(e.target.value))}
          />
        </div>
        
        {(isToday || isFuture) && (
          <>
            <p className="section-label">Predictions</p>
            <PredictionsTable predictions={filteredPredictions} loading={loading} error={error} />
            <p className="section-label">Edges</p>
            <EdgesTable edges={filteredEdges} loading={edgesLoading} error={edgesError} />
          </>
        )}

        {(isYesterday || (isToday && results.length > 0)) && (
          <>
            <p className="section-label">Results</p>
            <ResultsTable results={results} loading={resultsLoading} error={resultsError} />
          </>
        )}
      </div>
    </div>
  )
}

export default App
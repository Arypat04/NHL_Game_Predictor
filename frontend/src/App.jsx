import { useState, useEffect, useMemo, useCallback } from "react"
import DateTabs from "./components/DateTabs"
import PredictionsTable from "./components/PredictionsTable"
import ResultsTable from "./components/ResultsTable"
import StatsBar from "./components/StatsBar"
import EdgesTable from "./components/EdgesTable"
import BetTypeTabs from "./components/BetTypeTabs"
import SportTabs from "./components/SportTabs"
import { getPredictions, getResults, getStatus, toAPIDate, getEdges } from "./services/api"
import './App.css'

function App() {
  const todayStr = useMemo(() => toAPIDate(new Date()), [])
  const yesterdayStr = useMemo(() => {
    const d = new Date()
    d.setDate(d.getDate() - 1)
    return toAPIDate(d)
  }, [])

  // Read date and sport from URL on load
  const getInitialDate = () => {
    const params = new URLSearchParams(window.location.search)
    return params.get("date") || todayStr
  }

  const getInitialSport = () => {
    const params = new URLSearchParams(window.location.search)
    const s = params.get("sport")
    return s === "mlb" ? "mlb" : "nhl"  // default to nhl
  }

  const [selectedDate, setSelectedDate] = useState(getInitialDate)
  const [sport, setSport] = useState(getInitialSport)
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

  const [minConfidence, setMinConfidence] = useState(50)

  const isToday  = selectedDate === todayStr
  const isPast   = selectedDate < todayStr
  const isFuture = selectedDate > todayStr

  // Update selected date and push to URL
  const handleDateSelect = useCallback((date) => {
    setSelectedDate(date)
    const url = new URL(window.location)
    url.searchParams.set("date", date)
    window.history.pushState({}, "", url)
  }, [])

  // Update sport, push to URL, reset all data
  const handleSportSelect = useCallback((newSport) => {
    setSport(newSport)
    setPredictions([])
    setResults([])
    setEdges([])
    setStatus(null)
    const url = new URL(window.location)
    url.searchParams.set("sport", newSport)
    window.history.pushState({}, "", url)
  }, [])

  // Apply theme to document
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme)
  }, [theme])

  // Fetch predictions, results, edges when date or sport changes
  useEffect(() => {
    let cancelled = false

    async function fetchData() {
      setLoading(true)
      setResultsLoading(true)
      setEdgesLoading(true)
      setError(null)
      setResultsError(null)
      setEdgesError(null)
      setPredictions([])
      setResults([])
      setEdges([])

      try {
        const data = await getPredictions(selectedDate, sport)
        if (!cancelled) setPredictions(data)
      } catch (err) {
        if (!cancelled) setError(err.message)
      } finally {
        if (!cancelled) setLoading(false)
      }

      try {
        const data = await getResults(selectedDate, sport)
        if (!cancelled) setResults(data)
      } catch (err) {
        if (!cancelled) setResultsError(err.message)
      } finally {
        if (!cancelled) setResultsLoading(false)
      }

      try {
        const data = await getEdges(selectedDate, sport)
        if (!cancelled) setEdges(data)
      } catch (err) {
        if (!cancelled) setEdgesError(err.message)
      } finally {
        if (!cancelled) setEdgesLoading(false)
      }
    }

    fetchData()
    return () => { cancelled = true }
  }, [selectedDate, sport])

  // Fetch status when sport changes
  useEffect(() => {
    async function fetchStatus() {
      setStatusLoading(true)
      setStatusError(null)
      try {
        const data = await getStatus(sport)
        setStatus(data)
      } catch (err) {
        setStatusError(err.message)
      } finally {
        setStatusLoading(false)
      }
    }
    fetchStatus()
  }, [sport])

  // Polling — re-fetch results and edges every 5 minutes when today is selected
  useEffect(() => {
    if (!isToday) return

    const interval = setInterval(async () => {
      try {
        const [resultsData, edgesData] = await Promise.all([
          getResults(todayStr, sport),
          getEdges(todayStr, sport),
        ])
        setResults(resultsData)
        setEdges(edgesData)
      } catch {
        // silent fail
      }
    }, 5 * 60 * 1000)

    return () => clearInterval(interval)
  }, [isToday, todayStr, sport])

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
        <button
          className="theme-toggle"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        >
          {theme === "dark" ? "☀ Light" : "☾ Dark"}
        </button>
      </header>

      <div className="content">
        <StatsBar status={status} loading={statusLoading} error={statusError} />
        <SportTabs selectedSport={sport} onSportSelect={handleSportSelect} />
        <DateTabs selectedDate={selectedDate} onDateSelect={handleDateSelect} />
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

        {/* Predictions + edges for today and future */}
        {(isToday || isFuture) && (
          <>
            <p className="section-label">Predictions</p>
            <PredictionsTable
              predictions={filteredPredictions}
              loading={loading}
              error={error}
            />
            <p className="section-label">Edges</p>
            <EdgesTable
              edges={filteredEdges}
              loading={edgesLoading}
              error={edgesError}
            />
          </>
        )}

        {/* Results for past dates or today if games have finished */}
        {(isPast || (isToday && results.length > 0)) && (
          <>
            <p className="section-label">Results</p>
            <ResultsTable
              results={results}
              loading={resultsLoading}
              error={resultsError}
            />
          </>
        )}
      </div>
    </div>
  )
}

export default App
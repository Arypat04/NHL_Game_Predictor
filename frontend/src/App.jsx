import { useState, useEffect, useMemo, useCallback } from "react"
import DateTabs from "./components/DateTabs"
import BoardTable from "./components/BoardTable"
import ResultsTable from "./components/ResultsTable"
import StatsBar from "./components/StatsBar"
import BetTypeTabs from "./components/BetTypeTabs"
import SportTabs from "./components/SportTabs"
import Legend from "./components/Legend"
import { IconSun, IconMoon } from "./components/Icons"
import { getPredictions, getResults, getStatus, toAPIDate, getEdges } from "./services/api"
import "./App.css"

function App() {
  const todayStr = useMemo(() => toAPIDate(new Date()), [])

  const params = () => new URLSearchParams(window.location.search)
  const getInitialDate = () => params().get("date") || todayStr
  const getInitialSport = () => (params().get("sport") === "mlb" ? "mlb" : "nhl")
  const getInitialMode = () =>
    ["value", "confidence", "time"].includes(params().get("sort")) ? params().get("sort") : "value"
  const getInitialConf = () => {
    const c = parseInt(params().get("conf"), 10)
    return Number.isFinite(c) ? Math.min(90, Math.max(50, c)) : 50
  }
  const getInitialSplits = () => params().get("splits") === "1"
  // write a view preference to the URL without cluttering Back history
  const setParam = (key, value, remove = false) => {
    const url = new URL(window.location)
    if (remove) url.searchParams.delete(key)
    else url.searchParams.set(key, value)
    window.history.replaceState({}, "", url)
  }

  const [selectedDate, setSelectedDate] = useState(getInitialDate)
  const [sport, setSport] = useState(getInitialSport)
  const [theme, setTheme] = useState(() => {
    try {
      const saved = localStorage.getItem("linelab-theme")
      if (saved === "light" || saved === "dark") return saved
      if (window.matchMedia?.("(prefers-color-scheme: light)").matches) return "light"
    } catch { /* storage blocked */ }
    return "dark"
  })
  const [betType, setBetType] = useState("moneyline")

  const [predictions, setPredictions] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const [results, setResults] = useState([])
  const [resultsLoading, setResultsLoading] = useState(false)
  const [resultsError, setResultsError] = useState(null)

  const [edges, setEdges] = useState([])
  const [edgesError, setEdgesError] = useState(null)

  const [status, setStatus] = useState(null)
  const [lastUpdated, setLastUpdated] = useState(null)
  const [justRefreshed, setJustRefreshed] = useState(false)
  const [reloadKey, setReloadKey] = useState(0)
  const retry = useCallback(() => setReloadKey((k) => k + 1), [])

  const [minConfidence, setMinConfidenceState] = useState(getInitialConf)
  const setMinConfidence = useCallback((v) => {
    setMinConfidenceState(v)
    setParam("conf", v, v === 50)
  }, [])
  const clearFilter = useCallback(() => setMinConfidence(50), [setMinConfidence])
  const [boardMode, setBoardModeState] = useState(getInitialMode)   // value | confidence | time
  const setBoardMode = useCallback((m) => {
    setBoardModeState(m)
    setParam("sort", m, m === "value")
  }, [])
  const [splitsOnly, setSplitsOnlyState] = useState(getInitialSplits)
  const toggleSplits = useCallback(() => {
    setSplitsOnlyState((s) => { setParam("splits", "1", s); return !s })
  }, [])

  const isToday  = selectedDate === todayStr
  const isPast   = selectedDate < todayStr
  const isFuture = selectedDate > todayStr

  const handleDateSelect = useCallback((date) => {
    setSelectedDate(date)
    const url = new URL(window.location)
    url.searchParams.set("date", date)
    window.history.pushState({}, "", url)
  }, [])

  const handleSportSelect = useCallback((newSport) => {
    setSport(newSport)
    setPredictions([]); setResults([]); setEdges([]); setStatus(null)
    const url = new URL(window.location)
    url.searchParams.set("sport", newSport)
    window.history.pushState({}, "", url)
  }, [])

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme)
    try { localStorage.setItem("linelab-theme", theme) } catch { /* storage blocked */ }
  }, [theme])

  // keep app state in sync with browser Back/Forward
  useEffect(() => {
    const onPop = () => {
      setSelectedDate(getInitialDate())
      setSport(getInitialSport())
      setBoardModeState(getInitialMode())
      setMinConfidenceState(getInitialConf())
      setSplitsOnlyState(getInitialSplits())
    }
    window.addEventListener("popstate", onPop)
    return () => window.removeEventListener("popstate", onPop)
  }, [todayStr])

  useEffect(() => {
    let cancelled = false
    async function fetchData() {
      setLoading(true); setResultsLoading(true)
      setError(null); setResultsError(null); setEdgesError(null)
      setPredictions([]); setResults([]); setEdges([])

      try { const d = await getPredictions(selectedDate, sport); if (!cancelled) setPredictions(d) }
      catch (err) { if (!cancelled) setError(err) }
      finally { if (!cancelled) setLoading(false) }

      try { const d = await getResults(selectedDate, sport); if (!cancelled) setResults(d) }
      catch (err) { if (!cancelled) setResultsError(err) }
      finally { if (!cancelled) setResultsLoading(false) }

      try { const d = await getEdges(selectedDate, sport); if (!cancelled) setEdges(d) }
      catch (err) { if (!cancelled) setEdgesError(err) }

      if (!cancelled) setLastUpdated(new Date())
    }
    fetchData()
    return () => { cancelled = true }
  }, [selectedDate, sport, reloadKey])

  // while the model is warming (503 cold start), poll until it answers — so the
  // "readings will appear automatically" promise is actually kept
  useEffect(() => {
    if (error?.kind !== "warming") return
    let cancelled = false
    const id = setInterval(async () => {
      try {
        const d = await getPredictions(selectedDate, sport)
        if (cancelled) return
        setPredictions(d); setError(null)
        try { setEdges(await getEdges(selectedDate, sport)) } catch { /* edges optional */ }
        setLastUpdated(new Date())
      } catch { /* still warming */ }
    }, 5000)
    return () => { cancelled = true; clearInterval(id) }
  }, [error, selectedDate, sport])

  useEffect(() => {
    async function fetchStatus() {
      try { setStatus(await getStatus(sport)) } catch { setStatus(null) }
    }
    fetchStatus()
  }, [sport])

  useEffect(() => {
    if (!isToday) return
    const interval = setInterval(async () => {
      try {
        const [r, e] = await Promise.all([getResults(todayStr, sport), getEdges(todayStr, sport)])
        setResults(r); setEdges(e); setLastUpdated(new Date())
        setJustRefreshed(true)
        setTimeout(() => setJustRefreshed(false), 2000)
      } catch { /* silent */ }
    }, 5 * 60 * 1000)
    return () => clearInterval(interval)
  }, [isToday, todayStr, sport])

  // one board: each game once, with its market edge attached where odds exist
  const edgeByGame = {}
  for (const e of edges) edgeByGame[`${e.Away}@${e.Home}`] = e
  const board = predictions.map((p) => ({ ...p, edge: edgeByGame[`${p.Away}@${p.Home}`] || null }))
  // a "split" = the model's best-value bet is a different team than its pick
  const isSplit = (g) => g.edge && ((g.edge.Best_Bet === "Home") !== (g.Predicted_Winner === g.Home))
  const confBoard = board.filter((g) => g.Confidence * 100 >= minConfidence)
  const filteredBoard = splitsOnly ? confBoard.filter(isSplit) : confBoard
  const oddsUnavailable = edgesError?.kind === "quota"
  const oddsLive = !!(status && status.odds_api_configured) && !oddsUnavailable
  // value sort is meaningless with no odds — fall back to confidence
  const effectiveMode = oddsUnavailable && boardMode === "value" ? "confidence" : boardMode

  const showBoard = isToday || isFuture
  const showResults = isPast || (isToday && results.length > 0)
  const confFill = `${((minConfidence - 50) / (90 - 50)) * 100}%`

  const liveMsg =
    error?.kind === "warming" ? "Model is warming up — readings will appear shortly."
    : error?.kind === "offline" ? "Can't reach the server."
    : showBoard ? `${filteredBoard.length} games on the board for ${selectedDate}`
    : showResults ? `${results.length} results for ${selectedDate}` : ""

  return (
    <div className="app">
      <header className="shell-header">
        <div className="shell-bar">
          <h1 className="wordmark">
            Line<span className="lab">Lab</span>
            <span className="caret">// {sport.toUpperCase()}</span>
          </h1>
          <div className="status-strip"><StatsBar status={status} oddsLive={oddsLive} /></div>
          <button
            className="theme-toggle"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          >
            {theme === "dark" ? <IconSun size={17} /> : <IconMoon size={17} />}
          </button>
        </div>
      </header>

      <main className="content" id="linelab-panel" role="tabpanel" aria-label={`${sport.toUpperCase()} moneyline — ${selectedDate}`}>
        <div className="status-strip-mobile"><StatsBar status={status} oddsLive={oddsLive} /></div>

        <div className="controls-row">
          <SportTabs selectedSport={sport} onSportSelect={handleSportSelect} />
          <BetTypeTabs selectedBetType={betType} onBetTypeSelect={setBetType} />
        </div>

        <DateTabs selectedDate={selectedDate} onDateSelect={handleDateSelect} todayStr={todayStr} />

        <div className="filter">
          <span className="filter-k">Min confidence</span>
          <span className="filter-v mono">{minConfidence}%</span>
          <input
            type="range" min="50" max="90" step="1" value={minConfidence}
            style={{ "--fill": confFill }}
            onChange={(e) => setMinConfidence(Number(e.target.value))}
            aria-label={`Minimum confidence filter, ${minConfidence} percent`}
          />
          {minConfidence > 50 && (
            <button className="filter-clear" onClick={clearFilter}>Clear</button>
          )}
          {showBoard && (
            <button
              className={`splits-toggle ${splitsOnly ? "is-active" : ""}`}
              aria-pressed={splitsOnly}
              title="Show only games where the best value bet differs from the model's pick"
              onClick={toggleSplits}
            >Splits only</button>
          )}
          {lastUpdated && (
            <span className={`updated mono ${justRefreshed ? "is-fresh" : ""}`} title={lastUpdated.toLocaleString()}>
              {justRefreshed && <span className="dot dot-live" />}
              Updated {lastUpdated.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" })}
            </span>
          )}
        </div>

        <Legend />

        {/* screen-reader announcement of counts / state changes */}
        <div className="sr-only" aria-live="polite">{liveMsg}</div>

        {showBoard && (
          <section className="surface">
            <div className="surface-head">
              <span className="surface-tick" />
              <h2 className="surface-title">The Board</h2>
              {!loading && !error && (
                <span className="surface-count mono">{filteredBoard.length}</span>
              )}
              {!loading && !error && filteredBoard.length > 0 && (
                <div className="sort" role="group" aria-label="Sort the board">
                  <span className="sort-k">Sort</span>
                  {[["value", "Value"], ["confidence", "Conf"], ["time", "Time"]].map(([k, l]) => {
                    const disabled = k === "value" && oddsUnavailable
                    return (
                      <button
                        key={k}
                        className={`sort-btn ${effectiveMode === k ? "is-active" : ""}`}
                        aria-pressed={effectiveMode === k}
                        disabled={disabled}
                        title={disabled ? "No edges to rank by while odds are unavailable" : undefined}
                        onClick={() => setBoardMode(k)}
                      >{l}</button>
                    )
                  })}
                </div>
              )}
            </div>
            <BoardTable
              board={filteredBoard} loading={loading} error={error}
              rawCount={board.length} threshold={minConfidence} onClear={clearFilter}
              mode={effectiveMode} oddsUnavailable={oddsUnavailable}
              splitsEmpty={splitsOnly && filteredBoard.length === 0 && confBoard.length > 0}
              onShowAll={toggleSplits} onRetry={retry}
            />
          </section>
        )}

        {showResults && (
          <section className="surface">
            <div className="surface-head">
              <span className="surface-tick" />
              <h2 className="surface-title">Results</h2>
              {!resultsLoading && !resultsError && (
                <span className="surface-count mono">{results.length}</span>
              )}
              <span className="surface-note">predicted vs. actual — model calibration</span>
            </div>
            <ResultsTable results={results} loading={resultsLoading} error={resultsError} />
          </section>
        )}
      </main>
    </div>
  )
}

export default App

import ProbScale from "./ProbScale"
import { IconUp, IconDown, IconWarn, IconFilter } from "./Icons"
import { byTime, splitTime, fmtPct, fmtOdds, fmtEdge } from "./format"
import { ReadingSkeleton, surfaceStateFor } from "./States"

const betEdgeOf = (g) =>
  g.edge ? (g.edge.Best_Bet === "Home" ? g.edge.Home_Edge : g.edge.Away_Edge) : -Infinity

function sorter(mode) {
  if (mode === "confidence") return (a, b) => b.Confidence - a.Confidence
  if (mode === "time") return byTime
  return (a, b) => betEdgeOf(b) - betEdgeOf(a)   // value (default)
}

function BoardTable({ board, loading, error, rawCount = 0, threshold, onClear, mode = "value", oddsUnavailable = false, splitsEmpty = false, onShowAll, onRetry }) {
  if (loading) return <ReadingSkeleton />

  if (!error && splitsEmpty) {
    return (
      <div className="state is-filtered">
        <div className="state-icon"><IconFilter size={24} /></div>
        <div className="state-title">No split plays right now</div>
        <p className="state-msg">On every game the model's best value is also its straight-up pick — nothing diverges.</p>
        <button className="state-action" onClick={onShowAll}>Show all games</button>
      </div>
    )
  }

  const items = board || []
  const state = surfaceStateFor(error, {
    empty: rawCount === 0,
    filtered: rawCount > 0 && items.length === 0
      ? { hiddenCount: rawCount, threshold, onClear, noun: "games" }
      : undefined,
    onRetry,
  })
  if (state) return state

  const rows = [...items].sort(sorter(mode))

  return (
    <>
      {oddsUnavailable && (
        <div className="board-note" role="status">
          <IconWarn size={14} />
          Live odds are out of free-tier requests — showing model picks without edges until the quota resets.
        </div>
      )}
      <div className="readings">
        {rows.map((g, i) => {
          const { t, ap } = splitTime(g.Time)
          const pickHome = g.Predicted_Winner === g.Home
          const pickCode = pickHome ? g.Home : g.Away
          const e = g.edge
          // Market prob on the same no-vig basis the backend used for the edge
          // (model − edge), so the scale ring and the equation always agree.
          const marketHome = e ? g.Home_Win_Prob - e.Home_Edge : null
          const hasOdds = marketHome != null
          const betHome = e ? e.Best_Bet === "Home" : pickHome
          const betCode = betHome ? g.Home : g.Away
          const betEdge = e ? (betHome ? e.Home_Edge : e.Away_Edge) : null
          const betOdds = e ? (betHome ? e.Home_Odds : e.Away_Odds) : null
          const divergent = hasOdds && betHome !== pickHome
          // Model win prob for the side we're betting, and the market baseline it
          // beats. Derive market as (model − edge) so the shown equation always
          // resolves exactly to the displayed edge, with no rounding drift.
          const modelBetPct = betHome ? g.Home_Win_Prob : (g.Away_Win_Prob ?? 1 - g.Home_Win_Prob)
          const marketShown = hasOdds ? modelBetPct - betEdge : null

          return (
            <div className="reading" key={i}>
              <div className="r-time">{t}<span className="r-ampm"> {ap}</span></div>

              <div className="r-match">
                <div className={`r-team ${!pickHome ? "is-pick" : ""}`}>
                  <span className="r-code mono">{g.Away}</span>
                  {!pickHome && <span className="r-pickflag">PICK</span>}
                </div>
                <div className={`r-team ${pickHome ? "is-pick" : ""}`}>
                  <span className="r-at">@</span>
                  <span className="r-code mono">{g.Home}</span>
                  {pickHome && <span className="r-pickflag">PICK</span>}
                </div>
              </div>

              <div className="r-scale">
                <ProbScale
                  modelHome={g.Home_Win_Prob}
                  marketHome={marketHome}
                  edge={hasOdds ? fmtEdge(betEdge) : null}
                  edgeNegative={hasOdds && betEdge < 0}
                  homeCode={g.Home}
                  awayCode={g.Away}
                />
              </div>

              <div className="r-read">
                {hasOdds ? (
                  <>
                    <div
                      className={`r-metric mono ${betEdge > 0 ? "is-for" : betEdge < 0 ? "is-against" : ""}`}
                      title="Edge = model win probability − market implied probability, in percentage points."
                    >
                      {betEdge !== 0 && (
                        <span className={`r-edgedir ${betEdge > 0 ? "is-for" : "is-against"}`}>
                          {betEdge > 0 ? <IconUp size={13} /> : <IconDown size={13} />}
                        </span>
                      )}
                      {fmtEdge(betEdge)}<span className="r-unit">pts</span>
                    </div>

                    <div className="r-eq" title="How the edge is calculated: the model's win probability for the bet, minus the market's implied probability.">
                      <span className="r-op"><span className="r-op-k">model</span>{fmtPct(modelBetPct)}</span>
                      <span className="r-minus" aria-label="minus">−</span>
                      <span className="r-op"><span className="r-op-k">market</span>{fmtPct(marketShown)}</span>
                    </div>

                    <div className={`r-sub ${divergent ? "is-diverge" : ""}`}>
                      {divergent && <span className="split-tag">SPLIT</span>}
                      {divergent && <span className="r-betk">bet</span>}
                      <b className="mono">{betCode}</b>
                      <span className="r-dot-sep">·</span>
                      <span className="mono">{fmtOdds(betOdds)}</span>
                      <span className="r-dot-sep">·</span>{e.Bookmaker}
                    </div>

                    {divergent && (
                      <div className="r-sub r-pickline">
                        <span className="r-betk">pick</span>
                        <b className="mono">{pickCode}</b>
                        <span className="r-dot-sep">·</span>
                        <span className="mono">{fmtPct(g.Confidence)}</span>
                      </div>
                    )}
                  </>
                ) : (
                  <>
                    <div className="r-metric mono">{fmtPct(g.Confidence)}</div>
                    <div className="r-sub"><span className="r-betk">pick</span><b className="mono">{pickCode}</b> confidence</div>
                  </>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </>
  )
}

export default BoardTable

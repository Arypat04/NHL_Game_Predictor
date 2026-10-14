import { IconPulse, IconWarn, IconClock, IconFilter } from "./Icons"

export function ReadingSkeleton({ rows = 4 }) {
  return (
    <div className="readings" aria-busy="true" aria-label="Loading readings">
      {Array.from({ length: rows }).map((_, i) => (
        <div className="sk-reading" key={i}>
          <div className="sk" style={{ width: 40 }} />
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <div className="sk" style={{ width: "70%" }} />
            <div className="sk" style={{ width: "45%" }} />
          </div>
          <div className="sk tall" />
          <div className="sk" style={{ width: 70, justifySelf: "end" }} />
        </div>
      ))}
    </div>
  )
}

export function EmptyState({ title = "No games on the slate", msg = "Nothing scheduled for this date. Try another day." }) {
  return (
    <div className="state">
      <div className="state-icon"><IconClock size={28} /></div>
      <div className="state-title">{title}</div>
      <p className="state-msg">{msg}</p>
    </div>
  )
}

export function WarmingState() {
  return (
    <div className="state is-warming">
      <div className="pulse-ring"><IconPulse size={22} /></div>
      <div className="state-title">Instrument warming up</div>
      <p className="state-msg">
        The model is spinning up on a cold server — this takes a few seconds on the
        first request. The readings will appear automatically.
      </p>
    </div>
  )
}

export function DegradedState({
  title = "Live odds unavailable",
  msg = "The odds feed is out of free-tier requests for now. Predictions still work; edges return when the quota resets.",
}) {
  return (
    <div className="state is-degraded">
      <div className="state-icon"><IconWarn size={26} /></div>
      <div className="state-title">{title}</div>
      <p className="state-msg">{msg}</p>
    </div>
  )
}

export function ErrorState({ title = "Couldn't load this", msg = "Something went wrong fetching this.", onRetry }) {
  return (
    <div className="state is-error">
      <div className="state-icon"><IconWarn size={26} /></div>
      <div className="state-title">{title}</div>
      <p className="state-msg">{msg}</p>
      {onRetry && <button className="state-action" onClick={onRetry}>Retry</button>}
    </div>
  )
}

/* Games exist but the confidence filter hid them all — never say "no games". */
export function FilteredEmptyState({ hiddenCount, threshold, onClear, noun = "games" }) {
  return (
    <div className="state is-filtered">
      <div className="state-icon"><IconFilter size={24} /></div>
      <div className="state-title">
        {hiddenCount} {noun} below your {threshold}% filter
      </div>
      <p className="state-msg">They're on the slate — just under your minimum-confidence threshold.</p>
      <button className="state-action" onClick={onClear}>Clear filter</button>
    </div>
  )
}

/* Pick the right state from an error object's kind (set in services/api.js). */
// eslint-disable-next-line react-refresh/only-export-components
export function surfaceStateFor(error, { empty = false, emptyProps, filtered, onRetry } = {}) {
  if (error) {
    if (error.kind === "warming") return <WarmingState />
    if (error.kind === "quota") return <DegradedState />
    if (error.kind === "offline")
      return <ErrorState title="Can't reach Line Lab" msg="We can't reach the service right now — check your connection and try again." onRetry={onRetry} />
    return <ErrorState msg={error.message} onRetry={onRetry} />
  }
  if (filtered) return <FilteredEmptyState {...filtered} />
  if (empty) return <EmptyState {...emptyProps} />
  return null
}

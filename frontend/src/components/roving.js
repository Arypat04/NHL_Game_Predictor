import { useRef } from "react"

/*
 * Manual-activation roving tablist: one tab stop for the whole group, arrow
 * keys move focus (Home/End jump), activation stays on click/Enter/Space so we
 * don't fire a data fetch on every arrow press. Disabled tabs are skipped.
 */
export function useRovingTablist() {
  const refs = useRef([])

  const onKeyDown = (e, idx, count) => {
    const btns = refs.current
    const step = (from, dir) => {
      let n = from
      for (let i = 0; i < count; i++) {
        n = (n + dir + count) % count
        if (btns[n] && !btns[n].disabled) return n
      }
      return from
    }
    let next
    if (e.key === "ArrowRight" || e.key === "ArrowDown") next = step(idx, 1)
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = step(idx, -1)
    else if (e.key === "Home") next = step(-1, 1)
    else if (e.key === "End") next = step(0, -1)
    else return
    e.preventDefault()
    btns[next]?.focus()
  }

  return { refs, onKeyDown }
}

/* Authored icon set — one consistent 1.5px stroke, 24px grid. No emoji/glyphs. */

function Svg({ size = 16, children, fill = "none", ...rest }) {
  return (
    <svg
      width={size} height={size} viewBox="0 0 24 24"
      fill={fill} stroke="currentColor" strokeWidth="1.5"
      strokeLinecap="round" strokeLinejoin="round"
      aria-hidden="true" focusable="false" {...rest}
    >
      {children}
    </svg>
  )
}

export const IconHit = (p) => (
  <Svg {...p}><path d="M20 6 9 17l-5-5" /></Svg>
)

export const IconMiss = (p) => (
  <Svg {...p}><path d="M6 6l12 12M18 6 6 18" /></Svg>
)

/* oscilloscope pulse — the model "reading" mark */
export const IconPulse = (p) => (
  <Svg {...p}><path d="M2 12h4l2.5-7 4 14L15 9l1.5 3H22" /></Svg>
)

export const IconSun = (p) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </Svg>
)

export const IconMoon = (p) => (
  <Svg {...p}><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" /></Svg>
)

export const IconLock = (p) => (
  <Svg {...p}>
    <rect x="5" y="11" width="14" height="9" rx="1.5" />
    <path d="M8 11V8a4 4 0 0 1 8 0v3" />
  </Svg>
)

export const IconWarn = (p) => (
  <Svg {...p}>
    <path d="M12 3.5 1.8 20.5h20.4L12 3.5z" />
    <path d="M12 10v4.5M12 17.6h.01" />
  </Svg>
)

/* up/down deviation arrows — reinforce edge sign beyond color */
export const IconUp = (p) => (
  <Svg {...p}><path d="M12 19V5M6 11l6-6 6 6" /></Svg>
)
export const IconDown = (p) => (
  <Svg {...p}><path d="M12 5v14M6 13l6 6 6-6" /></Svg>
)

export const IconClock = (p) => (
  <Svg {...p}><circle cx="12" cy="12" r="8.5" /><path d="M12 7.5V12l3 2" /></Svg>
)

/* level scores — an equals mark for a live tie */
export const IconTie = (p) => (
  <Svg {...p}><path d="M5 9.5h14M5 14.5h14" /></Svg>
)

export const IconFilter = (p) => (
  <Svg {...p}><path d="M3 5h18M6 12h12M10 19h4" /></Svg>
)

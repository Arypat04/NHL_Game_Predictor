function TableSkeleton({ rows = 5, cols = 7 }) {
  return (
    <>
      {Array.from({ length: rows }).map((_, i) => (
        <tr key={i} className="skeleton-row">
          {Array.from({ length: cols }).map((_, j) => (
            <td key={j}>
              <div className="skeleton-cell" style={{ 
                width: j === 0 ? "60px" : j === cols - 1 ? "80px" : "100%"
              }} />
            </td>
          ))}
        </tr>
      ))}
    </>
  )
}



export default TableSkeleton
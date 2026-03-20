import { useState, useEffect } from "react"
import DateTabs from "./components/DateTabs"
import { getPredictions, toAPIDate } from "./services/api"

function App() {
  const [selectedDate, setSelectedDate] = useState(toAPIDate(new Date()))
  const [predictions, setPredictions] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function fetchData() {
      setLoading(true)
      setError(null)
      try {
        const data = await getPredictions(selectedDate)
        setPredictions(data)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [selectedDate])  

  return (
    <div>
      <h1>NHL Predictor</h1>
      <DateTabs selectedDate={selectedDate} onDateSelect={setSelectedDate} />
      {loading && <p>Loading...</p>}
      {error && <p>Error: {error}</p>}
      {/* predictions table will go here */}
      <pre>{JSON.stringify(predictions, null, 2)}</pre>
    </div>
  )
}

export default App
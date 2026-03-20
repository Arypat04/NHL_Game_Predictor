import { toAPIDate } from  '../services/api'

function getLabel(date, index) {
    if (index === 0) return 'Yesterday'  // i = -1, index 0
    if (index === 1) return 'Today'      // i = 0, index 1
    if (index === 2) return 'Tomorrow'   // i = 1, index 2
    return date.toLocaleDateString('en-US', { weekday: 'short', day: 'numeric' })
}



function DateTabs({ selectedDate, onDateSelect }) { 
    const today = new Date()


    const dates = [] 
    for(let i = -1; i<=6; i++) {
        const date = new Date()
        date.setDate(today.getDate() + i)
        dates.push(date)
    }

    return(
        <div className="date-tabs">
            {dates.map((date, index) => (
                <button 
                    key={toAPIDate(date)}
                    className = {selectedDate === toAPIDate(date) ? 'tab active' : 'tab'}
                    onClick={() => onDateSelect(toAPIDate(date))}
                >
                    {getLabel(date, index)}
                </button>
            ))}
        </div>
    )
}


export default DateTabs
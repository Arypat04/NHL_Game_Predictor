const BASE_URL = 'http://localhost:8000';

export async function getPredictions(date) {
    const response = await fetch(`${BASE_URL}/predictions?date=${date}`);
    if (!response.ok) {
        throw new Error(`Error fetching predictions: ${response.statusText}`);
    }  

    return await response.json();
    
}

export async function getEdges(date) {
    const response = await fetch(`${BASE_URL}/edges?date=${date}`);
    if (!response.ok) {
        throw new Error(`Error fetching edges: ${response.statusText}`);
    }

    return await response.json();
}

export async function getResults(date) {
    const response = await fetch(`${BASE_URL}/results?date=${date}`);
    if (!response.ok) {
        throw new Error(`Error fetching results: ${response.statusText}`);
    } 
    return await response.json();
}

export async function getStatus() {
    const response = await fetch(`${BASE_URL}/status`);
    if (!response.ok) {
        throw new Error(`Error fetching status: ${response.statusText}`);
    } 
    return await response.json();
    
}

export function toAPIDate(date) {
    const year = date.getFullYear()
    const month = String(date.getMonth() + 1).padStart(2, '0')
    const day = String(date.getDate()).padStart(2, '0')
    return `${year}-${month}-${day}`
}


// This file contains JavaScript code for rendering charts using Chart.js to visualize emotional trends.

const ctx = document.getElementById('emotionTrendChart').getContext('2d');
const emotionTrendChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [], // Labels for the x-axis (e.g., dates)
        datasets: [{
            label: 'Emotion Trend',
            data: [], // Data points for the y-axis (e.g., emotion scores)
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderWidth: 1,
            fill: true,
        }]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: 'Emotion Score'
                }
            },
            x: {
                title: {
                    display: true,
                    text: 'Date'
                }
            }
        }
    }
});

// Function to update the chart with new data
function updateChart(labels, data) {
    emotionTrendChart.data.labels = labels;
    emotionTrendChart.data.datasets[0].data = data;
    emotionTrendChart.update();
}

// Example of how to fetch data and update the chart
fetch('/api/emotion-trends')
    .then(response => response.json())
    .then(data => {
        const labels = data.map(entry => entry.date);
        const emotionScores = data.map(entry => entry.score);
        updateChart(labels, emotionScores);
    })
    .catch(error => console.error('Error fetching emotion trends:', error));
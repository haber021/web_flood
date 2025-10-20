document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('barangay-search');
    const barangaysBody = document.getElementById('affected-barangays');
    const refreshButton = document.getElementById('refresh-prediction');

    // Historical chart variables
    let historicalChart = null;
    let currentChartType = 'rainfall';
    let currentPeriod = '10';

    // Initialize variables that were missing
    let municipalityId = null;
    let barangayId = null;

    // Check if elements exist before adding event listeners
    if (searchInput) {
        searchInput.addEventListener('keyup', function(event) {
            const searchTerm = event.target.value.toLowerCase();
            const rows = Array.from(barangaysBody.getElementsByTagName('tr'));

            rows.forEach(function(row) {
                const firstCell = row.querySelector('td:first-child');
                if (firstCell) {
                    const barangayName = firstCell.textContent.toLowerCase();
                    if (barangayName.includes(searchTerm)) {
                        row.style.display = '';
                    } else {
                        row.style.display = 'none';
                    }
                }
            });
        });
    }

    // Initialize historical chart
    initializeHistoricalChart();

    // Historical chart button handlers - check if elements exist
    const rainfallBtn = document.getElementById('btn-rainfall-history');
    const waterLevelBtn = document.getElementById('btn-water-level-history');
    
    if (rainfallBtn) {
        rainfallBtn.addEventListener('click', function() {
            currentChartType = 'rainfall';
            updateChartButtons('rainfall');
            loadHistoricalData();
            fetchHistoricalSuggestion(municipalityId, barangayId, currentChartType, currentPeriod);
        });
    }

    if (waterLevelBtn) {
        waterLevelBtn.addEventListener('click', function() {
            currentChartType = 'water_level';
            updateChartButtons('water_level');
            loadHistoricalData();
            fetchHistoricalSuggestion(municipalityId, barangayId, currentChartType, currentPeriod);
        });
    }

    // Period button handlers
    document.querySelectorAll('.btn-group-sm .btn[data-period]').forEach(button => {
        button.addEventListener('click', function() {
            currentPeriod = this.getAttribute('data-period');
            updatePeriodButtons(currentPeriod);
            loadHistoricalData();
            fetchHistoricalSuggestion(municipalityId, barangayId, currentChartType, currentPeriod);
        });
    });

    // Refresh Prediction functionality
    if (refreshButton) {
        refreshButton.addEventListener('click', function() {
            refreshPrediction();
        });
    }

    function refreshPrediction() {
        // Show loading state
        refreshButton.disabled = true;
        refreshButton.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i> Refreshing...';

        // Get current location filters (if any)
        const urlParams = new URLSearchParams(window.location.search);
        municipalityId = urlParams.get('municipality_id');
        barangayId = urlParams.get('barangay_id');

        // Build API URL
        let apiUrl = '/api/prediction/';
        const params = new URLSearchParams();
        if (municipalityId) params.append('municipality_id', municipalityId);
        if (barangayId) params.append('barangay_id', barangayId);
        if (params.toString()) apiUrl += '?' + params.toString();

        // Make API call for prediction
        fetch(apiUrl)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                // Update prediction display
                updatePredictionDisplay(data);

                // Update affected barangays
                updateAffectedBarangays(data.affected_barangays || []);

                // Update summary stats
                updateSummaryStats(data);

                // Update last prediction time
                const now = new Date();
                const lastPredictionTime = document.getElementById('last-prediction-time');
                if (lastPredictionTime) {
                    lastPredictionTime.textContent = now.toLocaleString('en-US', {
                        year: 'numeric',
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                    });
                }

                // Update prediction status
                const predictionStatus = document.getElementById('prediction-status');
                if (predictionStatus) {
                    predictionStatus.textContent = 'Updated';
                    predictionStatus.className = 'badge bg-success text-white';
                }

                // Auto-fill Predicted Flood Time and Scheduled Send Time inputs
                try {
                    const toLocalInputValue = (d) => {
                        const pad = n => String(n).padStart(2, '0');
                        return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
                    };
                    const predictedInput = document.getElementById('id_predicted_flood_time');
                    const scheduleInput = document.getElementById('id_scheduled_send_time');
                    if (data.flood_time) {
                        const floodTime = new Date(data.flood_time);
                        if (predictedInput) {
                            predictedInput.value = toLocalInputValue(new Date(floodTime));
                        }
                        if (scheduleInput) {
                            const nowLocal = new Date();
                            let sched = new Date(floodTime.getTime() - 30 * 60 * 1000); // 30 minutes before
                            if (sched < nowLocal) {
                                // If predicted is too soon, set schedule 5 minutes from now
                                sched = new Date(nowLocal.getTime() + 5 * 60 * 1000);
                            }
                            scheduleInput.value = toLocalInputValue(sched);
                        }
                    }
                } catch (e) {
                    console.warn('Could not auto-fill predicted/scheduled times:', e);
                }

                // Fetch historical suggestion data
                fetchHistoricalSuggestion(municipalityId, barangayId);

                // Load historical chart data
                loadHistoricalData();
            })
            .catch(error => {
                console.error('Error refreshing prediction:', error);
                const predictionStatus = document.getElementById('prediction-status');
                if (predictionStatus) {
                    predictionStatus.textContent = 'Error';
                    predictionStatus.className = 'badge bg-danger text-white';
                }
                // Still try to fetch historical suggestion even if prediction fails
                fetchHistoricalSuggestion(municipalityId, barangayId);
            })
            .finally(() => {
                // Reset button state
                if (refreshButton) {
                    refreshButton.disabled = false;
                    refreshButton.innerHTML = '<i class="fas fa-sync-alt me-1"></i> Refresh';
                }
            });
    }

    function updatePredictionDisplay(data) {
        // Update flood probability gauge
        const probabilityElement = document.getElementById('flood-probability');
        if (probabilityElement) {
            probabilityElement.textContent = Math.round(data.probability || 0) + '%';

            // Update gauge color based on probability
            const gaugeCircle = probabilityElement.closest('.gauge-circle');
            if (gaugeCircle) {
                gaugeCircle.className = 'gauge-circle';
                const probability = data.probability || 0;
                if (probability >= 75) {
                    gaugeCircle.classList.add('high-risk');
                } else if (probability >= 50) {
                    gaugeCircle.classList.add('medium-risk');
                } else if (probability >= 25) {
                    gaugeCircle.classList.add('low-risk');
                }
            }
        }

        // Update impact
        const impactElement = document.getElementById('prediction-impact');
        if (impactElement) {
            impactElement.textContent = data.impact || 'Unknown';
        }

        // Update ETA
        const etaElement = document.getElementById('prediction-eta');
        if (etaElement) {
            if (data.hours_to_flood) {
                etaElement.textContent = `In approximately ${data.hours_to_flood} hours`;
            } else {
                etaElement.textContent = 'No immediate threat detected';
            }
        }

        // Update contributing factors
        const factorsList = document.getElementById('contributing-factors');
        if (factorsList) {
            factorsList.innerHTML = '';
            if (data.contributing_factors && data.contributing_factors.length > 0) {
                data.contributing_factors.forEach(factor => {
                    const li = document.createElement('li');
                    li.textContent = factor;
                    factorsList.appendChild(li);
                });
            } else {
                const li = document.createElement('li');
                li.textContent = 'No significant contributing factors identified';
                factorsList.appendChild(li);
            }
        }
    }

    function updateAffectedBarangays(barangays) {
        if (!barangaysBody) return;
        
        barangaysBody.innerHTML = '';

        if (barangays.length === 0) {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td colspan="4" class="text-center p-4">
                    <i class="fas fa-check-circle text-success fa-2x mb-2"></i>
                    <p class="mb-0">No barangays currently at risk</p>
                </td>
            `;
            barangaysBody.appendChild(row);
            return;
        }

        barangays.forEach(barangay => {
            const row = document.createElement('tr');

            // Determine risk level styling
            let riskClass = 'table-light';
            let riskText = barangay.risk_level || 'Low';
            let riskBadgeClass = 'secondary';
            
            if (barangay.risk_level === 'High') {
                riskClass = 'table-danger';
                riskBadgeClass = 'danger';
            } else if (barangay.risk_level === 'Medium') {
                riskClass = 'table-warning';
                riskBadgeClass = 'warning';
            }

            row.className = riskClass;
            row.innerHTML = `
                <td>${barangay.name || 'Unknown'}</td>
                <td>${barangay.population ? barangay.population.toLocaleString() : 'N/A'}</td>
                <td><span class="badge bg-${riskBadgeClass}">${riskText}</span></td>
                <td>${barangay.evacuation_centers || 1}</td>
            `;
            barangaysBody.appendChild(row);
        });
    }

    function updateSummaryStats(data) {
        // Update 24h rainfall
        const rainfallElement = document.getElementById('rainfall-24h');
        if (rainfallElement && data.rainfall_24h !== undefined) {
            rainfallElement.textContent = `${data.rainfall_24h.toFixed(1)} mm`;
        }

        // Update current water level
        const waterLevelElement = document.getElementById('current-water-level');
        if (waterLevelElement && data.water_level !== undefined) {
            waterLevelElement.textContent = `${data.water_level.toFixed(2)} m`;
        }
    }

    function fetchHistoricalSuggestion(municipalityId, barangayId, chartType = currentChartType, period = currentPeriod) {
        // Build API URL for historical suggestion
        let suggestionUrl = '/api/historical_suggestion/';
        const params = new URLSearchParams();
        params.append('type', chartType || 'rainfall');
        // Use fixed 7-day period for historical comparison
        params.append('days', '7');
        if (municipalityId) params.append('municipality_id', municipalityId);
        if (barangayId) params.append('barangay_id', barangayId);
        suggestionUrl += '?' + params.toString();

        // Make API call for historical suggestion
        fetch(suggestionUrl)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                // Update decision support section
                updateDecisionSupport(data);
            })
            .catch(error => {
                console.error('Error fetching historical suggestion:', error);
                // Update with error state
                updateDecisionSupport({
                    subject: 'Decision Support: Unable to Load',
                    level: 'Error',
                    level_numeric: 0,
                    suggested_action: 'Please check system status or try again later.',
                    reasons: ['Failed to retrieve historical data for decision support.']
                });
            });
    }

    function updateDecisionSupport(data) {
        if (!data) return;

        // Update subject
        const subjectElement = document.getElementById('suggestion-subject');
        if (subjectElement) {
            subjectElement.textContent = data.subject || 'No subject available';
        }

        // Update level with appropriate badge
        const levelElement = document.getElementById('suggestion-level');
        if (levelElement) {
            levelElement.textContent = data.level || 'Unknown';
            levelElement.className = 'badge';
            const levelNumeric = data.level_numeric || 0;
            if (levelNumeric >= 4) {
                levelElement.classList.add('bg-danger');
            } else if (levelNumeric >= 3) {
                levelElement.classList.add('bg-warning');
            } else if (levelNumeric >= 1) {
                levelElement.classList.add('bg-info');
            } else {
                levelElement.classList.add('bg-secondary');
            }
        }

        // Update suggested action
        const actionElement = document.getElementById('suggested-action');
        if (actionElement) {
            actionElement.textContent = data.suggested_action || 'No action suggested';
        }

        // Update reasons list
        const reasonsList = document.getElementById('suggestion-reasons');
        if (reasonsList) {
            reasonsList.innerHTML = '';
            if (data.reasons && data.reasons.length > 0) {
                data.reasons.forEach(reason => {
                    const li = document.createElement('li');
                    li.textContent = reason;
                    reasonsList.appendChild(li);
                });
            } else {
                const li = document.createElement('li');
                li.textContent = 'No reasons available';
                reasonsList.appendChild(li);
            }
        }
    }

    // Auto-refresh every 30 minutes
    setInterval(refreshPrediction, 30 * 60 * 1000);

    // Load prediction data on page load
    refreshPrediction();

    function initializeHistoricalChart() {
        const chartCanvas = document.getElementById('historical-chart');
        if (!chartCanvas) return;

        const ctx = chartCanvas.getContext('2d');
        historicalChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Rainfall (mm)',
                    data: [],
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    tension: 0.1,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                },
                plugins: {
                    legend: {
                        display: true
                    }
                }
            }
        });
    }

    function updateChartButtons(activeType) {
        const rainfallBtn = document.getElementById('btn-rainfall-history');
        const waterLevelBtn = document.getElementById('btn-water-level-history');
        
        if (rainfallBtn) rainfallBtn.classList.toggle('active', activeType === 'rainfall');
        if (waterLevelBtn) waterLevelBtn.classList.toggle('active', activeType === 'water_level');
    }

    function updatePeriodButtons(activePeriod) {
        document.querySelectorAll('.btn-group-sm .btn[data-period]').forEach(button => {
            button.classList.toggle('active', button.getAttribute('data-period') === activePeriod);
        });
    }

    function loadHistoricalData() {
        const urlParams = new URLSearchParams(window.location.search);
        const municipalityId = urlParams.get('municipality_id');
        const barangayId = urlParams.get('barangay_id');

        let apiUrl = `/api/chart-data/?type=${currentChartType}`;
        if (currentPeriod === '10') {
            apiUrl += `&limit=10`;
        } else {
            apiUrl += `&range=${currentPeriod}d`;
        }
        if (municipalityId) apiUrl += `&municipality_id=${municipalityId}`;
        if (barangayId) apiUrl += `&barangay_id=${barangayId}`;

        fetch(apiUrl)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                updateHistoricalChart(data);
            })
            .catch(error => {
                console.error('Error loading historical data:', error);
            });
    }

    function updateHistoricalChart(data) {
        if (!historicalChart) return;

        const labels = data.labels_manila || data.labels || [];
        const values = data.values || [];

        historicalChart.data.labels = labels;
        historicalChart.data.datasets[0].data = values;
        historicalChart.data.datasets[0].label = currentChartType === 'rainfall' ? 'Rainfall (mm)' : 'Water Level (m)';
        historicalChart.data.datasets[0].borderColor = currentChartType === 'rainfall' ? 'rgb(54, 162, 235)' : 'rgb(255, 99, 132)';
        historicalChart.data.datasets[0].backgroundColor = currentChartType === 'rainfall' ? 'rgba(54, 162, 235, 0.1)' : 'rgba(255, 99, 132, 0.1)';

        historicalChart.update();
    }
});
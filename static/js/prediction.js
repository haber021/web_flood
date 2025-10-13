// static/js/prediction.js

document.addEventListener('DOMContentLoaded', function() {
    // --- STATE ---
    const state = {
        historicalChart: null,
        historicalDataType: 'rainfall', // 'rainfall' or 'water_level'
        historicalPeriod: '7', // '7', '30', '365'
        municipalityId: null,
        barangayId: null,
    };

    // --- INITIALIZATION ---
    function initialize() {
        initLocationSelectors();
        initHistoricalChart();
        bindHistoricalChartControls();
        bindPredictionControls();

        // Initial data load
        refreshAll();
    }

    // --- LOCATION SELECTORS ---
    function initLocationSelectors() {
        // This would be similar to the dashboard, but for simplicity, we'll just read from it if it exists
        // For a standalone prediction page, you'd populate these dropdowns.
        // We will assume the global `location-select` and `barangay-select` from `base.html` are used.
        const muniSelect = document.getElementById('location-select');
        const brgySelect = document.getElementById('barangay-select');

        if (muniSelect) {
            muniSelect.addEventListener('change', () => {
                state.municipalityId = muniSelect.value || null;
                state.barangayId = null; // Reset barangay
                if (brgySelect) brgySelect.value = '';
                updateLocationDisplay();
                refreshAll();
            });
        }

        if (brgySelect) {
            brgySelect.addEventListener('change', () => {
                state.barangayId = brgySelect.value || null;
                updateLocationDisplay();
                refreshAll();
            });
        }
    }

    function updateLocationDisplay() {
        const display = document.getElementById('current-location-display');
        if (!display) return;

        const muniSelect = document.getElementById('location-select');
        const brgySelect = document.getElementById('barangay-select');

        let text = 'All Areas';
        if (state.municipalityId && muniSelect) {
            const muniOption = muniSelect.options[muniSelect.selectedIndex];
            text = muniOption ? muniOption.text : 'All Areas';
        }
        if (state.barangayId && brgySelect) {
            const brgyOption = brgySelect.options[brgySelect.selectedIndex];
            if (brgyOption && brgyOption.value) {
                text += ` > ${brgyOption.text}`;
            }
        }
        display.textContent = text;
    }

    // --- DATA FETCHING & UI UPDATES ---
    function refreshAll() {
        updateSummaryStats();
        updateHistoricalChart();
        updatePredictionModel();
        updateAffectedBarangays();
        updateDecisionSupport(); // This is the new function to fix the issue
    }

    async function updateSummaryStats() {
        const rainfallEl = document.getElementById('rainfall-24h');
        const waterLevelEl = document.getElementById('current-water-level');
        if (!rainfallEl || !waterLevelEl) return;

        rainfallEl.textContent = '...';
        waterLevelEl.textContent = '...';

        let url = `/api/parameter-status/`;
        const params = buildLocationParams();
        if (params) url += `?${params}`;

        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error('Failed to fetch summary stats');
            const data = await response.json();

            const rainfall = data.items.find(i => i.parameter === 'rainfall');
            const waterLevel = data.items.find(i => i.parameter === 'water_level');

            rainfallEl.textContent = rainfall && rainfall.latest !== null ? `${rainfall.latest.toFixed(1)} mm` : '--';
            waterLevelEl.textContent = waterLevel && waterLevel.latest !== null ? `${waterLevel.latest.toFixed(2)} m` : '--';

        } catch (error) {
            console.error("Error updating summary stats:", error);
            rainfallEl.textContent = 'Error';
            waterLevelEl.textContent = 'Error';
        }
    }

    async function updatePredictionModel() {
        const probabilityEl = document.getElementById('flood-probability');
        const impactEl = document.getElementById('prediction-impact');
        const etaEl = document.getElementById('prediction-eta');
        const factorsEl = document.getElementById('contributing-factors');
        const statusEl = document.getElementById('prediction-status');
        const lastTimeEl = document.getElementById('last-prediction-time');

        if (!probabilityEl) return;

        statusEl.textContent = 'Calculating...';
        statusEl.className = 'badge bg-info text-dark';

        let url = `/api/prediction/`;
        const params = buildLocationParams();
        if (params) url += `?${params}`;

        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error('Prediction API request failed');
            const data = await response.json();

            // Update Gauge
            const probability = data.probability || 0;
            probabilityEl.textContent = `${probability.toFixed(0)}%`;
            const gauge = probabilityEl.closest('.prediction-gauge');
            if (gauge) {
                gauge.style.setProperty('--gauge-value', probability / 100);
            }

            // Update text fields
            impactEl.textContent = data.impact || 'Not available.';
            etaEl.textContent = data.hours_to_flood ? `Approx. ${data.hours_to_flood.toFixed(1)} hours` : 'Not applicable.';
            lastTimeEl.textContent = new Date().toLocaleTimeString();

            // Update contributing factors
            if (data.contributing_factors && data.contributing_factors.length > 0) {
                factorsEl.innerHTML = data.contributing_factors.map(f => `<li>${f}</li>`).join('');
            } else {
                factorsEl.innerHTML = '<li>No significant factors identified.</li>';
            }

            // Update status badge
            statusEl.textContent = data.severity_level > 0 ? getSeverityName(data.severity_level) : 'Normal';
            statusEl.className = `badge ${getSeverityClass(data.severity_level, 'bg')}`;

        } catch (error) {
            console.error("Error updating prediction model:", error);
            statusEl.textContent = 'Error';
            statusEl.className = 'badge bg-danger';
            probabilityEl.textContent = '--';
            impactEl.textContent = 'Could not calculate prediction.';
            etaEl.textContent = '--';
            factorsEl.innerHTML = '<li>Error fetching data.</li>';
        }
    }

    async function updateAffectedBarangays() {
        const tbody = document.getElementById('affected-barangays');
        if (!tbody) return;

        tbody.innerHTML = '<tr><td colspan="4" class="text-center">Loading...</td></tr>';

        let url = `/api/prediction/`;
        const params = buildLocationParams();
        if (params) url += `?${params}`;

        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error('Prediction API failed');
            const data = await response.json();

            if (data.affected_barangays && data.affected_barangays.length > 0) {
                tbody.innerHTML = data.affected_barangays.map(b => `
                    <tr>
                        <td>${b.name}</td>
                        <td>${b.population.toLocaleString()}</td>
                        <td><span class="badge ${getRiskClass(b.risk_level)}">${b.risk_level}</span></td>
                        <td>${b.evacuation_centers}</td>
                    </tr>
                `).join('');
            } else {
                tbody.innerHTML = '<tr><td colspan="4" class="text-center">No barangays predicted to be affected.</td></tr>';
            }

        } catch (error) {
            console.error("Error updating affected barangays:", error);
            tbody.innerHTML = '<tr><td colspan="4" class="text-center text-danger">Error loading data.</td></tr>';
        }
    }

    /**
     * THIS IS THE FIX: Update the Decision Support card with data from the backend.
     */
    async function updateDecisionSupport() {
        const levelEl = document.getElementById('suggestion-level');
        const subjectEl = document.getElementById('suggestion-subject');
        const actionEl = document.getElementById('suggested-action');
        const reasonsEl = document.getElementById('suggestion-reasons');

        if (!levelEl) return;

        levelEl.textContent = 'Loading...';
        subjectEl.textContent = 'Analyzing...';
        actionEl.textContent = 'Please wait...';
        reasonsEl.innerHTML = '<li>Loading...</li>';

        let url = `/api/historical-suggestion/?type=${state.historicalDataType}&days=${state.historicalPeriod}`;
        const params = buildLocationParams();
        if (params) url += `&${params}`;

        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error('Failed to fetch suggestion');
            const data = await response.json();

            levelEl.textContent = data.level || 'Unknown';
            levelEl.className = `badge ${getSeverityClass(data.level_numeric, 'bg')}`;
            subjectEl.textContent = data.subject || 'No suggestion available.';
            actionEl.textContent = data.suggested_action || 'Analysis inconclusive.';

            if (data.reasons && data.reasons.length > 0) {
                reasonsEl.innerHTML = data.reasons.map(r => `<li>${r}</li>`).join('');
            } else {
                reasonsEl.innerHTML = '<li>No specific reasons provided.</li>';
            }

        } catch (error) {
            console.error("Error updating decision support:", error);
            levelEl.textContent = 'Error';
            levelEl.className = 'badge bg-danger';
            subjectEl.textContent = 'Error fetching suggestion.';
            actionEl.textContent = 'Could not connect to the decision support service.';
            reasonsEl.innerHTML = '<li>An error occurred.</li>';
        }
    }

    // --- HISTORICAL CHART ---
    function initHistoricalChart() {
        const ctx = document.getElementById('historical-chart');
        if (!ctx || !window.Chart) return;

        state.historicalChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Current Period',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        data: [],
                        fill: true,
                    },
                    {
                        label: 'Historical (Last Year)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        data: [],
                        fill: true,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Rainfall (mm)' }
                    }
                },
                plugins: {
                    legend: { position: 'top' }
                }
            }
        });
    }

    function bindHistoricalChartControls() {
        const typeButtons = document.querySelectorAll('#btn-rainfall-history, #btn-water-level-history');
        const periodButtons = document.querySelectorAll('.btn-group[data-period] button');

        typeButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                typeButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                state.historicalDataType = btn.id === 'btn-rainfall-history' ? 'rainfall' : 'water_level';
                updateHistoricalChart();
                updateDecisionSupport();
            });
        });

        periodButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                periodButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                state.historicalPeriod = btn.getAttribute('data-period');
                updateHistoricalChart();
                updateDecisionSupport();
            });
        });
    }

    async function updateHistoricalChart() {
        if (!state.historicalChart) return;

        const chart = state.historicalChart;
        const unit = state.historicalDataType === 'rainfall' ? 'mm' : 'm';
        chart.options.scales.y.title.text = `${state.historicalDataType.replace('_', ' ')} (${unit})`;

        let url = `/api/chart-data/?type=${state.historicalDataType}&days=${state.historicalPeriod}`;
        const params = buildLocationParams();
        if (params) url += `&${params}`;

        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error('Failed to fetch chart data');
            const data = await response.json();

            chart.data.labels = data.labels.map(l => new Date(l).toLocaleDateString());
            chart.data.datasets[0].data = data.values;
            chart.data.datasets[0].label = `Current (${unit})`;

            // Fetch historical data
            const histUrl = `${url}&historical=true`;
            const histResponse = await fetch(histUrl);
            if (!histResponse.ok) throw new Error('Failed to fetch historical chart data');
            const histData = await histResponse.json();
            chart.data.datasets[1].data = histData.historical_values;
            chart.data.datasets[1].label = `Last Year (${unit})`;

            chart.update();

        } catch (error) {
            console.error("Error updating historical chart:", error);
        }
    }

    // --- OTHER CONTROLS ---
    function bindPredictionControls() {
        const refreshBtn = document.getElementById('refresh-prediction');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', updatePredictionModel);
        }
    }

    // --- UTILITY HELPERS ---
    function buildLocationParams() {
        const params = [];
        if (state.municipalityId) params.push(`municipality_id=${state.municipalityId}`);
        if (state.barangayId) params.push(`barangay_id=${state.barangayId}`);
        return params.join('&');
    }

    function getSeverityName(level) {
        const names = { 1: 'Advisory', 2: 'Watch', 3: 'Warning', 4: 'Emergency', 5: 'Catastrophic' };
        return names[level] || 'Normal';
    }

    function getSeverityClass(level, prefix = 'alert') {
        if (level >= 4) return `${prefix}-danger`;
        if (level === 3) return `${prefix}-warning`;
        if (level >= 1) return `${prefix}-info`;
        return `${prefix}-success`;
    }

    function getRiskClass(riskLevel) {
        const level = (riskLevel || '').toLowerCase();
        if (level === 'high') return 'bg-danger';
        if (level === 'moderate') return 'bg-warning text-dark';
        return 'bg-success';
    }

    // --- START ---
    initialize();
});
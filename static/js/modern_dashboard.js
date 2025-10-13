// modern_dashboard.js
// Powers the modern dashboard UI with live, database-driven data using existing API endpoints

(function() {
  const state = {
    municipalityId: null,
    barangayId: null,
    map: null,
    mapLayers: {
      zones: null,
      sensors: null,
      barangays: null,
    },
    barangayLayerById: new Map(), // Add this to store barangay layers
    chart: null,
    normalize: false, // when true, normalize all lines to 0–100 for alignment
    _rawSeries: null, // keep last fetched raw series to re-apply normalization without refetch
    _trendsFetchInFlight: false, // guard against overlapping fetches
    _alertsReqToken: 0, // monotonic token to ensure latest alert response wins
    trendsRange: 'latest',
    lastAlertId: null, // Track the last seen alert to play sound only for new ones
    lastCombinedLevel: 0, // Track the last severity level to trigger popups only on change
    mapDisplayParam: 'overall', // New state for map parameter
  };
  let audioContext = null; // Audio context for playing sounds

  function initAudioContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
  }

  document.body.addEventListener('click', initAudioContext, { once: true });

  document.addEventListener('DOMContentLoaded', () => {
    setupLocationSelector();
    initChart();
    initMap();
    bindApplyThresholdsButton();
    setupMapParamSelector(); // New: Bind the map parameter selector
    setupTrendsRangeControls();
    // Ensure any previous chart overlay from older versions is removed
    try { clearChartOverlay(); } catch (e) {}

    // Bind align-lines toggle if present
    const alignToggle = document.getElementById('align-lines-toggle');
    if (alignToggle) {
      state.normalize = !!alignToggle.checked;
      alignToggle.addEventListener('change', () => {
        state.normalize = !!alignToggle.checked;
        // Re-apply scaling using the cached raw series
        if (state.chart && state._rawSeries && Array.isArray(state.chart.data.labels)) {
          applyChartScaling(state._rawSeries);
          try {
            // Temporarily disable events to avoid hover processing during update
            const prevEvents = state.chart.options && state.chart.options.events ? state.chart.options.events.slice() : undefined;
            if (state.chart.options) state.chart.options.events = [];
            hardenScaleOptions(state.chart);
            sanitizeChartData(state.chart);
            hardenDatasets(state.chart);
            deepPlainChartConfig(state.chart);
            state.chart.update();
            if (prevEvents) state.chart.options.events = prevEvents; else if (state.chart.options) delete state.chart.options.events;
          } catch (e) {
            recreateChart([], { t:[], h:[], r:[], wl:[], ws:[] });
          }
        } else {
          // Fallback: reload
          loadTrendsChart();
        }
      });
    }

    // Modal event listeners
    const modalCloseBtn = document.getElementById('modal-close');
    if (modalCloseBtn) {
        modalCloseBtn.addEventListener('click', closeAlertModal);
    }
    const modalOverlay = document.getElementById('alert-modal');
    if (modalOverlay) {
        modalOverlay.addEventListener('click', (e) => {
            if (e.target === modalOverlay) {
                closeAlertModal();
            }
        });
    }

    // Initial loads
    refreshAll();

    // Periodic refreshes (guard to prevent double interval setup on hot-reloads)
    if (!state._intervalsSet) {
      setInterval(updateSensorValues, 60 * 1000);
      setInterval(updateAlerts, 30 * 1000);
      setInterval(updateMapData, 5 * 60 * 1000); // Refresh map data every 5 minutes
      state._intervalsSet = true;
    }
    // When the tab becomes visible again, refresh immediately
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') {
        loadTrendsChart();
      }
    });

    // Responsive: on window resize, resize chart and re-apply scaling
    window.addEventListener('resize', () => {
      if (state.chart) {
        if (state._rawSeries) applyChartScaling(state._rawSeries);
        try { state.chart.resize(); } catch(e) {}
      }
    });
    // Responsive: observe container resize
    const chartContainer = document.querySelector('.chart-container-modern');
    if (window.ResizeObserver && chartContainer) {
      try {
        const ro = new ResizeObserver(() => {
          if (state.chart) {
            if (state._rawSeries) applyChartScaling(state._rawSeries);
            try { state.chart.resize(); } catch(e) {}
          }
        });
        ro.observe(chartContainer);
        state._chartResizeObserver = ro;
      } catch (e) { /* ignore */ }
    }
  });

  // ===== Helpers defined at IIFE scope (not inside DOMContentLoaded/handlers) =====

  // Ensure global point defaults exist to avoid undefined hitRadius/hoverRadius
  function ensureChartGlobalPointDefaults() {
    try {
      if (!window.Chart || !Chart.defaults) return;
      const p = (Chart.defaults.elements = Chart.defaults.elements || {}).point = (Chart.defaults.elements.point || {});
      if (typeof p.radius !== 'number') p.radius = 4;
      if (typeof p.hoverRadius !== 'number') p.hoverRadius = 6;
      if (typeof p.hitRadius !== 'number') p.hitRadius = 6;
      if (typeof p.borderWidth !== 'number') p.borderWidth = 2;
      if (!p.backgroundColor) p.backgroundColor = '#ffffff';
    } catch (e) { /* ignore */ }
  }

  async function fetchHeatmapPoints() {
    try {
      const params = [];
      if (state.municipalityId) params.push(`municipality_id=${state.municipalityId}`);
      if (state.barangayId) params.push(`barangay_id=${state.barangayId}`);
      const url = `/api/heatmap/${params.length ? ('?' + params.join('&')) : ''}`;
      const res = await fetch(url, { headers: { 'Accept': 'application/json' }});
      if (!res.ok) return null;
      return await res.json();
    } catch (e) { return null; }
  }

  function bindHeatmapToggleUI() {
    const cb = document.getElementById('heatmap-toggle');
    if (!cb) return;
    // Initialize checkbox state
    cb.checked = !!state.heatEnabled;
    cb.addEventListener('change', () => {
      const enable = !!cb.checked;
      setHeatmapEnabled(enable);
    });
  }

  // Convert API data to heatmap points [lat, lng, intensity]
  function buildHeatPoints(data) {
    if (data && Array.isArray(data.points)) {
        return data.points;
    }
    return [];
  }

  // Add a simple heatmap toggle control
  function addHeatToggleControl(map) {
    if (!window.L) return;
    if (!document.getElementById('flood-heat-style')) {
      const st = document.createElement('style');
      st.id = 'flood-heat-style';
      st.textContent = `.leaflet-control-heat a{background:#fff;border:1px solid #dcdcdc;border-radius:4px;display:inline-block;width:28px;height:28px;line-height:28px;text-align:center;font-size:16px;color:#333;box-shadow:0 1px 3px rgba(0,0,0,.2);} .leaflet-control-heat a.active{background:#e0f2fe;border-color:#7dd3fc;color:#0369a1}`;
      document.head.appendChild(st);
    }
    const HeatCtrl = L.Control.extend({
      options: { position: 'topleft' },
      onAdd: function() {
        const c = L.DomUtil.create('div', 'leaflet-control leaflet-bar leaflet-control-heat');
        const a = L.DomUtil.create('a', '', c);
        a.href = '#'; a.title = 'Toggle heatmap'; a.innerHTML = '🔥';
        // Make absolutely sure it's visible
        c.style.zIndex = '1000';
        c.style.display = 'block';
        c.style.marginTop = '4px';
        if (state.heatEnabled) a.classList.add('active');
        L.DomEvent.on(a, 'click', L.DomEvent.stop)
          .on(a, 'click', () => setHeatmapEnabled(!state.heatEnabled));
        return c;
      }
    });
    map.addControl(new HeatCtrl());
  }

  // Central function to control heatmap state
  function setHeatmapEnabled(enabled) {
    state.heatEnabled = !!enabled;
    // Sync the map button
    const mapBtn = document.querySelector('.leaflet-control-heat a');
    if (mapBtn) mapBtn.classList.toggle('active', state.heatEnabled);
    // Sync the sidebar checkbox
    const sidebarCheck = document.getElementById('heatmap-toggle');
    if (sidebarCheck) sidebarCheck.checked = state.heatEnabled;

    // Update the layer
    updateHeatLayer();
  }

  // ---------------- Fetch all barangays in selected municipality ----------------
  async function fetchAllBarangaysInMunicipality() {
    try {
      let url = '/api/map-data/';
      const params = [];
      if (state.municipalityId) params.push(`municipality_id=${state.municipalityId}`);
      if (state.barangayId) params.push(`barangay_id=${state.barangayId}`);
      if (params.length) url += '?' + params.join('&');

      const res = await fetch(url, { headers: { 'Accept': 'application/json' }});
      if (!res.ok) return [];

      const data = await res.json();
      const barangays = data.barangays || [];

      // Adjust severity based on selected map parameter
      return barangays.map(b => {
        let severity = b.severity || 0;
        if (state.mapDisplayParam !== 'overall' && b.param_severities && b.param_severities[state.mapDisplayParam] !== undefined) {
          severity = b.param_severities[state.mapDisplayParam];
        }
        return { ...b, severity };
      });
    } catch (e) {
      return [];
    }
  }

  // ---------------- Affected Areas updater ----------------
  function updateAffectedAreas(affectedBarangays, severityLevel = 0) {
    const tbody = document.getElementById('affected-areas-body');
    if (!tbody) return;

    if (!affectedBarangays || affectedBarangays.length === 0) {
      tbody.innerHTML = '<tr><td colspan="3" style="color: var(--gray)">No barangays currently affected by floods.</td></tr>';
      return;
    }

    // If affectedBarangays is an array of IDs (legacy), convert to objects with severity
    let barangaysData = affectedBarangays;
    if (typeof affectedBarangays[0] === 'number' || typeof affectedBarangays[0] === 'string') {
      // Array of IDs, fetch details
      const promises = affectedBarangays.map(id =>
        fetch(`/api/barangays/${id}/`, { headers: { 'Accept': 'application/json' }})
          .then(r => r.ok ? r.json() : Promise.reject(new Error(`Failed to fetch barangay ${id}`)))
      );

      Promise.allSettled(promises)
        .then(results => {
          const barangays = results
            .filter(r => r.status === 'fulfilled' && r.value)
            .map(r => ({ ...r.value, severity: severityLevel }));

          renderAffectedAreasTable(barangays);
        })
        .catch(() => {
          tbody.innerHTML = '<tr><td colspan="3">Unable to load affected areas at this time.</td></tr>';
        });
    } else {
      // Array of barangay objects with severity
      renderAffectedAreasTable(barangaysData);
    }
  }

  function renderAffectedAreasTable(barangays) {
    const tbody = document.getElementById('affected-areas-body');
    if (!tbody) return;

    // Update the header to reflect the selected parameter
    const headerEl = document.querySelector('.modern-card .card-title-modern');
    if (headerEl && headerEl.textContent.startsWith('Affected Areas')) {
      const paramLabel = state.mapDisplayParam === 'overall' ? 'Overall Risk' :
                        state.mapDisplayParam === 'rainfall' ? 'Rainfall' :
                        state.mapDisplayParam === 'water_level' ? 'Water Level' :
                        state.mapDisplayParam === 'temperature' ? 'Temperature' :
                        state.mapDisplayParam === 'humidity' ? 'Humidity' :
                        state.mapDisplayParam === 'wind_speed' ? 'Wind Speed' : 'Overall Risk';
      headerEl.textContent = `Affected Areas (${paramLabel})`;
    }

    if (barangays.length > 0) {
      tbody.innerHTML = barangays.map(b => {
        const severityLevel = b.severity || 0;
        const riskLevel = getSeverityText(severityLevel);
        const riskClass =
            severityLevel >= 4 ? 'status-danger' :
            severityLevel >= 3 ? 'status-warning' :
            severityLevel >= 1 ? 'status-info' :
            'status-normal';

        return `<tr>
          <td>${escapeHtml(b.name || '—')}</td>
          <td>${Number(b.population || 0).toLocaleString()}</td>
          <td><span class="status-badge ${riskClass}">${riskLevel}</span></td>
        </tr>`;
      }).join('');
    } else {
      tbody.innerHTML = '<tr><td colspan="3" style="color: var(--gray)">No barangays currently affected by floods.</td></tr>';
    }
  }

  // ---------------- Current Location card updater ----------------
  function updateCurrentLocationCard() {
    try {
      const muniEl = document.getElementById('current-muni');
      const brgyEl = document.getElementById('current-brgy');
      const noteEl = document.getElementById('current-location-note');
      const muniSel = document.getElementById('location-select');
      const brgySel = document.getElementById('barangay-select');
      if (!muniEl || !brgyEl) return; // card not present
      const muniName = (muniSel && muniSel.selectedIndex > -1) ? muniSel.options[muniSel.selectedIndex].text : 'All Municipalities';
      const brgyName = (brgySel && brgySel.selectedIndex > 0) ? brgySel.options[brgySel.selectedIndex].text : 'All Barangays';
      muniEl.textContent = muniName || 'All Municipalities';
      brgyEl.textContent = brgyName || 'All Barangays';
      if (noteEl) {
        if (state.barangayId || state.municipalityId) {
          noteEl.textContent = 'Monitoring environmental conditions for the selected location.';
        } else {
          noteEl.textContent = 'Monitoring environmental conditions continuously.';
        }

        // Also update the trends chart location note
        const trendsNoteEl = document.getElementById('trends-location-note');
        if (trendsNoteEl) {
            let locationText = 'For All Locations';
            if (brgyName && brgyName !== 'All Barangays') locationText = `For ${brgyName}`;
            else if (muniName && muniName !== 'All Municipalities') locationText = `For ${muniName}`;
            trendsNoteEl.textContent = locationText;
        }
      }
    } catch (e) { /* ignore */ }
  }

  function refreshAll() {
    updateSensorValues();
    updateAlerts();
    updateMapData();
    loadTrendsChart();
  }

  // Bind Latest / 1W / 1M / 1Y controls and update state.trendsRange
  function setupTrendsRangeControls() {
    const group = document.getElementById('trends-range');
    if (!group) return;
    const buttons = Array.from(group.querySelectorAll('button[data-range]'));
    const setActive = (val) => {
      buttons.forEach(b => b.classList.toggle('active', b.getAttribute('data-range') === val));
    };
    // Initialize selection
    const initial = (state.trendsRange || 'latest');
    setActive(initial);
    // Click bindings
    buttons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        const r = btn.getAttribute('data-range') || 'latest';
        state.trendsRange = r;
        setActive(r);
        // Update the small description under the header
        const descEl = document.getElementById('trends-desc');
        if (descEl) {
          let timeText = 'Showing latest 10 readings';
          if (r === '1w') timeText = 'Showing readings from the last week';
          else if (r === '1m') timeText = 'Showing readings from the last month';
          else if (r === '1y') timeText = 'Showing readings from the last year';

          const muniSel = document.getElementById('location-select');
          const brgySel = document.getElementById('barangay-select');
          const muniName = (muniSel && muniSel.selectedIndex > 0) ? muniSel.options[muniSel.selectedIndex].text : '';
          const brgyName = (brgySel && brgySel.selectedIndex > 0) ? brgySel.options[brgySel.selectedIndex].text : '';

          let locationText = 'for all locations';
          if (brgyName) locationText = `for ${brgyName}`;
          else if (muniName) locationText = `for ${muniName}`;

          descEl.textContent = `${timeText} ${locationText}.`;
        }
        loadTrendsChart();
      });
    });
  }

  // ---------------- Apply Thresholds (Server-side evaluation) ----------------
  function bindApplyThresholdsButton() {
    const btn = document.getElementById('apply-thresholds-btn');
    if (!btn) return;
    btn.addEventListener('click', async () => {
      try {
        btn.disabled = true;
        btn.textContent = 'Applying...';
        await applyThresholdsNow();
      } catch (e) {
        console.error('Error applying thresholds:', e);
        alert('Unable to apply thresholds. Please make sure you are logged in and try again.');
      } finally {
        btn.disabled = false;
        btn.textContent = 'Apply Thresholds';
      }
    });
  }

  async function applyThresholdsNow() {
    try {
      const body = buildApplyThresholdsBody();
      const res = await fetch('/api/apply-thresholds/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'X-CSRFToken': getCSRFToken(),
        },
        body: JSON.stringify(body),
        credentials: 'same-origin',
      });
      if (!res.ok) {
        // Silently ignore 401/403 (not logged in) for automatic attempts
        if (res.status === 401 || res.status === 403) return;
        const txt = await res.text().catch(() => '');
        throw new Error(`Failed to apply thresholds (${res.status}): ${txt || res.statusText}`);
      }
      // On success, refresh alerts UI and map data
      updateAlerts();
      updateMapData();
    } catch (e) {
      // Quietly log for automatic path
      console.warn('[Apply Thresholds] Auto-apply failed:', e.message || e);
    }
  }

  function buildApplyThresholdsBody() {
    const body = { dry_run: false };
    if (state.barangayId) {
      body.process_scope = 'barangay';
      body.barangay_id = state.barangayId;
    } else if (state.municipalityId) {
      body.process_scope = 'municipality';
      body.municipality_id = state.municipalityId;
    } else {
      body.process_scope = 'all';
    }
    return body;
  }

  function getCSRFToken() {
    // Standard Django CSRF cookie name is 'csrftoken'
    const name = 'csrftoken=';
    const cookies = document.cookie ? document.cookie.split(';') : [];
    for (let i = 0; i < cookies.length; i++) {
      const c = cookies[i].trim();
      if (c.startsWith(name)) return decodeURIComponent(c.substring(name.length));
    }
    return '';
  }

  // ---------------- Location selector ----------------
  function setupLocationSelector() {
    const muniSel = document.getElementById('location-select');
    const brgySel = document.getElementById('barangay-select');
    if (!muniSel) return;

    // Populate municipalities
    fetch('/api/municipalities/?limit=200')
      .then(r => r.ok ? r.json() : Promise.reject(r))
      .then(data => {
        const results = data.results || [];
        // Clear existing non-default options
        muniSel.querySelectorAll('option:not([selected])').forEach(o => o.remove());
        results.sort((a,b)=>a.name.localeCompare(b.name)).forEach(m => {
          const opt = document.createElement('option');
          opt.value = m.id;
          opt.textContent = m.name;
          muniSel.appendChild(opt);
        });
        // Restore persisted selection
        const savedMuni = sessionStorage.getItem('dashboard_municipality_id');
        const savedBrgy = sessionStorage.getItem('dashboard_barangay_id');
        if (savedMuni) {
          muniSel.value = savedMuni;
          state.municipalityId = savedMuni;
          populateBarangays(savedMuni).then(() => {
            if (savedBrgy && brgySel) {
              brgySel.value = savedBrgy;
              state.barangayId = savedBrgy;
              brgySel.disabled = false;
              updateCurrentLocationCard();
              refreshAll();
              applyThresholdsNow();
            } else {
              updateCurrentLocationCard();
              refreshAll();
              applyThresholdsNow();
            }
          });
        } else {
          // No saved selection; still reflect defaults in the card
          updateCurrentLocationCard();
        }
      })
      .catch(() => {});

    // On municipality change, load barangays
    muniSel.addEventListener('change', () => {
      const val = muniSel.value;
      state.municipalityId = val || null;
      sessionStorage.setItem('dashboard_municipality_id', state.municipalityId || '');
      state.barangayId = null;
      sessionStorage.setItem('dashboard_barangay_id', ''); // Clear barangay selection
      if (brgySel) {
        brgySel.innerHTML = '<option value="" selected>All Barangays</option>';
        brgySel.disabled = !state.municipalityId;
      }
      updateCurrentLocationCard();
      // Reset Alert Status UI and invalidate any in-flight alert requests
      const badge = document.getElementById('alert-status-badge');
      const title = document.getElementById('alert-title');
      const msg = document.getElementById('alert-message');
      if (badge) { badge.textContent = 'Normal'; badge.classList.remove('status-warning','status-danger'); badge.classList.add('status-normal'); }
      if (title) title.textContent = 'Loading status…';
      if (msg) msg.textContent = 'Fetching latest advisory for the selected location.';
      const list = document.getElementById('param-status-list');
      if (list) list.innerHTML = '';
      state._alertsReqToken++;
      if (state.municipalityId) {
        // A municipality is selected. Find its details to center the map.
        fetch(`/api/municipalities/${state.municipalityId}/`)
          .then(r => r.ok ? r.json() : Promise.reject(r))
          .then(municipality => {
            if (state.map && municipality.latitude && municipality.longitude) {
              // Zoom to the municipality's center. The zoom level (e.g., 12) can be adjusted.
              state.map.setView([municipality.latitude, municipality.longitude], 12);
            }
            // Now that the map is centered, refresh other data.
            populateBarangays(state.municipalityId).then(() => {
              // After populating barangays, refresh all dashboard components.
              refreshAll();
            });
          })
          .catch(() => {
            // Fallback if fetching municipality details fails
            populateBarangays(state.municipalityId).then(() => refreshAll());
          });

        // Automatically apply thresholds for the new scope (municipality-wide for all barangays)
        applyThresholdsNow();
      } else {
        if (state.heatEnabled) updateHeatLayer();
        refreshAll();
      }
    });

    // On barangay change, just set barangayId and refresh
    if (brgySel) {
      brgySel.addEventListener('change', () => {
        state.barangayId = brgySel.value || null;
        sessionStorage.setItem('dashboard_barangay_id', state.barangayId || '');
        updateCurrentLocationCard();
        // Reset Alert Status UI immediately to avoid stale badge while loading
        const badge = document.getElementById('alert-status-badge');
        const title = document.getElementById('alert-title');
        const msg = document.getElementById('alert-message');
        if (badge) { badge.textContent = 'Normal'; badge.classList.remove('status-warning','status-danger'); badge.classList.add('status-normal'); }
        if (title) title.textContent = 'Loading status…';
        if (msg) msg.textContent = 'Fetching latest advisory for the selected barangay.';
        const list = document.getElementById('param-status-list');
        if (list) list.innerHTML = '';
        // Bump token to invalidate in-flight requests for previous selection
        state._alertsReqToken++;
        refreshAll();
        // Explicitly update map data to trigger zoom for the selected barangay
        updateMapData();
        // Automatically apply thresholds for selected barangay
        applyThresholdsNow();
      });
    }
  }

  function populateBarangays(municipalityId) {
    const brgySel = document.getElementById('barangay-select');
    if (!brgySel || !municipalityId) return Promise.resolve();
    brgySel.disabled = true;
    return fetch(`/api/all-barangays/?municipality_id=${encodeURIComponent(municipalityId)}`)
      .then(r => r.ok ? r.json() : Promise.reject(r))
      .then(d => {
        const items = d.barangays || [];
        brgySel.innerHTML = '<option value="" selected>All Barangays</option>';
        items.sort((a,b)=>a.name.localeCompare(b.name)).forEach(b => {
          const opt = document.createElement('option');
          opt.value = b.id;
          opt.textContent = b.name;
          brgySel.appendChild(opt);
        });
        brgySel.disabled = false;
      })
      .catch(() => { brgySel.disabled = false; });
  }

  // ---------------- Sensors (Weather Conditions) ----------------
  function updateSensorValues() {
    const weatherIDs = ['temperature-value', 'humidity-value', 'rainfall-value', 'water-level-value', 'wind-speed-value'];
    weatherIDs.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = '...';
    });

    let url = '/api/parameter-status/';
    const params = [];
    if (state.municipalityId) params.push(`municipality_id=${state.municipalityId}`);
    if (state.barangayId) params.push(`barangay_id=${state.barangayId}`);
    params.push(`_=${new Date().getTime()}`); // Cache buster
    if (params.length) url += '?' + params.join('&');

    fetch(url, { headers: { 'Accept': 'application/json' }})
      .then(r => r.ok ? r.json() : Promise.reject(r))
      .then(data => {
        const items = data.items || [];
        const latest = {};
        items.forEach(item => {
            latest[item.parameter] = item.latest;
        });

        setValue('temperature-value', latest.temperature, '°C');
        setValue('humidity-value', latest.humidity, '%');
        setValue('rainfall-value', latest.rainfall, 'mm');
        setValue('water-level-value', latest.water_level, 'm');
        setValue('wind-speed-value', latest.wind_speed, 'km/h');
        
        const ts = new Date();
        const lastUpdated = document.getElementById('map-last-updated');
        if (lastUpdated) lastUpdated.textContent = ts.toLocaleString();
        updateWeatherSeverityStyles();
      })
      .catch(() => {
        setValue('temperature-value', null, '°C');
        setValue('humidity-value', null, '%');
        setValue('rainfall-value', null, 'mm');
        setValue('water-level-value', null, 'm');
        setValue('wind-speed-value', null, 'km/h');
      });
  }

  // Build a compact HTML block listing each parameter's status, showing 'Normal' when not breached.
  function buildParameterStatusHTML(sev) {
    try {
      const map = {};
      (sev.items || []).forEach(it => { map[it.parameter] = it; });
      const order = [
        {key:'temperature', label:'Temperature'},
        {key:'humidity', label:'Humidity'},
        {key:'rainfall', label:'Rainfall'},
        {key:'water_level', label:'Water Level'},
        {key:'wind_speed', label:'Wind Speed'}
      ];
      const badge = lvl => {
        const n = Number(lvl)||0;
        if (n>=5) return 'CATASTROPHIC';
        if (n>=4) return 'EMERGENCY';
        if (n>=3) return 'WARNING';
        if (n>=2) return 'WATCH';
        if (n>=1) return 'ADVISORY';
        return 'Normal';
      };
      const row = it => {
        const unit = it.unit || '';
        const lvl = Number(it.level)||0;
        const thr = it.thresholds || {};
        const thrMap = {1: thr.advisory, 2: thr.watch, 3: thr.warning, 4: thr.emergency, 5: thr.catastrophic};
        const ref = thrMap[lvl] != null ? thrMap[lvl] : '';
        const latest = (it.latest != null) ? Number(it.latest).toFixed(unit === '%' ? 0 : 2).replace(/\.00$/,'') : '—';
        const refText = ref !== '' ? Number(ref).toFixed(2).replace(/\.00$/,'') : '';
        const statusText = badge(lvl);
        const color = (lvl>=4)?'#dc2626':(lvl>=3)?'#d97706':(lvl>=1)?'#0ea5e9':'#16a34a';
        const extra = (lvl>0 && refText!=='') ? ` (>= ${refText} ${unit})` : '';
        return `<div style="display:flex; justify-content:space-between; gap:10px; padding:4px 0;">
          <span>${paramLabel(it.parameter)}</span>
          <span style="white-space:nowrap; color:${color}; font-weight:600;">${statusText}</span>
          <span style="white-space:nowrap; color:var(--gray)">Latest: ${latest} ${unit}${extra}</span>
        </div>`;
      };
      const html = order.map(o => row(map[o.key] || { parameter:o.key, unit:unitFor(o.key), level:0, latest:null, thresholds:{} })).join('');
      return `<div style="margin-top:8px; border-top:1px dashed #e5e7eb; padding-top:6px;"><strong>Parameter status</strong>${html}</div>`;
    } catch (e) {
      return '';
    }
  }

  function unitFor(key){
    const u={temperature:'°C',humidity:'%',rainfall:'mm',water_level:'m',wind_speed:'km/h'}; return u[key]||'';
  }

  function setValue(id, value, suffix) {
    const el = document.getElementById(id);
    if (!el) return;
    if (value === null || value === undefined || isNaN(value)) {
      el.textContent = '--';
      return;
    }
    // Display the original data without rounding/conversion
    const text = Number(value);
    el.textContent = `${text}${suffix}`;
  }

  // Apply severity styles to Weather Conditions based on threshold endpoint
  async function updateWeatherSeverityStyles() {
    try {
      const sev = await fetchThresholdSeverity();
      if (!sev || !sev.items) return;
      const levelByParam = {};
      sev.items.forEach(it => { levelByParam[it.parameter] = it.level || 0; });
      
      // Decorate container cards with severity classes.
      // This will automatically handle the text color of the values via CSS.
      setSeverityClass('temperature-value', levelByParam.temperature || 0);
      setSeverityClass('humidity-value', levelByParam.humidity || 0);
      setSeverityClass('rainfall-value', levelByParam.rainfall || 0);
      setSeverityClass('water-level-value', levelByParam.water_level || 0);
      setSeverityClass('wind-speed-value', levelByParam.wind_speed || 0);

      // Apply color to the value text itself.
      applySeverityStyle('temperature-value', levelByParam.temperature || 0);
      applySeverityStyle('humidity-value', levelByParam.humidity || 0);
      applySeverityStyle('rainfall-value', levelByParam.rainfall || 0);
      applySeverityStyle('water-level-value', levelByParam.water_level || 0);
      applySeverityStyle('wind-speed-value', levelByParam.wind_speed || 0);

      // Update status chips and extra text using returned items
      sev.items.forEach(it => {
        const key = it.parameter; // rainfall | water_level | temperature | humidity
        const idBase = paramIdBase(key); // e.g., 'water-level'
        const latest = it.latest;
        const unit = it.unit || '';
        setStatusChip(`${idBase}-status`, it.level || 0);
        setExtraText(`${idBase}-extra`, latest, unit);
      });
    } catch (e) { /* ignore */ }
  }

  function applySeverityStyle(id, level) {
    const el = document.getElementById(id);
    if (!el) return;
    const lvl = Math.max(0, Math.min(5, Number(level) || 0));
    let color = '#16a34a'; // sev-0
    if (lvl >= 4) color = '#dc2626'; // sev-4, sev-5
    else if (lvl === 3) color = '#d97706'; // sev-3
    else if (lvl >= 1) color = '#0ea5e9'; // sev-1, sev-2
    el.style.color = color;
  }

  // Add/remove .sev-* classes on the container .weather-item for modern styles
  function setSeverityClass(valueElementId, level) {
    const el = document.getElementById(valueElementId);
    if (!el) return;
    // Find the nearest .weather-item container
    let container = el.closest ? el.closest('.weather-item') : el.parentElement;
    if (!container) return;
    // Remove existing sev-* classes
    for (let i = 0; i <= 5; i++) {
      container.classList.remove(`sev-${i}`);
    }
    // Clamp level between 0 and 5 and apply
    const lvl = Math.max(0, Math.min(5, Number(level) || 0));
    container.classList.add(`sev-${lvl}`);
  }

  function paramIdBase(key) {
    if (!key) return '';
    // Convert API parameter keys like 'water_level' and 'wind_speed' to DOM id base 'water-level', 'wind-speed'
    return String(key).replace(/_/g, '-');
  }

  function setStatusChip(chipId, level) {
    const el = document.getElementById(chipId);
    if (!el) return;
    // Reset classes
    el.classList.remove('normal','info','warning','danger');
    // Choose label and class
    const lvl = Number(level) || 0;
    let cls = 'normal';
    if (lvl >= 4) cls = 'danger';
    else if (lvl >= 3) cls = 'warning';
    else if (lvl >= 1) cls = 'info';
    el.classList.add(cls);
    el.textContent = (lvl === 0) ? 'Normal' : severityName(lvl);
  }

  function setExtraText(extraId, latest, unit) {
    const el = document.getElementById(extraId);
    if (!el) return;
    if (latest === null || latest === undefined || isNaN(latest)) {
      el.textContent = '';
      return;
    }
    const val = Number(latest);
    const fixed = (unit === 'mm' || unit === 'm') ? val.toFixed(1) : val.toFixed(0);
    el.textContent = `Latest: ${fixed} ${unit}`.trim();
  }

  function playAlertSound() {
    if (!audioContext) {
        console.warn("AudioContext not initialized. Click on the page to enable sound.");
        return;
    }

    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    oscillator.type = 'sine';
    oscillator.frequency.setValueAtTime(880, audioContext.currentTime);
    gainNode.gain.setValueAtTime(0.5, audioContext.currentTime);

    oscillator.start();
    oscillator.stop(audioContext.currentTime + 0.5);
  }

  // New: Setup map parameter selector dropdown
  function setupMapParamSelector() {
    const selector = document.getElementById('map-param-select');
    if (!selector) return;

    // Populate options
    const options = [
      { value: 'overall', label: 'Overall Risk' },
      { value: 'rainfall', label: 'Rainfall' },
      { value: 'water_level', label: 'Water Level' },
      { value: 'temperature', label: 'Temperature' },
      { value: 'humidity', label: 'Humidity' },
      { value: 'wind_speed', label: 'Wind Speed' }
    ];

    selector.innerHTML = options.map(opt =>
      `<option value="${opt.value}" ${state.mapDisplayParam === opt.value ? 'selected' : ''}>${opt.label}</option>`
    ).join('');

    // Handle changes
    selector.addEventListener('change', (e) => {
      state.mapDisplayParam = e.target.value;
      // Refresh map data to update colors
      updateMapData();
      // Also update affected areas to reflect the new parameter selection
      fetchAllBarangaysInMunicipality().then(barangays => {
        updateAffectedAreas(barangays, 0);
      }).catch(() => {
        updateAffectedAreas([], 0);
      });
    });
  }
  function openAlertModal(alert) {
    if (!alert) return;
    const modal = document.getElementById('alert-modal');
    const header = modal.querySelector('.modal-header');
    if (!modal || !header) return;

    modal.querySelector('#modal-title').textContent = alert.title;
    modal.querySelector('#modal-description').textContent = alert.description;
    modal.querySelector('#modal-severity').textContent = severityName(alert.severity_level);

    header.classList.remove('sev-3', 'sev-4', 'sev-5');
    if (alert.severity_level >= 3) header.classList.add(`sev-${alert.severity_level}`);

    modal.style.display = 'flex';
  }

  function closeAlertModal() {
    document.getElementById('alert-modal').style.display = 'none';
  }

  // ---------------- Alerts ----------------
  function updateAlerts() {
    const token = ++state._alertsReqToken; // capture a new token for this invocation
    let url = '/api/flood-alerts/?active=true';
    if (state.municipalityId) url += `&municipality_id=${state.municipalityId}`;
    if (state.barangayId) url += `&barangay_id=${state.barangayId}`;

    fetch(url, { headers: { 'Accept': 'application/json' }})
      .then(r => r.ok ? r.json() : Promise.reject(r))
      .then(async data => {
        // Ensure this response belongs to the latest request
        if (token !== state._alertsReqToken) return; // stale response; ignore
        let results = (data.results || []);
        // If a barangay is selected, only consider alerts that affect that barangay
        if (state.barangayId) {
          const selId = String(state.barangayId);
          results = results.filter(a => Array.isArray(a.affected_barangays) && a.affected_barangays.map(String).includes(selId));
        }
        results = results.sort((a,b) => b.severity_level - a.severity_level);
        let highest = results[0] || null;
        const badge = document.getElementById('alert-status-badge');
        const title = document.getElementById('alert-title');
        const msg = document.getElementById('alert-message');
        if (!badge || !title || !msg) return;

        // Always compute threshold-based severity for context
        const sev = await fetchThresholdSeverity();
        // If selection changed while awaiting thresholds, drop this update
        if (token !== state._alertsReqToken) return;
        const thresholdLevel = (sev && typeof sev.level === 'number') ? sev.level : 0;
        const combinedLevel = Math.max(highest ? (highest.severity_level || 0) : 0, thresholdLevel);

        if (combinedLevel >= 3 && state.lastCombinedLevel < 3) {
            playAlertSound();
            // If the alert is from a FloodAlert object, use that.
            if (highest && highest.severity_level >= 3) {
                openAlertModal(highest);
            } else {
                // Otherwise, create a synthetic alert object for the modal.
                const topParam = sev.items.sort((a,b) => b.level - a.level)[0];
                const syntheticAlert = {
                    title: `${severityName(topParam.level)}: ${paramLabel(topParam.parameter)} Threshold Exceeded`,
                    description: `The latest reading for ${paramLabel(topParam.parameter)} has exceeded the configured threshold for your selected location.`,
                    severity_level: topParam.level
                };
                openAlertModal(syntheticAlert);
            }
        }
        state.lastCombinedLevel = combinedLevel;

        // Badge color and text from combinedLevel
        const levels = {1:'ADVISORY',2:'WATCH',3:'WARNING',4:'EMERGENCY',5:'CATASTROPHIC'};
        const levelText = combinedLevel > 0 ? (levels[combinedLevel] || severityName(combinedLevel)) : 'Normal';
        badge.textContent = levelText;
        badge.classList.remove('status-normal','status-warning','status-danger');
        if (combinedLevel >= 4) badge.classList.add('status-danger');
        else if (combinedLevel >= 2) badge.classList.add('status-warning');
        else badge.classList.add('status-normal');

        // Resolve display location text from selectors
        const muniSel = document.getElementById('location-select');
        const brgySel = document.getElementById('barangay-select');
        const muniName = (muniSel && muniSel.selectedIndex > -1) ? muniSel.options[muniSel.selectedIndex].text : '';
        const brgyName = (brgySel && brgySel.selectedIndex > -1) ? brgySel.options[brgySel.selectedIndex].text : '';
        const locText = state.barangayId ? brgyName : (state.municipalityId ? muniName : 'All Locations');

        // Title and concise message with compact threshold details (full list remains below)
        if (highest && (highest.severity_level || 0) >= thresholdLevel) {
          // Prioritize active alert severity, but display current selection name
          // If the original title already contains the location name, use it. Otherwise, construct a new one.
          const originalTitle = highest.title || '';
          if (originalTitle.toLowerCase().includes(locText.toLowerCase())) {
            title.textContent = originalTitle;
          } else {
            title.textContent = `${levels[highest.severity_level] || 'ALERT'}: Automated Alert for ${locText}`;
          }
          const lines = [];
          // Use a location-aware description instead of the stored alert description,
          // which may reference a different barangay.
          const desc = (state.barangayId || state.municipalityId)
            ? `The system is monitoring environmental conditions in ${escapeHtml(locText)}.`
            : 'The system is monitoring environmental conditions continuously.';
          lines.push(desc);
          if (sev && Array.isArray(sev.items)) {
            const top = sev.items.filter(it => (it.level||0)>0).sort((a,b)=> (b.level||0)-(a.level||0)).slice(0,3);
            if (top.length) {
              lines.push(buildThresholdDetails(top));
            }
          }
          msg.innerHTML = lines.join('<br>');
        } else if (thresholdLevel > 0 && sev) {
          // Threshold-driven status only
          const text = severityName(thresholdLevel);
          const top = (sev.items || []).sort((a,b)=>b.level-a.level)[0];
          const topLabel = top ? paramLabel(top.parameter) : 'Threshold';
          title.textContent = `${text}: ${topLabel} for ${locText}`;
          const lines = ['Conditions exceed configured thresholds.'];
          const topItems = (sev.items || []).filter(it => (it.level||0)>0).sort((a,b)=> (b.level||0)-(a.level||0)).slice(0,3);
          if (topItems.length) lines.push(buildThresholdDetails(topItems));
          msg.innerHTML = lines.join('<br>');
        } else {
          // No alerts and no threshold breach -> Normal
          title.textContent = 'No Active Alerts';
          msg.textContent = 'The system is monitoring environmental conditions continuously.';
        }

        // Render parameter status list using compact endpoint
        try {
          const p = await fetchParameterStatus();
          if (token !== state._alertsReqToken) return;
          renderParamStatusList(p && p.items ? p.items : []);
          // If we couldn't add compact details earlier (sev may have failed), try to append from parameter-status
          try {
            const msgEl = document.getElementById('alert-message');
            if (msgEl && p && Array.isArray(p.items) && msgEl.innerHTML.indexOf('<ul>') === -1) {
              const mapped = p.items
                .filter(x => (x.level || 0) > 0)
                .map(x => ({
                  parameter: x.parameter || x.name || '',
                  unit: x.unit || '',
                  latest: (x.latest != null) ? x.latest : (x.value != null ? x.value : (x.latest_value != null ? x.latest_value : null)),
                  level: x.level || 0,
                  thresholds: x.thresholds || {}
                }));
              if (mapped.length) {
                msgEl.innerHTML += '<br>' + buildThresholdDetails(mapped.slice(0,3));
              }
            }
          } catch (e) { /* ignore */ }
        } catch (e) {
          // ignore
        }
      })
      .catch(() => {
        // Leave as-is on error
      });

   // Always fetch and display all barangays in the selected municipality
   fetchAllBarangaysInMunicipality().then(barangays => {
     updateAffectedAreas(barangays, 0);
   }).catch(() => {
     updateAffectedAreas([], 0);
   });
  }

  // Build a compact HTML list of top breached thresholds
  function buildThresholdDetails(items) {
    const sevLabel = lvl => (lvl>=5?'CATASTROPHIC':lvl>=4?'EMERGENCY':lvl>=3?'WARNING':lvl>=2?'WATCH':lvl>=1?'ADVISORY':'NORMAL');
    const rows = items.map(it => {
      const lbl = paramLabel(it.parameter);
      const unit = it.unit || '';
      const thr = it.thresholds || {};
      const thrMap = {1: thr.advisory, 2: thr.watch, 3: thr.warning, 4: thr.emergency, 5: thr.catastrophic};
      const ref = thrMap[it.level] != null ? thrMap[it.level] : '';
      const latest = (it.latest != null) ? Number(it.latest).toFixed(unit === '%' ? 0 : 2).replace(/\.00$/,'') : '—';
      const refText = ref !== '' ? Number(ref).toFixed(unit === '%' ? 0 : 2).replace(/\.00$/, '') : '—';
      return `<li><strong>${lbl}:</strong> ${latest} ${unit} > <em>${sevLabel(it.level)}</em> (${refText} ${unit})`;
    });
    return `<ul style="margin:6px 0 0 18px; padding:0; color:#334155; font-size:13px; line-height:1.35;">${rows.join('')}</ul>`;
  }

  async function fetchParameterStatus() {
    try {
      const params = [];
      // Request a compact per-parameter status for the current selection
      if (state.municipalityId) params.push(`municipality_id=${state.municipalityId}`);
      if (state.barangayId) params.push(`barangay_id=${state.barangayId}`);
      const url = `/api/parameter-status/${params.length ? ('?' + params.join('&')) : ''}`;
      const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
      if (!res.ok) return null;
      return await res.json();
    } catch (e) {
      return null;
    }
  }

  async function fetchThresholdSeverity() {
    try {
      const params = [];
      params.push('parameter=rainfall,water_level,temperature,humidity,wind_speed');
      if (state.municipalityId) params.push(`municipality_id=${state.municipalityId}`);
      if (state.barangayId) params.push(`barangay_id=${state.barangayId}`);
      const url = `/api/threshold-visualization/?${params.join('&')}`;
      const res = await fetch(url, { headers: { 'Accept': 'application/json' }});
      if (!res.ok) return null;
      const data = await res.json();
      const items = (data.data || []).map(it => ({
        parameter: it.parameter,
        unit: it.unit,
        latest: it.latest ? it.latest.value : null,
        level: it.severity ? (it.severity.level || 0) : 0,
        thresholds: it.thresholds || {}
      }));
      const maxLevel = items.reduce((m, it) => Math.max(m, it.level || 0), 0);
      return { level: maxLevel, items };
    } catch (e) { return null; }
  }


  // New helper function to get color based on severity
  function getThresholdColor(level) {
    const l = Number(level) || 0;
    if (l >= 4) return '#ef4444'; // High Risk (Danger)
    if (l >= 2) return '#f59e0b'; // Medium Risk (Warning)
    return '#10b981'; // Low Risk (Success)
  }

  // New helper to get severity text
  function getSeverityText(level) {
    const l = Number(level) || 0;
    if (l >= 4) return 'High Risk';
    if (l >= 2) return 'Medium Risk';
    return 'Low Risk';
  }

  // ---------------- Map ----------------
  function initMap() {
    const container = document.getElementById('flood-map');
    if (!container || !window.L) return;
    if (!L.heatLayer) {
        console.error("Leaflet.heat plugin not found. Heatmap will not be available.");
        return;
    }

    // --- Performance Patch for Leaflet.heat ---
    L.HeatLayer.prototype._initCanvas = function () {
        const canvas = L.DomUtil.create('canvas', 'leaflet-heatmap-layer leaflet-layer');
        const origin = L.DomUtil.getStyle(this._map.getPanes().overlayPane, 'transform');
        canvas.style.transform = origin;
        canvas.width = this._width;
        canvas.height = this._height;
        this._canvas = canvas;
        this._ctx = canvas.getContext('2d', { willReadFrequently: true });
    };
    // --- End of Patch ---

    state.map = L.map('flood-map').setView([17.135678, 120.437203], 12);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(state.map);

    state.mapLayers.zones = L.layerGroup().addTo(state.map);
    state.mapLayers.sensors = L.layerGroup().addTo(state.map);
    state.mapLayers.barangays = L.layerGroup().addTo(state.map);
    state.mapLayers.heat = null; // heat layer holder

    addFullscreenControl(state.map, 'flood-map');

    // ESC to exit fullscreen
    window.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape') {
        const el = document.getElementById('flood-map');
        if (el && el.dataset.fullscreen === '1') {
          toggleMapFullscreen('flood-map', state.map);
        }
      }
    });

    // Ensure scales are replaced with plain objects up-front
    try { hardenScaleOptions(state.chart); } catch (_) {}

    // Debounced/observed resize handling to fix tile alignment in grids
    const invalidate = () => { try { state.map && state.map.invalidateSize(true); } catch (e) {} };
    // Initial invalidate passes after render to avoid fractional height gaps
    setTimeout(invalidate, 50);
    setTimeout(invalidate, 250);
    setTimeout(invalidate, 600);
    setTimeout(invalidate, 1000);
    // Window resize
    let resizeTimer = null;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(invalidate, 120);
    });
    // Container resize
    if (window.ResizeObserver) {
      try {
        const ro = new ResizeObserver(() => {
          clearTimeout(resizeTimer);
          resizeTimer = setTimeout(invalidate, 60);
        });
        ro.observe(container);
        state._mapResizeObserver = ro;
      } catch (e) { /* ignore */ }
    }
  }

  function updateMapData() {
    if (!state.map) return;
    let url = '/api/map-data/';
    const params = [];
    if (state.municipalityId) params.push(`municipality_id=${state.municipalityId}`);
    if (state.barangayId) params.push(`barangay_id=${state.barangayId}`);
    if (params.length) url += '?' + params.join('&');

    const lastUpdated = document.getElementById('map-last-updated');
    if (lastUpdated) lastUpdated.textContent = 'Loading data...';

    fetch(url, { headers: { 'Accept': 'application/json' }})
      .then(r => r.ok ? r.json() : Promise.reject(r))
      .then(async data => {
        clearMapLayers();
        drawZones(data.zones || []);
        drawSensors(data.sensors || []);
        drawBarangays(data.barangays || []);
        renderLocationsList(data.barangays || []);
        if (lastUpdated) lastUpdated.textContent = new Date().toLocaleString();
        // Ensure map tiles realign after layer updates
        try {
          state.map && state.map.invalidateSize(true);
          // a couple of delayed passes to remove any hairline gaps
          setTimeout(() => { try { state.map && state.map.invalidateSize(true); } catch (e) {} }, 120);
          setTimeout(() => { try { state.map && state.map.invalidateSize(true); } catch (e) {} }, 300);
        } catch (e) {}
      })
      .catch(() => {
        if (lastUpdated) lastUpdated.textContent = 'Unable to load map data';
      });
  }

  function clearMapLayers() {
    Object.values(state.mapLayers).forEach(layer => layer && layer.clearLayers());
    state.barangayLayerById.clear();
  }

  function drawZones(zones) {
    zones.forEach(z => {
      try {
        const gj = typeof z.geojson === 'string' ? JSON.parse(z.geojson) : z.geojson;
        if (gj) {
          L.geoJSON(gj, {
            style: feature => ({
              color: zoneColor(z.risk_level || z.severity || 'low'),
              weight: 2,
              fillOpacity: 0.2,
            })
          }).addTo(state.mapLayers.zones);
        }
      } catch (e) { /* ignore malformed */ }
    });
  }

  function drawSensors(sensors) {
    sensors.forEach(s => {
      if (!s.lat || !s.lng) return;
      const marker = L.circleMarker([s.lat, s.lng], {
        radius: 6,
        color: '#0d6efd',
        fillColor: '#0d6efd',
        fillOpacity: 0.8
      });
      const valueText = (s.latest_reading && s.latest_reading.value != null) ? s.latest_reading.value : '—';
      marker.bindPopup(
        `<strong>${escapeHtml(s.name || 'Sensor')}</strong><br>` +
        `Type: ${escapeHtml((s.type || '').toString())}<br>` +
        `Value: ${escapeHtml(valueText.toString())}`
      );
      marker.addTo(state.mapLayers.sensors);
    });
  }

  function drawBarangays(items) {
    const coords = [];
    if (state.barangayLayerById) state.barangayLayerById.clear();
    items.forEach(b => {
      if (!b.lat || !b.lng) return;
      coords.push([b.lat, b.lng]);
      
      // Use the severity level from the backend API, which is calculated based on
      // threshold analysis of sensor readings (with active alerts taking priority).
      let severityLevel = 0; // Default to 0 (Normal)
      if (state.mapDisplayParam === 'overall') {
        if (b.param_severities) {
          // For overall risk, use the maximum severity across all parameters
          severityLevel = Math.max(...Object.values(b.param_severities));
        } else {
          severityLevel = b.severity || 0;
        }
      } else if (b.param_severities && b.param_severities[state.mapDisplayParam] !== undefined) {
        // Use the specific parameter's severity if it exists
        severityLevel = b.param_severities[state.mapDisplayParam];
      }
      const color = getThresholdColor(severityLevel);

      const radius = 200 + Math.min(800, Math.sqrt(b.population || 1));
      const circle = L.circle([b.lat, b.lng], {
        radius,
        color,
        weight: severityLevel >= 3 ? 2 : 1.5, // Thicker border for higher risk
        fillColor: color,
        fillOpacity: 0.4, // Increased opacity for better visibility
        dashArray: severityLevel >= 3 ? null : '4, 4' // Dashed line for lower risk
      });
      const severityText = getSeverityText(severityLevel);
      circle.bindPopup(
        `<strong>${escapeHtml(b.name || 'Barangay')}</strong><br>` +
        `Population: ${Number(b.population||0).toLocaleString()}<br>`+
        `Risk Level: <span style="color:${color}; font-weight:bold;">${severityText}</span>`
      );
      circle.addTo(state.mapLayers.barangays);

      // Keep reference by id for focusing and interaction
      if (b.id != null) {
        state.barangayLayerById.set(String(b.id), circle);
      }

      // Clicking a barangay circle updates the dropdown and filters dashboard
      circle.on('click', () => {
        const brgySel = document.getElementById('barangay-select');
        if (b.id != null) {
          state.barangayId = String(b.id);
          if (brgySel) brgySel.value = String(b.id);
          sessionStorage.setItem('dashboard_barangay_id', state.barangayId || '');
          refreshAll();
          // Automatically apply thresholds when selecting via map
          applyThresholdsNow();
        }
      });
    });

    // If a specific barangay is selected, zoom tighter and open popup
    if (state.barangayId && state.barangayLayerById.has(String(state.barangayId))) {
      const layer = state.barangayLayerById.get(String(state.barangayId));
      const ll = layer.getLatLng();
      state.map.setView(ll, 17);
      layer.openPopup();
      // Brief highlight pulse
      const pulse = L.circleMarker(ll, { radius: 18, color: '#0d6efd', fillColor: '#0d6efd', fillOpacity: 0.3, weight: 2 });
      pulse.addTo(state.mapLayers.barangays);
      setTimeout(() => { state.mapLayers.barangays.removeLayer(pulse); }, 1500);
    } else if (state.barangayId && coords.length === 1) {
      state.map.setView(coords[0], 17);
    } else if (coords.length > 0) {
      state.map.fitBounds(coords, { padding: [20, 20] });
    }
  }

  function zoneColor(level) {
    const l = (typeof level === 'string') ? level.toLowerCase() : level;
    if (l === 'high' || l === 3) return '#ef4444';
    if (l === 'medium' || l === 2) return '#f59e0b';
    return '#10b981';
  }

  // ---------------- Trends chart ----------------
  function initChart() {
    const canvas = document.getElementById('trends-chart');
    if (!canvas || !window.Chart) return;
    ensureChartGlobalPointDefaults();
    // If an old chart exists (e.g., from hot reload), destroy it first
    try { if (state.chart && state.chart.destroy) state.chart.destroy(); } catch (_) {}
    state.chart = new Chart(canvas, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                themedDataset('Temperature (°C)', 'rgba(239,68,68,1)', 'y'),
                themedDataset('Humidity (%)', 'rgba(14,165,233,1)', 'y'),
                themedDataset('Rainfall (mm)', 'rgba(59,130,246,1)', 'y1'),
                themedDataset('Water Level (m)', 'rgba(16,185,129,1)', 'y1'),
                themedDataset('Wind Speed (km/h)', 'rgba(168,85,247,1)', 'y1'),
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            spanGaps: true, // Draw lines over null data points
            interaction: { mode: 'index', intersect: false },
            elements: {
                line: { tension: 0.2, borderWidth: 2 },
                point: { radius: 4, hoverRadius: 6, hitRadius: 6, borderWidth: 2, backgroundColor: '#ffffff' }
            },
            plugins: {
                legend: { position: 'top' },
                tooltip: {
                    enabled: true, // Enable tooltips
                    callbacks: {
                        title: function (tooltipItems) {
                            if (!tooltipItems.length) return '';
                            const item = tooltipItems[0];
                            // The raw label is the ISO timestamp, format it for display
                            const isoString = state.chart.data.isoLabels[item.dataIndex];
                            return formatManilaFull(isoString);
                        }
                    }
                }
            },
            scales: {
                x: { type: 'category', grid: { display: false } },
                y: {
                    type: 'linear', position: 'left',
                    title: { display: true, text: 'Temperature (°C) / Humidity (%)' }
                },
                y1: {
                    type: 'linear', position: 'right',
                    title: { display: true, text: 'Rainfall (mm) / Water Level (m) / Wind (km/h)' },
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });

    // Helper to create themed dataset with outlined points
    function themedDataset(label, color, axis) {
        return {
            label,
            data: [],
            borderColor: color,
            backgroundColor: 'transparent',
            cubicInterpolationMode: 'monotone',
            yAxisID: axis,
            pointRadius: 4,
            pointHoverRadius: 6,
            pointHitRadius: 6,
            pointBorderColor: color,
            pointBackgroundColor: '#ffffff',
            pointBorderWidth: 2,
            borderWidth: 2,
            fill: false
        };
    }
  }

  function loadTrendsChart() {
    if (!state.chart || state._trendsFetchInFlight) return;
    state._trendsFetchInFlight = true;
    setChartOverlay('Loading trends data...');

    const queries = [
        fetchChart('temperature'),
        fetchChart('humidity'),
        fetchChart('rainfall'),
        fetchChart('water_level'),
        fetchChart('wind_speed'),
    ];

    Promise.allSettled(queries).then(results => {
      const temp = (results[0].status === 'fulfilled') ? results[0].value : { labels: [], values: [] };
      const hum  = (results[1].status === 'fulfilled') ? results[1].value : { labels: [], values: [] };
      const rain = (results[2].status === 'fulfilled') ? results[2].value : { labels: [], values: [] };
      const water= (results[3].status === 'fulfilled') ? results[3].value : { labels: [], values: [] };
      const wind = (results[4].status === 'fulfilled') ? results[4].value : { labels: [], values: [] };

      const merged = mergeSeries([
        { labels: temp.labels,  values: temp.values,  key: 't' },
        { labels: hum.labels,   values: hum.values,   key: 'h' },
        { labels: rain.labels,  values: rain.values,  key: 'r' },
        { labels: water.labels, values: water.values, key: 'wl' },
        { labels: wind.labels,  values: wind.values,  key: 'ws' },
      ]);

      const anyData = (() => {
        const s = merged.series || {};
        const keys = ['t','h','r','wl','ws'];
        for (const k of keys) {
          const arr = s[k] || [];
          if (arr.some(v => typeof v === 'number' && !isNaN(v))) return true;
        }
        return false;
      })();

      if (!merged.labels.length || !anyData) {
        showTrendsNoData();
      }

      renderTrendsFromMerged(merged);

    }).finally(() => {
      state._trendsFetchInFlight = false;
      const el = document.getElementById('trends-updated-at');
      if (el) {
          el.textContent = `Last updated: ${formatManilaFull(new Date().toISOString())}`;
      }
    });
  }

  function showTrendsNoData() {
    try {
      if (!state.chart) return;
      state.chart.data.labels = [];
      state.chart.data.isoLabels = [];
      state.chart.data.datasets.forEach(ds => { ds.data = []; });
      state.chart.update();
      setChartOverlay('No data available for the selected location and time range.');
      const el = document.getElementById('trends-updated-at');
      if (el) el.textContent = 'Last updated: —';
    } catch (e) { /* ignore */ }
  }

  function renderTrendsFromMerged(merged) {
    if (!state.chart) return;

    // Store raw series for normalization toggle
    state._rawSeries = merged.series;
    // Store original ISO labels for tooltips
    state.chart.data.isoLabels = merged.labels;

    // Create display labels (formatted time)
    state.chart.data.labels = merged.labels.map(l => formatManilaShort(l));

    // Apply scaling (either raw values or normalized 0-100)
    applyChartScaling(merged.series);

    // Update the chart
    state.chart.update();

    // Update timestamp
    const el = document.getElementById('trends-updated-at');
    if (el) {
        const lastIso = merged.labels.length > 0 ? merged.labels[merged.labels.length - 1] : new Date().toISOString();
        el.textContent = `Last updated: ${formatManilaFull(lastIso)}`;
    }

    // Clear overlay if data is present
    const hasData = (() => {
      const s = merged.series || {};
      for (const key of ['t', 'h', 'r', 'wl', 'ws']) {
        const arr = s[key] || [];
        if (arr.some(v => v != null && typeof v === 'number' && !isNaN(v))) return true;
      }
      return false;
    })();

    if (hasData) {
      clearChartOverlay();
    } else {
      showTrendsNoData();
    }
  }

  // ------- Chart overlay helpers (loading / no data) -------
  function ensureChartOverlayHost() {
    const container = document.querySelector('.chart-container-modern');
    if (!container) return null;
    let host = container.querySelector('.chart-overlay-host');
    if (!host) {
      host = document.createElement('div');
      host.className = 'chart-overlay-host';
      Object.assign(host.style, {
        position: 'absolute', inset: '0', display: 'flex', alignItems: 'center', justifyContent: 'center',
        pointerEvents: 'none', color: 'var(--gray)', fontSize: '14px',
      });
      // Ensure parent is positioned
      const cs = window.getComputedStyle(container);
      if (cs.position === 'static') container.style.position = 'relative';
      container.appendChild(host);
    }
    return host;
  }

  function setChartOverlay(text) {
    const host = ensureChartOverlayHost();
    if (!host) return;
    host.textContent = text || '';
    host.style.display = 'flex';
  }

  function clearChartOverlay() {
    const container = document.querySelector('.chart-container-modern');
    const host = container ? container.querySelector('.chart-overlay-host') : null;
    if (host) host.remove();
  }

  // Filter merged series by a range key ('latest' | '1w' | '1m' | '1y').
  // Keeps label alignment and nulls across all series.
  function filterMergedByRange(merged, rangeKey) {
    try {
      const out = { labels: [], series: { t:[], h:[], r:[], wl:[], ws:[] } };
      const labels = Array.isArray(merged.labels) ? merged.labels : [];
      const series = merged.series || {};
      const mapDays = { '1w':7, '1m':30, '1y':365 };
      if (!labels.length || rangeKey === 'latest' || !mapDays[rangeKey]) {
        return { labels: labels.slice(), series: {
          t: (series.t||[]).slice(),
          h: (series.h||[]).slice(),
          r: (series.r||[]).slice(),
          wl:(series.wl||[]).slice(),
          ws:(series.ws||[]).slice(),
        }};
      }
      const lastIso = labels[labels.length - 1];
      const lastDate = new Date(lastIso);
      if (isNaN(lastDate.getTime())) {
        return { labels: labels.slice(), series: {
          t: (series.t||[]).slice(), h:(series.h||[]).slice(), r:(series.r||[]).slice(), wl:(series.wl||[]).slice(), ws:(series.ws||[]).slice()
        }};
      }
      const days = mapDays[rangeKey];
      const cutoffMs = lastDate.getTime() - days * 24 * 60 * 60 * 1000;
      for (let i = 0; i < labels.length; i++) {
        const d = new Date(labels[i]);
        if (!isNaN(d.getTime()) && d.getTime() >= cutoffMs) {
          out.labels.push(labels[i]);
          out.series.t.push((series.t||[])[i] ?? null);
          out.series.h.push((series.h||[])[i] ?? null);
          out.series.r.push((series.r||[])[i] ?? null);
          out.series.wl.push((series.wl||[])[i] ?? null);
          out.series.ws.push((series.ws||[])[i] ?? null);
        }
      }
      return out;
    } catch (e) {
      return merged;
    }
  }

  // Apply scaling based on state.normalize. When true, normalize each series to 0–100 keeping nulls intact.
  function applyChartScaling(series) {
    try {
      if (!state.chart || !state.chart.options || !state.chart.options.scales) return;
      const y = state.chart.options.scales.y;
      const y1 = state.chart.options.scales.y1;
      if (state.normalize) {
      const tN  = normalizeArray(series.t);
      const hN  = normalizeArray(series.h);
      const rN  = normalizeArray(series.r);
      const wlN = normalizeArray(series.wl);
      const wsN = normalizeArray(series.ws);
      state.chart.data.datasets[0].data = tN;
      state.chart.data.datasets[1].data = hN;
      state.chart.data.datasets[2].data = rN;
      state.chart.data.datasets[3].data = wlN;
      state.chart.data.datasets[4].data = wsN;
      // Single axis 0–100 for all
      if (y) { y.suggestedMin = 0; y.suggestedMax = 100; }
      // Do not toggle axis display dynamically to avoid scriptable recursion
      } else {
      // Restore raw data
      state.chart.data.datasets[0].data = series.t;
      state.chart.data.datasets[1].data = series.h;
      state.chart.data.datasets[2].data = series.r;
      state.chart.data.datasets[3].data = series.wl;
      state.chart.data.datasets[4].data = series.ws;
      // Compute dynamic axis bounds as before
      const tVals  = (series.t  || []).filter(v => v != null);
      const hVals  = (series.h  || []).filter(v => v != null);
      const rVals  = (series.r  || []).filter(v => v != null);
      const wlVals = (series.wl || []).filter(v => v != null);
      const wsVals = (series.ws || []).filter(v => v != null);
      const leftAll = [...tVals, ...hVals];
      if (leftAll.length) {
        const min = Math.min(...leftAll);
        const max = Math.max(...leftAll);
        if (y) { y.suggestedMin = Math.floor(Math.min(0, min - 1)); y.suggestedMax = Math.ceil(max + 1); }
      } else {
        if (y) { y.suggestedMin = 0; y.suggestedMax = 100; }
      }
      if (y1) {
        const rightAll = [
          rVals.length ? Math.max(...rVals) : 0,
          wlVals.length ? Math.max(...wlVals) : 0,
          wsVals.length ? Math.max(...wsVals) : 0,
        ];
        const rightMax = Math.max(...rightAll);
        y1.suggestedMin = 0;
        y1.suggestedMax = Math.ceil((rightMax || 1) * 1.2);
      }
    }
    } catch (e) {
      // On failure, just log it. Avoid recreating the chart which is unstable.
      console.error("Error applying chart scaling:", e);
    }
  }

  function normalizeArray(arr) {
    if (!Array.isArray(arr) || arr.length === 0) return [];
    const nums = arr.filter(v => typeof v === 'number');
    if (!nums.length) return arr.map(() => null);
    const min = Math.min(...nums);
    const max = Math.max(...nums);
    const range = max - min;
    if (range === 0) {
      // All equal; map non-null values to 50 so the line is visible and aligned
      return arr.map(v => (v == null ? null : 50));
    }
    return arr.map(v => (v == null ? null : ((v - min) / range) * 100));
  }

  function fetchChart(type, opts = {}) {
    // Map UI range to backend params
    const range = state.trendsRange || 'latest';
    const rangeToDays = (r) => r === '1w' ? 7 : r === '1m' ? 30 : r === '1y' ? 365 : null;
    const days = rangeToDays(range);
    // Build URL: use limit for 'latest', use days otherwise
    let url = `/api/chart-data/?type=${encodeURIComponent(type)}`;
    if (days) {
      url += `&days=${days}`;
    } else {
      url += `&limit=10`;
    }
    if (state.municipalityId) url += `&municipality_id=${state.municipalityId}`;
    if (state.barangayId) url += `&barangay_id=${state.barangayId}`;
    try { console.debug('[Trends] GET', url); } catch(e) {}

    const doFetch = (u) => fetch(u, { headers: { 'Accept': 'application/json' }})
      .then(r => {
        if (!r.ok) {
          const err = new Error(`chart-data HTTP ${r.status}`);
          err.status = r.status; err.url = u;
          throw err;
        }
        return r.json();
      })
      .then(d => {
        // Some endpoints provide labels_manila only. Fallback to that when labels are empty.
        let labels = Array.isArray(d.labels) ? d.labels.slice() : [];
        const labelsManila = Array.isArray(d.labels_manila) ? d.labels_manila.slice() : [];
        if (!labels.length && labelsManila.length) labels = labelsManila.slice();
        let values = Array.isArray(d.values) ? d.values.slice() : [];
        // Coerce to numbers; keep nulls for non-numeric entries
        values = values.map(v => {
          const n = typeof v === 'number' ? v : parseFloat(v);
          return Number.isFinite(n) ? n : null;
        });
        // Ensure labels and values have the same length
        const n = Math.min(labels.length, values.length);
        return { labels: labels.slice(0, n), labelsManila: labelsManila.slice(0, Math.min(labelsManila.length, n)), values: values.slice(0, n) };
      });

    return doFetch(url)
      .catch(err => {
        try { console.warn('[Trends] chart-data fetch failed:', err && err.message ? err.message : err); } catch(e) {}
        return { labels: [], labelsManila: [], values: [] };
      });
  }

  function mergeSeries(arr) {
    const labelSet = new Set();
    arr.forEach(s => (s.labels || []).forEach(l => labelSet.add(l)));
    const labels = Array.from(labelSet).sort((a,b) => new Date(a) - new Date(b));
    const series = {};
    arr.forEach(s => {
      const map = new Map();
      (s.labels || []).forEach((l, i) => map.set(l, s.values[i]));
      series[s.key] = labels.map(l => (map.has(l) ? map.get(l) : null));
    });
    return { labels, series };
  }

  // ---------------- Utils ----------------
  function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // Fullscreen helpers for Flood Risk Map
  function addFullscreenControl(map, containerId) {
    if (!window.L || !map) return;
    // Inject minimal styles once
    if (!document.getElementById('flood-fullscreen-style')) {
      const css = `
      .leaflet-control-fullscreen a{background:#fff;border:1px solid #dcdcdc;border-radius:4px;display:inline-block;width:28px;height:28px;line-height:28px;text-align:center;font-size:16px;color:#333;box-shadow:0 1px 3px rgba(0,0,0,.2);}
      .leaflet-control-fullscreen a:hover{background:#f5f5f5}
      `;
      const st = document.createElement('style');
      st.id = 'flood-fullscreen-style';
      st.textContent = css;
      document.head.appendChild(st);
    }

    const FullC = L.Control.extend({
      options: { position: 'topleft' },
      onAdd: function() {
        const c = L.DomUtil.create('div', 'leaflet-control leaflet-control-fullscreen');
        const a = L.DomUtil.create('a', '', c);
        a.href = '#'; a.title = 'Toggle full screen'; a.innerHTML = '⤢';
        L.DomEvent.on(a, 'click', L.DomEvent.stop)
          .on(a, 'click', () => toggleMapFullscreen(containerId, map));
        return c;
      },
      onRemove: function() {}
    });
    map.addControl(new FullC());
  }

  function toggleMapFullscreen(containerId, map) {
    const el = document.getElementById(containerId);
    if (!el) return;
    const isFs = el.dataset.fullscreen === '1';
    if (!isFs) {
      // Enter fullscreen-like mode via fixed positioning
      el.dataset.fullscreen = '1';
      el.dataset.prevStyle = el.getAttribute('style') || '';
      el.style.position = 'fixed';
      el.style.top = '0';
      el.style.left = '0';
      el.style.width = '100vw';
      el.style.height = '100vh';
      el.style.zIndex = '10000';
      el.style.background = '#fff';
    } else {
      // Exit fullscreen
      el.dataset.fullscreen = '0';
      const prev = el.dataset.prevStyle || '';
      if (prev) el.setAttribute('style', prev); else el.removeAttribute('style');
    }
    setTimeout(() => map && map.invalidateSize(true), 60);
  }

  // Formatters for Manila-localized labels
  function formatManilaShort(iso) {
    try {
      if (!iso) return '';
      const d = new Date(iso);
      // Short: HH:mm in Asia/Manila
      return d.toLocaleTimeString('en-PH', { hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'Asia/Manila' });
    } catch (e) { return iso; }
  }
  function formatManilaFull(iso) {
    try {
      if (!iso) return '';
      const d = new Date(iso);
      // Full: e.g., 21 Sep 2025, 13:45 Manila Time
      const date = d.toLocaleDateString('en-PH', { day: '2-digit', month: 'short', year: 'numeric', timeZone: 'Asia/Manila' });
      const time = d.toLocaleTimeString('en-PH', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false, timeZone: 'Asia/Manila' });
      return `${date} ${time}`;
    } catch (e) { return iso; }
  }

  // Severity and label helpers
  function severityName(level) {
    const levels = {1:'ADVISORY',2:'WATCH',3:'WARNING',4:'EMERGENCY',5:'CATASTROPHIC'};
    return levels[level] || 'ALERT';
  }
  function paramLabel(key) {
    const map = { rainfall: 'Rainfall', water_level: 'Water Level', temperature: 'Temperature', humidity: 'Humidity', wind_speed: 'Wind Speed' };
    return map[key] || (key ? (key.charAt(0).toUpperCase() + key.slice(1)) : 'Parameter');
  }
})();
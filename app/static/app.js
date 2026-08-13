const $ = (id) => document.getElementById(id);
const num = (value) => new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(value);
let priceChart, backtestChart, importanceChart;

const baseOptions = (axisTitle) => ({
  responsive: true, maintainAspectRatio: false,
  plugins: { legend: { labels: { color: '#c9daf5' } }, tooltip: { mode: 'index', intersect: false } },
  scales: { x: { ticks: { color: '#97a9c7', maxTicksLimit: 7 }, grid: { color: '#183854' } }, y: { ticks: { color: '#97a9c7' }, grid: { color: '#183854' }, title: { display: true, text: axisTitle, color: '#97a9c7' } } }
});

function drawCharts(charts) {
  [priceChart, backtestChart, importanceChart].forEach(chart => chart?.destroy());
  priceChart = new Chart($('priceChart'), { type: 'line', data: { labels: charts.price_history.map(x => x.date), datasets: [{ label: 'Close', data: charts.price_history.map(x => x.close), borderColor: '#4db5ff', pointRadius: 0, tension: .2 }] }, options: baseOptions('Price') });
  backtestChart = new Chart($('backtestChart'), { type: 'line', data: { labels: charts.backtest.map(x => x.date), datasets: [{ label: 'Actual close', data: charts.backtest.map(x => x.actual), borderColor: '#55e7a6', pointRadius: 0, tension: .2 }, { label: 'Model estimate', data: charts.backtest.map(x => x.predicted), borderColor: '#ffcf67', pointRadius: 0, tension: .2, borderDash: [6, 4] }] }, options: baseOptions('Price') });
  importanceChart = new Chart($('importanceChart'), { type: 'bar', data: { labels: charts.feature_importance.map(x => x.name), datasets: [{ label: 'Importance', data: charts.feature_importance.map(x => x.value), backgroundColor: '#4db5ff' }] }, options: { ...baseOptions('Importance (%)'), indexAxis: 'y', plugins: { legend: { display: false } } } });
}

async function loadSentiment() {
  try {
    const data = await fetch('/api/sentiment').then(r => r.json());
    $('sentiment').textContent = `${data.label} (${data.score ?? 0}) · refreshed every 30 minutes`;
    $('headlines').replaceChildren(...(data.headlines || []).slice(0, 4).map(text => { const li = document.createElement('li'); li.textContent = text; return li; }));
  } catch { $('sentiment').textContent = 'Temporarily unavailable'; }
}

$('form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const symbol = $('symbol').value.trim().toUpperCase();
  $('status').textContent = 'Fetching live prices, macro context, and calculating forecast…';
  $('result').hidden = true; $('charts').hidden = true;
  try {
    const response = await fetch(`/api/forecast?symbol=${encodeURIComponent(symbol)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Request failed');
    $('current').textContent = num(data.current_close); $('prediction').textContent = num(data.predicted_next_close); $('mae').textContent = num(data.validation_mae);
    $('change').textContent = `${data.expected_change_pct >= 0 ? '+' : ''}${data.expected_change_pct}% estimated change`;
    $('change').className = data.expected_change_pct >= 0 ? 'up' : 'down';
    $('tech').textContent = `RSI(14): ${data.indicators.rsi_14} · MACD: ${data.indicators.macd} · 20-day volatility: ${data.indicators.volatility_20d}%`;
    $('status').textContent = `${data.symbol} · data as of ${data.as_of} · ${data.cached ? 'served from 10-minute cache' : 'fresh model run'}`;
    $('result').hidden = false; $('charts').hidden = false; drawCharts(data.charts);
  } catch (error) { $('status').textContent = error.message; }
});
loadSentiment();

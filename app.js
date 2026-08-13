const $ = (id) => document.getElementById(id);
const num = (value) => new Intl.NumberFormat(undefined, {maximumFractionDigits: 2}).format(value);

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
  $('result').hidden = true;
  try {
    const response = await fetch(`/api/forecast?symbol=${encodeURIComponent(symbol)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Request failed');
    $('current').textContent = num(data.current_close);
    $('prediction').textContent = num(data.predicted_next_close);
    $('mae').textContent = num(data.validation_mae);
    $('change').textContent = `${data.expected_change_pct >= 0 ? '+' : ''}${data.expected_change_pct}% estimated change`;
    $('change').className = data.expected_change_pct >= 0 ? 'up' : 'down';
    $('tech').textContent = `RSI(14): ${data.indicators.rsi_14} · MACD: ${data.indicators.macd} · 20-day volatility: ${data.indicators.volatility_20d}%`;
    $('status').textContent = `${data.symbol} · data as of ${data.as_of} · ${data.cached ? 'served from 10-minute cache' : 'fresh model run'}`;
    $('result').hidden = false;
  } catch (error) { $('status').textContent = error.message; }
});
loadSentiment();

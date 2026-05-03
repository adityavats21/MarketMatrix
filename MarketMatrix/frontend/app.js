const state = {
  apiBase: localStorage.getItem("marketmatrixApiBase") || "https://marketmatrix-backend.onrender.com",
  oneDayMetrics: [],
  sixtyDayMetrics: [],
  dataset: null,
};

const modelLabels = {
  lr: "Linear Regression",
  arima: "ARIMA",
  lstm: "LSTM",
};

const oneDayGraphs = {
  lr: "/api/lr/graphs/01_actual_vs_predicted",
  arima: "/api/arima/graphs/01_actual_vs_predicted",
  lstm: "/api/lstm/graphs/01_actual_vs_predicted",
};

const sixtyDayGraphs = {
  lr: "/api/experiment60/graphs/linear_regression/actual_vs_predicted",
  arima: "/api/experiment60/graphs/arima/actual_vs_predicted",
  lstm: "/api/experiment60/graphs/lstm/actual_vs_predicted",
};

function formatNumber(value, digits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return number.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function apiUrl(path) {
  return `${state.apiBase}${path}`;
}

async function fetchJson(path) {
  const response = await fetch(apiUrl(path));
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `${response.status} ${response.statusText}`);
  return data;
}

function setStatus(ok, text) {
  const status = document.querySelector("#apiStatus");
  status.textContent = text;
  status.classList.toggle("ok", ok);
}

function setImage(id, path) {
  const image = document.querySelector(`#${id}`);
  image.src = `${apiUrl(path)}?t=${Date.now()}`;
}

function metricFor(metrics, modelName) {
  return metrics.find((row) => row.Model.toLowerCase() === modelName.toLowerCase());
}

function tableHtml(rows) {
  return `
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>MAE</th>
          <th>RMSE</th>
          <th>R2</th>
          <th>MAPE</th>
        </tr>
      </thead>
      <tbody>
        ${rows
          .map(
            (row) => `
              <tr>
                <td>${row.Model}</td>
                <td>${formatNumber(row.MAE, 3)}</td>
                <td>${formatNumber(row.RMSE, 3)}</td>
                <td>${formatNumber(row.R2, 4)}</td>
                <td>${formatNumber(row.MAPE, 3)}%</td>
              </tr>
            `
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function renderComparisonCards(rows) {
  const target = document.querySelector("#comparisonCards");
  target.innerHTML = rows
    .map(
      (row) => `
        <article class="metric-card">
          <span>${row.Model}</span>
          <strong>MAE ${formatNumber(row.MAE, 2)}</strong>
          <p>RMSE ${formatNumber(row.RMSE, 2)} | R2 ${formatNumber(row.R2, 4)} | MAPE ${formatNumber(row.MAPE, 2)}%</p>
        </article>
      `
    )
    .join("");
}

function configureCompare(mode) {
  const isSixty = mode === "sixty";
  const rows = isSixty ? state.sixtyDayMetrics : state.oneDayMetrics;
  const graphPrefix = isSixty ? "/api/experiment60/graphs" : "/api/comparison/graphs";
  const metricOptions = isSixty
    ? [
        ["mae_comparison", "MAE"],
        ["rmse_comparison", "RMSE"],
        ["r2_comparison", "R2"],
        ["mape_comparison", "MAPE"],
        ["metrics_heatmap", "Heatmap"],
      ]
    : [
        ["mae_comparison", "MAE"],
        ["rmse_comparison", "RMSE"],
        ["r2_comparison", "R2"],
        ["mape_comparison", "MAPE"],
        ["metrics_heatmap", "Heatmap"],
        ["radar_chart", "Radar"],
        ["summary_table", "Summary Table"],
      ];

  renderComparisonCards(rows);
  document.querySelector("#comparisonTable").innerHTML = tableHtml(rows);
  document.querySelector("#overlayTitle").textContent = isSixty
    ? "60-day horizon line comparison"
    : "Next-day line comparison";
  setImage(
    "overlayGraph",
    isSixty ? "/api/experiment60/graphs/actual_vs_models_60day" : "/api/comparison/graphs/actual_vs_models"
  );

  const metricSelect = document.querySelector("#metricGraphSelect");
  metricSelect.innerHTML = metricOptions
    .map(([value, label]) => `<option value="${graphPrefix}/${value}">${label}</option>`)
    .join("");
  setImage("metricGraph", metricSelect.value);
}

function latestPredictionRow(rows) {
  return rows[rows.length - 1] || null;
}

async function runPrediction() {
  const model = document.querySelector("#predictModel").value;
  const horizon = Number(document.querySelector("#predictHorizon").value);
  const days = Number(document.querySelector("#predictDays").value || horizon);
  const forecastList = document.querySelector("#forecastList");
  forecastList.innerHTML = "";

  try {
    if (horizon === 60 || days >= 60) {
      const rows = await fetchJson(`/api/experiment60/predictions/${model}`);
      const latest = latestPredictionRow(rows);
      if (!latest) throw new Error("No 60-day prediction rows found");

      document.querySelector("#predictedClose").textContent = formatNumber(latest.Predicted, 2);
      document.querySelector("#actualClose").textContent = formatNumber(latest.Actual, 2);
      document.querySelector("#predictionError").textContent = formatNumber(
        Number(latest.Actual) - Number(latest.Predicted),
        2
      );
      document.querySelector("#predictionChartTitle").textContent = `${modelLabels[model]} 60-day experiment`;
      setImage("predictionGraph", model === "lr" ? sixtyDayGraphs.lr : model === "arima" ? sixtyDayGraphs.arima : sixtyDayGraphs.lstm);

      forecastList.innerHTML = rows
        .slice(-5)
        .reverse()
        .map(
          (row) => `
            <div class="forecast-item">
              <span>${String(row.Date).slice(0, 10)}</span>
              <strong>${formatNumber(row.Predicted, 2)}</strong>
            </div>
          `
        )
        .join("");
      return;
    }

    if (model === "lr") {
      const rows = await fetchJson("/api/predictions/lr");
      const latest = latestPredictionRow(rows);
      document.querySelector("#predictedClose").textContent = formatNumber(latest.Predicted, 2);
      document.querySelector("#actualClose").textContent = formatNumber(latest.Actual, 2);
      document.querySelector("#predictionError").textContent = formatNumber(
        Number(latest.Actual) - Number(latest.Predicted),
        2
      );
      document.querySelector("#predictionChartTitle").textContent = "Linear Regression next-day behavior";
      setImage("predictionGraph", oneDayGraphs.lr);
      forecastList.innerHTML = `<div class="forecast-item"><span>Latest test prediction</span><strong>${formatNumber(
        latest.Predicted,
        2
      )}</strong></div>`;
      return;
    }

    const response = await fetch(apiUrl(`/api/predict/${model}`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ days }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Prediction failed");

    const lastForecast = result.forecast[result.forecast.length - 1];
    document.querySelector("#predictedClose").textContent = formatNumber(lastForecast, 2);
    document.querySelector("#actualClose").textContent = "Future";
    document.querySelector("#predictionError").textContent = "-";
    document.querySelector("#predictionChartTitle").textContent = `${modelLabels[model]} next-day forecast`;
    setImage("predictionGraph", model === "arima" ? oneDayGraphs.arima : oneDayGraphs.lstm);

    forecastList.innerHTML = result.forecast
      .map(
        (value, index) => `
          <div class="forecast-item">
            <span>Day ${index + 1}</span>
            <strong>${formatNumber(value, 2)}</strong>
          </div>
        `
      )
      .join("");
  } catch (error) {
    forecastList.innerHTML = `<div class="forecast-item"><span>${error.message}</span></div>`;
  }
}

async function loadData() {
  state.apiBase = document.querySelector("#apiBase").value.replace(/\/$/, "");
  localStorage.setItem("marketmatrixApiBase", state.apiBase);
  try {
    const [health, dataset, oneDayMetrics, sixtyDayMetrics] = await Promise.all([
      fetchJson("/api/health"),
      fetchJson("/api/dataset/summary"),
      fetchJson("/api/comparison/metrics"),
      fetchJson("/api/experiment60/metrics"),
    ]);

    state.dataset = dataset;
    state.oneDayMetrics = oneDayMetrics;
    state.sixtyDayMetrics = sixtyDayMetrics;
    setStatus(true, `${health.status.toUpperCase()} connected`);

    document.querySelector("#datasetRange").textContent = `${dataset.start_date} to ${dataset.end_date}`;
    document.querySelector("#datasetRows").textContent = dataset.rows.toLocaleString();

    const best = oneDayMetrics.reduce((current, row) => (row.MAE < current.MAE ? row : current), oneDayMetrics[0]);
    document.querySelector("#winnerBadge").textContent = `${best.Model} leading`;

    configureCompare(document.querySelector("#comparisonMode").value);
    setImage("predictionGraph", "/api/lstm/graphs/01_actual_vs_predicted");
    setImage("edaGraph", document.querySelector("#edaGraphSelect").value);
    runPrediction();
  } catch (error) {
    setStatus(false, "API offline");
  }
}

document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((item) => item.classList.remove("active"));
    document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
    button.classList.add("active");
    document.querySelector(`#${button.dataset.view}`).classList.add("active");
    document.querySelector("#viewTitle").textContent =
      button.dataset.view === "predict"
        ? "Prediction Machine"
        : button.dataset.view.charAt(0).toUpperCase() + button.dataset.view.slice(1);
  });
});

document.querySelector("#refreshBtn").addEventListener("click", loadData);
document.querySelector("#predictBtn").addEventListener("click", runPrediction);
document.querySelector("#predictionGraphSelect").addEventListener("change", (event) => {
  setImage("predictionGraph", event.target.value);
});
document.querySelector("#comparisonMode").addEventListener("change", (event) => {
  configureCompare(event.target.value);
});
document.querySelector("#metricGraphSelect").addEventListener("change", (event) => {
  setImage("metricGraph", event.target.value);
});
document.querySelector("#edaGraphSelect").addEventListener("change", (event) => {
  setImage("edaGraph", event.target.value);
});
document.querySelector("#predictHorizon").addEventListener("change", (event) => {
  document.querySelector("#predictDays").value = event.target.value;
});

document.querySelector("#apiBase").value = state.apiBase;
loadData();

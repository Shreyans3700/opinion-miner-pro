const STORAGE_KEY = "opinion_miner_api_key";

function byId(id) {
  return document.getElementById(id);
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setMessage(targetId, text, kind) {
  const el = byId(targetId);
  if (!text) {
    el.className = "message hidden";
    el.textContent = "";
    return;
  }
  el.className = `message ${kind}`;
  el.textContent = text;
}

function setLoading(buttonId, isLoading, loadingText) {
  const btn = byId(buttonId);
  if (!btn.dataset.defaultText) {
    btn.dataset.defaultText = btn.textContent;
  }
  btn.disabled = isLoading;
  btn.textContent = isLoading ? loadingText : btn.dataset.defaultText;
}

function getApiKey() {
  return (sessionStorage.getItem(STORAGE_KEY) || "").trim();
}

function saveApiKey() {
  const value = byId("apiKeyInput").value.trim();
  if (!value) {
    setMessage("globalMessage", "API key cannot be empty.", "error");
    return;
  }
  sessionStorage.setItem(STORAGE_KEY, value);
  setMessage("globalMessage", "API key saved for current browser session.", "ok");
}

function requireApiKey(messageTarget) {
  const key = getApiKey();
  if (!key) {
    setMessage(
      messageTarget,
      "Please save your API key first.",
      "error"
    );
    return null;
  }
  return key;
}

async function parseError(response) {
  try {
    const data = await response.json();
    return data.detail || JSON.stringify(data);
  } catch (_) {
    return `Request failed with status ${response.status}.`;
  }
}

function renderAnalyseResult(payload) {
  const result = byId("analyseResult");
  const confidenceValue = payload.confidence != null
    ? payload.confidence.toFixed(4)
    : "N/A";
  const marginValue = payload.decision_margin != null
    ? payload.decision_margin.toFixed(4)
    : "N/A";
  const positiveScore = payload.positive_score != null
    ? payload.positive_score
    : null;
  const negativeScore = payload.negative_score != null
    ? payload.negative_score
    : null;
  const positivePct = positiveScore != null
    ? (positiveScore * 100).toFixed(1)
    : "N/A";
  const negativePct = negativeScore != null
    ? (negativeScore * 100).toFixed(1)
    : "N/A";
  const aspectSentiments = payload.aspect_sentiments || {};
  const featurePoints = Array.isArray(payload.main_feature_points)
    ? payload.main_feature_points
    : [];
  const aspectRows = Object.entries(aspectSentiments)
    .map(([aspect, sentiment]) => `<li><strong>${escapeHtml(aspect)}</strong>: ${escapeHtml(sentiment)}</li>`)
    .join("");
  const featureRows = featurePoints
    .map((point) => {
      const evidence = point.evidence ? `<div class="muted">${escapeHtml(point.evidence)}</div>` : "";
      const toneClass = String(point.sentiment || "").toLowerCase() === "negative"
        ? "negative"
        : "positive";
      const scoreText = point.scores
        ? `P ${(Number(point.scores.positive || 0) * 100).toFixed(1)}% / N ${(Number(point.scores.negative || 0) * 100).toFixed(1)}%`
        : "No score";
      return `
        <article class="feature-card">
          <div class="feature-title">${escapeHtml(point.feature || "feature")}</div>
          <div class="feature-tone ${toneClass}">${escapeHtml(point.sentiment || "unknown")} - ${escapeHtml(scoreText)}</div>
          ${evidence}
        </article>
      `;
    })
    .join("");
  const aspectBlock = aspectRows
    ? `<div class="aspect-box"><strong>Aspect Sentiments</strong><ul>${aspectRows}</ul></div>`
    : `<div class="aspect-box"><strong>Aspect Sentiments</strong><div class="muted">No aspects found.</div></div>`;
  const featureBlock = featureRows
    ? `<div class="aspect-box"><strong>Main Feature Points</strong><div class="feature-grid">${featureRows}</div></div>`
    : `<div class="aspect-box"><strong>Main Feature Points</strong><div class="muted">No feature points found.</div></div>`;
  const positiveWidth = positiveScore != null ? Math.max(0, Math.min(100, positiveScore * 100)) : 0;
  const negativeWidth = negativeScore != null ? Math.max(0, Math.min(100, negativeScore * 100)) : 0;
  const overall = payload.overall_sentiment || payload.predicted_label || "unknown";

  result.innerHTML = `
    <div class="chips">
      <span class="chip">Overall: ${escapeHtml(overall)}</span>
      <span class="chip">Positive: ${positivePct}${positiveScore != null ? "%" : ""}</span>
      <span class="chip">Negative: ${negativePct}${negativeScore != null ? "%" : ""}</span>
      <span class="chip">Confidence: ${confidenceValue}</span>
    </div>
    <div class="score-bars">
      <div class="score-row">
        <span class="score-label">Positive</span>
        <div class="score-track"><div class="score-fill positive" style="width:${positiveWidth}%"></div></div>
        <span class="score-value">${positivePct}${positiveScore != null ? "%" : ""}</span>
      </div>
      <div class="score-row">
        <span class="score-label">Negative</span>
        <div class="score-track"><div class="score-fill negative" style="width:${negativeWidth}%"></div></div>
        <span class="score-value">${negativePct}${negativeScore != null ? "%" : ""}</span>
      </div>
    </div>
    ${aspectBlock}
    ${featureBlock}
  `;
  result.classList.remove("hidden");
}

async function handleAnalyse() {
  setMessage("analyseMessage", "", "");
  byId("analyseResult").classList.add("hidden");

  const apiKey = requireApiKey("analyseMessage");
  if (!apiKey) return;

  const review = byId("reviewInput").value.trim();
  if (!review) {
    setMessage("analyseMessage", "Review text is required.", "error");
    return;
  }

  setLoading("analyseBtn", true, "Analysing...");
  try {
    const response = await fetch("/analyse", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
      },
      body: JSON.stringify({ review }),
    });

    if (!response.ok) {
      const detail = await parseError(response);
      throw new Error(detail);
    }

    const payload = await response.json();
    renderAnalyseResult(payload);
    setMessage("analyseMessage", "Analysis completed.", "ok");
  } catch (error) {
    setMessage("analyseMessage", error.message || "Analysis failed.", "error");
  } finally {
    setLoading("analyseBtn", false, "Analysing...");
  }
}

let bulkResultsData = null;
let activeTooltipCell = null;
let bulkTooltipEl = null;
let bulkTooltipHideTimer = null;

function ensureBulkTooltip() {
  if (bulkTooltipEl) return bulkTooltipEl;

  bulkTooltipEl = document.createElement("div");
  bulkTooltipEl.className = "bulk-floating-tooltip";
  document.body.appendChild(bulkTooltipEl);
  return bulkTooltipEl;
}

function positionTooltip(cell, tooltip) {
  if (!tooltip) return;

  const rect = cell.getBoundingClientRect();
  const tooltipRect = tooltip.getBoundingClientRect();
  const width = tooltipRect.width || 300;
  const height = tooltipRect.height || 250;

  let top = rect.top - 15 - height;
  let left = rect.left + rect.width / 2 - width / 2;

  if (top < 20) {
    top = rect.bottom + 15;
  }
  if (top + height > window.innerHeight - 20) {
    top = Math.max(20, window.innerHeight - height - 20);
  }
  if (left < 20) left = 20;
  if (left + width > window.innerWidth - 20) {
    left = window.innerWidth - width - 20;
  }

  tooltip.style.top = `${top}px`;
  tooltip.style.left = `${left}px`;
}

function showTooltip(cell) {
  if (!cell) return;

  const template = cell.querySelector(".tooltip");
  if (!template) return;

  if (activeTooltipCell && activeTooltipCell !== cell) {
    hideTooltip(activeTooltipCell);
  }

  if (bulkTooltipHideTimer) {
    window.clearTimeout(bulkTooltipHideTimer);
    bulkTooltipHideTimer = null;
  }

  const tooltip = ensureBulkTooltip();
  tooltip.innerHTML = template.innerHTML;
  tooltip.style.display = "block";
  positionTooltip(cell, tooltip);
  activeTooltipCell = cell;
}

function hideTooltip(cell) {
  if (!cell) return;

  if (bulkTooltipHideTimer) {
    window.clearTimeout(bulkTooltipHideTimer);
  }

  bulkTooltipHideTimer = window.setTimeout(() => {
    if (bulkTooltipEl) {
      bulkTooltipEl.style.display = "none";
    }
    if (activeTooltipCell === cell) {
      activeTooltipCell = null;
    }
    bulkTooltipHideTimer = null;
  }, 40);
}

function bindTooltipHandlers() {
  document.querySelectorAll(".sentiment-cell").forEach((cell) => {
    cell.addEventListener("mouseenter", () => showTooltip(cell));
    cell.addEventListener("mouseleave", () => hideTooltip(cell));
  });
}

async function handleBulkAnalyse() {
  setMessage("bulkMessage", "", "");
  byId("bulkResults").classList.add("hidden");

  const apiKey = requireApiKey("bulkMessage");
  if (!apiKey) return;

  const fileInput = byId("bulkFileInput");
  const file = fileInput.files[0];
  const textColumn = byId("textColumnInput").value.trim();

  if (!file) {
    setMessage("bulkMessage", "Please select a CSV file.", "error");
    return;
  }
  if (!textColumn) {
    setMessage("bulkMessage", "Please provide text column name.", "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("text_column", textColumn);
  formData.append("page", "1");
  formData.append("per_page", "50");

  setLoading("bulkBtn", true, "Processing...");
  try {
    const response = await fetch("/bulkAnalyseResults", {
      method: "POST",
      headers: { "x-api-key": apiKey },
      body: formData,
    });

    if (!response.ok) {
      const detail = await parseError(response);
      throw new Error(detail);
    }

    const data = await response.json();
    bulkResultsData = data;
    renderBulkResults(data);
    byId("exportBulkBtn").classList.remove("hidden");
    setMessage("bulkMessage", `Analysis completed. ${data.total} reviews processed.`, "ok");
  } catch (error) {
    setMessage("bulkMessage", error.message || "Bulk analysis failed.", "error");
  } finally {
    setLoading("bulkBtn", false, "Processing...");
  }
}

function renderBulkResults(data) {
  const { total, page, per_page, results } = data;

  if (bulkTooltipEl) {
    bulkTooltipEl.style.display = "none";
  }
  if (bulkTooltipHideTimer) {
    window.clearTimeout(bulkTooltipHideTimer);
    bulkTooltipHideTimer = null;
  }
  activeTooltipCell = null;

  byId("bulkCount").textContent = `${total} results`;
  byId("pageInfo").textContent = `Page ${page} of ${Math.ceil(total / per_page)}`;

  byId("prevPageBtn").disabled = page <= 1;
  byId("nextPageBtn").disabled = page * per_page >= total;

  const tbody = byId("bulkTableBody");
  tbody.innerHTML = results.map((row) => {
    const sentiment = row.overall_sentiment || "unknown";
    const isPositive = sentiment.toLowerCase() === "positive";
    const posPct = (row.positive_score * 100).toFixed(1);
    const negPct = (row.negative_score * 100).toFixed(1);
    const confidence = (row.confidence * 100).toFixed(1);

    const featuresHtml = row.features && row.features.length
      ? row.features.map(f => `
          <div class="tt-feature">
            <div class="tt-feature-header">
              <span class="tt-feature-name">${escapeHtml(f.feature)}</span>
              <span class="tt-feature-sent ${f.sentiment.toLowerCase()}">${escapeHtml(f.sentiment)}</span>
            </div>
            ${f.evidence ? `<div class="tt-feature-evidence">"${escapeHtml(f.evidence)}"</div>` : ''}
            ${f.scores ? `<div class="tt-feature-scores">P: ${(f.scores.positive * 100).toFixed(0)}% / N: ${(f.scores.negative * 100).toFixed(0)}%</div>` : ''}
          </div>
        `).join("")
      : '<div class="tt-muted">No features detected</div>';

    const tooltipHtml = `
      <div class="tooltip-content">
        <div class="tt-section">
          <div class="tt-label">Scores</div>
          <div class="tt-bars">
            <div class="tt-bar-row">
              <span class="tt-bar-label">Positive</span>
              <div class="tt-bar-track"><div class="tt-bar-fill positive" style="width:${posPct}%"></div></div>
              <span class="tt-bar-value">${posPct}%</span>
            </div>
            <div class="tt-bar-row">
              <span class="tt-bar-label">Negative</span>
              <div class="tt-bar-track"><div class="tt-bar-fill negative" style="width:${negPct}%"></div></div>
              <span class="tt-bar-value">${negPct}%</span>
            </div>
          </div>
        </div>
        <div class="tt-section">
          <div class="tt-label">Confidence: ${confidence}%</div>
        </div>
        <div class="tt-section">
          <div class="tt-label">Features</div>
          <div class="tt-features">${featuresHtml}</div>
        </div>
      </div>
    `;

    const truncated = row.review.length > 80 ? row.review.substring(0, 80) + "..." : row.review;

    return `
      <tr>
        <td class="review-cell" title="${escapeHtml(row.review)}">${escapeHtml(truncated)}</td>
        <td class="sentiment-cell">
          <span class="sentiment-badge ${isPositive ? 'positive' : 'negative'}">${escapeHtml(sentiment)}</span>
          <div class="tooltip">${tooltipHtml}</div>
        </td>
      </tr>
    `;
  }).join("");

  bindTooltipHandlers();
  byId("bulkResults").classList.remove("hidden");
}

async function changeBulkPage(delta) {
  if (!bulkResultsData) return;

  const newPage = bulkResultsData.page + delta;
  if (newPage < 1) return;
  if (newPage > Math.ceil(bulkResultsData.total / bulkResultsData.per_page)) return;

  const apiKey = getApiKey();
  if (!apiKey) return;

  const fileInput = byId("bulkFileInput");
  const file = fileInput.files[0];
  const textColumn = byId("textColumnInput").value.trim();

  const formData = new FormData();
  formData.append("file", file);
  formData.append("text_column", textColumn);
  formData.append("page", String(newPage));
  formData.append("per_page", "50");

  try {
    const response = await fetch("/bulkAnalyseResults", {
      method: "POST",
      headers: { "x-api-key": apiKey },
      body: formData,
    });

    if (response.ok) {
      const data = await response.json();
      bulkResultsData = data;
      renderBulkResults(data);
    }
  } catch (e) {}
}

async function exportBulkCSV() {
  const apiKey = getApiKey();
  if (!apiKey) return;

  const fileInput = byId("bulkFileInput");
  const file = fileInput.files[0];
  const textColumn = byId("textColumnInput").value.trim();

  const formData = new FormData();
  formData.append("file", file);
  formData.append("text_column", textColumn);

  try {
    const response = await fetch("/bulkAnalyse", {
      method: "POST",
      headers: { "x-api-key": apiKey },
      body: formData,
    });

    if (response.ok) {
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "bulk_analysis.csv";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }
  } catch (e) {}
}

async function handleTrain() {
  setMessage("trainMessage", "", "");
  byId("trainDetails").classList.add("hidden");

  const apiKey = requireApiKey("trainMessage");
  if (!apiKey) return;

  setLoading("trainBtn", true, "Training...");
  try {
    const response = await fetch("/train", {
      method: "POST",
      headers: { "x-api-key": apiKey },
    });

    if (!response.ok) {
      const detail = await parseError(response);
      throw new Error(detail);
    }

    const payload = await response.json();
    setMessage("trainMessage", "Training completed successfully.", "ok");
    byId("trainOutput").textContent = JSON.stringify(payload, null, 2);
    byId("trainDetails").classList.remove("hidden");
  } catch (error) {
    setMessage("trainMessage", error.message || "Training failed.", "error");
  } finally {
    setLoading("trainBtn", false, "Training...");
  }
}

function setupTabs() {
  const tabs = Array.from(document.querySelectorAll(".tab"));
  const panels = Array.from(document.querySelectorAll(".tab-panel"));

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      panels.forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      byId(tab.dataset.tab).classList.add("active");
    });
  });
}

function activateTab(tabName) {
  const tabs = Array.from(document.querySelectorAll(".tab"));
  const panels = Array.from(document.querySelectorAll(".tab-panel"));
  tabs.forEach((t) => t.classList.remove("active"));
  panels.forEach((p) => p.classList.remove("active"));

  const targetTab = document.querySelector(`.tab[data-tab="${tabName}"]`);
  const targetPanel = byId(tabName);
  if (targetTab) targetTab.classList.add("active");
  if (targetPanel) targetPanel.classList.add("active");
}

function setupSamples() {
  const buttons = Array.from(document.querySelectorAll(".sample-btn"));
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const text = btn.dataset.sample || "";
      byId("reviewInput").value = text;
      activateTab("analyse");
      byId("reviewInput").focus();
    });
  });
}

function restoreApiKey() {
  const savedKey = getApiKey();
  if (savedKey) {
    byId("apiKeyInput").value = savedKey;
  }
}

function bootstrap() {
  restoreApiKey();
  setupTabs();
  setupSamples();
  byId("saveKeyBtn").addEventListener("click", saveApiKey);
  byId("analyseBtn").addEventListener("click", handleAnalyse);
  byId("bulkBtn").addEventListener("click", handleBulkAnalyse);
  byId("exportBulkBtn").addEventListener("click", exportBulkCSV);
  byId("prevPageBtn").addEventListener("click", () => changeBulkPage(-1));
  byId("nextPageBtn").addEventListener("click", () => changeBulkPage(1));
  byId("trainBtn").addEventListener("click", handleTrain);

}

document.addEventListener("DOMContentLoaded", bootstrap);

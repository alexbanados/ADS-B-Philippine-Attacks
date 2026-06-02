const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("csv-file");
const stateBox = document.getElementById("state");
const logBox = document.getElementById("log");
const allButtons = Array.from(document.querySelectorAll("button"));
const pages = Array.from(document.querySelectorAll(".page"));
const stepLabels = Array.from(document.querySelectorAll("[data-step-label]"));
const attackTypeInput = document.getElementById("attack-type");
const attackStartInput = document.getElementById("attack-start");
const attackDurationInput = document.getElementById("attack-duration");
const attackStartRandomInput = document.getElementById("attack-start-random");
const attackDurationRandomInput = document.getElementById("attack-duration-random");
const attackStartValue = document.getElementById("attack-start-value");
const attackDurationValue = document.getElementById("attack-duration-value");
const attackEnvelopeInput = document.getElementById("attack-envelope");
const ATTACK_BIN_COUNT = 2000;
const ROUTE_NAMES = {
  ceb: "Manila-Cebu",
  dvo: "Manila-Davao",
  ilo: "Manila-Iloilo",
  mph: "Manila-Malay, Aklan",
  pps: "Manila-Puerto Princesa",
};

let currentState = {};
let poisonWanted = false;
let attackDebounce = null;

function appendLog(text) {
  if (!text) return;
  const prefix = logBox.textContent ? "\n\n" : "";
  logBox.textContent += prefix + text;
  logBox.scrollTop = logBox.scrollHeight;
}

function setBusy(isBusy) {
  allButtons.forEach((button) => {
    button.disabled = isBusy;
  });
}

function showPage(name) {
  pages.forEach((page) => page.classList.toggle("active", page.id === `page-${name}`));
  stepLabels.forEach((label) => {
    label.classList.toggle("active", label.dataset.stepLabel === name);
  });
  requestAnimationFrame(() => {
    window.scrollTo({ top: 0, left: 0, behavior: "auto" });
    document.querySelector(".workspace")?.scrollTo({ top: 0, left: 0, behavior: "auto" });
  });
}

function updateState(state) {
  currentState = state || currentState || {};
  if (!currentState || Object.keys(currentState).length === 0) {
    stateBox.textContent = "No file uploaded.";
    return;
  }

  const lines = [];
  if (currentState.flight_id) lines.push(`flight: ${currentState.flight_id}`);
  if (currentState.canonical_name) lines.push(`file: ${currentState.canonical_name}`);
  if (currentState.route) lines.push(`route: ${routeName(currentState.route)}`);
  if (currentState.attack_type) lines.push(`attack: ${currentState.attack_type}`);
  if (currentState.poison_generator) lines.push(`poison generator: ${currentState.poison_generator}`);
  if (currentState.current_path) lines.push(`current: ${currentState.current_path}`);
  if (currentState.auth_path) lines.push(`auth: ${currentState.auth_path}`);
  if (currentState.mod_path) lines.push(`mod: ${currentState.mod_path}`);
  if (currentState.poison_path) lines.push(`poison: ${currentState.poison_path}`);
  stateBox.textContent = lines.join("\n");
}

function routeName(routeCode) {
  return ROUTE_NAMES[String(routeCode || "").toLowerCase()] || routeCode;
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw data;
  return data;
}

async function runAction(payload) {
  const data = await postJson("/api/run", payload);
  updateState(data.state);
  appendLog(data.log || data.message);
  return data;
}

function wantsAttack() {
  return document.querySelector('input[name="want-attack"]:checked').value === "yes";
}

function freshAttackSeed() {
  const randomPart = Math.floor(Math.random() * 1_000_000);
  return (Date.now() + randomPart) % 2_147_483_647;
}

function percentToBins(percent) {
  return Math.round((Number(percent) / 100) * ATTACK_BIN_COUNT);
}

function updateAttackSliderLabels() {
  const startBins = percentToBins(attackStartInput.value);
  const durationBins = percentToBins(attackDurationInput.value);
  attackStartInput.disabled = attackStartRandomInput.checked;
  attackDurationInput.disabled = attackDurationRandomInput.checked;
  attackStartValue.textContent = attackStartRandomInput.checked
    ? "Random"
    : `${attackStartInput.value}% (bin ${startBins})`;
  attackDurationValue.textContent = attackDurationRandomInput.checked
    ? "Random"
    : `${attackDurationInput.value}% (${durationBins} bins)`;
}

function attackPayload(seed = 42) {
  return {
    action: "attack",
    attack_type: wantsAttack() ? attackTypeInput.value : "authentic",
    attack_seed: seed,
    attack_start: attackStartRandomInput.checked ? "" : percentToBins(attackStartInput.value),
    attack_duration: attackDurationRandomInput.checked ? "" : percentToBins(attackDurationInput.value),
    attack_envelope: attackEnvelopeInput.value,
  };
}

function scheduleAttackPreview() {
  if (!wantsAttack()) return;
  clearTimeout(attackDebounce);
  attackDebounce = setTimeout(async () => {
    document.getElementById("poison-question").classList.add("hidden");
    setBusy(true);
    try {
      const data = await runAction(attackPayload());
      renderAttackReport(data.report);
    } catch (error) {
      updateState(error.state);
      appendLog(`ERROR: ${error.message || error}`);
    } finally {
      setBusy(false);
    }
  }, 650);
}

function renderStepReport(container, report) {
  container.classList.remove("hidden", "ok", "error");
  container.classList.add(report.status === "ok" ? "ok" : "error");

  const steps = (report.steps || [])
    .map((step) => `<li>${step.name}: ${step.status}</li>`)
    .join("");
  const routeLine = report.route
    ? `<p><strong>Route classified:</strong> ${routeName(report.route)}</p>`
    : "";
  const visual = report.plot_data
    ? interactivePlotShell()
    : report.image_url
      ? `<img src="${report.image_url}" alt="Segmented flight plot" />`
      : "";
  const action =
    report.status === "ok"
      ? `<button id="preprocess-ok">OK</button>`
      : `<button id="preprocess-reupload">Re-upload</button>`;

  container.innerHTML = `
    <h3>${report.status === "ok" ? "Preprocessing Complete" : "Preprocessing Error"}</h3>
    <p>${report.message || ""}</p>
    ${routeLine}
    <ol class="step-list">${steps}</ol>
    ${visual}
    <div class="button-row">${action}</div>
  `;

  const plotElement = container.querySelector(".interactive-plot");
  if (plotElement && report.plot_data) setupInteractivePlot(plotElement, report.plot_data);
  const ok = document.getElementById("preprocess-ok");
  if (ok) ok.addEventListener("click", () => showPage("attack"));
  const reupload = document.getElementById("preprocess-reupload");
  if (reupload) reupload.addEventListener("click", () => showPage("upload"));
}

function zoomableImage(url, altText) {
  return `
    <div class="zoom-viewer">
      <div class="zoom-toolbar">
        <button type="button" data-zoom-action="out" title="Zoom out">-</button>
        <button type="button" data-zoom-action="reset" title="Reset zoom">1:1</button>
        <button type="button" data-zoom-action="in" title="Zoom in">+</button>
      </div>
      <div class="zoom-stage">
        <img src="${url}" alt="${altText}" draggable="false" />
      </div>
    </div>
  `;
}

function setupZoomViewer(viewer) {
  const stage = viewer.querySelector(".zoom-stage");
  const image = viewer.querySelector("img");
  const buttons = viewer.querySelectorAll("[data-zoom-action]");
  const state = {
    scale: 1,
    x: 0,
    y: 0,
    isDragging: false,
    startX: 0,
    startY: 0,
    baseX: 0,
    baseY: 0,
  };

  function applyTransform() {
    if (state.scale <= 1) {
      state.x = 0;
      state.y = 0;
    }
    image.style.transform = `translate(${state.x}px, ${state.y}px) scale(${state.scale})`;
    stage.classList.toggle("is-zoomed", state.scale > 1);
  }

  function setScale(nextScale) {
    state.scale = Math.min(6, Math.max(1, nextScale));
    applyTransform();
  }

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const action = button.dataset.zoomAction;
      if (action === "in") setScale(state.scale * 1.25);
      if (action === "out") setScale(state.scale / 1.25);
      if (action === "reset") {
        state.scale = 1;
        state.x = 0;
        state.y = 0;
        applyTransform();
      }
    });
  });

  stage.addEventListener("wheel", (event) => {
    event.preventDefault();
    setScale(state.scale * (event.deltaY < 0 ? 1.12 : 1 / 1.12));
  });

  stage.addEventListener("pointerdown", (event) => {
    if (state.scale <= 1) return;
    state.isDragging = true;
    state.startX = event.clientX;
    state.startY = event.clientY;
    state.baseX = state.x;
    state.baseY = state.y;
    stage.classList.add("is-dragging");
    stage.setPointerCapture(event.pointerId);
  });

  stage.addEventListener("pointermove", (event) => {
    if (!state.isDragging) return;
    state.x = state.baseX + event.clientX - state.startX;
    state.y = state.baseY + event.clientY - state.startY;
    applyTransform();
  });

  stage.addEventListener("pointerup", (event) => {
    state.isDragging = false;
    stage.classList.remove("is-dragging");
    if (stage.hasPointerCapture(event.pointerId)) {
      stage.releasePointerCapture(event.pointerId);
    }
  });

  stage.addEventListener("pointercancel", () => {
    state.isDragging = false;
    stage.classList.remove("is-dragging");
  });

  applyTransform();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function interactivePlotShell() {
  return `
    <div class="interactive-plot">
      <div class="plot-toolbar">
        <button type="button" data-plot-action="out" title="Zoom out">-</button>
        <button type="button" data-plot-action="reset" title="Reset zoom">Reset</button>
        <button type="button" data-plot-action="in" title="Zoom in">+</button>
      </div>
      <div class="plot-grid"></div>
    </div>
  `;
}

function infoGrid(items = []) {
  if (!items.length) return "";
  const rows = items
    .map(
      (item) => `
        <div class="info-item">
          <span>${escapeHtml(item.label)}</span>
          <strong>${escapeHtml(item.value)}</strong>
        </div>
      `
    )
    .join("");
  return `<div class="info-grid">${rows}</div>`;
}

function createSvgElement(name, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => {
    element.setAttribute(key, String(value));
  });
  return element;
}

function formatTick(value) {
  const absValue = Math.abs(value);
  if (absValue >= 1000) return value.toFixed(0);
  if (absValue >= 100) return value.toFixed(1);
  if (absValue >= 10) return value.toFixed(2);
  return value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function niceTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return [min];
  const span = max - min;
  const rawStep = span / Math.max(1, count - 1);
  const power = 10 ** Math.floor(Math.log10(rawStep));
  const multiples = [1, 2, 5, 10];
  const step = multiples.find((multiple) => multiple * power >= rawStep) * power;
  const first = Math.ceil(min / step) * step;
  const ticks = [];
  for (let value = first; value <= max + step * 0.5; value += step) {
    ticks.push(value);
  }
  return ticks;
}

function domainForPanel(panel) {
  const xs = [];
  const ys = [];
  (panel.series || []).forEach((series) => {
    (series.points || []).forEach(([x, y]) => {
      if (Number.isFinite(x) && Number.isFinite(y)) {
        xs.push(x);
        ys.push(y);
      }
    });
  });

  function expand(values) {
    if (!values.length) return { min: 0, max: 1 };
    let min = Math.min(...values);
    let max = Math.max(...values);
    if (min === max) {
      const pad = Math.abs(min || 1) * 0.05;
      min -= pad;
      max += pad;
    } else {
      const pad = (max - min) * 0.045;
      min -= pad;
      max += pad;
    }
    return { min, max };
  }

  const xDomain = expand(xs);
  const yDomain = expand(ys);
  return {
    xMin: xDomain.min,
    xMax: xDomain.max,
    yMin: yDomain.min,
    yMax: yDomain.max,
  };
}

function setupInteractivePlot(plotElement, plotData) {
  const grid = plotElement.querySelector(".plot-grid");
  const panels = (plotData.panels || []).slice(0, 3);
  const states = panels.map((panel, index) => {
    const initialDomain = domainForPanel(panel);
    const panelElement = document.createElement("div");
    panelElement.className = `plot-panel plot-panel-${panel.id || index}`;
    if (panel.id === "position") panelElement.classList.add("plot-panel-position");
    panelElement.innerHTML = "<svg></svg>";
    grid.appendChild(panelElement);
    return {
      panel,
      panelElement,
      svg: panelElement.querySelector("svg"),
      initialDomain,
      domain: { ...initialDomain },
      drag: null,
      clipId: `plot-clip-${Date.now()}-${index}-${Math.floor(Math.random() * 10000)}`,
    };
  });

  const margin = { top: 30, right: 18, bottom: 46, left: 62 };

  function dimensions(state) {
    const rect = state.panelElement.getBoundingClientRect();
    const width = Math.max(320, rect.width || 640);
    const height = Math.max(240, rect.height || 320);
    return {
      width,
      height,
      plotLeft: margin.left,
      plotRight: width - margin.right,
      plotTop: margin.top,
      plotBottom: height - margin.bottom,
      plotWidth: width - margin.left - margin.right,
      plotHeight: height - margin.top - margin.bottom,
    };
  }

  function scaleFor(state, dims) {
    const { domain } = state;
    return {
      x: (value) =>
        dims.plotLeft +
        ((value - domain.xMin) / (domain.xMax - domain.xMin || 1)) * dims.plotWidth,
      y: (value) =>
        dims.plotBottom -
        ((value - domain.yMin) / (domain.yMax - domain.yMin || 1)) * dims.plotHeight,
      dataX: (pixel) =>
        domain.xMin + ((pixel - dims.plotLeft) / dims.plotWidth) * (domain.xMax - domain.xMin),
      dataY: (pixel) =>
        domain.yMax - ((pixel - dims.plotTop) / dims.plotHeight) * (domain.yMax - domain.yMin),
    };
  }

  function pathForPoints(points, scales) {
    return (points || [])
      .map(([x, y], index) => `${index === 0 ? "M" : "L"} ${scales.x(x)} ${scales.y(y)}`)
      .join(" ");
  }

  function drawPanel(state) {
    const dims = dimensions(state);
    const scales = scaleFor(state, dims);
    const svg = state.svg;
    svg.replaceChildren();
    svg.setAttribute("viewBox", `0 0 ${dims.width} ${dims.height}`);

    const defs = createSvgElement("defs");
    const clipPath = createSvgElement("clipPath", { id: state.clipId });
    clipPath.appendChild(
      createSvgElement("rect", {
        x: dims.plotLeft,
        y: dims.plotTop,
        width: dims.plotWidth,
        height: dims.plotHeight,
      })
    );
    defs.appendChild(clipPath);
    svg.appendChild(defs);

    svg.appendChild(
      createSvgElement("rect", {
        x: dims.plotLeft,
        y: dims.plotTop,
        width: dims.plotWidth,
        height: dims.plotHeight,
        fill: "#fff",
        stroke: "#d0d7de",
      })
    );

    niceTicks(state.domain.xMin, state.domain.xMax).forEach((tick) => {
      const x = scales.x(tick);
      svg.appendChild(
        createSvgElement("line", {
          x1: x,
          y1: dims.plotTop,
          x2: x,
          y2: dims.plotBottom,
          stroke: "#e5e7eb",
        })
      );
      const text = createSvgElement("text", {
        x,
        y: dims.plotBottom + 18,
        "text-anchor": "middle",
        class: "plot-axis-text",
      });
      text.textContent = formatTick(tick);
      svg.appendChild(text);
    });

    niceTicks(state.domain.yMin, state.domain.yMax).forEach((tick) => {
      const y = scales.y(tick);
      svg.appendChild(
        createSvgElement("line", {
          x1: dims.plotLeft,
          y1: y,
          x2: dims.plotRight,
          y2: y,
          stroke: "#e5e7eb",
        })
      );
      const text = createSvgElement("text", {
        x: dims.plotLeft - 8,
        y: y + 4,
        "text-anchor": "end",
        class: "plot-axis-text",
      });
      text.textContent = formatTick(tick);
      svg.appendChild(text);
    });

    (state.panel.series || []).forEach((series) => {
      if (!series.points || series.points.length === 0) return;
      svg.appendChild(
        createSvgElement("path", {
          d: pathForPoints(series.points, scales),
          fill: "none",
          stroke: series.color || "#111827",
          "stroke-width": 2.2,
          "clip-path": `url(#${state.clipId})`,
        })
      );
    });

    const title = createSvgElement("text", {
      x: dims.width / 2,
      y: 18,
      "text-anchor": "middle",
      class: "plot-title",
    });
    title.textContent = state.panel.title || "";
    svg.appendChild(title);

    const xLabel = createSvgElement("text", {
      x: (dims.plotLeft + dims.plotRight) / 2,
      y: dims.height - 10,
      "text-anchor": "middle",
      class: "plot-axis-label",
    });
    xLabel.textContent = state.panel.x_label || "";
    svg.appendChild(xLabel);

    const yLabel = createSvgElement("text", {
      x: 16,
      y: (dims.plotTop + dims.plotBottom) / 2,
      "text-anchor": "middle",
      transform: `rotate(-90 16 ${(dims.plotTop + dims.plotBottom) / 2})`,
      class: "plot-axis-label",
    });
    yLabel.textContent = state.panel.y_label || "";
    svg.appendChild(yLabel);

    let legendY = dims.plotTop + 16;
    (state.panel.series || [])
      .filter((series) => series.label && !String(series.label).startsWith("_"))
      .forEach((series) => {
      svg.appendChild(
        createSvgElement("line", {
          x1: dims.plotRight - 130,
          y1: legendY - 4,
          x2: dims.plotRight - 104,
          y2: legendY - 4,
          stroke: series.color || "#111827",
          "stroke-width": 3,
        })
      );
      const label = createSvgElement("text", {
        x: dims.plotRight - 98,
        y: legendY,
        class: "plot-legend-text",
      });
      label.textContent = series.label || "";
      svg.appendChild(label);
      legendY += 18;
    });
  }

  function drawAll() {
    states.forEach(drawPanel);
  }

  function zoomState(state, factor, pivotX, pivotY) {
    const { domain } = state;
    domain.xMin = pivotX - (pivotX - domain.xMin) * factor;
    domain.xMax = pivotX + (domain.xMax - pivotX) * factor;
    domain.yMin = pivotY - (pivotY - domain.yMin) * factor;
    domain.yMax = pivotY + (domain.yMax - pivotY) * factor;
    drawPanel(state);
  }

  function zoomAll(factor) {
    states.forEach((state) => {
      const pivotX = (state.domain.xMin + state.domain.xMax) / 2;
      const pivotY = (state.domain.yMin + state.domain.yMax) / 2;
      zoomState(state, factor, pivotX, pivotY);
    });
  }

  plotElement.querySelector("[data-plot-action='in']").addEventListener("click", () => zoomAll(0.8));
  plotElement.querySelector("[data-plot-action='out']").addEventListener("click", () => zoomAll(1.25));
  plotElement.querySelector("[data-plot-action='reset']").addEventListener("click", () => {
    states.forEach((state) => {
      state.domain = { ...state.initialDomain };
    });
    drawAll();
  });

  states.forEach((state) => {
    state.panelElement.addEventListener("wheel", (event) => {
      event.preventDefault();
      const dims = dimensions(state);
      const scales = scaleFor(state, dims);
      const rect = state.panelElement.getBoundingClientRect();
      const localX = Math.min(dims.plotRight, Math.max(dims.plotLeft, event.clientX - rect.left));
      const localY = Math.min(dims.plotBottom, Math.max(dims.plotTop, event.clientY - rect.top));
      zoomState(state, event.deltaY < 0 ? 0.82 : 1.22, scales.dataX(localX), scales.dataY(localY));
    });

    state.panelElement.addEventListener("pointerdown", (event) => {
      const dims = dimensions(state);
      state.drag = {
        startX: event.clientX,
        startY: event.clientY,
        domain: { ...state.domain },
        dims,
      };
      state.panelElement.classList.add("is-dragging");
      state.panelElement.setPointerCapture(event.pointerId);
    });

    state.panelElement.addEventListener("pointermove", (event) => {
      if (!state.drag) return;
      const { startX, startY, domain, dims } = state.drag;
      const dx = event.clientX - startX;
      const dy = event.clientY - startY;
      const xShift = (-dx / dims.plotWidth) * (domain.xMax - domain.xMin);
      const yShift = (dy / dims.plotHeight) * (domain.yMax - domain.yMin);
      state.domain = {
        xMin: domain.xMin + xShift,
        xMax: domain.xMax + xShift,
        yMin: domain.yMin + yShift,
        yMax: domain.yMax + yShift,
      };
      drawPanel(state);
    });

    state.panelElement.addEventListener("pointerup", (event) => {
      state.drag = null;
      state.panelElement.classList.remove("is-dragging");
      if (state.panelElement.hasPointerCapture(event.pointerId)) {
        state.panelElement.releasePointerCapture(event.pointerId);
      }
    });

    state.panelElement.addEventListener("pointercancel", () => {
      state.drag = null;
      state.panelElement.classList.remove("is-dragging");
    });

    state.panelElement.addEventListener("dblclick", () => {
      state.domain = { ...state.initialDomain };
      drawPanel(state);
    });
  });

  drawAll();
  window.addEventListener("resize", drawAll);
}

function renderImageReport(container, report, okCallback) {
  container.classList.remove("hidden", "ok", "error");
  container.classList.add(report.status === "ok" ? "ok" : "error");
  const info = infoGrid(report.info || []);
  const visual = report.plot_data
    ? interactivePlotShell()
    : report.image_url
      ? zoomableImage(report.image_url, "Generated plot")
      : "";
  container.innerHTML = `
    <h3>${report.message || "Done"}</h3>
    ${info}
    ${visual}
    <div class="button-row"><button id="${container.id}-ok">OK</button></div>
  `;
  container.querySelectorAll(".zoom-viewer").forEach(setupZoomViewer);
  const plotElement = container.querySelector(".interactive-plot");
  if (plotElement && report.plot_data) setupInteractivePlot(plotElement, report.plot_data);
  document.getElementById(`${container.id}-ok`).addEventListener("click", okCallback);
}

function renderAttackReport(report) {
  const container = document.getElementById("attack-report");
  document.getElementById("poison-question").classList.add("hidden");
  renderImageReport(container, report, () => {
    if (report.attack_type === "authentic") {
      poisonWanted = false;
      goToResults();
    } else {
      goToPoisonPage();
    }
  });
}

function renderPoisonReport(report) {
  const container = document.getElementById("poison-report");
  renderImageReport(container, report, () => goToResults());
}

function renderPoisonError(message) {
  const container = document.getElementById("poison-report");
  container.classList.remove("hidden", "ok");
  container.classList.add("error");
  container.innerHTML = `
    <h3>Poison Generation Error</h3>
    <div class="button-row">
      <button id="poison-error-back">Back to Attack</button>
      <button id="poison-error-results" class="secondary">Skip to Results</button>
    </div>
  `;
  document.getElementById("poison-error-back").addEventListener("click", () => {
    showPage("attack");
  });
  document.getElementById("poison-error-results").addEventListener("click", () => {
    poisonWanted = false;
    goToResults();
  });
}

function renderResults(sections, plots = []) {
  const container = document.getElementById("results-report");
  if (!sections || !sections.length) {
    container.innerHTML = "<p>No results.</p>";
    return;
  }

  const resultsHtml = sections
    .map((section) => {
      if (section.note) {
        return `
          <div class="result-card">
            <h3>${section.title}</h3>
            <p>${section.note}</p>
          </div>
        `;
      }

      const rows = section.rows
        .map((row) => {
          const tag = row.correct ? "[CORRECT]" : "[WRONG]";
          const tagClass = row.correct ? "tag-correct" : "tag-wrong";
          const scores = row.scores || {};
          const scoreLabels = {
            authentic: "authentic",
            modified_altitude: "modified_altitude",
            modified_speed: "modified_speed",
            modified_position: "modified_position",
          };
          const highestScore = Object.entries(scoreLabels)
            .map(([key, label]) => ({ label, value: Number(scores[key]) }))
            .filter((score) => Number.isFinite(score.value))
            .sort((a, b) => b.value - a.value)[0];
          const scoreText = highestScore
            ? ` (${highestScore.value.toFixed(4)})`
            : "-";
          return `
            <tr>
              <td class="${tagClass}">${tag}</td>
              <td>${row.file}</td>
              <td>${row.prediction}${scoreText === "-" ? "" : scoreText}</td>
              <td>${row.true_class}</td>
            </tr>
          `;
        })
        .join("");

      return `
        <div class="result-card">
          <h3>${section.title}</h3>
          <p>${section.subtitle || ""}</p>
          <table>
            <thead>
              <tr>
                <th>Status</th>
                <th>File</th>
                <th>Prediction</th>
                <th>True Class</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      `;
    })
    .join("");

  const plotHtml = (plots || [])
    .map(
      (plot, index) => `
        <div class="result-card plot-result-card">
          <h3>${escapeHtml(plot.title || `Plot ${index + 1}`)}</h3>
          ${interactivePlotShell()}
        </div>
      `
    )
    .join("");

  container.innerHTML = `
    <div class="result-sections-grid">${resultsHtml}</div>
    ${plotHtml ? `<div class="result-plots-grid">${plotHtml}</div>` : ""}
  `;
  container.querySelectorAll(".interactive-plot").forEach((plotElement, index) => {
    setupInteractivePlot(plotElement, plots[index]);
  });
}

async function buildReadyFiles() {
  return runAction({ action: "ready" });
}

async function goToPoisonPage() {
  poisonWanted = true;
  setBusy(true);
  try {
    await buildReadyFiles();
    showPage("poison");
  } catch (error) {
    updateState(error.state);
    appendLog(`ERROR: ${error.message || error}`);
  } finally {
    setBusy(false);
  }
}

async function goToResults() {
  showPage("results");
  setBusy(true);
  try {
    await buildReadyFiles();
    const data = await runAction({ action: "evaluate" });
    renderResults(data.results, data.plots || []);
  } catch (error) {
    updateState(error.state);
    appendLog(`ERROR: ${error.message || error}`);
    document.getElementById("results-report").innerHTML = `<p class="tag-wrong">ERROR: ${
      error.message || error
    }</p>`;
  } finally {
    setBusy(false);
  }
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!fileInput.files.length) return;

  const formData = new FormData();
  formData.append("csv_file", fileInput.files[0]);

  setBusy(true);
  try {
    const response = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) throw data;
    updateState(data.state);
    appendLog(data.message);
    document.getElementById("preprocess-report").classList.add("hidden");
    document.getElementById("attack-report").classList.add("hidden");
    document.getElementById("poison-question").classList.add("hidden");
    document.getElementById("poison-report").classList.add("hidden");
    document.getElementById("results-report").innerHTML = "";
    showPage("preprocess");
  } catch (error) {
    updateState(error.state);
    appendLog(`ERROR: ${error.message || error}`);
  } finally {
    setBusy(false);
  }
});

document.getElementById("run-preprocess").addEventListener("click", async () => {
  setBusy(true);
  try {
    const data = await runAction({ action: "preprocess_all" });
    renderStepReport(document.getElementById("preprocess-report"), data.report);
  } catch (error) {
    updateState(error.state);
    appendLog(`ERROR: ${error.message || error}`);
    renderStepReport(document.getElementById("preprocess-report"), {
      status: "error",
      message: error.message || String(error),
      steps: [],
    });
  } finally {
    setBusy(false);
  }
});

document.querySelectorAll('input[name="want-attack"]').forEach((input) => {
  input.addEventListener("change", () => {
    document
      .getElementById("attack-options")
      .classList.toggle("hidden", input.value !== "yes" || !input.checked);
    document.getElementById("attack-report").classList.add("hidden");
    document.getElementById("poison-question").classList.add("hidden");
    if (input.value === "yes" && input.checked) scheduleAttackPreview();
  });
});

[attackTypeInput, attackEnvelopeInput].forEach((input) => {
  input.addEventListener("change", scheduleAttackPreview);
});

[attackStartInput, attackDurationInput].forEach((input) => {
  input.addEventListener("input", () => {
    updateAttackSliderLabels();
    scheduleAttackPreview();
  });
});

[attackStartRandomInput, attackDurationRandomInput].forEach((input) => {
  input.addEventListener("change", () => {
    updateAttackSliderLabels();
    scheduleAttackPreview();
  });
});

document.getElementById("attack-continue").addEventListener("click", async () => {
  clearTimeout(attackDebounce);
  setBusy(true);
  try {
    const data = await runAction(attackPayload(freshAttackSeed()));
    renderAttackReport(data.report);
  } catch (error) {
    updateState(error.state);
    appendLog(`ERROR: ${error.message || error}`);
  } finally {
    setBusy(false);
  }
});

document.getElementById("go-poison").addEventListener("click", async () => {
  await goToPoisonPage();
});

document.getElementById("skip-poison").addEventListener("click", () => {
  poisonWanted = false;
  goToResults();
});

document.getElementById("generate-poison").addEventListener("click", async () => {
  setBusy(true);
  try {
    const data = await runAction({
      action: "poison",
      poison_generator: document.getElementById("poison-generator").value,
    });
    renderPoisonReport(data.report);
  } catch (error) {
    updateState(error.state);
    appendLog(`ERROR: ${error.message || error}`);
    renderPoisonError(error.message || String(error));
  } finally {
    setBusy(false);
  }
});

document.getElementById("back-to-attack").addEventListener("click", () => {
  showPage("attack");
});

document.getElementById("run-results").addEventListener("click", async () => {
  await goToResults();
});

document.getElementById("reset-workflow").addEventListener("click", async () => {
  setBusy(true);
  try {
    const data = await runAction({ action: "reset" });
    updateState(data.state);
    fileInput.value = "";
    logBox.textContent = "";
    document.getElementById("preprocess-report").classList.add("hidden");
    document.getElementById("attack-report").classList.add("hidden");
    document.getElementById("poison-question").classList.add("hidden");
    document.getElementById("poison-report").classList.add("hidden");
    document.getElementById("results-report").innerHTML = "";
    document.querySelector('input[name="want-attack"][value="no"]').checked = true;
    document.getElementById("attack-options").classList.add("hidden");
    showPage("upload");
  } catch (error) {
    appendLog(`ERROR: ${error.message || error}`);
  } finally {
    setBusy(false);
  }
});

updateAttackSliderLabels();
showPage("upload");

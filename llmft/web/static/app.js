"use strict";

/*
 * The browser half of the front-end.
 *
 * The parameter form is generated from /api/parameters rather than written out
 * here, so the labels, explanations and permitted ranges are the same ones the
 * CLI and the manual use. Adding a tunable to the engine adds it to this form.
 *
 * Every long stage is started with a POST that returns immediately, then
 * watched over server-sent events. One stage runs at a time; the server refuses
 * a second and says so.
 */

const state = {
  modelId: "",
  dataset: "",
  specs: { lora: [], sft: [] },
  quantTypes: [],
  estimateOk: false,
};

// --- helpers ---------------------------------------------------------------

const $ = (id) => document.getElementById(id);

// --- steps -----------------------------------------------------------------
//
// The fine-tune flow is a sequence of <details>. Answering one marks it done,
// collapses it and opens the next. Nothing is ever locked shut: a <summary> is
// always clickable, so any step can be reopened to change an earlier answer.

const STEPS = ["model", "dataset", "params", "estimate", "run"];

const stepOf = (name) => $(`step-${name}`);

function completeStep(name, answer) {
  const step = stepOf(name);
  step.dataset.state = "done";
  const summary = $(`answer-${name}`);
  if (summary) summary.textContent = answer || "";

  const next = STEPS[STEPS.indexOf(name) + 1];
  if (!next) return;
  step.open = false;
  const following = stepOf(next);
  following.open = true;
  if (following.dataset.state !== "done") following.dataset.state = "active";
}

/* Undo every step after `name`: an earlier answer has changed, so the ones that
   depended on it are no longer answered. Without this a new model would keep a
   stale estimate, and the Run button would still be enabled. */
function resetStepsAfter(name) {
  STEPS.slice(STEPS.indexOf(name) + 1).forEach((later) => {
    const step = stepOf(later);
    step.dataset.state = "";
    step.open = false;
    const summary = $(`answer-${later}`);
    if (summary) summary.textContent = "";
  });
  state.estimateOk = false;
  $("train-run").disabled = true;
  $("estimate-report").textContent = "";
  $("estimate-report").className = "report";
}

async function getJson(url) {
  const response = await fetch(url);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Request failed: ${response.status}`);
  return body;
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Request failed: ${response.status}`);
  return body;
}

function humanBytes(size) {
  if (!size) return "";
  const units = ["B", "KB", "MB", "GB"];
  let scaled = size;
  let index = 0;
  while (scaled >= 1024 && index < units.length - 1) { scaled /= 1024; index += 1; }
  return `${scaled.toFixed(1)} ${units[index]}`;
}

function fill(select, items, { value, label, placeholder }) {
  select.innerHTML = "";
  if (placeholder) select.append(new Option(placeholder, ""));
  items.forEach((item) => select.append(new Option(label(item), value(item))));
}

// --- tabs ------------------------------------------------------------------

$("tabs").addEventListener("click", (event) => {
  const tab = event.target.closest(".tab");
  if (!tab) return;
  document.querySelectorAll(".tab").forEach((each) => each.setAttribute("aria-selected", String(each === tab)));
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.hidden = panel.id !== `panel-${tab.dataset.panel}`;
  });
  if (tab.dataset.panel === "documentation") loadDocumentation();
});

// The manual is fetched the first time it is opened rather than on start-up:
// it is the largest thing the page can hold and most sessions never open it.
let documentationLoaded = false;
async function loadDocumentation() {
  if (documentationLoaded) return;
  documentationLoaded = true;
  try {
    const response = await fetch("/documentation");
    if (!response.ok) throw new Error(`The manual could not be loaded (${response.status}).`);
    $("documentation").innerHTML = await response.text();
    initDocumentation();
  } catch (error) {
    documentationLoaded = false;
    $("documentation").innerHTML = `<p class="report bad">${error.message} Run: python docs/build_docs.py</p>`;
  }
}

// --- searching the manual --------------------------------------------------
//
// Chapters are <details>, closed on arrival. A search opens the ones that match,
// marks every occurrence inside them, and hides the rest along with their
// contents entries. Each chapter's original markup is kept so highlighting can
// be undone exactly rather than by trying to strip <mark> elements out again.

const MINIMUM_SEARCH_LENGTH = 2;
const chapters = [];

function initDocumentation() {
  chapters.length = 0;
  document.querySelectorAll(".doc-section").forEach((section) => {
    chapters.push({ section, markup: section.innerHTML, entry: contentsEntryFor(section.id) });
  });

  $("doc-search").addEventListener("input", (event) => searchDocumentation(event.target.value.trim()));

  // A contents link points at a closed chapter, so open it on the way.
  document.querySelectorAll("#doc-contents-list a").forEach((link) => {
    link.addEventListener("click", () => {
      const target = $(link.getAttribute("href").slice(1));
      if (target) target.open = true;
    });
  });
}

function contentsEntryFor(anchor) {
  return document.querySelector(`#doc-contents-list li[data-target="${anchor}"]`);
}

function searchDocumentation(query) {
  clearSearch();
  if (query.length < MINIMUM_SEARCH_LENGTH) {
    $("doc-search-status").textContent = "";
    return;
  }

  const needle = query.toLowerCase();
  let matchedChapters = 0;
  let matches = 0;
  chapters.forEach(({ section, entry }) => {
    const found = section.textContent.toLowerCase().includes(needle) ? highlightWithin(section, query) : 0;
    section.hidden = found === 0;
    section.open = found > 0;
    if (entry) entry.hidden = found === 0;
    if (found > 0) {
      matchedChapters += 1;
      matches += found;
    }
  });

  $("doc-search-status").textContent = matches
    ? `${matches} match${matches === 1 ? "" : "es"} in ${matchedChapters} chapter${matchedChapters === 1 ? "" : "s"}`
    : "No matches";
}

function clearSearch() {
  chapters.forEach(({ section, markup, entry }) => {
    section.innerHTML = markup;
    section.hidden = false;
    section.open = false;
    if (entry) entry.hidden = false;
  });
}

/* Wrap every occurrence in <mark>, walking text nodes so the surrounding markup
   is left intact: a search-and-replace over innerHTML would happily corrupt a
   tag whose name or attribute contained the query. */
function highlightWithin(root, query) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const needle = query.toLowerCase();
  const hits = [];
  while (walker.nextNode()) {
    if (walker.currentNode.nodeValue.toLowerCase().includes(needle)) hits.push(walker.currentNode);
  }

  let count = 0;
  const pattern = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi");
  hits.forEach((node) => {
    const fragment = document.createDocumentFragment();
    node.nodeValue.split(pattern).forEach((piece, index) => {
      if (!piece) return;
      if (index % 2 === 1) {
        const mark = document.createElement("mark");
        mark.textContent = piece;
        fragment.append(mark);
        count += 1;
      } else {
        fragment.append(document.createTextNode(piece));
      }
    });
    node.replaceWith(fragment);
  });
  return count;
}

// --- environment -----------------------------------------------------------

async function loadCapabilities() {
  const data = await getJson("/api/capabilities");
  state.quantTypes = data.quantTypes;

  $("gpu").textContent = data.gpu
    ? `${data.gpu.name} — ${data.gpu.freeMemoryMb} MiB free of ${data.gpu.totalMemoryMb} MiB`
    : "No CUDA GPU detected";
  if (data.gpu && data.gpu.heavilyOccupied) {
    $("gpu").textContent += ` (${data.gpu.occupiedMemoryMb} MiB held by another application)`;
    $("gpu").classList.add("warn");
  }

  $("features").innerHTML =
    "<tr><th>Operation</th><th>Available</th><th>Reason</th></tr>" +
    data.features
      .map((item) => `<tr><td>${item.label}</td><td>${item.available ? "yes" : "no"}</td><td>${item.reason}</td></tr>`)
      .join("");

  fill($("quantize-type"), data.quantTypes, { value: (q) => q.name, label: (q) => q.label });
  $("quantize-type").value = data.defaultQuantType;
  showQuantExplanation();

  const usable = data.targets.filter((target) => target.available);
  fill($("deploy-runner"), usable, { value: (t) => t.key, label: (t) => t.label });
  if (usable.length === 0) {
    $("deploy-runner").append(new Option("No local model runner is installed", ""));
    $("deploy-run").disabled = true;
  }
}

function showQuantExplanation() {
  const chosen = state.quantTypes.find((item) => item.name === $("quantize-type").value);
  $("quantize-explanation").textContent = chosen ? chosen.explanation : "";
}
$("quantize-type").addEventListener("change", showQuantExplanation);

// --- model selection -------------------------------------------------------

$("local-only").addEventListener("change", async (event) => {
  const local = event.target.checked;
  $("model-remote").hidden = local;
  $("model-local").hidden = !local;
  state.modelId = "";
  updateChosen();
  if (local) await loadLocalModels();
});

async function loadLocalModels() {
  const models = await getJson("/api/models/local");
  fill($("model-cached"), models, {
    value: (m) => m.modelId,
    label: (m) => `${m.modelId} — ${m.description}`,
    placeholder: models.length ? "Choose a model…" : "No fine-tunable models are cached",
  });
}

$("model-cached").addEventListener("change", (event) => {
  state.modelId = event.target.value;
  updateChosen();
  if (state.modelId) completeStep("model", state.modelId);
});

// The dropdown is refreshed on every input change, as specified. Requests are
// debounced so a fast typist issues one search per pause, not one per keystroke.
let searchTimer = null;
$("model-search").addEventListener("input", (event) => {
  const query = event.target.value.trim();
  state.modelId = query;
  updateChosen();
  clearTimeout(searchTimer);
  if (query.length < 2) return;
  searchTimer = setTimeout(() => refreshSuggestions(query), 250);
});

async function refreshSuggestions(query) {
  let found = [];
  try {
    found = await getJson(`/api/models/search?q=${encodeURIComponent(query)}`);
  } catch (error) {
    return; // Suggestions are optional; a full identifier can still be typed.
  }
  $("model-options").innerHTML = found
    .map((item) => {
      const marks = [item.isLocal ? "downloaded" : "", `${item.downloads.toLocaleString()} downloads`]
        .filter(Boolean)
        .join(" · ");
      return `<option value="${item.modelId}">${marks}</option>`;
    })
    .join("");
}

function updateChosen() {
  $("model-chosen").textContent = state.modelId ? `Selected: ${state.modelId}` : "";
  $("model-continue").disabled = !state.modelId;
  resetStepsAfter("model");
  if (stepOf("model").dataset.state === "done") stepOf("model").dataset.state = "active";
}

$("model-continue").addEventListener("click", () => {
  if (state.modelId) completeStep("model", state.modelId);
});

// --- dataset ---------------------------------------------------------------

$("dataset-file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const body = new FormData();
  body.append("file", file);
  const response = await fetch("/api/dataset", { method: "POST", body });
  const report = await response.json();
  renderDataset(report);
});

function renderDataset(report) {
  if (report.valid) {
    state.dataset = report.path;
    $("dataset-report").className = "report ok";
    $("dataset-report").textContent = report.summary;
    completeStep("dataset", `${report.recordCount} conversations`);
    return;
  }
  // Reset from the dataset onwards, not from the model: collapsing this step
  // would hide the errors the user has just been shown.
  state.dataset = "";
  stepOf("dataset").dataset.state = "active";
  resetStepsAfter("dataset");
  $("dataset-report").className = "report bad";
  const issues = (report.issues || []).map((issue) => `<li>${issue.location}: ${issue.problem}</li>`).join("");
  $("dataset-report").innerHTML = `<p>${report.error || "This file cannot be used:"}</p><ul>${issues}</ul>`;
}

// --- parameters ------------------------------------------------------------

async function loadParameters() {
  state.specs = await getJson("/api/parameters");
  $("params-lora").innerHTML = `<h3>LoRA adapter</h3>${state.specs.lora.map(field).join("")}`;
  $("params-sft").innerHTML = `<h3>Training schedule</h3>${state.specs.sft.map(field).join("")}`;
  await loadPresetNames();
}

function field(spec) {
  const id = `param-${spec.name}`;
  return `<div class="param">
      <label for="${id}">${spec.label}</label>
      ${input(spec, id)}
      <p class="explanation">${spec.explanation}</p>
    </div>`;
}

function input(spec, id) {
  if (spec.kind === "bool") {
    return `<input type="checkbox" id="${id}" ${spec.default ? "checked" : ""}>`;
  }
  if (spec.kind === "choice") {
    const options = spec.choices
      .map((choice) => `<option ${choice === spec.default ? "selected" : ""}>${choice}</option>`)
      .join("");
    return `<select id="${id}">${options}</select>`;
  }
  if (spec.kind === "multi_choice") {
    const options = spec.choices
      .map((choice) => `<option ${spec.default.includes(choice) ? "selected" : ""}>${choice}</option>`)
      .join("");
    return `<select id="${id}" multiple size="${spec.choices.length}">${options}</select>`;
  }
  const step = spec.kind === "float" ? "any" : "1";
  return `<input type="number" id="${id}" value="${spec.default}" step="${step}"
            min="${spec.minimum ?? ""}" max="${spec.maximum ?? ""}">`;
}

function readGroup(specs) {
  const values = {};
  specs.forEach((spec) => {
    const element = $(`param-${spec.name}`);
    if (spec.kind === "bool") values[spec.name] = element.checked;
    else if (spec.kind === "multi_choice") values[spec.name] = [...element.selectedOptions].map((o) => o.value);
    else if (spec.kind === "int") values[spec.name] = parseInt(element.value, 10);
    else if (spec.kind === "float") values[spec.name] = parseFloat(element.value);
    else values[spec.name] = element.value;
  });
  return values;
}

function currentSettings() {
  return { lora: readGroup(state.specs.lora), sft: readGroup(state.specs.sft) };
}

function applySettings(settings) {
  [...state.specs.lora, ...state.specs.sft].forEach((spec) => {
    const value = settings.lora[spec.name] ?? settings.sft[spec.name];
    if (value === undefined) return;
    const element = $(`param-${spec.name}`);
    if (spec.kind === "bool") element.checked = value;
    else if (spec.kind === "multi_choice") {
      [...element.options].forEach((option) => { option.selected = value.includes(option.value); });
    } else element.value = value;
  });
}

// --- presets ---------------------------------------------------------------

async function loadPresetNames() {
  const names = await getJson("/api/presets");
  fill($("preset-list"), names, { value: (n) => n, label: (n) => n, placeholder: "Load a preset…" });
}

$("preset-list").addEventListener("change", async (event) => {
  if (!event.target.value) return;
  const preset = await getJson(`/api/presets/${encodeURIComponent(event.target.value)}`);
  applySettings(preset);
  $("preset-status").textContent = `Loaded '${preset.name}'.`;
});

$("preset-save").addEventListener("click", async () => {
  const name = $("preset-name").value.trim();
  if (!name) return;
  try {
    await postJson("/api/presets", { name, ...currentSettings() });
    $("preset-status").textContent = `Saved '${name}'.`;
    await loadPresetNames();
  } catch (error) {
    $("preset-status").textContent = error.message;
  }
});

// --- estimate --------------------------------------------------------------

$("params-continue").addEventListener("click", () => completeStep("params", "ready"));

$("estimate-run").addEventListener("click", async () => {
  if (!state.modelId) {
    $("estimate-report").className = "report bad";
    $("estimate-report").textContent = "Choose a model first.";
    return;
  }
  $("estimate-report").className = "report";
  $("estimate-report").textContent = "Estimating…";
  try {
    const estimate = await postJson("/api/estimate", { modelId: state.modelId, ...currentSettings() });
    renderEstimate(estimate);
  } catch (error) {
    $("estimate-report").className = "report bad";
    $("estimate-report").textContent = error.message;
  }
});

function renderEstimate(estimate) {
  state.estimateOk = estimate.fits;
  $("train-run").disabled = !estimate.fits;
  if (estimate.fits) completeStep("estimate", "fits");
  else stepOf("estimate").dataset.state = "active";
  $("estimate-report").className = `report ${estimate.fits ? "ok" : "bad"}`;
  const remedies = estimate.remedies.map((remedy) => `<li>${remedy}</li>`).join("");
  const size = (estimate.parameterCount / 1e9).toFixed(2);
  $("estimate-report").innerHTML =
    `<p>${size}B parameters, vocabulary ${estimate.vocabSize.toLocaleString()}</p>` +
    `<p>${estimate.headline}</p>` +
    (estimate.fits ? "" : `<p>Short by ${estimate.shortfall}.</p><ul>${remedies}</ul>`);
}

// --- running a stage -------------------------------------------------------

$("train-run").addEventListener("click", () =>
  start("/api/jobs/train", { modelId: state.modelId, dataset: state.dataset, ...currentSettings() }, "Fine-tuning")
);

$("quantize-run").addEventListener("click", () =>
  start("/api/jobs/quantize", { modelId: $("quantize-model").value, quantType: $("quantize-type").value }, "Quantizing")
);

$("deploy-run").addEventListener("click", () => {
  const option = $("deploy-gguf").selectedOptions[0];
  if (!option) return;
  start(
    "/api/jobs/deploy",
    {
      path: option.value,
      runner: $("deploy-runner").value,
      name: $("deploy-name").value.trim(),
      modelId: option.dataset.modelId,
      quantType: option.dataset.quantType,
    },
    "Deploying"
  );
});

async function start(url, payload, title) {
  showStatus(title, "Starting…", 0);
  try {
    await postJson(url, payload);
  } catch (error) {
    finishStatus(title, error.message, false);
    return;
  }
  watch(title);
}

function watch(title) {
  const source = new EventSource("/api/jobs/stream");
  source.addEventListener("progress", (event) => {
    const data = JSON.parse(event.data);
    showStatus(title, data.message, data.fraction);
  });
  source.addEventListener("done", (event) => {
    const job = JSON.parse(event.data);
    source.close();
    finishStatus(title, job.state === "succeeded" ? job.result : job.error, job.state === "succeeded");
    refreshArtefacts();
  });
  source.onerror = () => {
    source.close();
    finishStatus(title, "Lost contact with the server.", false);
  };
}

function showStatus(title, message, fraction) {
  $("status").hidden = false;
  $("status").className = "status";
  $("status-dismiss").hidden = true;
  $("status-title").textContent = title;
  $("status-message").textContent = message;
  $("status-fill").style.width = fraction === null || fraction === undefined ? "0%" : `${fraction * 100}%`;
  $("status-fill").classList.toggle("indeterminate", fraction === null || fraction === undefined);
}

function finishStatus(title, message, succeeded) {
  $("status").className = `status ${succeeded ? "ok" : "bad"}`;
  $("status-title").textContent = `${title} ${succeeded ? "finished" : "failed"}`;
  $("status-message").textContent = message;
  $("status-fill").style.width = succeeded ? "100%" : "0%";
  $("status-fill").classList.remove("indeterminate");
  $("status-dismiss").hidden = false;
}

$("status-dismiss").addEventListener("click", () => { $("status").hidden = true; });

// --- artefact lists --------------------------------------------------------

async function refreshArtefacts() {
  const merged = await getJson("/api/models/merged");
  fill($("quantize-model"), merged, {
    value: (m) => m.modelId,
    label: (m) => `${m.modelId} (${humanBytes(m.sizeOnDisk)})`,
    placeholder: merged.length ? null : "No merged models yet",
  });

  const gguf = await getJson("/api/models/gguf");
  $("deploy-gguf").innerHTML = "";
  if (gguf.length === 0) $("deploy-gguf").append(new Option("No quantized models yet", ""));
  gguf.forEach((item) => {
    const option = new Option(`${item.name} (${humanBytes(item.sizeOnDisk)})`, item.path);
    option.dataset.modelId = item.modelId;
    option.dataset.quantType = item.quantType;
    $("deploy-gguf").append(option);
  });
}

// --- start -----------------------------------------------------------------

(async function initialise() {
  try {
    await loadCapabilities();
    await loadParameters();
    await refreshArtefacts();
  } catch (error) {
    document.body.insertAdjacentHTML("afterbegin", `<p class="report bad">${error.message}</p>`);
  }
})();

/* LLM Finetune - Fine-tune */

/** The steps, in the order they are answered. */
const STEPS = ['model', 'dataset', 'params', 'estimate', 'run'];

/** Matches MAX_DATASET_SIZE in llmft/web/api.py. */
const MAX_DATASET_SIZE = 32 * 1024 * 1024;

/** Shorter queries match too much of Hugging Face to be worth a request. */
const MINIMUM_QUERY_LENGTH = 2;

/** One search per pause in typing, not one per keystroke. */
const SEARCH_DELAY = 250;

const state = {
    modelId: '',
    dataset: '',
    specs: { lora: [], sft: [] },
    suggestions: [],
    highlighted: -1,
};

/* ---------- Steps ----------
   Answering a step collapses it, marks it done with its answer, and opens the
   next. Changing an earlier answer un-answers every step that depended on it,
   so a new model cannot leave the previous model's estimate behind. */

/**
 * Mark a step answered and open the one after it.
 * @param {string} name
 * @param {string} answer - shown collapsed, in the step header
 */
function completeStep(name, answer) {
    const step = document.getElementById(`step-${name}`);
    step.dataset.state = 'done';
    document.getElementById(`answer-${name}`).textContent = answer || '';

    const next = STEPS[STEPS.indexOf(name) + 1];
    if (!next) return;

    step.open = false;
    const following = document.getElementById(`step-${next}`);
    following.open = true;
    if (following.dataset.state !== 'done') following.dataset.state = 'active';
}

/**
 * Un-answer every step after this one.
 * @param {string} name
 */
function resetStepsAfter(name) {
    STEPS.slice(STEPS.indexOf(name) + 1).forEach((later) => {
        const step = document.getElementById(`step-${later}`);
        step.dataset.state = '';
        step.open = false;
        document.getElementById(`answer-${later}`).textContent = '';
    });

    document.getElementById('run-btn').disabled = true;
    document.getElementById('estimate-verdict').style.display = 'none';
    document.getElementById('estimate-detail').style.display = 'none';
}

/**
 * Build one copyable detail row.
 * @param {string} label
 * @param {string} value
 * @param {string} valueClass - class for the value cell
 * @returns {string} the row markup
 */
function detailRow(label, value, valueClass = 'tech-id') {
    return `<tr>
        <td class="text-muted">${escapeHtml(label)}</td>
        <td class="${valueClass}">${escapeHtml(value)}</td>
        <td><button class="btn btn-sm copy-btn" data-value="${escapeAttribute(value)}" title="Copy">&#9112;</button></td>
    </tr>`;
}

/**
 * Wire every copy button inside a container.
 * @param {HTMLElement} container
 */
function bindCopyButtons(container) {
    container.querySelectorAll('.copy-btn').forEach((btn) => {
        btn.addEventListener('click', () => copyToClipboard(btn.dataset.value));
    });
}

/* ---------- Step 1: the base model ---------- */

/**
 * Reflect the current choice in the step header and the Continue button.
 */
function updateChosen() {
    const step = document.getElementById('step-model');
    document.getElementById('model-continue').disabled = !state.modelId;
    resetStepsAfter('model');
    if (step.dataset.state === 'done') step.dataset.state = 'active';
    document.getElementById('answer-model').textContent = '';
}

/**
 * Close the suggestion list.
 */
function closeSuggestions() {
    document.getElementById('model-dropdown').classList.remove('show');
    state.highlighted = -1;
}

/**
 * Move the highlight within the suggestion list.
 * @param {number} delta - +1 down, -1 up
 */
function moveHighlight(delta) {
    const items = document.querySelectorAll('#model-dropdown .autocomplete-item');
    if (items.length === 0) return;

    state.highlighted = (state.highlighted + delta + items.length) % items.length;
    items.forEach((item, index) => item.classList.toggle('selected', index === state.highlighted));
    items[state.highlighted].scrollIntoView({ block: 'nearest' });
}

/**
 * Adopt a suggestion as the chosen model.
 * @param {string} modelId
 */
function chooseModel(modelId) {
    state.modelId = modelId;
    document.getElementById('model-search').value = modelId;
    closeSuggestions();
    updateChosen();
    completeStep('model', modelId);
}

/**
 * Draw the suggestion list.
 * @param {Array<object>} models
 */
function renderSuggestions(models) {
    const dropdown = document.getElementById('model-dropdown');
    state.suggestions = models;
    state.highlighted = -1;

    if (models.length === 0) {
        closeSuggestions();
        return;
    }

    dropdown.innerHTML = models.map((model) => {
        const local = model.isLocal ? '<span class="badge badge-accent">downloaded</span>' : '';
        const downloads = `${model.downloads.toLocaleString()} downloads`;
        return `<div class="autocomplete-item" data-model-id="${escapeAttribute(model.modelId)}">
            <span class="item-id">${escapeHtml(model.modelId)}</span>
            ${local}
            <span class="item-name">${escapeHtml(downloads)}</span>
        </div>`;
    }).join('');

    dropdown.querySelectorAll('.autocomplete-item').forEach((item) => {
        item.addEventListener('mousedown', (event) => {
            event.preventDefault();
            chooseModel(item.dataset.modelId);
        });
    });

    dropdown.classList.add('show');
}

/**
 * Refresh the suggestions for what has been typed so far.
 * Suggestions are optional: a full identifier can always be typed by hand, so
 * a failed search closes the list rather than raising an error.
 */
async function refreshSuggestions() {
    const query = document.getElementById('model-search').value.trim();

    if (query.length < MINIMUM_QUERY_LENGTH) {
        closeSuggestions();
        return;
    }

    const response = await apiFetch(`/api/models/search?q=${encodeURIComponent(query)}`);
    if (response.status) renderSuggestions(response.data.models);
    else closeSuggestions();
}

/**
 * Load the models already in the local cache.
 */
async function loadLocalModels() {
    const select = document.getElementById('model-cached');
    const response = await apiFetch('/api/models/local');

    select.innerHTML = '';

    if (!response.status) {
        select.append(new Option(response.message || 'The local cache could not be read', ''));
        return;
    }

    const models = response.data.models;
    select.append(new Option(models.length ? 'Choose a model…' : 'No fine-tunable models are cached', ''));
    models.forEach((model) => select.append(new Option(`${model.modelId} — ${model.description}`, model.modelId)));
}

/**
 * Switch between the cached list and the Hugging Face search.
 */
async function handleLocalOnly() {
    const local = document.getElementById('local-only').checked;
    document.getElementById('model-remote').style.display = local ? 'none' : '';
    document.getElementById('model-local').style.display = local ? '' : 'none';
    state.modelId = '';
    updateChosen();
    if (local) await loadLocalModels();
}

/* ---------- Step 2: the dataset ---------- */

/**
 * Reset the drop zone to its initial state.
 */
function resetFileZone() {
    document.getElementById('file-input').value = '';
    document.getElementById('file-drop-content').style.display = '';
    document.getElementById('file-drop-selected').style.display = 'none';
}

/**
 * Render a validation report.
 * @param {object} report
 */
function renderDataset(report) {
    const verdict = document.getElementById('dataset-verdict');
    const issues = document.getElementById('dataset-issues');

    verdict.style.display = '';
    verdict.className = `verdict ${report.valid ? 'ok' : 'bad'}`;

    if (report.valid) {
        state.dataset = report.path;
        verdict.textContent = report.summary;
        issues.style.display = 'none';
        completeStep('dataset', `${report.recordCount} conversations`);
        return;
    }

    // Reset from here onwards, not from the model: collapsing this step would
    // hide the errors that have just been listed.
    state.dataset = '';
    document.getElementById('step-dataset').dataset.state = 'active';
    resetStepsAfter('dataset');
    document.getElementById('answer-dataset').textContent = '';

    verdict.textContent = `This file cannot be used: ${report.issues.length} problem${report.issues.length === 1 ? '' : 's'} found`;
    issues.style.display = '';
    const body = document.getElementById('dataset-issues-body');
    body.innerHTML = report.issues
        .map((issue) => detailRow(issue.location, issue.problem, 'text-muted'))
        .join('');
    bindCopyButtons(body);
}

/**
 * Upload and validate a chosen dataset.
 * @param {File} file
 */
async function loadDataset(file) {
    if (!file) return;

    if (file.size > MAX_DATASET_SIZE) {
        notify(`The dataset exceeds the ${formatSize(MAX_DATASET_SIZE)} limit`, 'error');
        return;
    }

    document.getElementById('file-name').textContent = `${file.name} (${formatSize(file.size)})`;
    document.getElementById('file-drop-content').style.display = 'none';
    document.getElementById('file-drop-selected').style.display = '';

    const body = new FormData();
    body.append('file', file);

    // No Content-Type header: the browser has to add the multipart boundary.
    const response = await fetch('/api/dataset', { method: 'POST', body });
    const envelope = await response.json();

    if (envelope.status) {
        renderDataset(envelope.data);
    } else {
        resetFileZone();
        notify(envelope.message || 'The dataset could not be read', 'error');
    }
}

/**
 * Wire the drop zone events.
 */
function initFileDropZone() {
    const dropZone = document.getElementById('file-drop-zone');
    const fileInput = document.getElementById('file-input');

    document.getElementById('file-size-hint').textContent = `Maximum size: ${formatSize(MAX_DATASET_SIZE)}`;

    fileInput.addEventListener('change', (event) => loadDataset(event.target.files[0]));

    document.getElementById('file-browse-link').addEventListener('click', (event) => {
        event.preventDefault();
        fileInput.click();
    });

    document.getElementById('file-change-link').addEventListener('click', (event) => {
        event.preventDefault();
        resetFileZone();
        fileInput.click();
    });

    dropZone.addEventListener('dragover', (event) => {
        event.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

    dropZone.addEventListener('drop', (event) => {
        event.preventDefault();
        dropZone.classList.remove('drag-over');
        loadDataset(event.dataTransfer.files[0]);
    });
}

/* ---------- Step 3: the parameters ----------
   The form is generated from /api/parameters rather than written out here, so
   the labels, explanations and permitted ranges are the ones the CLI and the
   manual use. Adding a tunable to the engine adds it to this form. */

/**
 * Build the control for one tunable.
 * @param {object} spec
 * @param {string} id
 * @returns {string} the control markup
 */
function control(spec, id) {
    if (spec.kind === 'bool') {
        return `<label class="toggle-row">
            <span class="toggle-switch">
                <input type="checkbox" id="${id}" ${spec.default ? 'checked' : ''}>
                <span class="toggle-slider"></span>
            </span>
            <span class="toggle-label">${escapeHtml(spec.label)}</span>
        </label>`;
    }

    if (spec.kind === 'choice' || spec.kind === 'multi_choice') {
        const many = spec.kind === 'multi_choice';
        const options = spec.choices.map((choice) => {
            const chosen = many ? spec.default.includes(choice) : choice === spec.default;
            return `<option value="${escapeAttribute(choice)}" ${chosen ? 'selected' : ''}>${escapeHtml(choice)}</option>`;
        }).join('');
        const attributes = many ? ` multiple size="${spec.choices.length}"` : '';
        return `<select class="form-control form-control-mono" id="${id}"${attributes}>${options}</select>`;
    }

    const step = spec.kind === 'float' ? 'any' : '1';
    const minimum = spec.minimum === null ? '' : ` min="${spec.minimum}"`;
    const maximum = spec.maximum === null ? '' : ` max="${spec.maximum}"`;
    return `<input type="number" class="form-control form-control-mono" id="${id}"
                   value="${escapeAttribute(spec.default)}" step="${step}"${minimum}${maximum}>`;
}

/**
 * Build the labelled field for one tunable.
 * @param {object} spec
 * @returns {string} the field markup
 */
function field(spec) {
    const id = `param-${spec.name}`;
    // A toggle carries its own label, so repeating it above would say it twice.
    const label = spec.kind === 'bool' ? '' : `<label for="${id}">${escapeHtml(spec.label)}</label>`;
    return `<div class="form-group">
        ${label}
        ${control(spec, id)}
        <p class="field-description">${escapeHtml(spec.explanation)}</p>
    </div>`;
}

/**
 * Read one group of tunables back out of the form.
 * @param {Array<object>} specs
 * @returns {object} name to value
 */
function readGroup(specs) {
    const values = {};
    specs.forEach((spec) => {
        const element = document.getElementById(`param-${spec.name}`);
        if (spec.kind === 'bool') values[spec.name] = element.checked;
        else if (spec.kind === 'multi_choice') values[spec.name] = [...element.selectedOptions].map((o) => o.value);
        else if (spec.kind === 'int') values[spec.name] = parseInt(element.value, 10);
        else if (spec.kind === 'float') values[spec.name] = parseFloat(element.value);
        else values[spec.name] = element.value;
    });
    return values;
}

/**
 * The parameters as the form currently holds them.
 * @returns {object} {lora, sft}
 */
function currentSettings() {
    return { lora: readGroup(state.specs.lora), sft: readGroup(state.specs.sft) };
}

/**
 * Write a stored set of parameters into the form.
 * @param {object} settings - {lora, sft}
 */
function applySettings(settings) {
    [...state.specs.lora, ...state.specs.sft].forEach((spec) => {
        const value = settings.lora[spec.name] ?? settings.sft[spec.name];
        if (value === undefined) return;

        const element = document.getElementById(`param-${spec.name}`);
        if (spec.kind === 'bool') element.checked = value;
        else if (spec.kind === 'multi_choice') {
            [...element.options].forEach((option) => { option.selected = value.includes(option.value); });
        } else element.value = value;
    });
}

/**
 * Load the tunables and build the form.
 */
async function loadParameters() {
    const response = await apiFetch('/api/parameters');

    if (!response.status) {
        notify(response.message || 'The parameters could not be loaded', 'error');
        return;
    }

    state.specs = response.data;
    document.getElementById('params-lora').innerHTML = state.specs.lora.map(field).join('');
    document.getElementById('params-sft').innerHTML = state.specs.sft.map(field).join('');
}

/* ---------- Presets ---------- */

/**
 * Refresh the list of stored presets.
 */
async function loadPresetNames() {
    const select = document.getElementById('preset-list');
    const response = await apiFetch('/api/presets');

    select.innerHTML = '';
    select.append(new Option('Load a preset…', ''));
    if (response.status) response.data.presets.forEach((name) => select.append(new Option(name, name)));
}

/**
 * Apply the selected preset to the form.
 */
async function handlePresetLoad() {
    const name = document.getElementById('preset-list').value;
    if (!name) return;

    const response = await apiFetch(`/api/presets/${encodeURIComponent(name)}`);

    if (response.status) {
        applySettings(response.data);
        notify(`Loaded '${response.data.name}'`, 'success');
    } else {
        notify(response.message || 'The preset could not be loaded', 'error');
    }
}

/**
 * Store the current parameters under the name given.
 */
async function handlePresetSave() {
    const name = document.getElementById('preset-name').value.trim();

    if (!name) {
        notify('Give the preset a name first', 'error');
        return;
    }

    const response = await apiFetch('/api/presets', {
        method: 'POST',
        body: { name, ...currentSettings() },
    });

    if (response.status) {
        notify(`Saved '${name}'`, 'success');
        await loadPresetNames();
    } else {
        notify(response.message || 'The preset could not be saved', 'error');
    }
}

/* ---------- Step 4: the estimate ---------- */

/**
 * Render a pre-flight verdict.
 * @param {object} estimate
 */
function renderEstimate(estimate) {
    const verdict = document.getElementById('estimate-verdict');
    const detail = document.getElementById('estimate-detail');
    const body = document.getElementById('estimate-body');

    verdict.style.display = '';
    verdict.className = `verdict ${estimate.fits ? 'ok' : 'bad'}`;
    verdict.textContent = estimate.headline;

    const rows = [
        detailRow('Parameters', `${(estimate.parameterCount / 1e9).toFixed(2)}B`),
        detailRow('Vocabulary', estimate.vocabSize.toLocaleString()),
        detailRow('Peak memory required', formatSize(estimate.requiredBytes)),
        detailRow('Available', formatSize(estimate.availableBytes)),
    ];
    if (!estimate.fits) rows.push(detailRow('Short by', estimate.shortfall));
    estimate.remedies.forEach((remedy) => rows.push(detailRow('Remedy', remedy, 'text-muted')));

    body.innerHTML = rows.join('');
    bindCopyButtons(body);
    detail.style.display = '';

    document.getElementById('run-btn').disabled = !estimate.fits;

    if (estimate.fits) {
        completeStep('estimate', 'fits');
    } else {
        document.getElementById('step-estimate').dataset.state = 'active';
    }
}

/**
 * Ask for a pre-flight estimate of this run.
 */
async function handleEstimate() {
    if (!state.modelId) {
        notify('Choose a base model first', 'error');
        return;
    }

    const button = document.getElementById('estimate-run');
    button.disabled = true;

    const response = await apiFetch('/api/estimate', {
        method: 'POST',
        body: { modelId: state.modelId, ...currentSettings() },
    });

    button.disabled = false;

    if (response.status) {
        renderEstimate(response.data);
    } else {
        const verdict = document.getElementById('estimate-verdict');
        verdict.style.display = '';
        verdict.className = 'verdict bad';
        verdict.textContent = response.message || 'The estimate could not be made.';
        document.getElementById('estimate-detail').style.display = 'none';
        document.getElementById('run-btn').disabled = true;
    }
}

/* ---------- Step 5: the run ---------- */

/**
 * Render the merged model a finished run produced.
 * @param {string} mergedPath
 */
function renderResults(mergedPath) {
    document.getElementById('results-verdict').textContent = 'Fine-tune and merge finished';

    const body = document.getElementById('results-body');
    body.innerHTML = detailRow('Base model', state.modelId) + detailRow('Merged model', mergedPath);
    bindCopyButtons(body);
}

/**
 * Start the fine-tune, which merges automatically when it succeeds.
 */
async function handleRun() {
    const button = document.getElementById('run-btn');
    button.disabled = true;

    const job = await runJob(
        '/api/jobs/train',
        { modelId: state.modelId, dataset: state.dataset, ...currentSettings() },
        'Fine-tuning',
    );

    button.disabled = false;

    if (job) {
        completeStep('run', 'finished');
        renderResults(job.result);
        showSection('results');
    }
}

/* ---------- Wiring ---------- */

document.addEventListener('DOMContentLoaded', () => {
    const search = document.getElementById('model-search');
    const debouncedSearch = debounce(refreshSuggestions, SEARCH_DELAY);

    document.getElementById('local-only').addEventListener('change', handleLocalOnly);

    document.getElementById('model-cached').addEventListener('change', (event) => {
        state.modelId = event.target.value;
        updateChosen();
        if (state.modelId) completeStep('model', state.modelId);
    });

    search.addEventListener('input', () => {
        state.modelId = search.value.trim();
        updateChosen();
        debouncedSearch();
    });

    search.addEventListener('keydown', (event) => {
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            event.preventDefault();
            moveHighlight(event.key === 'ArrowDown' ? 1 : -1);
        } else if (event.key === 'Enter') {
            event.preventDefault();
            if (state.highlighted >= 0) chooseModel(state.suggestions[state.highlighted].modelId);
            else if (state.modelId) chooseModel(state.modelId);
        } else if (event.key === 'Escape') {
            closeSuggestions();
        }
    });

    document.addEventListener('click', (event) => {
        if (!event.target.closest('.autocomplete-wrapper')) closeSuggestions();
    });

    document.getElementById('model-continue').addEventListener('click', () => {
        if (state.modelId) chooseModel(state.modelId);
    });

    initFileDropZone();

    document.getElementById('preset-list').addEventListener('change', handlePresetLoad);
    document.getElementById('preset-save').addEventListener('click', handlePresetSave);
    document.getElementById('preset-name').addEventListener('keydown', (event) => {
        if (event.key === 'Enter') handlePresetSave();
    });

    document.getElementById('params-continue').addEventListener('click', () => completeStep('params', 'ready'));
    document.getElementById('estimate-run').addEventListener('click', handleEstimate);
    document.getElementById('run-btn').addEventListener('click', handleRun);

    loadParameters().then(loadPresetNames);
});

/* LLM Finetune - Quantize */

/** Every quantization type this machine can produce, for the explanation line. */
let quantTypes = [];

/**
 * Fill a select with options, replacing whatever it held.
 * @param {HTMLSelectElement} select
 * @param {Array<object>} items
 * @param {Function} value - item to option value
 * @param {Function} label - item to option label
 * @param {string} empty - placeholder shown when there is nothing to choose
 */
function fillSelect(select, items, value, label, empty) {
    select.innerHTML = '';
    if (items.length === 0) {
        select.append(new Option(empty, ''));
        return;
    }
    items.forEach((item) => select.append(new Option(label(item), value(item))));
}

/**
 * Show the explanation of the currently selected quantization type.
 */
function handleTypeChange() {
    const chosen = quantTypes.find((item) => item.name === document.getElementById('type-select').value);
    document.getElementById('type-explanation').textContent = chosen ? chosen.explanation : '';
}

/** The name the original model goes by, without its organisation. */
function originalName(modelId) {
    return modelId.split('/').pop();
}

/**
 * Propose the selected model's own name as the quantized model's name.
 */
function handleModelChange() {
    const modelId = document.getElementById('model-select').value;
    document.getElementById('name-input').value = modelId ? originalName(modelId) : '';
    handleNameChange();
}

/**
 * Show the file the current name and type will produce.
 */
function handleNameChange() {
    const modelId = document.getElementById('model-select').value;
    const name = document.getElementById('name-input').value.trim() || (modelId ? originalName(modelId) : '');
    const type = document.getElementById('type-select').value;
    document.getElementById('name-preview').textContent = name && type ? `File: ${name}-${type}.gguf` : '';
}

/**
 * Load the merged models and the quantization types this machine supports.
 */
async function loadInputs() {
    const [capabilities, merged] = await Promise.all([loadCapabilities(), apiFetch('/api/models/merged')]);

    if (!capabilities.status || !merged.status) {
        showError(capabilities.message || merged.message || 'The available models could not be listed.');
        return;
    }

    quantTypes = capabilities.data.quantTypes;

    const models = merged.data.models;
    fillSelect(
        document.getElementById('model-select'),
        models,
        (item) => item.modelId,
        (item) => `${item.modelId}  (${formatSize(item.sizeOnDisk)})`,
        'No merged models yet — fine-tune one first',
    );

    const typeSelect = document.getElementById('type-select');
    fillSelect(typeSelect, quantTypes, (item) => item.name, (item) => item.label, 'No quantization type available');
    typeSelect.value = capabilities.data.defaultQuantType;
    handleTypeChange();
    handleModelChange();

    document.getElementById('run-btn').disabled = models.length === 0;
}

/**
 * Render the produced file.
 * @param {string} producedPath - the path the stage reported
 */
async function renderResults(producedPath) {
    const response = await apiFetch('/api/models/gguf');
    const produced = response.status
        ? response.data.files.find((file) => file.path === producedPath)
        : null;

    document.getElementById('results-verdict').textContent = 'Quantization finished';

    const name = escapeHtml(produced ? produced.name : producedPath);
    const size = produced ? formatSize(produced.sizeOnDisk) : '—';
    document.getElementById('results-body').innerHTML = `<tr>
        <td class="tech-id">${name}</td>
        <td class="text-mono">${size}</td>
        <td><button class="btn btn-sm copy-btn" data-value="${escapeAttribute(producedPath)}" title="Copy path">&#9112;</button></td>
    </tr>`;

    document.querySelectorAll('#results-body .copy-btn').forEach((btn) => {
        btn.addEventListener('click', () => copyToClipboard(btn.dataset.value));
    });
}

/**
 * Handle the quantize button click.
 */
async function handleRun() {
    const modelId = document.getElementById('model-select').value;

    if (!modelId) {
        notify('Choose a merged model first', 'error');
        return;
    }

    const runButton = document.getElementById('run-btn');
    runButton.disabled = true;

    const job = await runJob(
        '/api/jobs/quantize',
        {
            modelId,
            quantType: document.getElementById('type-select').value,
            name: document.getElementById('name-input').value.trim(),
        },
        'Quantizing',
    );

    runButton.disabled = false;

    if (job) {
        await renderResults(job.result);
        showSection('results');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('model-select').addEventListener('change', handleModelChange);
    document.getElementById('type-select').addEventListener('change', () => {
        handleTypeChange();
        handleNameChange();
    });
    document.getElementById('name-input').addEventListener('input', handleNameChange);
    document.getElementById('run-btn').addEventListener('click', handleRun);
    loadInputs();
});

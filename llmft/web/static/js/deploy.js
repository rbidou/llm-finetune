/* LLM Finetune - Deploy */

/** What to do with the model once it is registered, keyed by runner. */
const NEXT_STEPS = {
    ollama: (name) => ({ label: 'Run it with', value: `ollama run ${name}` }),
    lms: () => ({ label: 'Next step', value: 'Open LM Studio and pick the imported model' }),
};

/** Every deployment target this machine offers, available or not. */
let targets = [];

/**
 * Show why the selected runner behaves the way it does.
 */
function handleRunnerChange() {
    const chosen = targets.find((item) => item.key === document.getElementById('runner-select').value);
    const explanation = chosen && chosen.key === 'lms'
        ? 'LM Studio gets its own copy of the file; the original stays in this application\'s output directory.'
        : 'Ollama registers the file where it is, under the name given above.';
    document.getElementById('runner-explanation').textContent = chosen ? explanation : '';
}

/**
 * Load the quantized models and the runners installed on this machine.
 */
async function loadInputs() {
    const [capabilities, gguf] = await Promise.all([loadCapabilities(), apiFetch('/api/models/gguf')]);

    if (!capabilities.status || !gguf.status) {
        showError(capabilities.message || gguf.message || 'The available models could not be listed.');
        return;
    }

    targets = capabilities.data.targets.filter((item) => item.available);

    const select = document.getElementById('gguf-select');
    select.innerHTML = '';
    if (gguf.data.files.length === 0) {
        select.append(new Option('No quantized models yet — quantize one first', ''));
    }
    gguf.data.files.forEach((file) => {
        const option = new Option(`${file.name}  (${formatSize(file.sizeOnDisk)})`, file.path);
        option.dataset.modelId = file.modelId;
        option.dataset.quantType = file.quantType;
        select.append(option);
    });

    const runnerSelect = document.getElementById('runner-select');
    runnerSelect.innerHTML = '';
    targets.forEach((target) => runnerSelect.append(new Option(target.label, target.key)));
    if (targets.length === 0) {
        runnerSelect.append(new Option('No local model runner is installed', ''));
    }
    handleRunnerChange();

    document.getElementById('run-btn').disabled = targets.length === 0 || gguf.data.files.length === 0;
}

/**
 * Build one result row.
 * @param {string} label
 * @param {string} value
 * @returns {string} the row markup
 */
function resultRow(label, value) {
    return `<tr>
        <td class="text-muted">${escapeHtml(label)}</td>
        <td class="tech-id">${escapeHtml(value)}</td>
        <td><button class="btn btn-sm copy-btn" data-value="${escapeAttribute(value)}" title="Copy">&#9112;</button></td>
    </tr>`;
}

/**
 * Render what the deployment produced.
 * @param {string} outcome - the name or path the stage reported
 * @param {string} runner - the runner key it was deployed to
 * @param {string} label - the runner's display name
 */
function renderResults(outcome, runner, label) {
    document.getElementById('results-verdict').textContent = `Deployed to ${label}`;

    const next = NEXT_STEPS[runner] ? NEXT_STEPS[runner](outcome) : null;
    const body = document.getElementById('results-body');
    body.innerHTML = resultRow(runner === 'ollama' ? 'Model name' : 'Source file', outcome)
        + (next ? resultRow(next.label, next.value) : '');

    body.querySelectorAll('.copy-btn').forEach((btn) => {
        btn.addEventListener('click', () => copyToClipboard(btn.dataset.value));
    });
}

/**
 * Handle the deploy button click.
 */
async function handleRun() {
    const option = document.getElementById('gguf-select').selectedOptions[0];
    const runner = document.getElementById('runner-select').value;

    if (!option || !option.value) {
        notify('Choose a quantized model first', 'error');
        return;
    }
    if (!runner) {
        notify('No local model runner is installed', 'error');
        return;
    }

    const runButton = document.getElementById('run-btn');
    runButton.disabled = true;

    const job = await runJob(
        '/api/jobs/deploy',
        {
            path: option.value,
            runner,
            name: document.getElementById('name-input').value.trim(),
            modelId: option.dataset.modelId,
            quantType: option.dataset.quantType,
        },
        'Deploying',
    );

    runButton.disabled = false;

    if (job) {
        const label = (targets.find((item) => item.key === runner) || {}).label || runner;
        renderResults(job.result, runner, label);
        showSection('results');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('runner-select').addEventListener('change', handleRunnerChange);
    document.getElementById('run-btn').addEventListener('click', handleRun);
    document.getElementById('name-input').addEventListener('keydown', (event) => {
        if (event.key === 'Enter') handleRun();
    });
    loadInputs();
});

/* LLM Finetune - Setup */

/**
 * Show the stored settings in the form.
 * @param {object} settings - the settings payload from the API
 */
function renderSettings(settings) {
    document.getElementById('output-dir-input').value = settings.outputDir;
    document.getElementById('default-hint').textContent = `Default: ${settings.defaultOutputDir}`;
    document.getElementById('default-badge').style.display = settings.isDefault ? '' : 'none';
    document.getElementById('reset-btn').disabled = settings.isDefault;
}

/**
 * Load the current settings.
 */
async function loadSettings() {
    const response = await apiFetch('/api/settings');

    if (!response.status) {
        showError(response.message || 'The settings could not be loaded.');
        return;
    }
    renderSettings(response.data);
}

/**
 * Store a model directory; an empty value restores the default.
 * @param {string} outputDir
 */
async function saveOutputDir(outputDir) {
    const response = await apiFetch('/api/settings', { method: 'POST', body: { outputDir } });

    if (!response.status) {
        showError(response.message || 'The model directory could not be changed.');
        return;
    }
    showSection('none');
    renderSettings(response.data);
    notify(`Models will be written under ${response.data.outputDir}`, 'success');
}

/**
 * Handle the save button click.
 */
function handleSave() {
    const value = document.getElementById('output-dir-input').value.trim();

    if (!value) {
        notify('Enter a directory, or restore the default', 'error');
        return;
    }
    saveOutputDir(value);
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('save-btn').addEventListener('click', handleSave);
    document.getElementById('reset-btn').addEventListener('click', () => saveOutputDir(''));
    document.getElementById('output-dir-input').addEventListener('keydown', (event) => {
        if (event.key === 'Enter') handleSave();
    });
    loadSettings();
});

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

/* ---------- Folder browser ---------- */

/** The folder the browser is showing, as last listed by the server. */
let browsed = null;

/**
 * List one folder and show its subfolders.
 * Errors stay inside the browser: the folder on screen is still usable.
 * @param {string} path - empty opens on the current model directory
 */
async function browseTo(path) {
    const response = await apiFetch(`/api/folders?path=${encodeURIComponent(path)}`);
    const error = document.getElementById('browser-error');

    if (!response.status) {
        error.textContent = response.message || 'This folder cannot be listed.';
        return;
    }
    error.textContent = '';
    browsed = response.data;
    renderBrowser(browsed);
}

/**
 * Draw the drive selector, the current path and the subfolders.
 * @param {object} listing - the folder listing from the API
 */
function renderBrowser(listing) {
    document.getElementById('browser-path').textContent = listing.path;
    document.getElementById('up-btn').disabled = !listing.parent;

    const rootSelect = document.getElementById('root-select');
    rootSelect.innerHTML = '';
    listing.roots.forEach((root) => rootSelect.append(new Option(root, root)));
    const currentRoot = listing.roots.find((root) => listing.path.toLowerCase().startsWith(root.toLowerCase()));
    if (currentRoot) rootSelect.value = currentRoot;
    rootSelect.style.display = listing.roots.length > 1 ? '' : 'none';

    const body = document.getElementById('browser-body');
    body.innerHTML = listing.folders.length === 0
        ? '<tr><td class="text-muted">No subfolders</td></tr>'
        : listing.folders.map((folder) => `<tr class="clickable" data-path="${escapeAttribute(folder.path)}">
            <td>&#128193; ${escapeHtml(folder.name)}</td>
        </tr>`).join('');

    body.querySelectorAll('tr.clickable').forEach((row) => {
        row.addEventListener('click', () => browseTo(row.dataset.path));
    });
}

/**
 * Open the browser on whatever the field currently holds.
 */
function handleBrowse() {
    document.getElementById('browser').style.display = '';
    browseTo(document.getElementById('output-dir-input').value.trim());
}

/**
 * Close the browser without changing anything.
 */
function closeBrowser() {
    document.getElementById('browser').style.display = 'none';
}

/**
 * Put the browsed folder in the field and store it.
 */
function handleChoose() {
    if (!browsed) return;
    document.getElementById('output-dir-input').value = browsed.path;
    closeBrowser();
    saveOutputDir(browsed.path);
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('browse-btn').addEventListener('click', handleBrowse);
    document.getElementById('up-btn').addEventListener('click', () => browsed && browseTo(browsed.parent));
    document.getElementById('root-select').addEventListener('change', (event) => browseTo(event.target.value));
    document.getElementById('choose-btn').addEventListener('click', handleChoose);
    document.getElementById('cancel-btn').addEventListener('click', closeBrowser);
    document.getElementById('save-btn').addEventListener('click', handleSave);
    document.getElementById('reset-btn').addEventListener('click', () => saveOutputDir(''));
    document.getElementById('output-dir-input').addEventListener('keydown', (event) => {
        if (event.key === 'Enter') handleSave();
    });
    loadSettings();
});

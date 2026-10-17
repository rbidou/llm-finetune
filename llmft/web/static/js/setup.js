/* LLM Finetune - Setup */

/**
 * The section of the page that edits one directory setting.
 * @param {string} kind - the directory key, as the API names it
 * @returns {HTMLElement}
 */
function directoryPanel(kind) {
    return document.querySelector(`.directory-panel[data-kind="${kind}"]`);
}

/**
 * Show the stored settings in the form.
 * @param {object} settings - the settings payload from the API
 */
function renderSettings(settings) {
    Object.entries(settings.directories).forEach(([kind, directory]) => renderDirectory(kind, directory));
    renderServer(settings.server);
}

/**
 * Show one stored directory in its section.
 * @param {string} kind
 * @param {object} directory - path, default and isDefault
 */
function renderDirectory(kind, directory) {
    const panel = directoryPanel(kind);
    panel.querySelector('.directory-input').value = directory.path;
    panel.querySelector('.directory-default-hint').textContent = `Default: ${directory.default}`;
    panel.querySelector('.directory-default-badge').style.display = directory.isDefault ? '' : 'none';
    panel.querySelector('.directory-reset-btn').disabled = directory.isDefault;
}

/**
 * Show the stored web server address in the form.
 * @param {object} server - the server part of the settings payload
 */
function renderServer(server) {
    document.getElementById('host-input').value = server.host;
    document.getElementById('port-input').value = server.port;
    document.getElementById('server-default-hint').textContent = `Default: ${server.defaultHost}:${server.defaultPort}`;
    document.getElementById('server-default-badge').style.display = server.isDefault ? '' : 'none';
    document.getElementById('server-exposed-warning').style.display = server.isLoopback ? 'none' : '';
    document.getElementById('server-reset-btn').disabled = server.isDefault;
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
 * Store one directory; an empty path restores its default.
 * @param {string} kind
 * @param {string} path
 */
async function saveDirectory(kind, path) {
    const response = await apiFetch(`/api/settings/directories/${kind}`, { method: 'POST', body: { path } });

    if (!response.status) {
        showError(response.message || 'The directory could not be changed.');
        return;
    }
    showSection('none');
    renderSettings(response.data);
    notify(`Saved: ${response.data.directories[kind].path}`, 'success');
}

/**
 * Handle a directory section's save button.
 * @param {string} kind
 */
function handleDirectorySave(kind) {
    const value = directoryPanel(kind).querySelector('.directory-input').value.trim();

    if (!value) {
        notify('Enter a directory, or restore the default', 'error');
        return;
    }
    saveDirectory(kind, value);
}

/**
 * Store the web server address; empty values restore the defaults.
 * @param {string} host
 * @param {string} port
 */
async function saveServerAddress(host, port) {
    const response = await apiFetch('/api/settings/server', { method: 'POST', body: { host, port } });

    if (!response.status) {
        showError(response.message || 'The web server address could not be changed.');
        return;
    }
    showSection('none');
    renderSettings(response.data);
    notify(`The web server will listen on ${response.data.server.url} from its next start`, 'success');
}

/**
 * Handle the web server save button click.
 */
function handleServerSave() {
    saveServerAddress(
        document.getElementById('host-input').value.trim(),
        document.getElementById('port-input').value.trim(),
    );
}

/* ---------- Folder browser ---------- */

/** The folder the browser is showing, as last listed by the server. */
let browsed = null;

/** The directory section the browser was opened from. */
let browsingFor = '';

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
 * Open the browser inside one directory section, on what its field holds.
 * @param {string} kind
 */
function handleBrowse(kind) {
    const panel = directoryPanel(kind);
    const browser = document.getElementById('browser');

    browsingFor = kind;
    panel.querySelector('.panel-body').insertBefore(browser, panel.querySelector('.directory-actions'));
    browser.style.display = '';
    browseTo(panel.querySelector('.directory-input').value.trim());
}

/**
 * Close the browser without changing anything.
 */
function closeBrowser() {
    document.getElementById('browser').style.display = 'none';
}

/**
 * Put the browsed folder in the field it was opened for, and store it.
 */
function handleChoose() {
    if (!browsed || !browsingFor) return;
    directoryPanel(browsingFor).querySelector('.directory-input').value = browsed.path;
    closeBrowser();
    saveDirectory(browsingFor, browsed.path);
}

/**
 * Wire one directory section's controls.
 * @param {HTMLElement} panel
 */
function initDirectoryPanel(panel) {
    const { kind } = panel.dataset;

    panel.querySelector('.directory-browse-btn').addEventListener('click', () => handleBrowse(kind));
    panel.querySelector('.directory-save-btn').addEventListener('click', () => handleDirectorySave(kind));
    panel.querySelector('.directory-reset-btn').addEventListener('click', () => saveDirectory(kind, ''));
    panel.querySelector('.directory-input').addEventListener('keydown', (event) => {
        if (event.key === 'Enter') handleDirectorySave(kind);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.directory-panel').forEach(initDirectoryPanel);
    document.getElementById('up-btn').addEventListener('click', () => browsed && browseTo(browsed.parent));
    document.getElementById('root-select').addEventListener('change', (event) => browseTo(event.target.value));
    document.getElementById('choose-btn').addEventListener('click', handleChoose);
    document.getElementById('cancel-btn').addEventListener('click', closeBrowser);
    document.getElementById('server-save-btn').addEventListener('click', handleServerSave);
    document.getElementById('server-reset-btn').addEventListener('click', () => saveServerAddress('', ''));
    ['host-input', 'port-input'].forEach((id) => {
        document.getElementById(id).addEventListener('keydown', (event) => {
            if (event.key === 'Enter') handleServerSave();
        });
    });
    loadSettings();
});

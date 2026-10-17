/* LLM Finetune - Shared JavaScript utilities */

/** Bytes in each unit step, largest first, for formatSize. */
const SIZE_UNITS = ['TB', 'GB', 'MB', 'KB'];

/**
 * Fetch JSON from an API endpoint.
 * Object bodies are serialized automatically.
 * @param {string} url
 * @param {object} options - fetch options
 * @returns {Promise<object>} the parsed API envelope
 */
async function apiFetch(url, options = {}) {
    const defaults = {
        headers: { 'Content-Type': 'application/json' },
    };
    const merged = { ...defaults, ...options };
    if (merged.body && typeof merged.body === 'object') {
        merged.body = JSON.stringify(merged.body);
    }
    const response = await fetch(url, merged);
    return response.json();
}

/**
 * Show a transient notification toast in the top-right corner.
 * @param {string} message
 * @param {'success'|'error'|'info'} type
 */
function notify(message, type = 'info') {
    const el = document.createElement('div');
    el.className = `notification ${type}`;
    el.textContent = message;
    document.body.appendChild(el);
    setTimeout(() => {
        el.style.opacity = '0';
        el.style.transition = 'opacity 0.3s';
        setTimeout(() => el.remove(), 300);
    }, 3000);
}

/**
 * Copy text to the clipboard and confirm with a notification.
 * @param {string} text
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        notify('Copied to clipboard', 'success');
    }).catch(() => {
        notify('Failed to copy', 'error');
    });
}

/**
 * Escape HTML special characters for safe interpolation into innerHTML.
 * @param {string} str
 * @returns {string}
 */
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str === null || str === undefined ? '' : str;
    return div.innerHTML;
}

/**
 * Escape a value for interpolation into a double-quoted HTML attribute.
 * escapeHtml() covers element content, where a quote is harmless; inside an
 * attribute the quote is what delimits the value, so it has to go too.
 * @param {string} str
 * @returns {string}
 */
function escapeAttribute(str) {
    return escapeHtml(str).replace(/"/g, '&quot;');
}

/**
 * Trigger a client-side file download.
 * @param {string|Blob} content
 * @param {string} filename
 * @param {string} mimeType
 */
function downloadFile(content, filename, mimeType = 'application/octet-stream') {
    const blob = content instanceof Blob ? content : new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

/**
 * Format a byte count in human-readable form.
 * Carries GB and TB as well as the reference MB: the artefacts this
 * application produces are model weights, and a 7B merge is 14 GB.
 * @param {number} bytes
 * @returns {string}
 */
function formatSize(bytes) {
    if (!bytes) return '0 B';
    let scale = 1024 ** SIZE_UNITS.length;
    for (const unit of SIZE_UNITS) {
        if (bytes >= scale) return `${(bytes / scale).toFixed(scale > 1024 * 1024 ? 2 : 1)} ${unit}`;
        scale /= 1024;
    }
    return `${bytes} B`;
}

/**
 * Debounce a function call.
 * @param {Function} fn
 * @param {number} delay - milliseconds
 * @returns {Function}
 */
function debounce(fn, delay = 300) {
    let timer = null;
    return function (...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), delay);
    };
}

/* ---------- The standard page regions ----------
   Every page carrying an operation uses the same four IDs, so the show/hide
   logic is written once here rather than copied into each page script. */

/**
 * Show one page section and hide the others.
 * @param {'loading'|'results'|'error'|'none'} section
 */
function showSection(section) {
    ['loading', 'results', 'error'].forEach((name) => {
        const el = document.getElementById(`${name}-section`);
        if (el) el.style.display = section === name ? '' : 'none';
    });
}

/**
 * Render an API failure inline, where the page recipe puts it.
 * @param {string} message
 */
function showError(message) {
    document.getElementById('error-message').textContent = message;
    showSection('error');
}

/* ---------- Capabilities ----------
   What this machine can do is asked for once per page load and shared: the
   sidebar wants the GPU line, and three of the pages want the same answer's
   feature gates, quantization types or deployment targets. */

let capabilitiesPromise = null;

/**
 * Load this machine's capabilities, once per page.
 * @returns {Promise<object>} the API envelope
 */
function loadCapabilities() {
    if (!capabilitiesPromise) capabilitiesPromise = apiFetch('/api/capabilities');
    return capabilitiesPromise;
}

/**
 * Fill the sidebar's GPU line.
 */
async function initSidebarStatus() {
    const target = document.getElementById('gpu-status');
    if (!target) return;

    const response = await loadCapabilities();
    const gpu = response.status ? response.data.gpu : null;

    if (!gpu) {
        target.textContent = 'none detected';
        return;
    }

    target.textContent = `${gpu.name}\n${gpu.freeMemoryMb} / ${gpu.totalMemoryMb} MiB free`;
    target.style.whiteSpace = 'pre-line';
    if (gpu.heavilyOccupied) {
        target.classList.add('warn');
        target.textContent += `\n${gpu.occupiedMemoryMb} MiB held elsewhere`;
    }
}

/* ---------- Long stages ----------
   Training, merging, quantizing and deploying all take far longer than a
   request may be held open. The POST returns a job identifier immediately and
   the stage reports over server-sent events, which every page renders into the
   same progress panel. */

/**
 * Update the progress panel.
 * @param {string} title
 * @param {string} message
 * @param {number|null} fraction - null when the stage cannot report a total
 */
function setProgress(title, message, fraction) {
    const known = fraction !== null && fraction !== undefined;
    document.getElementById('progress-title').textContent = title;
    document.getElementById('progress-message').textContent = message;
    const fill = document.getElementById('progress-fill');
    fill.classList.toggle('indeterminate', !known);
    fill.style.width = known ? `${fraction * 100}%` : '';
}

/**
 * Watch the running stage until it stops.
 * @param {string} title - shown in the progress panel
 * @returns {Promise<object>} the finished job payload
 */
function watchJob(title) {
    return new Promise((resolve) => {
        const source = new EventSource('/api/jobs/stream');

        source.addEventListener('progress', (event) => {
            const data = JSON.parse(event.data);
            setProgress(title, data.message, data.fraction);
        });

        source.addEventListener('done', (event) => {
            source.close();
            resolve(JSON.parse(event.data));
        });

        source.onerror = () => {
            source.close();
            resolve({ state: 'failed', error: 'Lost contact with the server.' });
        };
    });
}

/**
 * Start a stage and watch it through to its outcome.
 * Failures are rendered inline; the caller only handles success.
 * @param {string} url - the job endpoint
 * @param {object} payload - the request body
 * @param {string} title - shown in the progress panel and the toasts
 * @returns {Promise<object|null>} the succeeded job, or null on any failure
 */
async function runJob(url, payload, title) {
    showSection('loading');
    setProgress(title, 'Starting…', null);
    notify(`${title} started`, 'info');

    const response = await apiFetch(url, { method: 'POST', body: payload });

    if (!response.status) {
        showError(response.message || `${title} could not start.`);
        return null;
    }

    const job = await watchJob(title);

    if (job.state !== 'succeeded') {
        showError(job.error || `${title} failed.`);
        notify(`${title} failed`, 'error');
        return null;
    }

    notify(`${title} finished`, 'success');
    return job;
}

/**
 * Open or close the drawer menu of the compact layout.
 * @param {boolean} open
 * @param {boolean} restoreFocus - give the focus back to the toggle when closing
 */
function setNavOpen(open, restoreFocus = true) {
    const toggle = document.getElementById('nav-toggle');
    const wasOpen = document.body.classList.contains('nav-open');
    document.body.classList.toggle('nav-open', open);
    toggle.setAttribute('aria-expanded', String(open));
    toggle.setAttribute('aria-label', open ? 'Close the menu' : 'Open the menu');
    if (open) {
        const first = document.querySelector('#sidebar .sidebar-nav a');
        if (first) setTimeout(() => first.focus(), 50);
    } else if (wasOpen && restoreFocus) {
        toggle.focus();
    }
}

/**
 * Wire the drawer menu of the compact layout (< 1400px): toggle button, backdrop, Esc,
 * closing on navigation and when the window becomes wide again.
 */
function initNavDrawer() {
    const toggle = document.getElementById('nav-toggle');
    if (!toggle) return;
    toggle.addEventListener('click', () => setNavOpen(!document.body.classList.contains('nav-open')));
    document.getElementById('nav-backdrop').addEventListener('click', () => setNavOpen(false));
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && document.body.classList.contains('nav-open')) setNavOpen(false);
    });
    document.querySelectorAll('#sidebar a').forEach(link => {
        link.addEventListener('click', () => setNavOpen(false, false));
    });
    // Same breakpoint as the compact layout in app.css
    window.matchMedia('(min-width: 1400px)').addEventListener('change', (e) => {
        if (e.matches) setNavOpen(false, false);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    initSidebarStatus();
    initNavDrawer();
});

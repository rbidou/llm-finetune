/* LLM Finetune - Environment */

/**
 * Build one availability row.
 * @param {object} item - a feature or a deployment target
 * @returns {string} the row markup
 */
function availabilityRow(item) {
    const label = escapeHtml(item.label);
    const reason = escapeHtml(item.available ? '' : item.reason);
    const badge = item.available
        ? '<span class="badge badge-success">yes</span>'
        : '<span class="badge badge-danger">no</span>';
    const copyValue = escapeAttribute(item.available ? `${item.label}: available` : `${item.label}: ${item.reason}`);

    return `<tr>
        <td>${label}</td>
        <td>${badge}</td>
        <td class="text-muted">${reason || '&mdash;'}</td>
        <td><button class="btn btn-sm copy-btn" data-value="${copyValue}" title="Copy">&#9112;</button></td>
    </tr>`;
}

/**
 * Wire every copy button inside a table body.
 * @param {HTMLElement} tbody
 */
function bindCopyButtons(tbody) {
    tbody.querySelectorAll('.copy-btn').forEach((btn) => {
        btn.addEventListener('click', () => copyToClipboard(btn.dataset.value));
    });
}

/**
 * Render the capability report.
 * @param {object} data
 */
function renderResults(data) {
    const gpu = data.gpu;
    const usable = data.features.filter((item) => item.available).length;

    document.getElementById('stat-gpu').textContent = gpu ? gpu.name : 'none detected';
    document.getElementById('stat-free').textContent = gpu ? `${gpu.freeMemoryMb} MiB` : '—';
    document.getElementById('stat-total').textContent = gpu ? `${gpu.totalMemoryMb} MiB` : '—';
    document.getElementById('stat-features').textContent = `${usable}/${data.features.length}`;

    const features = document.getElementById('features-body');
    features.innerHTML = data.features.map(availabilityRow).join('');
    bindCopyButtons(features);

    const targets = document.getElementById('targets-body');
    targets.innerHTML = data.targets.map(availabilityRow).join('');
    bindCopyButtons(targets);
}

/**
 * Load and display what this machine can do.
 */
async function handleLoad() {
    const response = await loadCapabilities();

    if (response.status) {
        renderResults(response.data);
        showSection('results');
    } else {
        showError(response.message || 'This machine could not be inspected.');
    }
}

document.addEventListener('DOMContentLoaded', handleLoad);

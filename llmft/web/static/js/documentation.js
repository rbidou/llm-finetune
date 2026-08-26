/* LLM Finetune - Documentation */

/** Shorter queries match almost every chapter, so they are ignored. */
const MINIMUM_SEARCH_LENGTH = 2;

/** Characters that would otherwise be read as regular expression syntax. */
const REGEX_SPECIALS = /[.*+?^${}()|[\]\\]/g;

/**
 * Every chapter, with the markup it arrived with.
 * Keeping the original lets highlighting be undone exactly, rather than by
 * trying to strip <mark> elements back out of edited markup.
 */
const chapters = [];

/**
 * The contents entry pointing at a chapter.
 * @param {string} anchor - the chapter's id
 * @returns {HTMLElement|null}
 */
function contentsEntryFor(anchor) {
    return document.querySelector(`#doc-contents-list li[data-target="${anchor}"]`);
}

/**
 * Wrap every occurrence of a query in <mark>.
 * Text nodes are walked rather than the markup searched and replaced: a
 * replacement over innerHTML would happily corrupt a tag whose name or
 * attribute contained the query.
 * @param {HTMLElement} root
 * @param {string} query
 * @returns {number} how many occurrences were marked
 */
function highlightWithin(root, query) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const needle = query.toLowerCase();
    const hits = [];

    while (walker.nextNode()) {
        if (walker.currentNode.nodeValue.toLowerCase().includes(needle)) hits.push(walker.currentNode);
    }

    let count = 0;
    const pattern = new RegExp(`(${query.replace(REGEX_SPECIALS, '\\$&')})`, 'gi');

    hits.forEach((node) => {
        const fragment = document.createDocumentFragment();
        node.nodeValue.split(pattern).forEach((piece, index) => {
            if (!piece) return;
            if (index % 2 === 1) {
                const mark = document.createElement('mark');
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

/**
 * Restore every chapter to its unhighlighted, closed, visible state.
 */
function clearSearch() {
    chapters.forEach(({ section, markup, entry }) => {
        section.innerHTML = markup;
        section.hidden = false;
        section.open = false;
        if (entry) entry.hidden = false;
    });
}

/**
 * Show only the chapters matching the query, with every occurrence marked.
 * @param {string} query
 */
function searchDocumentation(query) {
    clearSearch();

    const status = document.getElementById('doc-search-status');

    if (query.length < MINIMUM_SEARCH_LENGTH) {
        status.textContent = '';
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

    status.textContent = matches
        ? `${matches} match${matches === 1 ? '' : 'es'} in ${matchedChapters} chapter${matchedChapters === 1 ? '' : 's'}`
        : 'No matches';
}

/**
 * Index the chapters and wire the search box and the contents links.
 */
function initDocumentation() {
    const search = document.getElementById('doc-search');
    if (!search) return;

    document.querySelectorAll('.doc-section').forEach((section) => {
        chapters.push({ section, markup: section.innerHTML, entry: contentsEntryFor(section.id) });
    });

    search.addEventListener('input', (event) => searchDocumentation(event.target.value.trim()));

    // A contents link points at a closed chapter, so open it on the way.
    document.querySelectorAll('#doc-contents-list a').forEach((link) => {
        link.addEventListener('click', () => {
            const target = document.getElementById(link.getAttribute('href').slice(1));
            if (target) target.open = true;
        });
    });
}

document.addEventListener('DOMContentLoaded', initDocumentation);

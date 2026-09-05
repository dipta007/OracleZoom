// Adapted from the Academic Project Page Template, trimmed to what this page uses:
// the More Works dropdown, the BibTeX copy button, scroll-to-top, and sticky-nav highlighting.

function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');

    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
    button.setAttribute('aria-expanded', dropdown.classList.contains('show'));
}

// Close dropdown when clicking outside
document.addEventListener('click', function (event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');

    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
        button.setAttribute('aria-expanded', 'false');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        if (dropdown) dropdown.classList.remove('show');
        if (button) { button.classList.remove('active'); button.setAttribute('aria-expanded', 'false'); }
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    if (!bibtexElement) return;

    const done = function () {
        button.classList.add('copied');
        copyText.textContent = 'Copied';
        setTimeout(function () {
            button.classList.remove('copied');
            copyText.textContent = 'Copy';
        }, 2000);
    };

    navigator.clipboard.writeText(bibtexElement.textContent.trim()).then(done).catch(function () {
        const textArea = document.createElement('textarea');
        textArea.value = bibtexElement.textContent.trim();
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        done();
    });
}

// Scroll to top
function scrollToTop() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Show the scroll button and sticky nav after the hero, and mark the section being read
window.addEventListener('scroll', function () {
    const scrollButton = document.querySelector('.scroll-to-top');
    const stickyNav = document.getElementById('stickyNav');

    if (window.pageYOffset > 300) {
        if (scrollButton) scrollButton.classList.add('visible');
        if (stickyNav) stickyNav.classList.add('visible');
    } else {
        if (scrollButton) scrollButton.classList.remove('visible');
        if (stickyNav) stickyNav.classList.remove('visible');
    }

    if (stickyNav) {
        const sections = document.querySelectorAll('section[id]');
        const scrollPos = window.pageYOffset + 80;
        sections.forEach(function (section) {
            const top = section.offsetTop;
            const height = section.offsetHeight;
            const id = section.getAttribute('id');
            const link = stickyNav.querySelector('a[href="#' + id + '"]');
            if (link) {
                if (scrollPos >= top && scrollPos < top + height) {
                    link.classList.add('active');
                } else {
                    link.classList.remove('active');
                }
            }
        });
    }
});

// ─────────────────────────────────────────────────────────────────
// dashboard/landing.js — public landing page logic
//
// Handles these things, all scoped to landing.html only:
//   1. Logged-in vs logged-out nav/CTA state (reuses the same
//      /api/me check app.js already uses on the dashboard).
//   2. Mobile menu toggle.
//   3. Nav background-on-scroll effect.
//   4. One-time hero entrance sequence on page load (fade-in-down
//      title/subtitle, fade-in-left/right CTA buttons).
//   5. Scroll reveal for .lp-reveal / .lp-stagger sections, including
//      sequenced sibling delays (About, Features, How It Works, Tech)
//      and per-child stagger (Feature cards, How It Works steps,
//      tech pills).
//   6. "Login Required" modal, shown when a logged-out visitor
//      clicks "Try Live Demo" or "Get Started".
//   7. "Watch Demo" smooth scroll to the About section.
//   8. About section video card play/pause.
// ─────────────────────────────────────────────────────────────────

// ── Hero entrance sequence (fires once, on page load) ────────────
// Adds .lp-enter-play to <body>, which triggers the staggered
// fade-in-down (title/subtitle) and fade-in-left/right (buttons)
// keyframes defined in landing.css. A tiny rAF delay (rather than
// firing synchronously) avoids the very first frame sometimes
// skipping the animation in some browsers when the class is present
// before the page has painted anything yet.
requestAnimationFrame(() => {
  requestAnimationFrame(() => {
    document.body.classList.add('lp-enter-play');
  });
});

// IDs of every button that should open the demo / live-recognition
// experience: "Try Live Demo" (nav, desktop + mobile) and
// "Get Started" (hero + final CTA). Logged out -> Login Required
// modal. Logged in -> straight to the dashboard.
const DEMO_TRIGGER_IDS = [
  'nav-try-demo',
  'nav-try-demo-mobile',
  'hero-cta-primary',
  'cta-primary',
];

let isLoggedIn = false;

// ── Logged-in vs logged-out nav/CTA state ────────────────────────
// Reuses the same /api/me check app.js already uses on the dashboard,
// so a logged-in visitor sees an empty nav action area instead of
// "Try Live Demo" / "Get Started", and the hero/CTA buttons go
// straight to the dashboard instead of opening the Login Required modal.
(async function setNavState() {
  try {
    const res = await fetch('/api/me');
    if (!res.ok) return; // logged out: leave the default buttons + modal wiring below
    isLoggedIn = true;

    document.getElementById('lp-nav-actions').innerHTML = '';
    document.getElementById('lp-mobile-actions').innerHTML = '';

    // Hide Login / Register links in the footer and CTA section —
    // a logged-in visitor (including guest mode) has no use for them.
    document.querySelectorAll('a[href="login.html"], a[href="register.html"]')
      .forEach(a => { a.style.display = 'none'; });
  } catch (err) {
    // Network hiccup - fall back to the default logged-out state, no harm done.
  }
})();

// ── Login Required modal ─────────────────────────────────────────
const modalOverlay = document.getElementById('login-modal-overlay');
const modalEl       = modalOverlay ? modalOverlay.querySelector('.lp-modal') : null;
let lastFocusedEl   = null;

function openLoginModal() {
  if (!modalOverlay) return;
  lastFocusedEl = document.activeElement;
  modalOverlay.classList.add('show');
  modalOverlay.setAttribute('aria-hidden', 'false');
  document.body.style.overflow = 'hidden';
  if (modalEl) modalEl.focus();
}

function closeLoginModal() {
  if (!modalOverlay) return;
  modalOverlay.classList.remove('show');
  modalOverlay.setAttribute('aria-hidden', 'true');
  document.body.style.overflow = '';
  if (lastFocusedEl && typeof lastFocusedEl.focus === 'function') {
    lastFocusedEl.focus();
  }
}

// Clicking "Try Live Demo" / "Get Started" while logged out opens
// the modal instead of navigating anywhere. While logged in, those
// same buttons go straight to the dashboard (no modal). Bound
// immediately (not waiting on the /api/me check below) so a fast
// click during that brief round-trip still does something sensible -
// it defaults to "logged out" -> modal, which is always a safe
// default; isLoggedIn flips to true once the auth check resolves,
// and these same listeners then redirect to the dashboard instead.
function bindDemoTriggers() {
  DEMO_TRIGGER_IDS.forEach(id => {
    const elx = document.getElementById(id);
    if (!elx) return; // may not exist post logged-in DOM swap (nav buttons were replaced)
    elx.addEventListener('click', () => {
      if (isLoggedIn) {
        window.location.href = '/dashboard';
      } else {
        openLoginModal();
      }
    });
  });
}
bindDemoTriggers();

if (modalOverlay) {
  // Cancel button closes the modal.
  const cancelBtn = document.getElementById('login-modal-cancel');
  if (cancelBtn) cancelBtn.addEventListener('click', closeLoginModal);

  // "X" close button.
  const closeBtn = document.getElementById('login-modal-close');
  if (closeBtn) closeBtn.addEventListener('click', closeLoginModal);

  // Clicking the overlay itself (outside the modal card) closes it.
  modalOverlay.addEventListener('click', (e) => {
    if (e.target === modalOverlay) closeLoginModal();
  });

  // ESC key closes it.
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modalOverlay.classList.contains('show')) {
      closeLoginModal();
    }
  });

  // "Go to Login" is a real link to the existing login route - no
  // need to intercept its click, it navigates normally.
}

// ── Mobile menu toggle ────────────────────────────────────────────
const burger = document.getElementById('lp-burger');
const mobileMenu = document.getElementById('lp-mobile-menu');
burger.addEventListener('click', () => {
  mobileMenu.classList.toggle('show');
  burger.classList.toggle('active');
});
mobileMenu.querySelectorAll('a').forEach(a => {
  a.addEventListener('click', () => {
    mobileMenu.classList.remove('show');
    burger.classList.remove('active');
  });
});

// ── Nav background on scroll ──────────────────────────────────────
const nav = document.getElementById('lp-nav');
window.addEventListener('scroll', () => {
  nav.classList.toggle('lp-nav-scrolled', window.scrollY > 8);
}, { passive: true });

// ── Back to top button ──────────────────────────────────────────────
const backToTop = document.getElementById('lp-back-to-top');
if (backToTop) {
  window.addEventListener('scroll', () => {
    backToTop.classList.toggle('lp-visible', window.scrollY > 300);
  }, { passive: true });

  backToTop.addEventListener('click', () => {
    const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    window.scrollTo({ top: 0, behavior: reduceMotion ? 'auto' : 'smooth' });
  });
}

// ── Active nav underline (scrollspy) ──────────────────────────────
// A single .lp-nav-indicator bar slides between links (via left/width,
// animated in CSS) instead of each link drawing its own underline -
// that's what makes it glide on both click and scroll instead of
// snapping. Desktop links get the sliding bar; the mobile menu (a
// vertical list) just gets the .lp-active class for its own static
// per-link underline.
const navLinksDesktop = Array.from(document.querySelectorAll('.lp-links a'));
const navLinksMobile  = Array.from(document.querySelectorAll('.lp-mobile-menu a'));
const navIndicator    = document.getElementById('lp-nav-indicator');

const navSections = ['hero', 'about', 'features', 'how-it-works']
  .map(id => document.getElementById(id))
  .filter(Boolean);

function setActiveNav(id) {
  navLinksDesktop.forEach(a => a.classList.toggle('lp-active', a.getAttribute('href') === `#${id}`));
  navLinksMobile.forEach(a => a.classList.toggle('lp-active', a.getAttribute('href') === `#${id}`));

  const activeLink = navLinksDesktop.find(a => a.classList.contains('lp-active'));
  if (activeLink && navIndicator) {
    navIndicator.style.left = `${activeLink.offsetLeft}px`;
    navIndicator.style.width = `${activeLink.offsetWidth}px`;
    navIndicator.style.opacity = '1';
  }
}

// Position the indicator once layout/fonts have settled, so it isn't
// measuring against a pre-webfont-swap width.
window.addEventListener('load', () => setActiveNav('hero'));
requestAnimationFrame(() => setActiveNav('hero'));

if ('IntersectionObserver' in window && navSections.length) {
  // Fires when a section crosses the vertical middle of the viewport,
  // rather than on any overlap - keeps the indicator from flickering
  // between two sections while both are partially visible.
  const navSpyObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) setActiveNav(entry.target.id);
    });
  }, { rootMargin: '-45% 0px -45% 0px', threshold: 0 });

  navSections.forEach(section => navSpyObserver.observe(section));
}

// Instant feedback on click, ahead of the smooth scroll settling -
// the scrollspy above will simply confirm the same section moments
// later, so there's no fight between the two.
[...navLinksDesktop, ...navLinksMobile].forEach(a => {
  a.addEventListener('click', () => {
    const id = a.getAttribute('href').slice(1);
    setActiveNav(id);
  });
});

// Re-measure on resize - link widths/positions shift with the
// viewport (and at the 860px breakpoint the desktop nav disappears
// entirely, though offsetLeft/Width still resolve harmlessly).
window.addEventListener('resize', () => {
  const activeLink = navLinksDesktop.find(a => a.classList.contains('lp-active'));
  if (activeLink) setActiveNav(activeLink.getAttribute('href').slice(1));
});

// ── Scroll reveal ──────────────────────────────────────────────────
// Three layers, all driven by one IntersectionObserver:
//
// 1. Per-child stagger: any .lp-stagger container gets a
//    --lp-stagger-i index written on each direct child up front, so
//    landing.css's transition-delay (var(--lp-stagger-step)) staggers
//    them automatically once .lp-revealed is added to the parent.
//    Used for the 6 Feature cards (100ms/card), the 4 How It Works
//    steps + connecting lines (200ms/step), and the 8 tech pills
//    (50ms/pill).
//
// 2. Sequenced siblings: groups of elements sharing a
//    [data-reveal-sequence="name"] wrapper are revealed in
//    [data-reveal-step] order with a fixed delay between each step,
//    rather than all firing the instant the group scrolls into view.
//    Used for About (video first, then text - 300ms apart) and for
//    the Features / How It Works headers (header first, then the
//    staggered body - 250ms apart).
//
// 3. Everything else (plain .lp-reveal / .lp-reveal-left/right /
//    .lp-footer-fade not part of a sequence) just gets .lp-revealed
//    the moment it's 15% into the viewport, as before.

// Layer 1: assign stagger indices once, before anything is observed.
document.querySelectorAll('.lp-stagger').forEach(container => {
  Array.from(container.children).forEach((child, i) => {
    child.style.setProperty('--lp-stagger-i', i);
  });
});

// Layer 2: build a lookup of sequence groups, keyed by group name,
// each holding its steps in order with the delay to use between them.
const SEQUENCE_DELAY_MS = 300; // About: video -> text
const HEADER_THEN_BODY_DELAY_MS = 250; // Features / How It Works: header -> staggered body

const sequenceGroups = {};
document.querySelectorAll('[data-reveal-sequence]').forEach(group => {
  const name = group.dataset.revealSequence;
  const steps = Array.from(group.querySelectorAll('[data-reveal-step]'))
    .sort((a, b) => Number(a.dataset.revealStep) - Number(b.dataset.revealStep));
  if (steps.length) sequenceGroups[name] = steps;
});

function revealSequence(steps, delayMs) {
  steps.forEach((stepEl, i) => {
    setTimeout(() => stepEl.classList.add('lp-revealed'), i * delayMs);
  });
}

// Elements that belong to a sequence are revealed by revealSequence()
// above, not individually - skip them in the generic observer loop.
const sequencedElements = new Set(Object.values(sequenceGroups).flat());

function revealElement(elx) {
  if (sequencedElements.has(elx)) return; // handled by its sequence group instead
  elx.classList.add('lp-revealed');
}

if ('IntersectionObserver' in window) {
  // One observer for plain elements (Layer 3).
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        revealElement(entry.target);
        revealObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.15 });

  document.querySelectorAll('.lp-reveal, .lp-reveal-left, .lp-reveal-right, .lp-reveal-up, .lp-reveal-fade, .lp-stagger, .lp-footer-fade')
    .forEach(elx => revealObserver.observe(elx));

  // One observer per sequence group (Layer 2) - watches the group's
  // wrapper, and once it enters the viewport, fires every step in
  // order with the appropriate delay between them.
  document.querySelectorAll('[data-reveal-sequence]').forEach(group => {
    const name = group.dataset.revealSequence;
    const steps = sequenceGroups[name];
    if (!steps) return;

    const delay = name === 'about' ? SEQUENCE_DELAY_MS : HEADER_THEN_BODY_DELAY_MS;

    const seqObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          revealSequence(steps, delay);
          seqObserver.unobserve(entry.target);
        }
      });
    }, { threshold: 0.15 });

    seqObserver.observe(group);
  });
} else {
  // No IntersectionObserver support: just show everything immediately.
  document.querySelectorAll('.lp-reveal, .lp-reveal-left, .lp-reveal-right, .lp-reveal-up, .lp-reveal-fade, .lp-stagger, .lp-footer-fade, [data-reveal-step]')
    .forEach(elx => elx.classList.add('lp-revealed'));
}

// ── "Watch Demo" smooth scroll to the About section ──────────────
// href="#about" already works as a plain-HTML fallback; this adds
// the explicit scrollIntoView({behavior:'smooth'}) call on top of it
// and prevents the default jump-to-anchor so the smooth scroll is
// the only thing that happens (no double-scroll / URL hash flicker).
const watchDemoBtn = document.getElementById('hero-cta-secondary');
const aboutSection = document.getElementById('about');
if (watchDemoBtn && aboutSection) {
  watchDemoBtn.addEventListener('click', (e) => {
    e.preventDefault();
    aboutSection.scrollIntoView({ behavior: 'smooth' });
  });
}

// ── About section video card ──────────────────────────────────────
// Toggles play/pause on click. If no <source> has been added inside
// the <video> tag yet, clicking just does nothing harmful - the
// poster/play-button stay visible since play() on an empty video
// element resolves to nothing playable.
const videoCard = document.getElementById('about-video-card');
const videoEl    = document.getElementById('about-video');
const playBtn    = document.getElementById('about-video-play-btn');

if (videoCard && videoEl && playBtn) {
  const hasSource = videoEl.querySelector('source') !== null || !!videoEl.getAttribute('src');

  function toggleVideoPlay(e) {
    // Once playback has started, the poster overlay hides itself
    // (pointer-events: none) and the native <video controls> bar
    // becomes directly clickable. A click on the video element
    // itself at that point means the visitor is using those native
    // controls (play/pause/seek/volume/fullscreen) - let the browser
    // handle it natively instead of also toggling play here, which
    // would otherwise immediately undo whatever the native control
    // just did. Keyboard-triggered calls (Enter/Space below) pass no
    // event, so they're unaffected by this check.
    if (e && e.target === videoEl) return;
    if (!hasSource) return; // no video file wired up yet - nothing to play
    if (videoEl.paused) {
      videoEl.play();
      videoCard.classList.add('lp-video-playing');
    } else {
      videoEl.pause();
      videoCard.classList.remove('lp-video-playing');
    }
  }

  videoCard.addEventListener('click', toggleVideoPlay);
  videoCard.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      toggleVideoPlay();
    }
  });
  videoEl.addEventListener('ended', () => {
    videoCard.classList.remove('lp-video-playing');
  });
}

// Year in footer
document.getElementById("year").textContent = new Date().getFullYear();

// Theme toggle with persistence + system preference
(function () {
  const root = document.documentElement;
  const toggle = document.getElementById("themeToggle");
  const stored = localStorage.getItem("theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;

  function apply(theme) {
    root.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }

  apply(stored || (prefersDark ? "dark" : "light"));

  toggle.addEventListener("click", function () {
    const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    apply(next);
  });
})();

// Header border + scroll-progress bar
(function () {
  const header = document.querySelector(".site-header");
  const bar = document.getElementById("scrollProgress");
  const onScroll = () => {
    header.classList.toggle("scrolled", window.scrollY > 8);
    const max = document.documentElement.scrollHeight - window.innerHeight;
    bar.style.transform = "scaleX(" + (max > 0 ? window.scrollY / max : 0) + ")";
  };
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll, { passive: true });
})();

// Mobile navigation menu
(function () {
  const toggle = document.getElementById("navToggle");
  const links = document.getElementById("navLinks");
  if (!toggle || !links) return;

  const setOpen = (open) => {
    links.classList.toggle("open", open);
    toggle.classList.toggle("open", open);
    toggle.setAttribute("aria-expanded", String(open));
    toggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
  };

  toggle.addEventListener("click", () => setOpen(!links.classList.contains("open")));
  // Close when a link is tapped or on Escape
  links.querySelectorAll("a").forEach((a) => a.addEventListener("click", () => setOpen(false)));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") setOpen(false); });
})();

// Live GitHub star counts (graceful: falls back to hardcoded values)
(function () {
  const badges = document.querySelectorAll(".stars[data-repo]");
  badges.forEach((el) => {
    fetch("https://api.github.com/repos/" + el.dataset.repo)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d) => {
        if (typeof d.stargazers_count === "number") {
          const n = d.stargazers_count;
          el.textContent = "★ " + (n >= 1000 ? (n / 1000).toFixed(1) + "k" : n);
        }
      })
      .catch(() => {}); // keep the static value on any failure
  });
})();

// Reveal sections on scroll
(function () {
  const els = document.querySelectorAll(".section, .hero-text, .hero-photo");
  els.forEach((el) => el.classList.add("reveal"));
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add("in");
          io.unobserve(e.target);
        }
      });
    },
    { threshold: 0.12 }
  );
  els.forEach((el) => io.observe(el));
})();

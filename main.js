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

// Header border on scroll
(function () {
  const header = document.querySelector(".site-header");
  const onScroll = () => header.classList.toggle("scrolled", window.scrollY > 8);
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });
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

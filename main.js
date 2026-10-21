// Year in footer
const yearEl = document.getElementById("year");
if (yearEl) yearEl.textContent = new Date().getFullYear();

// Theme toggle. The initial theme is set by the inline script in <head>; a choice
// is only saved once the visitor clicks, so otherwise the system setting is followed.
(function () {
  const root = document.documentElement;
  const toggle = document.getElementById("themeToggle");
  const media = window.matchMedia("(prefers-color-scheme: dark)");

  const hasSavedChoice = () => {
    try { return localStorage.getItem("theme") !== null; } catch (e) { return false; }
  };

  const onSystemChange = (e) => {
    if (!hasSavedChoice()) root.setAttribute("data-theme", e.matches ? "dark" : "light");
  };
  // addListener is the fallback for Safari 13 and older
  if (media.addEventListener) media.addEventListener("change", onSystemChange);
  else if (media.addListener) media.addListener(onSystemChange);

  if (toggle) {
    toggle.addEventListener("click", () => {
      const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) {}
    });
  }
})();

// Live GitHub star counts (falls back to the static values)
(function () {
  document.querySelectorAll(".stars[data-repo]").forEach((el) => {
    fetch("https://api.github.com/repos/" + el.dataset.repo)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d) => {
        if (typeof d.stargazers_count === "number") {
          const n = d.stargazers_count;
          el.textContent = "★ " + (n >= 1000 ? (n / 1000).toFixed(1) + "k" : n);
        }
      })
      .catch(() => {});
  });
})();

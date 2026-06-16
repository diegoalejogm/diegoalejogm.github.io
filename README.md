# diegoalejogm.github.io

Personal website of **Diego Gomez** — Senior Machine Learning Engineer.

A single-page, dependency-free static site. No build step, no framework — just
HTML, CSS, and a touch of vanilla JavaScript, served directly by GitHub Pages.

## Structure

```
index.html    # markup and content
styles.css    # all styling (light + dark themes)
main.js       # theme toggle, scroll reveals, footer year
.nojekyll     # tell GitHub Pages to skip Jekyll processing
```

## Develop locally

It's plain static files, so just open `index.html` in a browser, or serve it:

```bash
python3 -m http.server 8000
# then visit http://localhost:8000
```

## Deploy

Push to the default branch. GitHub Pages serves the site automatically at
<https://diegoalejogm.github.io>.

## Editing content

All copy lives in `index.html`. Update the project cards, writing links, and
about text directly there. Colors and spacing are driven by CSS custom
properties at the top of `styles.css`.

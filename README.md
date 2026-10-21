# diegoalejogm.github.io

Personal website of **Diego Gomez**, machine learning engineer.

A single-page, dependency-free static site served directly by GitHub Pages.
No build step and no framework.

## Structure

```
index.html                      # all content, plus the pre-paint theme script
styles.css                      # styling, light and dark themes
main.js                         # theme toggle, footer year, live GitHub star counts
assets/img/photo.jpg            # profile photo (128x128)
assets/img/social/og-image.png  # link-preview image (1200x630)
robots.txt, sitemap.xml         # search engine hints
.nojekyll                       # serve files as-is, skip Jekyll
```

## Develop locally

```bash
python3 -m http.server 8000
# then visit http://localhost:8000
```

## Deploy

Push to the default branch. GitHub Pages serves the site at
<https://diegoalejogm.github.io>.

## Editing content

All copy lives in `index.html`, one section per heading (Experience, Research,
Open source, Community, Education). Colors are CSS custom properties at the top
of `styles.css`.

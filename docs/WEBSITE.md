# Team website

A static site in [`website/`](../website/): plain HTML, CSS and JavaScript, no build step and no
dependencies.

```text
website/
  index.html    page structure and copy
  styles.css    design tokens (colours on :root) and layout
  app.js        navigation, hero animation, interactive demo, charts, team cards
  team.js       team members -- the one file to edit
  metrics.js    measured numbers shown in the Metrics section
```

## Run it locally

```bash
cd website
python -m http.server 8000
```

Then open <http://localhost:8000>. Opening `index.html` directly from disk also works; the site
loads nothing from the network.

## Deploy

Any static host works: GitHub Pages (set the Pages source to the `website/` folder via a workflow or
copy it to a `gh-pages` branch), Netlify or Vercel (publish directory `website`), or any web server.

## Editing

* **Team members:** edit [`website/team.js`](../website/team.js). Entries with
  `placeholder: true` show a visible "Placeholder" badge; delete that line when the entry is real.
* **Metrics:** edit [`website/metrics.js`](../website/metrics.js) after re-running
  `python solution/experiments.py` and `python evaluation/run_evaluation.py --bootstrap 1000`. Keep
  `fullMatch.spellF1: null` for our system unless the organisers publish the score: the page then
  shows "Run locally with the supplied data". Never put an estimate there.
* **Colours:** the tokens at the top of `styles.css`.

## What is on the page

Hero with an animated tactical board · the challenge and the match timeline · the H/A/D/X labels ·
pipeline architecture and how it works · an interactive possession demo · baseline versus solution
metrics · data constraints and gated assets · team cards · links to the repository, challenge and
evaluation docs.

The **demo** generates a synthetic two-minute match in the browser (seeded, reproducible) and runs
the same ideas as the Python model — majority-vote baseline, smoothed evidence, Viterbi with a
hand-over cost, short-spell clean-up — with spellF1 computed exactly as the official scorer does,
against the synthetic truth. It uses no match data.

## Accessibility

* Semantic landmarks (`header`, `nav`, `main`, `section`, `footer`), one `h1`, ordered headings.
* Skip link; every control reachable by keyboard; visible `:focus-visible` outline.
* Mobile menu button with `aria-expanded`/`aria-controls`; Escape closes it and returns focus.
* Every SVG has `role="img"` with a `title` and, where it carries information, a `desc`.
* Colour is never the only carrier of meaning: rows and tables are labelled in text.
* `prefers-reduced-motion`: the hero animation stays on a still frame, the demo cursor is disabled,
  smooth scrolling and transitions are off.
* Text contrast on the dark pitch background is at least 4.5:1 for body text.
* No horizontal page scroll at 375 px; wide tables and diagrams scroll inside their own region.

Checked in a browser at desktop and 375×812: no console errors, no page-level horizontal overflow,
keyboard open/close of the menu.

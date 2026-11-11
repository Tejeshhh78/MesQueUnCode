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

* **Team members:** edit [`website/team.js`](../website/team.js): name, GitHub login (the avatar
  is loaded from `github.com/<login>.png`, falling back to initials), optional role, and links.
  `GitHub` and `LinkedIn` links get their icon automatically.
* **Metrics:** edit [`website/metrics.js`](../website/metrics.js) after re-running
  `python solution/evaluate_external.py` and `python solution/tune_external.py`. The page shows
  external-game results only; keep `publishedFullMatch: null` for our system unless the organisers
  publish the score. Never put an estimate there, and never put statistics of the provided match on
  the page.
* **Colours:** the tokens at the top of `styles.css`. The theme is a night match in Barcelona:
  navy base, blaugrana (blue/garnet) stripe and glows, a gold accent, a green pitch for the board.
  Data colours are fixed by meaning and were checked with the dataviz skill's palette validator on
  the chart surface `#111b3d`: H `#3987e5` (home, blue), A `#e66767` (away, red), D `#a7acb6`
  (neutral, plus a 45° hatch as a second cue). Charts use `#c98500` for our system and a recessive
  `#6b7aa1` for the baselines. Re-run the validator if you change any of them.

## What is on the page

Hero with an animated tactical board and the "Més que un club. Més que un code." motto · a
scrolling ticker of punchwords · the challenge, with count-up stats and the match timeline · the
H/A/D/X labels · pipeline architecture and how it works · the **Rondo lab** · an interactive
possession demo · baseline versus solution metrics · data constraints and gated assets · team cards ·
links to the repository, challenge and evaluation docs. Section eyebrows are in Catalan or Spanish
with an English gloss ("El repte · The challenge").

The **Rondo lab** is a zoomed attacking half where you drag the ball and players (or move the ball
with the arrow keys). It runs the model's own per-frame evidence live: nearest player per team,
the 1.0 m control kernel, the pass-speed discount, the out-of-play margin, and how many frames of
that evidence would pay for a hand-over. It uses no match data.

The **demo** generates a synthetic two-minute match in the browser (seeded, reproducible) and runs
the same ideas as the Python model — majority-vote baseline, smoothed evidence, Viterbi with a
hand-over cost, short-spell clean-up — with spellF1 computed exactly as the official scorer does,
against the synthetic truth. It uses no match data.

## Accessibility

* Semantic landmarks (`header`, `nav`, `main`, `section`, `footer`), one `h1`, ordered headings.
* Skip link; every control reachable by keyboard; visible `:focus-visible` outline.
* Mobile menu button with `aria-expanded`/`aria-controls`; Escape closes it and returns focus.
* Every SVG has `role="img"` with a `title` and, where it carries information, a `desc`.
* Colour is never the only carrier of meaning: rows and tables are labelled in text, charts have
  a legend and a table view, and D is hatched.
* Chart bars are keyboard-focusable and show the same tooltip on focus as on hover.
* `prefers-reduced-motion`: the hero animation stays on a still frame, the ticker stops and wraps,
  stats show their final value, the demo cursor is disabled, smooth scrolling and transitions are off.
* Text contrast on the dark pitch background is at least 4.5:1 for body text.
* No horizontal page scroll at 375 px; wide tables and diagrams scroll inside their own region.

Checked in a browser at desktop and 375×812: no console errors, no page-level horizontal overflow,
keyboard open/close of the menu.

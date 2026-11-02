/* Mes Que Un Code -- site behaviour. No dependencies; works from file:// or any static server. */
(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";
  var reducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function el(tag, attrs, text) {
    var node = document.createElementNS(SVG_NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function html(tag, attrs, text) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // Deterministic PRNG so "New synthetic match" sequences are reproducible.
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* -- navigation ------------------------------------------------------------------------------ */
  function setupNav() {
    var toggle = document.querySelector(".nav-toggle");
    var links = document.getElementById("nav-links");
    if (!toggle || !links) return;
    function close() { links.classList.remove("open"); toggle.setAttribute("aria-expanded", "false"); }
    toggle.addEventListener("click", function () {
      var open = links.classList.toggle("open");
      toggle.setAttribute("aria-expanded", String(open));
      if (open) { var first = links.querySelector("a"); if (first) first.focus(); }
    });
    links.addEventListener("click", function (e) { if (e.target.closest("a")) close(); });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && links.classList.contains("open")) { close(); toggle.focus(); }
    });

    if (!("IntersectionObserver" in window)) return;
    var anchors = Array.prototype.slice.call(links.querySelectorAll("a"));
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        anchors.forEach(function (a) {
          if (a.getAttribute("href") === "#" + entry.target.id) a.setAttribute("aria-current", "true");
          else a.removeAttribute("aria-current");
        });
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    anchors.forEach(function (a) {
      var target = document.querySelector(a.getAttribute("href"));
      if (target) observer.observe(target);
    });
  }

  /* -- hero tactical board --------------------------------------------------------------------- */
  function setupHero() {
    var group = document.getElementById("hero-players");
    var ball = document.getElementById("hero-ball");
    var halo = document.getElementById("hero-halo");
    var chip = document.getElementById("hero-state");
    if (!group || !ball) return;

    var rand = mulberry32(7);
    var home = [[90, 340], [250, 140], [230, 280], [230, 400], [250, 540], [420, 200], [400, 340], [420, 480], [600, 150], [640, 340], [600, 530]];
    var away = home.map(function (p) { return [1053 - p[0], p[1]]; });
    var players = [];
    home.concat(away).forEach(function (p, i) {
      var isHome = i < 11;
      var node = el("circle", {
        r: 13, fill: isHome ? "var(--home-fill)" : "var(--away-fill)",
        stroke: "#0b2a1c", "stroke-width": 2.5
      });
      group.appendChild(node);
      players.push({ node: node, home: isHome, bx: p[0], by: p[1], x: p[0], y: p[1],
                     ph: rand() * 6.28, sp: 0.5 + rand() * 0.6 });
    });

    var holder = 6, target = 6, t = 0, travel = 1, from = [0, 0];
    function teammates(i) {
      var isHome = players[i].home;
      return players.map(function (_, j) { return j; })
        .filter(function (j) { return j !== i && players[j].home === isHome && j % 11 !== 0; });
    }
    function step(dt) {
      t += dt;
      var shift = players[holder].home ? 60 : -60;   // the team on the ball pushes up
      players.forEach(function (p, i) {
        var drift = p.home === players[holder].home ? shift : shift * 0.6;
        p.x = p.bx + drift + Math.sin(t * p.sp + p.ph) * 26;
        p.y = p.by + Math.cos(t * p.sp * 0.8 + p.ph) * 22;
        p.node.setAttribute("cx", p.x.toFixed(1));
        p.node.setAttribute("cy", p.y.toFixed(1));
      });
      travel += dt * 1.4;
      var bx, by;
      if (travel < 1) {
        var to = players[target];
        bx = from[0] + (to.x - from[0]) * travel;
        by = from[1] + (to.y - from[1]) * travel;
      } else {
        if (holder !== target) holder = target;
        bx = players[holder].x + 10; by = players[holder].y + 6;
        if (travel > 2.2) {   // pass: mostly to a teammate, sometimes intercepted
          from = [bx, by];
          var mates = teammates(holder);
          target = rand() < 0.82 ? mates[Math.floor(rand() * mates.length)]
                                 : (holder < 11 ? 12 + Math.floor(rand() * 10) : 1 + Math.floor(rand() * 10));
          travel = 0;
        }
      }
      ball.setAttribute("cx", bx.toFixed(1));
      ball.setAttribute("cy", by.toFixed(1));
      halo.setAttribute("cx", players[holder].x.toFixed(1));
      halo.setAttribute("cy", players[holder].y.toFixed(1));
      halo.setAttribute("stroke", players[holder].home ? "var(--home-fill)" : "var(--gold)");
      chip.textContent = "Model state: " + (players[holder].home ? "H" : "A");
    }

    step(0);
    if (reducedMotion) return;
    var last = null, running = true;
    function frame(now) {
      if (last !== null && running) step(Math.min(0.05, (now - last) / 1000));
      last = now;
      requestAnimationFrame(frame);
    }
    document.addEventListener("visibilitychange", function () { running = !document.hidden; });
    requestAnimationFrame(frame);
  }

  /* -- possession maths shared by the demo -------------------------------------------------------- */
  var H = 0, A = 1, D = 2;

  function segments(codes) {
    var out = [], start = 0;
    for (var i = 1; i <= codes.length; i++) {
      if (i === codes.length || codes[i] !== codes[start]) { out.push([start, i, codes[start]]); start = i; }
    }
    return out;
  }

  function spells(codes) {
    return segments(codes).filter(function (s) { return s[2] === H || s[2] === A; });
  }

  // Same definition as evaluation/possession_metrics.py: one-to-one greedy IoU matching,
  // F1 averaged over IoU 0.3..0.7.
  function spellF1(truth, pred) {
    var gt = spells(truth), pr = spells(pred), pairs = [];
    gt.forEach(function (g, i) {
      pr.forEach(function (p, j) {
        if (g[2] !== p[2] || p[1] <= g[0] || g[1] <= p[0]) return;
        var inter = Math.min(g[1], p[1]) - Math.max(g[0], p[0]);
        var union = Math.max(g[1], p[1]) - Math.min(g[0], p[0]);
        pairs.push([inter / union, i, j]);
      });
    });
    pairs.sort(function (a, b) { return b[0] - a[0] || a[1] - b[1] || a[2] - b[2]; });
    var total = 0, thresholds = [0.3, 0.4, 0.5, 0.6, 0.7];
    thresholds.forEach(function (th) {
      var usedG = {}, usedP = {}, tp = 0;
      for (var k = 0; k < pairs.length && pairs[k][0] >= th; k++) {
        if (usedG[pairs[k][1]] || usedP[pairs[k][2]]) continue;
        usedG[pairs[k][1]] = usedP[pairs[k][2]] = true; tp++;
      }
      var prec = pr.length ? tp / pr.length : 0, rec = gt.length ? tp / gt.length : 0;
      total += prec + rec ? 2 * prec * rec / (prec + rec) : 0;
    });
    return total / thresholds.length;
  }

  function viterbi(scores, switchCost, deadCost) {
    var n = scores.length, back = new Int8Array(n * 3);
    var cost = [[0, switchCost, deadCost], [switchCost, 0, deadCost], [deadCost, deadCost, 0]];
    var total = [scores[0][0] - deadCost, scores[0][1] - deadCost, scores[0][2]];
    for (var f = 1; f < n; f++) {
      var next = [0, 0, 0];
      for (var j = 0; j < 3; j++) {
        var bi = j, best = total[j];
        for (var i = 0; i < 3; i++) {
          var c = total[i] - cost[i][j];
          if (c > best) { best = c; bi = i; }
        }
        next[j] = best + scores[f][j];
        back[f * 3 + j] = bi;
      }
      total = next;
    }
    var path = new Int8Array(n), s = total.indexOf(Math.max.apply(null, total));
    for (var g = n - 1; g >= 0; g--) { path[g] = s; s = back[g * 3 + s]; }
    return path;
  }

  function absorbShort(codes, minLen) {
    var out = Int8Array.from(codes), changed = true;
    while (changed) {
      changed = false;
      var segs = segments(out), k, shortest = -1;
      for (k = 0; k < segs.length; k++) {
        var s = segs[k];
        if ((s[2] === H || s[2] === A) && s[1] - s[0] < minLen &&
            (shortest < 0 || s[1] - s[0] < segs[shortest][1] - segs[shortest][0])) shortest = k;
      }
      if (shortest < 0) break;
      var seg = segs[shortest];
      var left = shortest > 0 ? segs[shortest - 1][2] : null;
      var right = shortest + 1 < segs.length ? segs[shortest + 1][2] : null;
      var team = [left, right].filter(function (x) { return x === H || x === A; });
      out.fill(team.length ? team[0] : D, seg[0], seg[1]);
      changed = true;
    }
    return out;
  }

  /* -- synthetic match ---------------------------------------------------------------------------- */
  function syntheticMatch(seed, visibility) {
    var rand = mulberry32(seed), n = 3000, truth = new Int8Array(n);
    var f = 0, team = rand() < 0.5 ? H : A;
    truth.fill(D, 0, 40); f = 40;
    while (f < n) {
      var len = Math.max(60, Math.round(-Math.log(1 - rand()) * 300));
      truth.fill(team, f, Math.min(n, f + len)); f += len;
      if (rand() < 0.3 && f < n) {
        var dead = 60 + Math.round(rand() * 180);
        truth.fill(D, f, Math.min(n, f + dead)); f += dead;
      }
      team = rand() < 0.85 ? 1 - team : team;
    }
    // Ball visibility: bursty, like the real tracker -- gaps of ~2.4 s on average.
    var visible = new Uint8Array(n), on = rand() < visibility, meanGap = 60;
    var meanRun = visibility >= 0.999 ? Infinity : meanGap * visibility / (1 - visibility);
    for (var i = 0; i < n; i++) {
      visible[i] = on ? 1 : 0;
      var p = on ? 1 / meanRun : 1 / meanGap;
      if (rand() < p) on = !on;
    }
    // Nearest-player team: right 75% of frames in play, a coin flip in a stoppage.
    var nearest = new Int8Array(n), deadSignal = new Uint8Array(n);
    for (i = 0; i < n; i++) {
      if (truth[i] === D) { nearest[i] = rand() < 0.5 ? H : A; deadSignal[i] = rand() < 0.7 ? 1 : 0; }
      else { nearest[i] = rand() < 0.75 ? truth[i] : 1 - truth[i]; deadSignal[i] = rand() < 0.04 ? 1 : 0; }
    }
    return { n: n, truth: truth, visible: visible, nearest: nearest, deadSignal: deadSignal };
  }

  function rawRow(m) {
    var out = new Int8Array(m.n);
    for (var i = 0; i < m.n; i++) out[i] = m.visible[i] ? m.nearest[i] : D;
    return out;
  }

  function voteRow(m, w) {
    var out = new Int8Array(m.n), ch = [0], ca = [0];
    for (var i = 0; i < m.n; i++) {
      ch.push(ch[i] + (m.visible[i] && m.nearest[i] === H ? 1 : 0));
      ca.push(ca[i] + (m.visible[i] && m.nearest[i] === A ? 1 : 0));
    }
    for (i = 0; i < m.n; i++) {
      var lo = Math.max(0, i - w), hi = Math.min(m.n, i + w + 1);
      var h = ch[hi] - ch[lo], a = ca[hi] - ca[lo];
      out[i] = h + a === 0 ? D : (h >= a ? H : A);
    }
    return out;
  }

  // Mirrors solution/possession_model.py: team evidence smoothed over +/-6 frames, dead evidence,
  // a small bias towards live play, Viterbi, then sub-second spells absorbed.
  function decoderRow(m, switchCost) {
    var raw = new Float64Array(m.n), cum = [0], scores = new Array(m.n), half = 6;
    for (var i = 0; i < m.n; i++) {
      raw[i] = m.visible[i] ? (m.nearest[i] === H ? 0.35 : -0.35) : 0;
      cum.push(cum[i] + raw[i]);
    }
    for (i = 0; i < m.n; i++) {
      var lo = Math.max(0, i - half), hi = Math.min(m.n, i + half + 1);
      var team = (cum[hi] - cum[lo]) / (hi - lo);
      var dead = m.deadSignal[i] ? 0.6 : -0.05;
      scores[i] = [0.5 * team - 0.5 * dead + 0.02, -0.5 * team - 0.5 * dead + 0.02,
                   0.5 * dead + 0.5 * Math.abs(team)];
    }
    return absorbShort(viterbi(scores, switchCost, 3), 25);
  }

  function setupDemo() {
    var svg = document.getElementById("timeline-svg");
    if (!svg) return;
    var vis = document.getElementById("ctl-visibility");
    var win = document.getElementById("ctl-window");
    var sw = document.getElementById("ctl-switch");
    var body = document.getElementById("demo-body");
    var live = document.getElementById("demo-live");
    var playBtn = document.getElementById("btn-play");
    var seed = 2027, match = null, cursor = null, playing = false, pos = 0;
    var colours = ["var(--home-fill)", "var(--away-fill)", "var(--dead)"];

    var layer = document.getElementById("timeline-rows");

    function drawRow(codes, y, label) {
      layer.appendChild(el("text", { x: 0, y: y + 22, fill: "var(--ink-soft)", "font-size": 15 }, label));
      segments(codes).forEach(function (s) {
        layer.appendChild(el("rect", {
          x: (160 + s[0] / codes.length * 835).toFixed(2), y: y,
          width: Math.max(0.6, (s[1] - s[0]) / codes.length * 835).toFixed(2), height: 34,
          fill: colours[s[2]]
        }));
      });
    }

    function render() {
      document.getElementById("out-visibility").textContent = vis.value + "%";
      document.getElementById("out-window").textContent = win.value === "0" ? "off" : "±" + win.value + " frames";
      document.getElementById("out-switch").textContent = Number(sw.value).toFixed(1);
      match = syntheticMatch(seed, Number(vis.value) / 100);
      var rows = [
        ["Synthetic truth", match.truth],
        ["Raw nearest", rawRow(match)],
        ["Vote baseline", voteRow(match, Number(win.value))],
        ["Our decoder", decoderRow(match, Number(sw.value))]
      ];
      while (layer.firstChild) layer.removeChild(layer.firstChild);
      rows.forEach(function (r, k) { drawRow(r[1], 12 + k * 58, r[0]); });
      cursor = el("line", { x1: 160, x2: 160, y1: 4, y2: 246, stroke: "var(--focus)", "stroke-width": 2,
                            visibility: playing ? "visible" : "hidden" });
      layer.appendChild(cursor);

      body.innerHTML = "";
      rows.forEach(function (r, k) {
        var sp = spells(r[1]), alive = 0;
        sp.forEach(function (s) { alive += s[1] - s[0]; });
        var tr = html("tr", k === 3 ? { "class": "ours" } : {});
        tr.appendChild(html("th", { scope: "row" }, r[0]));
        tr.appendChild(html("td", {}, String(sp.length)));
        tr.appendChild(html("td", {}, sp.length ? (alive / sp.length / 25).toFixed(1) + " s" : "–"));
        tr.appendChild(html("td", {}, k === 0 ? "1.000 (reference)" : spellF1(match.truth, r[1]).toFixed(3)));
        body.appendChild(tr);
      });
      live.textContent = "Synthetic match #" + (seed - 2026) + ": decoder finds " + spells(rows[3][1]).length +
        " possessions against " + spells(match.truth).length + " in the made-up truth.";
    }

    function tick() {
      if (!playing) return;
      pos = (pos + 8) % 3000;
      var x = 160 + pos / 3000 * 835;
      cursor.setAttribute("x1", x); cursor.setAttribute("x2", x);
      requestAnimationFrame(tick);
    }

    [vis, win, sw].forEach(function (input) { input.addEventListener("input", render); });
    document.getElementById("btn-new").addEventListener("click", function () { seed += 1; render(); });
    playBtn.addEventListener("click", function () {
      playing = !playing && !reducedMotion;
      playBtn.setAttribute("aria-pressed", String(playing));
      playBtn.textContent = playing ? "Pause cursor" : "Play cursor";
      cursor.setAttribute("visibility", playing ? "visible" : "hidden");
      if (playing) requestAnimationFrame(tick);
    });
    if (reducedMotion) { playBtn.disabled = true; playBtn.title = "Disabled: reduced motion is on"; }
    render();
  }

  /* -- metrics ---------------------------------------------------------------------------------- */
  function setupMetrics() {
    var data = window.MQUC_METRICS;
    if (!data) return;
    var tbody = document.getElementById("metrics-body");
    data.systems.forEach(function (s) {
      var tr = html("tr", s.kind === "ours" ? { "class": "ours" } : {});
      tr.appendChild(html("th", { scope: "row" }, s.label));
      tr.appendChild(html("td", {}, s.public.macroF1.toFixed(4)));
      tr.appendChild(html("td", {}, s.public.spellF1.toFixed(4) + " [" + s.public.spellCI[0].toFixed(2) + "–" + s.public.spellCI[1].toFixed(2) + "]"));
      tr.appendChild(html("td", {}, String(s.public.poss)));
      var full = html("td", {}, s.fullMatch.spellF1 === null ? "Run locally with the supplied data" : s.fullMatch.spellF1.toFixed(4) + " (published)");
      if (s.fullMatch.spellF1 === null) full.className = "unavailable";
      tr.appendChild(full);
      tr.appendChild(html("td", {}, String(s.fullMatch.possessions)));
      tr.appendChild(html("td", {}, s.fullMatch.meanPossS.toFixed(1) + " s"));
      tr.appendChild(html("td", {}, s.fullMatch.shareD.toFixed(1) + "%"));
      tbody.appendChild(tr);
    });

    barChart(document.getElementById("chart-public"), data.systems.map(function (s) {
      return { label: s.label, value: s.public.spellF1, ci: s.public.spellCI, ours: s.kind === "ours" };
    }), 1, function (v) { return v.toFixed(3); });
    var maxPoss = Math.max.apply(null, data.systems.map(function (s) { return s.fullMatch.possessions; }));
    barChart(document.getElementById("chart-poss"), data.systems.map(function (s) {
      return { label: s.label, value: s.fullMatch.possessions, ours: s.kind === "ours" };
    }), maxPoss * 1.1, function (v) { return String(v); });
  }

  function barChart(svg, rows, max, fmt) {
    if (!svg) return;
    var left = 170, width = 270, rowH = 56;
    rows.forEach(function (r, i) {
      var y = 14 + i * rowH, w = Math.max(2, r.value / max * width);
      var name = r.label.length > 20 ? r.label.slice(0, 19) + "…" : r.label;
      svg.appendChild(el("text", { x: left - 10, y: y + 22, "text-anchor": "end", fill: "var(--ink-soft)", "font-size": 15 }, name));
      svg.appendChild(el("rect", { x: left, y: y + 6, width: width, height: 24, rx: 6, fill: "var(--surface-2)" }));
      var bar = el("rect", { x: left, y: y + 6, width: w.toFixed(1), height: 24, rx: 6, fill: r.ours ? "var(--gold)" : "var(--pitch-b)" });
      bar.appendChild(el("title", {}, r.label + ": " + fmt(r.value)));
      svg.appendChild(bar);
      if (r.ci) {
        var x0 = left + r.ci[0] / max * width, x1 = left + r.ci[1] / max * width;
        svg.appendChild(el("line", { x1: x0, x2: x1, y1: y + 18, y2: y + 18, stroke: "var(--chalk)", "stroke-width": 2 }));
        svg.appendChild(el("line", { x1: x0, x2: x0, y1: y + 11, y2: y + 25, stroke: "var(--chalk)", "stroke-width": 2 }));
        svg.appendChild(el("line", { x1: x1, x2: x1, y1: y + 11, y2: y + 25, stroke: "var(--chalk)", "stroke-width": 2 }));
      }
      svg.appendChild(el("text", { x: left + 4, y: y + 48, fill: r.ours ? "var(--gold)" : "var(--chalk)", "font-size": 15, "font-weight": 700 }, fmt(r.value)));
    });
  }

  /* -- team -------------------------------------------------------------------------------------- */
  function setupTeam() {
    var grid = document.getElementById("team-grid");
    var team = window.MQUC_TEAM || [];
    team.forEach(function (m) {
      var li = html("li");
      var card = html("article", { "class": "team-card" });
      if (m.placeholder) card.appendChild(html("span", { "class": "placeholder-badge" }, "Placeholder"));
      var initials = (m.name || "?").split(/\s+/).map(function (w) { return w.charAt(0); }).join("").slice(0, 2).toUpperCase();
      card.appendChild(html("div", { "class": "avatar", "aria-hidden": "true" }, initials));
      if (m.position) card.appendChild(html("span", { "class": "position" }, m.position));
      card.appendChild(html("h3", {}, m.name || ""));
      card.appendChild(html("p", { "class": "role" }, m.role || ""));
      if (m.focus) card.appendChild(html("p", {}, m.focus));
      if (m.links && m.links.length) {
        var links = html("div", { "class": "team-links" });
        m.links.forEach(function (l) { links.appendChild(html("a", { href: l.href }, l.label)); });
        card.appendChild(links);
      }
      li.appendChild(card);
      grid.appendChild(li);
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    setupNav();
    setupHero();
    setupDemo();
    setupMetrics();
    setupTeam();
  });
})();

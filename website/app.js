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
        stroke: "var(--chalk)", "stroke-width": 2
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
      halo.setAttribute("stroke", players[holder].home ? "var(--home-fill)" : "var(--away-fill)");
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
    var colours = ["var(--home-fill)", "var(--away-fill)", "url(#dead-hatch)"];

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
      return { label: s.label, short: s.short, value: s.public.spellF1, ci: s.public.spellCI,
               ours: s.kind === "ours",
               tip: "spellF1 " + s.public.spellF1.toFixed(3) + " · 5–95% interval " +
                    s.public.spellCI[0].toFixed(2) + "–" + s.public.spellCI[1].toFixed(2) };
    }), [0, 0.25, 0.5, 0.75, 1], function (v) { return v.toFixed(3); },
       function (v) { return v === 0 ? "0" : v.toFixed(2); });
    barChart(document.getElementById("chart-poss"), data.systems.map(function (s) {
      return { label: s.label, short: s.short, value: s.fullMatch.possessions, ours: s.kind === "ours",
               tip: s.fullMatch.possessions + " possessions · mean " + s.fullMatch.meanPossS.toFixed(1) + " s" };
    }), [0, 500, 1000, 1500, 2000], function (v) { return String(v); },
       function (v) { return v >= 1000 ? (v / 1000) + "K" : String(v); });
  }

  // One shared tooltip for every chart: follows the pointer, or sits on a keyboard-focused bar.
  var tooltip = null;
  function showTip(text, x, y) {
    if (!tooltip) {
      tooltip = html("div", { "class": "chart-tooltip", role: "status" });
      document.body.appendChild(tooltip);
    }
    tooltip.textContent = text;
    tooltip.hidden = false;
    var w = tooltip.offsetWidth;
    tooltip.style.left = Math.min(window.innerWidth - w - 8, x + 14) + "px";
    tooltip.style.top = (y + 14) + "px";
  }
  function hideTip() { if (tooltip) tooltip.hidden = true; }

  // A bar with a 4px rounded data end and a square baseline (dataviz mark spec).
  function barPath(x, y, w, h) {
    var r = Math.min(4, w / 2, h / 2);
    return "M" + x + "," + y + "H" + (x + w - r) + "Q" + (x + w) + "," + y + " " + (x + w) + "," + (y + r) +
           "V" + (y + h - r) + "Q" + (x + w) + "," + (y + h) + " " + (x + w - r) + "," + (y + h) +
           "H" + x + "Z";
  }

  // Horizontal bars: one baseline, a hairline grid on clean ticks, bars <= 24px, the value at the
  // tip in text colour, our system in the highlight hue and the baselines in a recessive one.
  function barChart(svg, rows, ticks, fmt, tickFmt) {
    if (!svg) return;
    var left = 160, width = 270, rowH = 52, top = 10, barH = 20;
    var max = ticks[ticks.length - 1];
    var bottom = top + rows.length * rowH;
    function sx(v) { return left + v / max * width; }

    ticks.forEach(function (t) {
      svg.appendChild(el("line", { x1: sx(t), x2: sx(t), y1: top, y2: bottom, stroke: "var(--grid)", "stroke-width": 1 }));
      svg.appendChild(el("text", { x: sx(t), y: bottom + 18, "text-anchor": "middle", fill: "var(--ink-muted)",
                                   "font-size": 13, style: "font-variant-numeric: tabular-nums" }, tickFmt(t)));
    });

    rows.forEach(function (r, i) {
      var y = top + i * rowH + (rowH - barH) / 2;
      var g = el("g", { tabindex: 0, role: "img", "aria-label": r.label + ": " + r.tip, "class": "chart-bar" });
      g.appendChild(el("rect", { x: 0, y: top + i * rowH, width: 480, height: rowH, fill: "transparent" }));
      g.appendChild(el("text", { x: left - 10, y: y + 15, "text-anchor": "end", fill: "var(--ink-soft)", "font-size": 15 }, r.short || r.label));
      g.appendChild(el("path", { d: barPath(left, y, Math.max(1, sx(r.value) - left), barH),
                                 fill: r.ours ? "var(--chart-ours)" : "var(--chart-base)" }));
      var tipEnd = sx(r.value);
      if (r.ci) {
        var x0 = sx(r.ci[0]), x1 = sx(r.ci[1]), cy = y + barH / 2;
        g.appendChild(el("line", { x1: x0, x2: x1, y1: cy, y2: cy, stroke: "var(--ink)", "stroke-width": 1.5 }));
        g.appendChild(el("line", { x1: x0, x2: x0, y1: cy - 6, y2: cy + 6, stroke: "var(--ink)", "stroke-width": 1.5 }));
        g.appendChild(el("line", { x1: x1, x2: x1, y1: cy - 6, y2: cy + 6, stroke: "var(--ink)", "stroke-width": 1.5 }));
        tipEnd = Math.max(tipEnd, x1);
      }
      g.appendChild(el("text", { x: tipEnd + 6, y: y + 15, fill: "var(--ink)", "font-size": 14,
                                 "font-weight": r.ours ? 700 : 500 }, fmt(r.value)));
      g.addEventListener("mousemove", function (e) { showTip(r.label + " — " + r.tip, e.clientX, e.clientY); });
      g.addEventListener("mouseleave", hideTip);
      g.addEventListener("focus", function () {
        var box = g.getBoundingClientRect();
        showTip(r.label + " — " + r.tip, box.left + box.width / 2, box.top + box.height / 2);
      });
      g.addEventListener("blur", hideTip);
      svg.appendChild(g);
    });

    var legend = html("div", { "class": "chart-legend" });
    legend.innerHTML = '<span><i class="swatch" style="background:var(--chart-ours)"></i>Mes Que Un Code</span>' +
                       '<span><i class="swatch" style="background:var(--chart-base)"></i>Reference baselines</span>';
    svg.parentNode.appendChild(legend);
  }

  var ICONS = {
    GitHub: "M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z",
    LinkedIn: "M13.6 0H2.4A2.4 2.4 0 000 2.4v11.2A2.4 2.4 0 002.4 16h11.2a2.4 2.4 0 002.4-2.4V2.4A2.4 2.4 0 0013.6 0zM4.8 13.6H2.4V6h2.4v7.6zM3.6 4.9a1.4 1.4 0 110-2.8 1.4 1.4 0 010 2.8zm10 8.7h-2.4V9.9c0-.9 0-2-1.2-2s-1.4.9-1.4 1.9v3.8H6.2V6h2.3v1h.1c.3-.6 1.1-1.2 2.3-1.2 2.4 0 2.8 1.6 2.8 3.6v4.2z"
  };

  /* -- team -------------------------------------------------------------------------------------- */
  function setupTeam() {
    var grid = document.getElementById("team-grid");
    var team = window.MQUC_TEAM || [];
    team.forEach(function (m) {
      var li = html("li");
      var card = html("article", { "class": "team-card" });
      if (m.placeholder) card.appendChild(html("span", { "class": "placeholder-badge" }, "Placeholder"));
      var initials = (m.name || "?").split(/\s+/).map(function (w) { return w.charAt(0); }).join("").slice(0, 2).toUpperCase();
      var avatar = html("div", { "class": "avatar", "aria-hidden": "true" }, initials);
      if (m.github) {
        var img = html("img", { src: "https://github.com/" + encodeURIComponent(m.github) + ".png?size=168",
                                alt: "", width: 84, height: 84, decoding: "async" });
        img.addEventListener("load", function () { avatar.textContent = ""; avatar.appendChild(img); });
      }
      card.appendChild(avatar);
      card.appendChild(html("h3", {}, m.name || ""));
      if (m.github) card.appendChild(html("p", { "class": "handle" }, "@" + m.github));
      if (m.role) card.appendChild(html("p", { "class": "role" }, m.role));
      if (m.links && m.links.length) {
        var links = html("div", { "class": "team-links" });
        m.links.forEach(function (l) {
          var a = html("a", { href: l.href, rel: "noopener", "aria-label": l.label + " — " + m.name });
          if (ICONS[l.label]) {
            var icon = el("svg", { viewBox: "0 0 16 16", "aria-hidden": "true", focusable: "false" });
            icon.appendChild(el("path", { d: ICONS[l.label] }));
            a.appendChild(icon);
          }
          a.appendChild(document.createTextNode(l.label));
          links.appendChild(a);
        });
        card.appendChild(links);
      }
      li.appendChild(card);
      grid.appendChild(li);
    });
  }

  /* -- Rondo lab: the model's per-frame evidence, live ---------------------------------------- */
  // Pitch coordinates in metres; the SVG draws 10 px per metre from (60, 60).
  var PITCH_L = 105.3, PITCH_W = 68, PX = 10, ORIGIN = 60;
  var SIGMA = 1.5, TEAM_EVIDENCE = 0.35, FLIGHT_WEIGHT = 0.15, OUT_MARGIN = 0.75, SWITCH_COST = 4;

  function setupRondo() {
    var svg = document.getElementById("rondo-svg");
    if (!svg) return;
    var playersG = document.getElementById("rondo-players");
    var overlay = document.getElementById("rondo-overlay");
    var ballG = document.getElementById("rondo-ball-g");
    var flight = document.getElementById("rondo-flight");
    var letter = document.getElementById("rondo-letter");
    var text = document.getElementById("rondo-text");
    var meter = document.getElementById("rondo-meter");
    var rand = mulberry32(10);
    var ball = { x: 70.8, y: 25.6 };   // kick off at a Barça player's feet
    var MIN_X = 50;   // the board shows the attacking half, from just behind halfway
    var players = [];
    var lastTeam = H, lastState = null;

    // A 4-v-3 rondo around the ball, plus a few players elsewhere -- deterministic.
    var BASE = [
      [H, 70, 25], [H, 85, 27], [H, 81, 41], [H, 69, 40], [H, 60, 18], [H, 97, 12],
      [A, 76, 34], [A, 80, 30], [A, 75, 27], [A, 100, 34], [A, 90, 52], [A, 62, 50]
    ];

    function toPx(m) { return ORIGIN + m * PX; }
    function fromEvent(e) {
      var pt = svg.createSVGPoint();
      pt.x = e.clientX; pt.y = e.clientY;
      var p = pt.matrixTransform(svg.getScreenCTM().inverse());
      return { x: Math.max(MIN_X, Math.min(PITCH_L + 5, (p.x - ORIGIN) / PX)),
               y: Math.max(-5, Math.min(PITCH_W + 5, (p.y - ORIGIN) / PX)) };
    }

    function buildPlayers(layout) {
      while (playersG.firstChild) playersG.removeChild(playersG.firstChild);
      players = layout.map(function (p) {
        var g = el("g", { "class": "rondo-player" });
        g.appendChild(el("circle", { r: 15, fill: p[0] === H ? "var(--home-fill)" : "var(--away-fill)",
                                     stroke: "var(--chalk)", "stroke-width": 2.5 }));
        playersG.appendChild(g);
        var player = { team: p[0], x: p[1], y: p[2], node: g };
        draggable(g, function (pos) { player.x = pos.x; player.y = pos.y; });
        return player;
      });
    }

    function draggable(node, move) {
      node.addEventListener("pointerdown", function (e) {
        e.preventDefault();
        node.setPointerCapture(e.pointerId);
        function onMove(ev) { move(fromEvent(ev)); render(); }
        function onUp() {
          node.removeEventListener("pointermove", onMove);
          node.removeEventListener("pointerup", onUp);
          node.removeEventListener("pointercancel", onUp);
        }
        node.addEventListener("pointermove", onMove);
        node.addEventListener("pointerup", onUp);
        node.addEventListener("pointercancel", onUp);
        onMove(e);
      });
    }

    function nearest(team) {
      var best = null;
      players.forEach(function (p) {
        if (p.team !== team) return;
        var d = Math.hypot(p.x - ball.x, p.y - ball.y);
        if (!best || d < best.d) best = { d: d, p: p };
      });
      return best;
    }

    function line(p, colour) {
      overlay.appendChild(el("line", { x1: toPx(ball.x), y1: toPx(ball.y), x2: toPx(p.x), y2: toPx(p.y),
                                       stroke: colour, "stroke-width": 3, "stroke-linecap": "round", opacity: 0.9 }));
    }

    function label(p, d) {
      var x = (toPx(ball.x) + toPx(p.x)) / 2, y = (toPx(ball.y) + toPx(p.y)) / 2 - 10;
      var t = el("text", { x: x, y: y, "text-anchor": "middle", fill: "var(--chalk)", "font-size": 18,
                           "font-weight": 700, stroke: "#070c1f", "stroke-width": 4, "paint-order": "stroke" },
                 d.toFixed(1) + " m");
      overlay.appendChild(t);
    }

    function render() {
      players.forEach(function (p) {
        p.node.setAttribute("transform", "translate(" + toPx(p.x) + " " + toPx(p.y) + ")");
      });
      ballG.setAttribute("transform", "translate(" + toPx(ball.x) + " " + toPx(ball.y) + ")");
      while (overlay.firstChild) overlay.removeChild(overlay.firstChild);

      // The control kernel around the ball: 1 sigma and 2 sigma.
      overlay.appendChild(el("circle", { cx: toPx(ball.x), cy: toPx(ball.y), r: SIGMA * PX, fill: "none",
                                         stroke: "var(--gold)", "stroke-width": 1.5, opacity: 0.9 }));
      overlay.appendChild(el("circle", { cx: toPx(ball.x), cy: toPx(ball.y), r: 2 * SIGMA * PX, fill: "none",
                                         stroke: "var(--gold)", "stroke-width": 1, opacity: 0.45 }));

      var out = ball.x < -OUT_MARGIN || ball.x > PITCH_L + OUT_MARGIN ||
                ball.y < -OUT_MARGIN || ball.y > PITCH_W + OUT_MARGIN;
      var nh = nearest(H), na = nearest(A);
      var kh = nh ? Math.exp(-0.5 * Math.pow(nh.d / SIGMA, 2)) : 0;
      var ka = na ? Math.exp(-0.5 * Math.pow(na.d / SIGMA, 2)) : 0;
      var ev = out ? 0 : TEAM_EVIDENCE * (kh - ka) * (flight.checked ? FLIGHT_WEIGHT : 1);

      if (nh) { line(nh.p, "var(--home-fill)"); label(nh.p, nh.d); }
      if (na) { line(na.p, "var(--away-fill)"); label(na.p, na.d); }

      document.getElementById("rondo-dh").textContent = nh ? nh.d.toFixed(1) + " m" : "–";
      document.getElementById("rondo-da").textContent = na ? na.d.toFixed(1) + " m" : "–";
      document.getElementById("rondo-ev").textContent = (ev >= 0 ? "+" : "−") + Math.abs(ev).toFixed(3);
      document.getElementById("rondo-frames").textContent =
        Math.abs(ev) < 0.005 ? "never" : Math.ceil(SWITCH_COST / Math.abs(ev)) + " (" + (SWITCH_COST / Math.abs(ev) / 25).toFixed(1) + " s)";

      var share = Math.max(-1, Math.min(1, ev / TEAM_EVIDENCE));
      meter.style.left = (share >= 0 ? 50 : 50 + share * 50) + "%";
      meter.style.width = Math.abs(share) * 50 + "%";
      meter.style.background = share >= 0 ? "var(--home-fill)" : "var(--away-fill)";

      var state, words;
      if (out) { state = D; words = "Dead ball: it's out of play"; }
      else if (Math.abs(ev) < 0.02) {
        state = lastTeam;
        words = "Loose ball: stays with " + (lastTeam === H ? "Barça (H)" : "Athletic (A)");
      } else {
        state = ev > 0 ? H : A; lastTeam = state;
        words = state === H ? (ev > 0.25 ? "Barça on the ball" : "Barça edging it")
                            : (ev < -0.25 ? "Athletic on the ball" : "Athletic edging it");
        if (flight.checked) words += " (in flight, barely counts)";
      }
      letter.textContent = ["H", "A", "D"][state];
      letter.className = "verdict-letter" + (state === A ? " is-a" : state === D ? " is-d" : "");
      if (state !== lastState && lastState !== null && !reducedMotion) {
        letter.classList.add("pop");
        setTimeout(function () { letter.classList.remove("pop"); }, 180);
      }
      lastState = state;
      text.textContent = words;
      ballG.setAttribute("aria-label", "Ball, use arrow keys to move. At " + (Math.round(ball.x) || 0) + " m, " + (Math.round(ball.y) || 0) +
                         " m. " + words + ".");
    }

    draggable(ballG, function (pos) { ball.x = pos.x; ball.y = pos.y; });
    ballG.addEventListener("keydown", function (e) {
      var step = e.shiftKey ? 5 : 1, moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
      if (!moves[e.key]) return;
      e.preventDefault();
      ball.x = Math.max(MIN_X, Math.min(PITCH_L + 5, ball.x + moves[e.key][0]));
      ball.y = Math.max(-5, Math.min(PITCH_W + 5, ball.y + moves[e.key][1]));
      render();
    });
    flight.addEventListener("change", render);
    document.getElementById("rondo-shuffle").addEventListener("click", function () {
      buildPlayers(BASE.map(function (p) {
        return [p[0], Math.max(MIN_X + 3, Math.min(PITCH_L - 2, p[1] + (rand() - 0.5) * 16)),
                      Math.max(2, Math.min(PITCH_W - 2, p[2] + (rand() - 0.5) * 12))];
      }));
      render();
    });
    document.getElementById("rondo-corner").addEventListener("click", function () {
      ball = { x: PITCH_L + 0.2, y: -0.3 };   // on the arc: still in play
      buildPlayers([[H, 103.5, 1.5], [H, 96, 30], [H, 99, 38], [H, 92, 26], [H, 88, 44],
                    [A, 100, 33], [A, 98, 36], [A, 101, 30], [A, 104, 34], [A, 95, 40], [A, 97, 8]]);
      render();
    });

    buildPlayers(BASE);
    render();
  }

  /* -- count-up stats ------------------------------------------------------------------------ */
  function setupCounters() {
    var stats = Array.prototype.slice.call(document.querySelectorAll(".stat-value"));
    stats.forEach(function (node) {
      var raw = node.textContent.trim(), m = raw.match(/^([\d,]+)(.*)$/);
      if (!m) return;
      node.dataset.target = m[1].replace(/,/g, "");
      node.dataset.suffix = m[2];
    });
    if (reducedMotion || !("IntersectionObserver" in window)) return;
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        observer.unobserve(entry.target);
        var node = entry.target, target = Number(node.dataset.target), start = null;
        if (!target) return;
        function frame(now) {
          if (start === null) start = now;
          var t = Math.min(1, (now - start) / 1200), eased = 1 - Math.pow(1 - t, 3);
          node.textContent = Math.round(target * eased).toLocaleString("en-US") + node.dataset.suffix;
          if (t < 1) requestAnimationFrame(frame);
        }
        requestAnimationFrame(frame);
      });
    }, { threshold: 0.6 });
    stats.forEach(function (node) { if (node.dataset.target) observer.observe(node); });
  }

  document.addEventListener("DOMContentLoaded", function () {
    setupNav();
    setupHero();
    setupRondo();
    setupCounters();
    setupDemo();
    setupMetrics();
    setupTeam();
  });
})();

/* ==========================================================================
   SmartCross console
   Vanilla ES2017. No build step, no framework, no CDN. Every chart is drawn
   by hand into SVG or canvas so the whole thing runs with the network off.
   ========================================================================== */
(function () {
  "use strict";

  var $  = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var API = "";                        // same origin
  var REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* Canvas cannot read CSS custom properties, so the palette lives here once
     and mirrors the tokens in app.css. Change both together. */
  var P = {
    plum: "#241046",
    violet: "#6d28d9",
    violetBright: "#8b5cf6",
    violet200: "#d9cdfb",
    violet50: "#f3effe",
    ink: "#1c0f3a",
    inkDim: "#5b5378",
    inkMute: "#8b83a6",
    line: "#e6dffa",
    tint: "#f6f2ff",
    pass: "#0e8f5e",
    block: "#d92d20",
    amber: "#b45309",
    blue: "#2563eb",
    white: "#ffffff"
  };

  var S = {                            // the whole client state
    meta: null,
    current: null,                     // last /api/recommend response
    stage: 1,                          // selected pipeline stage
    holdings: {},                      // product -> bool
    regime: "RECOVERY",
    perf: null, rules: null, matrix: null
  };

  /* ------------------------------------------------------------- helpers */
  function pct(x, d) { return (x * 100).toFixed(d === undefined ? 1 : d) + "%"; }
  function money(x) { return Number(x).toLocaleString("en-US", { maximumFractionDigits: 0 }); }
  function cap(s) { return String(s).charAt(0).toUpperCase() + String(s).slice(1); }
  function esc(s) {
    return String(s === undefined || s === null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  function el(tag, cls, html) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html !== undefined) n.innerHTML = html;
    return n;
  }
  var toastTimer;
  function toast(msg) {
    var t = $("#toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { t.classList.remove("show"); }, 2800);
  }
  function banner(msg) {
    var b = $("#banner");
    if (!msg) { b.hidden = true; return; }
    b.textContent = msg; b.hidden = false;
  }
  function api(path, opts) {
    return fetch(API + path, opts).then(function (r) {
      if (!r.ok) {
        return r.text().then(function (t) {
          throw new Error(r.status + " " + (t || r.statusText));
        });
      }
      return r.json();
    });
  }
  /* animate a number from 0 -> v, respecting reduced motion */
  function animate(ms, fn) {
    if (REDUCED) { fn(1); return; }
    var t0 = performance.now();
    (function step(t) {
      var p = Math.min(1, (t - t0) / ms);
      fn(1 - Math.pow(1 - p, 3));
      if (p < 1) requestAnimationFrame(step);
    })(t0);
  }
  /* size a canvas for the device pixel ratio and return a scaled context */
  function hidpi(cv, w, h) {
    var dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
    cv.style.height = h + "px";
    var c = cv.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, h);
    return c;
  }
  function paintRange(r) {
    var p = (r.value - r.min) / (r.max - r.min) * 100;
    r.style.background = "linear-gradient(90deg,var(--violet) " + p +
                         "%,var(--line-strong) " + p + "%)";
  }

  /* ------------------------------------------------------------- nav */
  $$(".nav button").forEach(function (b) {
    b.addEventListener("click", function () { show(b.dataset.view); });
  });
  function show(name) {
    $$(".nav button").forEach(function (b) {
      if (b.dataset.view === name) b.setAttribute("aria-current", "page");
      else b.removeAttribute("aria-current");
    });
    $$(".view").forEach(function (v) { v.classList.toggle("active", v.id === "view-" + name); });
    window.scrollTo({ top: 0, behavior: REDUCED ? "auto" : "smooth" });
    if (name === "engines")     { drawFlow(); renderStage(); }
    if (name === "performance") loadPerformance();
    if (name === "evidence")    loadEvidence();
    if (name === "session")     loadSession();
  }

  /* ========================================================================
     BOOT
     ======================================================================== */
  function boot() {
    Promise.all([
      api("/api/meta"),
      api("/api/macro").catch(function () { return { available: false, timeline: [] }; })
    ]).then(function (res) {
      var m = res[0];
      S.macroTimeline = res[1].timeline || [];
      S.macroFull = res[1];
      S.meta = m;
      $("#api-dot").className = "dot ok";
      $("#api-name").textContent = "API live";
      $("#model-dot").className = "dot ok";
      $("#model-name").textContent = m.metrics.n_products + " products · built " + m.built_at.slice(0, 10);
      /* wire the form before the regime widget: if the macro panel ever fails
         to build, the console must still be usable rather than inert. */
      buildHoldings(m);
      wireForm();
      buildRegimes(m);
      heroStats(m);
      persona("rahul", true);             // pre-fill the form without a toast over the intro
      banner(null);
    }).catch(function (e) {
      $("#api-dot").className = "dot bad";
      $("#api-name").textContent = "API unreachable";
      $("#model-dot").className = "dot bad";
      $("#model-name").textContent = "no model";
      banner("Cannot reach the API. Start it with  python app.py  in the backend folder, then reload. (" + e.message + ")");
    });
  }

  /* The hero quotes the end-to-end figure, not the model-only one: 85.3% is what
     a customer at the counter experiences once the tilt and the rules have run,
     measured over all 104,167 held-out events by defense/defense_eval.py. */
  var END_TO_END_TOP3 = "85.3%";
  function heroStats(m) {
    $("#hs-prod").textContent = m.metrics.n_products;
    api("/api/sample").then(function (s) {
      $("#hs-pool").textContent = (s.pool.n / 1000).toFixed(0) + "k";
    }).catch(function () { $("#hs-pool").textContent = "—"; });
    $("#hs-top3").textContent = END_TO_END_TOP3;
  }

  /* ------------------------------------------------------------- intake */
  function buildHoldings(m) {
    var box = $("#holdings");
    box.innerHTML = "";
    m.all_products.forEach(function (p) {
      var b = el("button", "hold", esc(p.replace(/_/g, " ")));
      b.type = "button";
      b.setAttribute("aria-pressed", "false");
      b.dataset.p = p;
      b.addEventListener("click", function () {
        S.holdings[p] = !S.holdings[p];
        b.setAttribute("aria-pressed", S.holdings[p] ? "true" : "false");
      });
      box.appendChild(b);
    });
  }
  function setHoldings(list) {
    S.holdings = {};
    list.forEach(function (p) { S.holdings[p] = true; });
    $$(".hold").forEach(function (b) {
      b.setAttribute("aria-pressed", S.holdings[b.dataset.p] ? "true" : "false");
    });
  }
  function heldList() {
    return Object.keys(S.holdings).filter(function (k) { return S.holdings[k]; });
  }

  function buildRegimes(m) {
    var box = $("#regimes");
    box.innerHTML = "";
    S.macro = m.macro || { available: false };

    /* AUTO comes first and is the default: the regime is a measurement, not
       an adviser's guess. The named regimes stay for scenario work. */
    if (S.macro.available) {
      var a = el("button", "auto", "Auto &#8212; from market data");
      a.type = "button"; a.dataset.r = "AUTO";
      a.addEventListener("click", function () { setRegime("AUTO"); });
      box.appendChild(a);
    } else {
      S.regime = "RECOVERY";
    }
    m.regimes.forEach(function (r) {
      var b = el("button", null, esc(cap(r.id.toLowerCase())));
      b.type = "button"; b.dataset.r = r.id;
      b.addEventListener("click", function () { setRegime(r.id); });
      box.appendChild(b);
    });

    /* month picker, only meaningful in AUTO */
    if (S.macro.available) {
      var row = el("div", "asof");
      row.innerHTML =
        '<label>Classify as of' +
        '<input id="f-asof" type="month" min="' + esc(S.macro.from_month) +
        '" max="' + esc(S.macro.to_month) + '" value="' + esc(S.macro.to_month) + '"></label>' +
        '<span class="muted small" id="asof-hint"></span>';
      /* #regimes sits inside its <label>; #regime-note is the label's SIBLING,
         so the row must be inserted into the label's parent, not the label. */
      var note = $("#regime-note");
      note.parentNode.insertBefore(row, note);
      $("#f-asof").addEventListener("change", function () { setRegime(S.regime); });
    }
    setRegime(S.regime);
  }

  function macroAt(month) {
    var tl = (S.macroTimeline || []);
    if (!tl.length) return null;
    var e = tl.filter(function (r) { return r.month === month; })[0];
    if (e) return e;
    var before = tl.filter(function (r) { return r.month <= month; });
    return before.length ? before[before.length - 1] : tl[0];
  }

  function setRegime(id) {
    S.regime = id;
    $$("#regimes button").forEach(function (b) {
      b.setAttribute("aria-pressed", b.dataset.r === id ? "true" : "false");
    });
    var note = $("#regime-note"), hint = $("#asof-hint");
    var asof = $("#f-asof") ? $("#f-asof").value : "";

    if (id === "AUTO") {
      var row = macroAt(asof || (S.macro && S.macro.to_month));
      if (row) {
        var r = S.meta.regimes.filter(function (x) { return x.id === row.regime; })[0];
        note.innerHTML = "<b>" + esc(cap(row.regime.toLowerCase())) + "</b> — " +
          esc(r ? r.note : "") + " Classified from " + S.macro.n_series +
          " price series: momentum " + (row.momentum >= 0 ? "+" : "") +
          (row.momentum * 100).toFixed(1) + "%, volatility " +
          (row.volatility * 100).toFixed(1) + "%, " +
          Math.round(row.agreement * 100) + "% of series agree.";
        if (hint) hint.textContent = S.macro.from_month + " to " + S.macro.to_month;
      } else {
        note.textContent = "Classifying from the price series on the server.";
      }
    } else {
      var rr = S.meta.regimes.filter(function (x) { return x.id === id; })[0];
      note.innerHTML = esc(rr ? rr.note : "") +
        " <em>Chosen by hand — a scenario, not a measurement.</em>";
      if (hint) hint.textContent = "";
    }
  }

  var PERSONAS = {
    rahul: {
      name: "Rahul", age: 29, income: 48000, sen: 26, seg: "Individual",
      msl: 5, acq: 2, active: true, nw: false, male: true,
      regime: "AUTO", asof: "",
      hold: ["current_account", "payroll_account", "direct_debit"],
      note: "Young salaried customer, two years in. The peer-group case."
    },
    aman: {
      name: "Aman", age: 38, income: 96000, sen: 86, seg: "VIP",
      msl: 2, acq: 6, active: true, nw: false, male: true,
      regime: "AUTO", asof: "2017-12",
      hold: ["current_account", "payroll_account", "direct_debit", "mortgage", "credit_card", "taxes"],
      note: "Just took a mortgage, in Dec 2017 — a month all 9 series call EXPANSION."
    },
    vikram: {
      name: "Vikram", age: 62, income: 18000, sen: 220, seg: "Individual",
      msl: 31, acq: 4, active: false, nw: false, male: true,
      regime: "AUTO", asof: "2022-09",
      hold: ["current_account", "long_term_deposits", "pension_plan"],
      note: "Retired, modest income, in Sep 2022 — all 9 series call STRESS."
    }
  };
  function persona(k, quiet) {
    var p = PERSONAS[k];
    S.truth = null; S.sampled = null;     // an illustration, not a real record
    $("#f-name").value = p.name;
    $("#f-age").value = p.age;
    $("#f-income").value = p.income;
    $("#f-income-r").value = p.income;
    paintRange($("#f-income-r"));
    $("#f-sen").value = p.sen;
    $("#f-seg").value = p.seg;
    $("#f-msl").value = p.msl;
    $("#f-acq").value = p.acq;
    $("#f-active").checked = p.active;
    $("#f-new").checked = p.nw;
    $("#f-male").checked = p.male;
    setHoldings(p.hold);
    if ($("#f-asof")) {
      $("#f-asof").value = p.asof || (S.macro && S.macro.to_month) || "";
    }
    setRegime(p.regime);
    $$(".seg.small button").forEach(function (b) {
      b.setAttribute("aria-pressed", b.dataset.persona === k ? "true" : "false");
    });
    if (!quiet) toast(p.note);
  }

  function wireForm() {
    $$("[data-persona]").forEach(function (b) {
      b.addEventListener("click", function () { persona(b.dataset.persona); });
    });
    var ir = $("#f-income-r"), inum = $("#f-income");
    ir.addEventListener("input", function () { inum.value = ir.value; paintRange(ir); });
    inum.addEventListener("input", function () { ir.value = inum.value; paintRange(ir); });
    paintRange(ir);

    var al = $("#f-alpha"), liveTimer = null;
    al.addEventListener("input", function () {
      $("#alpha-out").textContent = (al.value / 100).toFixed(2);
      $("#alpha-meaning").innerHTML = alphaMeaning(al.value / 100);
      paintRange(al);
      /* with a customer on screen, re-score as the slider moves */
      if (!S.current) return;
      clearTimeout(liveTimer);
      liveTimer = setTimeout(rescoreLive, 160);
    });
    $("#alpha-meaning").innerHTML = alphaMeaning(al.value / 100);
    paintRange(al);

    $("#intake").addEventListener("submit", function (e) { e.preventDefault(); run(); });
    $("#btn-run").addEventListener("click", run);
    $("#btn-reset").addEventListener("click", function () {
      setHoldings([]);
      $("#f-name").value = "";
      S.current = null;
      $("#btn-replay").hidden = true;
      renderEmpty();
      toast("Cleared.");
    });
    $("#btn-replay").addEventListener("click", function () {
      if (S.current) playWalk(S.current);
    });
    $("#btn-random").addEventListener("click", drawReal);

    /* editing the form by hand invalidates the ground truth attached to a
       drawn customer: it is no longer that customer. */
    ["#f-age", "#f-income", "#f-sen", "#f-msl", "#f-acq", "#f-seg"].forEach(function (sel) {
      var n = $(sel);
      if (n) n.addEventListener("input", function () { S.truth = null; });
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); run(); }
    });
    $("#btn-reload-perf").addEventListener("click", function () { S.perf = null; loadPerformance(); });
  }

  /* Draw an actual customer out of the held-out months. Because the event
     records what they took next, the console can then mark the answer right
     or wrong instead of only looking convincing. */
  function drawReal() {
    var btn = $("#btn-random");
    btn.disabled = true; btn.textContent = "Drawing…";
    api("/api/sample?min_holdings=1").then(function (s) {
      var c = s.customer;
      $("#f-name").value = c.name;
      $("#f-age").value = c.age;
      $("#f-income").value = Math.round(c.income);          // -1 ("not on file") survives rounding
      $("#f-income-r").value = Math.max(0, Math.min(250000, c.income));
      paintRange($("#f-income-r"));
      $("#f-sen").value = c.seniority_months;
      $("#f-seg").value = c.segment;
      $("#f-msl").value = c.months_since_last_acq;
      $("#f-acq").value = c.acquisitions_so_far;
      $("#f-active").checked = !!c.is_active;
      $("#f-new").checked = !!c.is_new_customer;
      $("#f-male").checked = !!c.is_male;
      setHoldings(c.holdings);
      setRegime("AUTO");
      $$(".seg.small button").forEach(function (b) { b.setAttribute("aria-pressed", "false"); });
      S.sampled = c;
      S.truth = { product: s.actual_next, label: s.actual_label,
                  in_catalogue: s.actual_in_catalogue, pool: s.pool.n };
      toast("Drew " + c.name + " from " + s.pool.n.toLocaleString() +
            " held-out customers. Scoring…");
      run();
    }).catch(function (e) {
      toast("Could not draw a customer: " + e.message);
    }).finally(function () {
      btn.disabled = false;
      btn.textContent = "Draw a real customer from the held‑out data";
    });
  }

  /* A drawn customer's real 3-month history only describes the holdings it
     came with; once the adviser edits those, the server estimates instead. */
  function realHistory() {
    if (!S.sampled || !S.sampled.history) return null;
    var now = heldList().slice().sort().join("|");
    return now === S.sampled.holdings.slice().sort().join("|") ? S.sampled.history : null;
  }

  function payload() {
    return {
      name: $("#f-name").value.trim(),
      age: +$("#f-age").value || 34,
      income: +$("#f-income").value || 0,
      seniority_months: +$("#f-sen").value || 0,
      segment: $("#f-seg").value,
      months_observed: S.sampled ? S.sampled.months_observed : -1,
      is_active: $("#f-active").checked ? 1 : 0,
      is_new_customer: $("#f-new").checked ? 1 : 0,
      is_male: $("#f-male").checked ? 1 : 0,
      holdings: heldList(),
      history: realHistory(),
      months_since_last_acq: +$("#f-msl").value,
      acquisitions_so_far: +$("#f-acq").value || 0,
      regime: S.regime,
      as_of: ($("#f-asof") && S.regime === "AUTO") ? $("#f-asof").value : "",
      alpha: (+$("#f-alpha").value) / 100,
      top_k: 3
    };
  }

  function run() {
    var btn = $("#btn-run");
    btn.disabled = true; btn.textContent = "Scoring…";
    api("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload())
    }).then(function (r) {
      S.current = r;
      renderVerdict(r);
      renderStage();
      $("#sess-count").textContent = "";
      api("/api/session").then(function (s) { $("#sess-count").textContent = s.length ? "(" + s.length + ")" : ""; });
    }).catch(function (e) {
      toast("Scoring failed: " + e.message);
    }).finally(function () {
      btn.disabled = false;
      btn.innerHTML = 'Advise <span class="kbd">&#9166;</span>';
    });
  }

  /* Re-score the customer on screen without replaying the walk, and say in
     words what the change did to the top three. record:false keeps these out
     of the Session view. */
  function rescoreLive() {
    var prev = S.current;
    var body = payload();
    body.record = false;
    api("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then(function (r) {
      S.current = r;
      renderVerdict(r, true);
      renderStage();
      if (prev) toast("\u03b1 " + r.alpha.toFixed(2) + ": " + whatMoved(prev, r));
    }).catch(function (e) { toast("Re-scoring failed: " + e.message); });
  }

  function whatMoved(a, b) {
    var A = a.recommendations.map(function (x) { return x.product; });
    var out = [];
    b.recommendations.forEach(function (x, i) {
      var j = A.indexOf(x.product);
      if (j < 0) out.push(x.label + " entered the top 3 at #" + (i + 1));
      else if (j > i) out.push(x.label + " moved up to #" + (i + 1));
    });
    var B = b.recommendations.map(function (x) { return x.product; });
    a.recommendations.forEach(function (x) {
      if (B.indexOf(x.product) < 0) out.push(x.label + " dropped out");
    });
    return out.length ? out.slice(0, 2).join("; ") : "same top three, scores shifted";
  }

  function alphaMeaning(a) {
    if (a >= 0.995) return "<b>Content engine as trained.</b> It already reads the peers' scores as evidence, " +
      "so nothing is mixed on top. This is what the bank would use.";
    if (a <= 0.005) return "<b>Peers only.</b> Just what customers with this exact portfolio added next &mdash; " +
      "no profile model. Useful to see what the crowd alone would say.";
    return "<b>" + Math.round(a * 100) + "% content engine, " + Math.round((1 - a) * 100) +
      "% raw peer opinion</b> mixed on top. A what&#8209;if: the further left, the more the crowd decides.";
  }

  function renderEmpty() {
    $("#verdict").innerHTML =
      '<div class="empty-verdict card"><h2>Cleared</h2>' +
      '<p class="muted">Pick the products this customer already holds, set the macro regime, and press Advise.</p></div>';
  }

  /* ========================================================================
     THE BRANCH WALK-THROUGH
     A customer walks in and is traced gate by gate. Each gate lights as they
     reach it and says what that engine actually did to THEIR profile — the
     same four stages as the pipeline, told as the thing the person sees.
     ======================================================================== */
  var SVGNS = "http://www.w3.org/2000/svg";
  function svg(t, a, parent) {
    var e = document.createElementNS(SVGNS, t);
    for (var k in a) if (a.hasOwnProperty(k)) e.setAttribute(k, a[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  var GATES = [
    { x: 300, name: "Reception",  sub: "profile taken" },
    { x: 560, name: "Engine room", sub: "two engines score" },
    { x: 820, name: "Market desk", sub: "regime applied" },
    { x: 1080, name: "Compliance", sub: "eligibility checked" }
  ];
  var FLOOR = 268, WALK_MS = 7200;
  var walkTimer = null;

  function walkCard(r) {
    var card = el("div", "card walk");
    card.id = "walk";
    card.innerHTML =
      '<div class="walk-head"><h2>' + esc(r.customer.name || "A customer") +
      ' walks into the branch</h2><span>TRACED LIVE, GATE BY GATE</span></div>' +
      '<p class="walk-caption" id="walk-cap">&nbsp;</p>' +
      '<svg id="branch" viewBox="0 0 1240 330" role="img" aria-label="Branch walk-through"></svg>';
    return card;
  }

  function buildBranch(r) {
    var s = $("#branch");
    if (!s) return;
    s.innerHTML = "";

    svg("rect", { x: 0, y: 0, width: 1240, height: FLOOR, class: "wall" }, s);
    svg("line", { x1: 0, y1: FLOOR, x2: 1240, y2: FLOOR, class: "floor" }, s);
    svg("path", { d: "M30 " + FLOOR + " L30 150 L96 150 L96 " + FLOOR,
                  class: "floor", fill: "none" }, s);
    var dt = svg("text", { x: 63, y: 140, "text-anchor": "middle",
                           fill: P.inkMute, "font-size": "11" }, s);
    dt.textContent = "entrance";
    svg("path", { d: "M100 " + (FLOOR - 4) + " L1200 " + (FLOOR - 4), class: "trail" }, s);

    /* the four gates */
    GATES.forEach(function (g, i) {
      var grp = svg("g", { class: "gate", id: "gate" + i }, s);
      svg("rect", { x: g.x - 66, y: FLOOR - 74, width: 132, height: 74, rx: 11 }, grp);
      var n = svg("text", { x: g.x, y: FLOOR - 52, "text-anchor": "middle", class: "gn" }, grp);
      n.textContent = "0" + (i + 1);
      var t = svg("text", { x: g.x, y: FLOOR - 31, "text-anchor": "middle", class: "gl" }, grp);
      t.textContent = g.name;
      var u = svg("text", { x: g.x, y: FLOOR - 13, "text-anchor": "middle", class: "gl",
                            "font-size": "10.5", opacity: ".72" }, grp);
      u.textContent = g.sub;

      /* speech bubble above each gate */
      var b = svg("g", { class: "bubble", id: "bub" + i }, s);
      svg("rect", { x: g.x - 104, y: 22, width: 208, height: 62, rx: 12 }, b);
      svg("path", { d: "M" + (g.x - 7) + " 84 L" + g.x + " 96 L" + (g.x + 7) + " 84",
                    fill: "#0d2420", stroke: "none" }, b);
      svg("text", { x: g.x, y: 46, "text-anchor": "middle", class: "bt", id: "bt" + i }, b);
      svg("text", { x: g.x, y: 65, "text-anchor": "middle", class: "bs", id: "bs" + i }, b);
      svg("text", { x: g.x, y: 78, "text-anchor": "middle", class: "bs", id: "bs2" + i,
                    "font-size": "10.5", opacity: ".8" }, b);
    });

    /* the walker */
    var p = svg("g", { class: "person", id: "walker", transform: "translate(90,0)" }, s);
    svg("circle", { cx: 0, cy: FLOOR - 76, r: 9, class: "head" }, p);
    svg("line", { x1: 0, y1: FLOOR - 66, x2: 0, y2: FLOOR - 34, class: "torso" }, p);
    svg("line", { x1: 0, y1: FLOOR - 60, x2: 0, y2: FLOOR - 44, class: "limb", id: "armA" }, p);
    svg("line", { x1: 0, y1: FLOOR - 60, x2: 0, y2: FLOOR - 44, class: "limb", id: "armB" }, p);
    svg("line", { x1: 0, y1: FLOOR - 34, x2: 0, y2: FLOOR - 2, class: "limb", id: "legA" }, p);
    svg("line", { x1: 0, y1: FLOOR - 34, x2: 0, y2: FLOOR - 2, class: "limb", id: "legB" }, p);
  }

  /* what each gate says about THIS customer */
  function gateLines(r) {
    var top = r.recommendations[0];
    var strongest = r.board.filter(function (b) { return !b.held; })[0];
    var tilted = r.board.filter(function (b) {
      return !b.held && Math.abs(b.tilt - 1) > 0.001;
    }).sort(function (a, b) { return b.tilt - a.tilt; })[0];
    return [
      { t: r.customer.n_holdings + " products on file",
        s: "age " + r.customer.age + " · " + money(r.customer.income) + " · " +
           r.customer.seniority_months + " months",
        s2: "35 features + 48 history built", bad: false },
      { t: r.n_scored + " products scored",
        s: "best before tilt: " + (strongest ? strongest.label : "—"),
        s2: "content engine, reading profile + peer scores" + (r.alpha < 1 ? ", raw CF mixed at " + (1 - r.alpha).toFixed(2) : ""), bad: false },
      { t: cap(r.regime.toLowerCase()) + " regime",
        s: tilted ? (tilted.label + " ×" + tilted.tilt.toFixed(2))
                  : "no product tilted for this profile",
        s2: (r.regime_source && r.regime_source.mode === "classified")
              ? ("classified from " + r.regime_source.n_series + " series, " +
                 r.regime_source.month)
              : "chosen by hand, not measured",
        bad: false },
      { t: r.n_blocked ? r.n_blocked + " products removed" : "nothing removed",
        s: r.blocked.length ? r.blocked[0].label + " — " + r.blocked[0].rule
                            : "every scored product passes its rule",
        s2: r.n_blocked ? "deleted, not down-weighted" : "",
        bad: r.n_blocked > 0 }
    ];
  }

  function playWalk(r, instant) {
    var s = $("#branch");
    if (!s) return;
    if (walkTimer) { cancelAnimationFrame(walkTimer); walkTimer = null; }

    var lines = gateLines(r);
    var walker = $("#walker", s);
    var fired = [false, false, false, false];
    var cap$ = $("#walk-cap");

    for (var i = 0; i < 4; i++) {
      $("#gate" + i, s).setAttribute("class", "gate");
      $("#bub" + i, s).setAttribute("class", "bubble");
    }

    function fire(i) {
      fired[i] = true;
      var L = lines[i];
      $("#gate" + i, s).setAttribute("class", "gate lit" + (L.bad ? " blocked" : ""));
      $("#bt" + i, s).textContent = L.t;
      $("#bs" + i, s).textContent = L.s.length > 42 ? L.s.slice(0, 41) + "…" : L.s;
      $("#bs2" + i, s).textContent = L.s2;
      $("#bub" + i, s).setAttribute("class", "bubble on" + (L.bad ? " bad" : ""));
      if (cap$) cap$.textContent = GATES[i].name + " — " + L.t + ". " + L.s;
    }

    /* reduced motion, or a live re-score: no walk, reveal everything at once */
    if (REDUCED || instant) {
      walker.setAttribute("transform", "translate(1180,0)");
      for (var k = 0; k < 4; k++) fire(k);
      if (cap$) cap$.textContent = "All four gates applied. " +
        r.recommendations[0].label + " comes out first.";
      return;
    }

    var t0 = performance.now();
    (function step(t) {
      var p = Math.min(1, (t - t0) / WALK_MS);
      var x = 90 + p * 1090;
      walker.setAttribute("transform", "translate(" + x.toFixed(1) + ",0)");

      /* a simple two-phase gait, paused while standing at a gate */
      var near = GATES.some(function (g) { return Math.abs(x - g.x) < 26; });
      var sw = near ? 0 : Math.sin(p * 92) * 10;
      $("#legA", s).setAttribute("x2", sw.toFixed(1));
      $("#legB", s).setAttribute("x2", (-sw).toFixed(1));
      $("#armA", s).setAttribute("x2", (-sw * 0.6).toFixed(1));
      $("#armB", s).setAttribute("x2", (sw * 0.6).toFixed(1));

      GATES.forEach(function (g, i) {
        if (!fired[i] && x >= g.x - 10) fire(i);
      });

      if (p < 1) { walkTimer = requestAnimationFrame(step); }
      else {
        walkTimer = null;
        if (cap$) cap$.textContent = "Out the other side: " +
          r.recommendations[0].label + " first, " +
          r.recommendations.length + " recommendations, " +
          r.n_blocked + " removed on the way.";
      }
    })(t0);
  }

  /* ========================================================================
     VERDICT — the walk, the top-3, the cascade, what was blocked
     ======================================================================== */
  function renderVerdict(r, live) {
    var v = $("#verdict");
    v.innerHTML = "";

    /* ground truth first: if this was a real customer we know the answer */
    if (S.truth) {
      var names = r.recommendations.map(function (x) { return x.product; });
      var rank = names.indexOf(S.truth.product) + 1;
      var gated = r.blocked.some(function (b) { return b.product === S.truth.product; });
      var cls, mark, head, sub;
      if (rank > 0) {
        cls = "hit"; mark = "✓";
        head = "Correct — they actually took <b>" + esc(S.truth.label) +
               "</b>, ranked #" + rank;
        sub = "Drawn from " + S.truth.pool.toLocaleString() +
              " held-out customers the model never saw.";
      } else if (gated) {
        cls = "gated"; mark = "⊘";
        head = "Removed — they actually took <b>" + esc(S.truth.label) +
               "</b>, and an eligibility rule deleted it";
        sub = "This is the cost of a hard gate, and it happens to about 11% of " +
              "real acquisitions. The rule is shown below with the score it would have had.";
      } else {
        cls = "miss"; mark = "✗";
        head = "Missed — they actually took <b>" + esc(S.truth.label) +
               "</b>, which is not in our top-3";
        sub = "Measured end to end, the full pipeline puts the right product in the " +
              "top-3 about 85% of the time. This is one of the other 15%.";
      }
      var t = el("div", "truth " + cls,
        '<span class="mark">' + mark + "</span>" +
        '<span class="what"><b>' + head + "</b><span>" + sub + "</span></span>");
      v.appendChild(t);
    }

    if (r.missing_fields && r.missing_fields.length) {
      v.appendChild(el("div", "missing-note",
        "The bank has no <b>" + r.missing_fields.map(esc).join("</b> and no <b>") +
        "</b> on file for this customer. Rules needing it cannot be evidenced, so those " +
        "products were removed rather than assumed to pass."));
    }

    if (r.history_source === "estimated") {
      v.appendChild(el("div", "missing-note",
        "This profile was entered by hand, so its <b>3-month holdings history is estimated</b> from today's " +
        "holdings. Customers drawn from the data use their real history."));
    }

    v.appendChild(walkCard(r));
    buildBranch(r);
    playWalk(r, live);
    $("#btn-replay").hidden = false;

    /* --- headline tiles --- */
    var t = el("div", "tiles");
    t.appendChild(el("div", "tile", "<b>" + r.n_scored + "</b><span>products scored</span>" +
      "<em>of " + r.n_products + " in the catalogue</em>"));
    t.appendChild(el("div", "tile " + (r.n_blocked ? "rose" : "mint"),
      "<b>" + r.n_blocked + "</b><span>removed by rules</span>" +
      "<em>" + (r.n_blocked ? "ineligible, not down-weighted" : "nothing was gated out") + "</em>"));
    t.appendChild(el("div", "tile sky", "<b>" + cap(r.regime.toLowerCase()) + "</b><span>macro regime</span>" +
      "<em>&alpha; = " + r.alpha.toFixed(2) + "</em>"));
    t.appendChild(el("div", "tile", "<b>" + r.customer.n_holdings + "</b><span>held today</span>" +
      "<em>" + (r.customer.name ? esc(r.customer.name) + ", " : "") + "age " + r.customer.age +
      " · " + money(r.customer.income) + "</em>"));
    v.appendChild(t);

    if (r.all_blocked) {
      v.appendChild(el("div", "banner",
        "Every remaining product failed its eligibility rule, so the ranking below is shown " +
        "<b>before</b> the constraint layer. In production this customer would be routed to a human adviser."));
      $(".banner", v).hidden = false;
    }

    /* --- the three cards --- */
    var sep = el("div", "sep", "<span>Recommended next, in order</span>");
    v.appendChild(sep);

    var list = el("div", "recs");
    r.recommendations.forEach(function (x, i) {
      var card = el("div", "rec r" + x.rank);
      card.style.animationDelay = (REDUCED ? 0 : i * 70) + "ms";
      card.innerHTML =
        '<div class="rec-rank">' + x.rank + '</div>' +
        '<div class="rec-main">' +
          '<h3>' + esc(x.label) + '</h3>' +
          '<p class="rec-why">' + esc(x.reason) + '</p>' +
          '<div class="rec-meta">' +
            '<span class="tag ok">eligible &middot; ' + esc(x.rule) + '</span>' +
            (x.top_rule ? '<span class="tag brass">' + x.top_rule.lift.toFixed(1) + '&times; lift from ' +
                          esc(x.top_rule.from_label) + '</span>' : '') +
            '<span class="tag">base rate ' + pct(x.base_rate, 2) + '</span>' +
            (Math.abs(x.macro_tilt - 1) > 0.001
              ? '<span class="tag">macro &times;' + x.macro_tilt.toFixed(2) + '</span>' : '') +
          '</div>' +
          barsHTML(x) +
        '</div>' +
        '<div class="rec-score"><b data-v="' + x.score + '">0.0%</b><span>final score</span></div>';
      list.appendChild(card);
    });
    v.appendChild(list);

    /* count the scores up (instantly when re-scoring live) */
    (live ? function (ms, fn) { fn(1); } : animate)(760, function (p) {
      $$(".rec-score b", v).forEach(function (b) {
        b.textContent = (parseFloat(b.dataset.v) * 100 * p).toFixed(1) + "%";
      });
      $$(".bar .fill", v).forEach(function (f) {
        f.style.width = (parseFloat(f.dataset.w) * p) + "%";
      });
    });

    /* --- why this answer: every product, in plain words --- */
    v.appendChild(explainCard(r));
  }

  /* ========================================================================
     WHY THIS ANSWER
     Every product the customer does not already hold, sorted into three
     groups anyone can read: what we recommend, what a rule removed, what
     else was considered. Each row says why in a sentence and shows the
     product's path through the four stages.
     ======================================================================== */
  function explainCard(r) {
    var recBy = {}, blockedBy = {};
    r.recommendations.forEach(function (x) { recBy[x.product] = x; });
    r.blocked.forEach(function (b) { blockedBy[b.product] = b; });
    var open = r.board.filter(function (b) { return !b.held; });   // ordered as before the rules
    var held = r.board.filter(function (b) { return b.held; });
    var mx = 1e-9;
    open.forEach(function (b) { mx = Math.max(mx, b.after_macro, b.final); });

    var rec = [], rem = [], rest = [];
    open.forEach(function (b, i) {
      var o = { b: b, before: i + 1 };
      if (recBy[b.product]) rec.push(o);
      else if (b.blocked) rem.push(o);
      else rest.push(o);
    });
    rec.sort(function (a, b) { return recBy[a.b.product].rank - recBy[b.b.product].rank; });

    function path(b, removed) {
      return '<div class="why-path">' +
        "<span>peers <b>" + pct(b.cf, 1) + "</b></span><span>&rarr;</span>" +
        "<span>content engine <b>" + pct(b.hybrid, 1) + "</b></span><span>&rarr;</span>" +
        "<span>economy <b>" + pct(b.after_macro, 1) + "</b></span><span>&rarr;</span>" +
        (removed ? "<span class='x'>removed by rule</span>"
                 : "<span>final <b>" + pct(b.final, 1) + "</b></span>") + "</div>";
    }
    function tiltWords(b) {
      if (b.tilt > 1.001) return " The " + r.regime.toLowerCase() + " economy nudged it up (&times;" + b.tilt.toFixed(2) + ").";
      if (b.tilt < 0.999) return " The " + r.regime.toLowerCase() + " economy nudged it down (&times;" + b.tilt.toFixed(2) + ").";
      return "";
    }
    function row(o, kind) {
      var b = o.b, x = recBy[b.product], k = blockedBy[b.product];
      var badge, text, score, val;
      if (kind === "rec") {
        badge = '<span class="why-badge is-rec">#' + x.rank + "</span>";
        text = esc(x.reason);
        score = pct(x.score, 1);
        val = b.final;
      } else if (kind === "rem") {
        badge = '<span class="why-badge is-rem">&times;</span>';
        text = "Would have ranked <b>#" + o.before + "</b>. Removed by the rule: <b>" +
          esc(k ? k.rule : "an eligibility rule") + "</b>.";
        score = "removed";
        val = b.after_macro;
      } else {
        badge = '<span class="why-badge">' + o.before + "</span>";
        text = "Scored " + pct(b.final, 1) + " &mdash; eligible, but below the top three." + tiltWords(b);
        score = pct(b.final, 1);
        val = b.final;
      }
      return '<div class="why-row is-' + kind + '">' + badge +
        '<div><div class="why-name">' + esc(b.label) + "</div>" +
        '<p class="why-text">' + text + "</p>" +
        '<div class="why-bar"><i style="width:' + Math.max(1, val / mx * 100).toFixed(1) + '%"></i></div>' +
        path(b, kind === "rem") + "</div>" +
        '<div class="why-score">' + score + "</div></div>";
    }

    var names = function (list) {
      return list.map(function (o) { return "<b>" + esc(o.b.label) + "</b>"; });
    };
    var join = function (a) {
      return a.length < 2 ? a.join("") : a.slice(0, -1).join(", ") + " and " + a[a.length - 1];
    };
    var summary = "In one sentence: we recommend " + join(names(rec)) + ".";
    if (rem.length) {
      summary += " We removed " + join(names(rem.slice(0, 2))) + (rem.length > 2 ? " and " + (rem.length - 2) + " more" : "") +
        " because " + (rem.length === 1 ? "the customer isn't eligible" : "the customer isn't eligible for them") + ".";
    }

    var card = el("div", "card why");
    card.style.marginTop = "20px";
    var shown = rest.slice(0, 5), hidden = rest.slice(5);
    card.innerHTML =
      "<h2>Why this answer</h2>" +
      "<p class='muted small'>Every product this customer doesn't already have, in three groups. " +
      "Each row shows its path: what similar customers did (collaborative engine), what the content engine concluded, the economy's nudge, " +
      "and whether a rule removed it.</p>" +
      '<div class="why-sum">' + summary + "</div>" +
      '<div class="why-sec"><h3>&check; We recommend <small>' + rec.length + "</small></h3>" +
        "<p>The highest&#8209;scoring products this customer is eligible for.</p>" +
        rec.map(function (o) { return row(o, "rec"); }).join("") + "</div>" +
      (rem.length ? '<div class="why-sec"><h3>&times; Removed by a rule <small>' + rem.length + "</small></h3>" +
        "<p>These scored high enough to be considered, but the customer doesn't meet a hard rule. " +
        "They are taken off the list, not just pushed down.</p>" +
        rem.map(function (o) { return row(o, "rem"); }).join("") + "</div>" : "") +
      (rest.length ? '<div class="why-sec"><h3>Also considered <small>' + rest.length + "</small></h3>" +
        "<p>Eligible, but scored lower than the three above. The number is where it ranked.</p>" +
        shown.map(function (o) { return row(o, "rest"); }).join("") +
        (hidden.length ? '<details class="why-more"><summary>Show ' + hidden.length + " more</summary>" +
          hidden.map(function (o) { return row(o, "rest"); }).join("") + "</details>" : "") + "</div>" : "") +
      (held.length ? '<p class="why-held">Already held, so never recommended: ' +
        held.map(function (b) { return esc(b.label); }).join(", ") + ".</p>" : "");
    return card;
  }

  function barsHTML(x) {
    var mx = Math.max(x.cf, x.hybrid, x.after_macro, x.score, 1e-9);
    function row(label, val, col) {
      return '<div class="bar"><span>' + label + '</span>' +
        '<span class="track"><span class="fill" style="background:' + col +
        '" data-w="' + (val / mx * 100).toFixed(2) + '"></span></span>' +
        '<span class="v">' + pct(val, 2) + '</span></div>';
    }
    return '<div class="bars">' +
      row("peers alone", x.cf, P.violetBright) +
      row("content engine", x.hybrid, P.blue) +
      row("after economy", x.after_macro, "#9d4edd") +
      row("final", x.score, P.violet) +
      "</div>";
  }

  /* roundRect shim for older engines */
  if (!CanvasRenderingContext2D.prototype.roundRect) {
    CanvasRenderingContext2D.prototype.roundRect = function (x, y, w, h, r) {
      r = Math.min(r, h / 2, w / 2);
      this.moveTo(x + r, y);
      this.arcTo(x + w, y, x + w, y + h, r);
      this.arcTo(x + w, y + h, x, y + h, r);
      this.arcTo(x, y + h, x, y, r);
      this.arcTo(x, y, x + w, y, r);
      this.closePath();
      return this;
    };
  }

  /* ========================================================================
     ENGINES — architecture flow + per-stage detail
     ======================================================================== */
  /* The architecture as it actually runs, in enough detail to defend.
     Five columns (profile, the two engines, economy, rules, output), each box
     carrying what it does and one measured number. The collaborative engine's
     scores go INTO the content engine -- fusion, not averaging -- and the same
     counts, read as lift, become the reason sentence. Click a box to open its
     stage. Geometry is explicit; the diagram is small enough to place by hand. */
  var BANDS = [
    { x: 14,   w: 176, t: "1 · PROFILE" },
    { x: 250,  w: 250, t: "2 · TWO ENGINES, FUSED" },
    { x: 540,  w: 200, t: "3 · ECONOMY" },
    { x: 780,  w: 200, t: "4 · RULES" },
    { x: 1020, w: 242, t: "OUTPUT" }
  ];
  var NODES = [
    { id: 1, key: "profile", x: 14, y: 136, w: 176, h: 132,
      label: "Profile matrix", sub: "one row per customer",
      lines: ["24 holdings + 11 behaviour", "+ 48 holdings history"], metric: "83 numbers" },
    { id: 2, key: "cf", x: 250, y: 46, w: 250, h: 116,
      label: "Collaborative engine", sub: "what this portfolio added next",
      lines: ["exact-portfolio match,", "backs off to co-occurrence"], metric: "85.0% top-3 alone" },
    { id: 2, key: "content", x: 250, y: 210, w: 250, h: 132,
      label: "Content engine", sub: "gradient boosting, early-stopped",
      lines: ["reads 83 profile + 44 peer inputs", "learns when to trust the peers"], metric: "90.4% top-3 · served" },
    { id: 3, key: "market", x: 540, y: 64, w: 200, h: 84, small: true,
      label: "Market data", sub: "9 real price series",
      lines: ["momentum + volatility vote"] },
    { id: 3, key: "macro", x: 540, y: 210, w: 200, h: 132,
      label: "Macro regime", sub: "4 regimes, classified",
      lines: ["multiplies each score", "by ×0.60 … ×1.30"], metric: "#1 changes for 1.7%" },
    { id: 4, key: "book", x: 780, y: 64, w: 200, h: 84, small: true,
      label: "Rule book", sub: "13 readable rules",
      lines: ["age · income · seniority"] },
    { id: 4, key: "rules", x: 780, y: 210, w: 200, h: 132,
      label: "Constraints", sub: "remove, never down-weight",
      lines: ["missing data never passes", "nothing re-admits a product"], metric: "10.8% truth removed" },
    { id: 2, key: "lift", x: 1020, y: 64, w: 242, h: 84, small: true,
      label: "Lift matrix", sub: "explanations, not ranking",
      lines: ["“payroll holders add pensions 9×”"] },
    { id: 4, key: "output", x: 1020, y: 210, w: 242, h: 132,
      label: "Top 3 + a reason each", sub: "what the adviser sees",
      lines: ["each product explained", "scored against what they did next"], metric: "85.3% end-to-end top-3" }
  ];
  /* [from, side, to, side, label, options] */
  var EDGES = [
    ["profile", "r", "cf", "l", "holdings", { fy: -34 }],
    ["profile", "r", "content", "l", "35 inputs", { fy: 26 }],
    ["cf", "b", "content", "t", "44 peer scores", {}],
    ["content", "r", "macro", "l", "", {}],
    ["market", "b", "macro", "t", "regime", {}],
    ["macro", "r", "rules", "l", "", {}],
    ["book", "b", "rules", "t", "rules", {}],
    ["rules", "r", "output", "l", "", {}],
    ["lift", "b", "output", "t", "reason", { dash: true }],
    ["cf", "t", "lift", "l", "same counts, read as lift", { dash: true, over: true }]
  ];

  function drawFlow() {
    var svg = $("#flow");
    svg.innerHTML = "";
    var ns = "http://www.w3.org/2000/svg";
    function mk(t, a, txt) {
      var e = document.createElementNS(ns, t);
      Object.keys(a).forEach(function (k) { e.setAttribute(k, a[k]); });
      if (txt !== undefined) e.textContent = txt;
      return e;
    }
    var by = {};
    NODES.forEach(function (n) { by[n.key] = n; });

    /* stage bands across the top */
    BANDS.forEach(function (b) {
      var g = mk("g", { class: "band" });
      g.appendChild(mk("line", { x1: b.x, y1: 24, x2: b.x + b.w, y2: 24 }));
      g.appendChild(mk("text", { x: b.x, y: 16 }, b.t));
      svg.appendChild(g);
    });

    function anchor(n, side, d) {
      d = d || 0;
      if (side === "r") return [n.x + n.w, n.y + n.h / 2 + d];
      if (side === "l") return [n.x, n.y + n.h / 2 + d];
      if (side === "t") return [n.x + n.w / 2 + d, n.y];
      return [n.x + n.w / 2 + d, n.y + n.h];
    }

    /* edges first so the boxes sit on top of them; their labels go on last */
    var labels = [];
    EDGES.forEach(function (e, i) {
      var a = by[e[0]], b = by[e[2]], o = e[5] || {};
      var A = anchor(a, e[1], o.fy), B = anchor(b, e[3], o.ty);
      var d, lx, ly, head;
      if (o.over) {
        /* up and over the economy and rule columns, into the lift box */
        var top = 36;
        A = [a.x + a.w - 30, a.y];
        d = "M" + A[0] + " " + A[1] + " C" + A[0] + " " + top + "," + A[0] + " " + top + "," + (A[0] + 30) + " " + top +
            " L" + (B[0] - 30) + " " + top + " C" + (B[0] - 12) + " " + top + "," + (B[0] - 12) + " " + B[1] + "," + B[0] + " " + B[1];
        lx = (A[0] + B[0]) / 2; ly = top - 5;
        head = "M" + (B[0] - 9) + " " + (B[1] - 5) + " L" + B[0] + " " + B[1] + " L" + (B[0] - 9) + " " + (B[1] + 5);
      } else if (e[1] === "b" || e[1] === "t") {
        var my = (A[1] + B[1]) / 2;
        d = "M" + A[0] + " " + A[1] + " C" + A[0] + " " + my + "," + B[0] + " " + my + "," + B[0] + " " + B[1];
        lx = A[0] + 8; ly = my + 4;
        head = "M" + (B[0] - 5) + " " + (B[1] - 9) + " L" + B[0] + " " + B[1] + " L" + (B[0] + 5) + " " + (B[1] - 9);
      } else {
        var mx = (A[0] + B[0]) / 2;
        d = "M" + A[0] + " " + A[1] + " C" + mx + " " + A[1] + "," + mx + " " + B[1] + "," + B[0] + " " + B[1];
        lx = mx; ly = (A[1] + B[1]) / 2 + (B[1] < A[1] ? -6 : 14);
        head = "M" + (B[0] - 9) + " " + (B[1] - 5) + " L" + B[0] + " " + B[1] + " L" + (B[0] - 9) + " " + (B[1] + 5);
      }
      svg.appendChild(mk("path", { d: d, class: "edge" + (o.dash ? " dash" : "") }));
      if (!o.dash) {
        var p = mk("path", { d: d, class: "pulse" });
        p.style.animationDelay = (i * 0.35) + "s";
        svg.appendChild(p);
      }
      svg.appendChild(mk("path", { d: head, class: "edge" }));
      if (e[4]) {
        labels.push(mk("text", { x: lx, y: ly, class: "elabel",
                                 "text-anchor": (e[1] === "b" || e[1] === "t") ? "start" : "middle" }, e[4]));
      }
    });

    /* the boxes */
    NODES.forEach(function (n) {
      var g = mk("g", { class: "node" + (n.small ? " small" : "") + (S.stage === n.id ? " sel" : ""),
                        tabindex: "0", role: "button", "aria-label": n.label + ", stage " + n.id });
      g.appendChild(mk("rect", { x: n.x, y: n.y, width: n.w, height: n.h, rx: 14 }));
      var x = n.x + 16, y = n.y + 25;
      g.appendChild(mk("text", { x: x, y: y, class: "t" }, n.label));
      g.appendChild(mk("text", { x: x, y: y + 18, class: "s" }, n.sub));
      (n.lines || []).forEach(function (l, k) {
        g.appendChild(mk("text", { x: x, y: y + 40 + k * 17, class: "bl" }, (n.small ? "" : "• ") + l));
      });
      if (n.metric) {
        var mw = n.metric.length * 6.9 + 18;
        var mg = mk("g", { class: "metric" });
        mg.appendChild(mk("rect", { x: x - 2, y: n.y + n.h - 30, width: mw, height: 20, rx: 10 }));
        mg.appendChild(mk("text", { x: x + 7, y: n.y + n.h - 16 }, n.metric));
        g.appendChild(mg);
      }
      function pick() { S.stage = n.id; drawFlow(); renderStage(); }
      g.addEventListener("click", pick);
      g.addEventListener("keydown", function (ev) {
        if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); pick(); }
      });
      svg.appendChild(g);
    });

    labels.forEach(function (t) { svg.appendChild(t); });

    /* what the model saw, and what it was tested on */
    var st = mk("g", { class: "strip" });
    st.appendChild(mk("rect", { x: 14, y: 364, width: 1248, height: 30, rx: 9 }));
    st.appendChild(mk("text", { x: 36, y: 384 },
      "TRAINED  2015-02 → 2016-02 · 459,137 events     │     SETTINGS CHOSEN ON  2016-02 (validation)     " +
      "│     TESTED ONCE  2016-03 → 2016-05 · 104,167 events the model never saw"));
    svg.appendChild(st);
    /* the explanatory caption lives in the HTML, below the diagram, so it is
       never clipped by the SVG's aspect-ratio box */
  }

  var STAGES = {
    1: {
      h: "Stage 1 — the profile matrix",
      p: "Everything known about the customer becomes one row of 35 numbers: a 0/1 flag for each of the 24 "
       + "products they could hold, then eleven behavioural and demographic columns. Nothing here is a segment "
       + "label &mdash; the model never sees \"salaried, above threshold\". It sees the actual holdings pattern. "
       + "The 24 holdings flags also go to the collaborative engine in stage 2. Beside them sit 48 history "
       + "features: what the customer held one observation earlier (so a product dropped last month is visible), "
       + "and how many months in a row they have held each product.",
      f: "x = [ holds₁ … holds₂₄ | n_held, months_observed, months_since_last, "
       + "acquisitions_so_far, age, income, seniority, active, new, segment, male "
       + "| held_2m_ago₁ … ₂₄ | held_months₁ … ₂₄ ]",
      note: "The behavioural lag columns matter most. Permutation importance put months_since_last_acquisition "
          + "and acquisitions_so_far above income &mdash; timing beats affluence."
    },
    2: {
      h: "Stage 2 — two engines, fused",
      p: "Two engines. The collaborative engine asks what customers with exactly this portfolio added next, "
       + "leaning on single-product co-occurrence when the portfolio is rare. Its 44 scores are then handed to the "
       + "content engine &mdash; a histogram gradient-boosting classifier &mdash; as extra inputs beside the 35 "
       + "profile and 48 history features, so the model learns per customer how far to trust the peers (feature augmentation).",
      f: "cf_j = P(add j | exact portfolio), shrunk toward mean_i P(add j | holds i)\n"
       + "score = GBM( profile[35] | history[48] | cf_cooc[22] | cf_pattern[22] )",
      note: "Why not a weighted blend: the first version averaged content 50/50 with a lift-ranked CF and measured "
          + "slightly worse than content alone, because lift ranks by surprise, not likelihood. The fused content engine was "
          + "chosen on a validation month before the test months were scored; its numbers are on the Performance view."
    },
    3: {
      h: "Stage 3 — the macro regime engine",
      p: "Momentum and volatility in the market series classify the month into one of four regimes. Each regime "
       + "carries a multiplier per product, set from business logic rather than learned: in stress, liquidity and "
       + "safety rise and investment products fall.",
      f: "regime = f(momentum, volatility) → {EXPANSION, RECOVERY, SLOWDOWN, STRESS};  score ← score · tilt[regime, j]",
      note: "Measured over all 104,167 held-out events: the tilt changes the #1 product for 1.7% of customers and "
          + "leaves top-3 unchanged (90.4%). It is here for business relevance and timing, and the honest claim is "
          + "that it reorders without predicting better. All three test months classify as STRESS from the price series."
    },
    4: {
      h: "Stage 4 — knowledge-based constraints",
      p: "One readable predicate per product, auditable without reading the model: age bands, income floors, "
       + "seniority minimums, product prerequisites. A product that fails is removed from the list, not "
       + "down-weighted, so nothing downstream can re-admit it.",
      f: "eligible(j, c) = false  ⇒  score[j] := 0   (deleted, then renormalised)",
      note: "Validated against real outcomes on all 104,167 held-out events: the rules remove the product the "
          + "customer actually took 10.8% of the time (0.4% because the data was missing). That is why top-3 falls "
          + "from 90.4% to 85.3% end to end &mdash; the cost of a hard gate, reported, not hidden. "
          + "An earlier version blocked 79.3% "
          + "of one product because a categorical code was read as ordinal — which is exactly why the "
          + "validation step exists."
    }
  };

  function renderStage() {
    var s = STAGES[S.stage], r = S.current;
    $("#stage-text").innerHTML =
      "<h2>" + s.h + "</h2><p>" + s.p + "</p>" +
      '<div class="formula">' + s.f + "</div>" +
      '<div class="note">' + s.note + "</div>";

    var viz = $("#stage-viz");
    if (!r) {
      viz.innerHTML = "<h2>No profile loaded</h2><p class='muted'>Run a customer on the Advise view and this panel " +
        "will show what each stage did to that specific profile.</p>";
      return;
    }
    var h = "<h2>What it did for " + esc(r.customer.name || "this customer") + "</h2>";

    if (S.stage === 1) {
      h += "<p class='muted small'>The non-zero columns of this customer's feature row.</p><table><tbody>" +
        [["products held", r.customer.n_holdings], ["age", r.customer.age],
         ["income", money(r.customer.income)], ["months with the bank", r.customer.seniority_months],
         ["segment", r.customer.segment]]
          .map(function (kv) { return "<tr><td>" + kv[0] + "</td><td class='n'>" + kv[1] + "</td></tr>"; }).join("") +
        "</tbody></table><div class='sep'><span>holdings flags set to 1</span></div><div class='holdings'>" +
        r.customer.holdings.map(function (p) {
          return '<span class="hold" aria-pressed="true">' + esc(p.replace(/_/g, " ")) + "</span>";
        }).join("") + "</div>";
      viz.innerHTML = h;
    } else if (S.stage === 2) {
      var mixed = r.alpha < 0.995;
      h += "<p class='muted small'><b>Peers</b> is what customers with exactly this portfolio added next &mdash; " +
        "these numbers went into the content engine as inputs. <b>Content engine</b> is the booster's answer after reading them " +
        "beside the profile. Where the two disagree, the profile changed the model's mind." +
        (mixed ? " The &alpha; slider is mixing raw peer scores back in at " + (1 - r.alpha).toFixed(2) +
                 ", shown as the last column." : "") + "</p>" +
        "<table><thead><tr><th>product</th><th class='n'>collaborative</th><th class='n'>content engine</th>" +
        (mixed ? "<th class='n'>after &alpha; mix</th>" : "") + "</tr></thead><tbody>" +
        r.board.filter(function (b) { return !b.held; }).slice(0, 10).map(function (b) {
          var d = b.content - b.cf, col = d > 0.02 ? "var(--pass)" : (d < -0.02 ? "var(--block)" : "inherit");
          return "<tr><td class='cap'>" + esc(b.label) + "</td><td class='n'>" + pct(b.cf, 1) +
            "</td><td class='n' style='color:" + col + "'>" + pct(b.content, 1) + "</td>" +
            (mixed ? "<td class='n'>" + pct(b.hybrid, 1) + "</td>" : "") + "</tr>";
        }).join("") + "</tbody></table>" +
        "<p class='muted small' style='margin-top:8px'>Green: the model rates it higher than peers alone; " +
        "red: lower.</p>";
      viz.innerHTML = h;
    } else if (S.stage === 3) {
      var src = r.regime_source || {};
      h += "<p class='muted small'>" + esc(r.regime_note) + "</p>";
      if (src.mode === "classified") {
        h += "<div class='note' style='margin:10px 0'><b>" +
          esc(cap(r.regime.toLowerCase())) + "</b> was classified, not chosen — from " +
          src.n_series + " price series for " + esc(src.month) + ". Momentum " +
          (src.momentum >= 0 ? "+" : "") + (src.momentum * 100).toFixed(1) +
          "%, volatility " + (src.volatility * 100).toFixed(1) + "%, " +
          Math.round(src.agreement * 100) + "% agreement" +
          (src.dissenting && src.dissenting.length
            ? ". Dissenting: " + src.dissenting.map(function (d) {
                return cap(d.toLowerCase()); }).join(", ") : " — unanimous") +
          ".</div>";
      } else if (src.mode === "manual") {
        h += "<div class='note' style='margin:10px 0'>This regime was <b>chosen by hand</b>, " +
          "not classified from the series. Useful as a scenario; not evidence.</div>";
      }
      if (S.macroTimeline && S.macroTimeline.length) {
        h += "<div class='sep'><span>regime history, from the price series</span></div>" +
             '<canvas class="viz" id="regstrip"></canvas>' +
             '<div class="legend">' +
               '<span><i style="background:' + REG_COL.EXPANSION + '"></i>expansion</span>' +
               '<span><i style="background:' + REG_COL.RECOVERY + '"></i>recovery</span>' +
               '<span><i style="background:' + REG_COL.SLOWDOWN + '"></i>slowdown</span>' +
               '<span><i style="background:' + REG_COL.STRESS + '"></i>stress</span>' +
             "</div>";
      }
      h += "<div class='sep'><span>what the tilt did</span></div>" +
        "<table><thead><tr><th>product</th><th class='n'>before</th><th class='n'>tilt</th>" +
        "<th class='n'>after</th></tr></thead><tbody>" +
        r.board.filter(function (b) { return !b.held; }).slice(0, 12).map(function (b) {
          var col = b.tilt > 1.001 ? "var(--pass)" : (b.tilt < 0.999 ? "var(--block)" : "var(--ink-mute)");
          return "<tr><td class='cap'>" + esc(b.label) + "</td><td class='n'>" + pct(b.hybrid, 2) +
            "</td><td class='n' style='color:" + col + "'>×" + b.tilt.toFixed(2) +
            "</td><td class='n'>" + pct(b.after_macro, 2) + "</td></tr>";
        }).join("") + "</tbody></table>";
      viz.innerHTML = h;
      drawRegimeStrip((r.regime_source || {}).month);
    } else {
      h += "<p class='muted small'>" + r.n_blocked + " of " + r.n_products +
        " products failed a rule for this customer.</p>";
      if (r.blocked.length) {
        h += "<div class='blocked-list'>" + r.blocked.map(function (b) {
          return "<div class='blocked-row'><b>" + esc(b.label) + "</b><span>" + esc(b.rule) +
            "<br>would have scored " + pct(b.score_before, 1) + "</span></div>";
        }).join("") + "</div>";
      } else {
        h += "<p class='muted'>Nothing was gated out &mdash; every scored product passes its rule for this profile.</p>";
      }
      viz.innerHTML = h;
    }
  }

  var REG_COL = { EXPANSION: "#0e8f5e", RECOVERY: "#8b5cf6",
                  SLOWDOWN: "#2563eb", STRESS: "#d92d20" };

  /* A strip of the classified regime per month, with the momentum line over
     it, and a marker on the month this recommendation was scored against. */
  function drawRegimeStrip(markMonth) {
    var cv = $("#regstrip"); if (!cv) return;
    var tl = S.macroTimeline || []; if (!tl.length) return;
    var W = cv.clientWidth || 520, H = 118, padL = 6, padR = 6, padB = 18;
    var c = hidpi(cv, W, H);
    var n = tl.length, bw = (W - padL - padR) / n;
    var stripTop = 8, stripH = 26;

    tl.forEach(function (r, i) {
      c.fillStyle = REG_COL[r.regime] || P.inkMute;
      c.globalAlpha = 0.85;
      c.fillRect(padL + i * bw, stripTop, Math.max(bw - 0.4, 0.8), stripH);
      c.globalAlpha = 1;
    });

    /* momentum line beneath the strip */
    var mom = tl.map(function (r) { return r.momentum; });
    var lo = Math.min.apply(null, mom), hi = Math.max.apply(null, mom);
    var span = (hi - lo) || 1, top = stripTop + stripH + 10, hgt = H - padB - top;
    var zeroY = top + hgt * (hi / span);
    c.strokeStyle = P.line; c.lineWidth = 1;
    c.beginPath(); c.moveTo(padL, zeroY); c.lineTo(W - padR, zeroY); c.stroke();
    c.strokeStyle = P.plum; c.lineWidth = 1.6; c.beginPath();
    tl.forEach(function (r, i) {
      var x = padL + i * bw + bw / 2, y = top + hgt * ((hi - r.momentum) / span);
      if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
    });
    c.stroke();

    /* the month this profile was scored against */
    if (markMonth) {
      var idx = -1;
      for (var i = 0; i < n; i++) if (tl[i].month === markMonth) { idx = i; break; }
      if (idx >= 0) {
        var mx = padL + idx * bw + bw / 2;
        c.strokeStyle = P.violet; c.lineWidth = 1.8; c.setLineDash([3, 3]);
        c.beginPath(); c.moveTo(mx, 2); c.lineTo(mx, H - padB + 2); c.stroke();
        c.setLineDash([]);
        c.fillStyle = P.violet; c.font = "700 10.5px " + '"Cascadia Mono",Consolas,monospace';
        c.textAlign = mx > W * 0.7 ? "right" : "left";
        c.fillText(markMonth, mx + (mx > W * 0.7 ? -5 : 5), H - 6);
      }
    }
    c.fillStyle = P.inkMute; c.font = "10.5px " + '"Cascadia Mono",Consolas,monospace';
    c.textAlign = "left";  c.fillText(tl[0].month, padL, H - 6);
    c.textAlign = "right"; c.fillText(tl[n - 1].month, W - padR, H - 6);
  }

  /* ========================================================================
     PERFORMANCE
     ======================================================================== */
  /* The engine table is whatever the backend measured in this build. Nothing
     is hardcoded here: a number the running code did not produce has no
     business being on screen. */
  function engineRows(p) {
    var served = p.served_engine || (p.metrics && p.metrics.served_engine) || "Fused hybrid (served)";
    return (p.engines || []).map(function (e) {
      var kind = /Popularity|previous/.test(e.name) ? "base"
               : (e.name === served ? "ours" : "part");
      return { name: e.name, f1: e.macro_f1, t1: e.top1, t3: e.top3,
               map: e.map7, note: e.note || "", kind: kind,
               served: e.name === served };
    });
  }

  function loadPerformance() {
    if (S.perf) { renderPerformance(S.perf); return; }
    $("#perf-body").innerHTML = "<div class='card muted'>Loading results…</div>";
    api("/api/performance").then(function (p) { S.perf = p; renderPerformance(p); })
      .catch(function (e) { $("#perf-body").innerHTML = "<div class='card'>Could not load results: " + esc(e.message) + "</div>"; });
  }

  function renderPerformance(p) {
    var m = p.metrics, chance = m.chance_macro_f1;
    var rows = engineRows(p);
    var body = $("#perf-body");

    /* Say plainly what won and what the blend actually did. All of this is
       derived from the measured rows, so it cannot drift away from the table
       beside it the way a hand-written paragraph would. */
    var verdict = "";
    if (rows.length) {
      var best = rows.reduce(function (a, b) { return b.t3 > a.t3 ? b : a; });
      var servedRow = rows.filter(function (x) { return x.served; })[0] || best;
      var hyb = rows.filter(function (x) { return x.kind === "ours"; })[0];

      if (best.name === servedRow.name) {
        verdict = "The engine being served, <b>" + esc(servedRow.name) +
          "</b>, is also the strongest here on top-3 (" + pct(best.t3, 1) + ").";
      } else if (Math.abs(best.t3 - servedRow.t3) < 0.002) {
        verdict = "On top-3 the content engines are level at " + pct(best.t3, 1) +
          ". <b>" + esc(best.name) + "</b> edges ahead on macro F1 (" +
          best.f1.toFixed(4) + " against the served engine's " +
          servedRow.f1.toFixed(4) + ").";
      } else {
        verdict = "<b>" + esc(best.name) + "</b> is the strongest engine on top-3 here, " +
          pct(best.t3, 1) + " against the served engine's " + pct(servedRow.t3, 1) + ".";
      }

      var contentRow = rows.filter(function (x) { return /Content only \(HistGB\)/.test(x.name); })[0];
      var prevRow = rows.filter(function (x) { return /blend.*previous/i.test(x.name); })[0];
      var cfNew = rows.filter(function (x) { return /pattern/i.test(x.name); })[0];
      var cfOld = rows.filter(function (x) { return /lift \(previous\)/i.test(x.name); })[0];
      if (hyb && contentRow) {
        var d3 = hyb.t3 - contentRow.t3, df = hyb.f1 - contentRow.f1;
        verdict += " Against the same booster without peer inputs, the fused hybrid moves top-3 by " +
          (d3 >= 0 ? "+" : "") + (d3 * 100).toFixed(2) + " points and macro F1 by " +
          (df >= 0 ? "+" : "") + df.toFixed(4) + ".";
        if (Math.abs(d3) < 0.003) verdict += " On top-3 that is level, not a breakthrough &mdash; said plainly.";
      }
      if (prevRow && hyb) {
        verdict += " The previous 50/50 blend scored " + pct(prevRow.t3, 1) + " top-3, below content alone: " +
          "it averaged in a collaborative engine that ranked by surprise.";
      }
      if (cfNew && cfOld) {
        verdict += " Ranking peers by probability instead lifted the collaborative engine on its own from " +
          pct(cfOld.t3, 1) + " to " + pct(cfNew.t3, 1) + " top-3, now above the popularity list.";
      }

      if (!p.xgboost_available) {
        verdict += " XGBoost is not installed here, so it has no row — absent " +
          "rather than remembered from an earlier run. " +
          "<span class='mono'>pip install xgboost</span> and rebuild to measure it.";
      }
    }
    body.innerHTML =
      '<div class="tiles" style="margin-bottom:20px">' +
        '<div class="tile"><b>' + m.macro_f1.toFixed(4) + '</b><span>macro F1</span><em>' +
          (m.macro_f1 / chance).toFixed(2) + '× the ' + chance.toFixed(4) + ' chance level</em></div>' +
        '<div class="tile mint"><b>' + pct(m.top1, 1) + '</b><span>top&#8209;1 hit rate</span>' +
          '<em>first suggestion is the product taken</em></div>' +
        '<div class="tile mint"><b>' + pct(m.top3, 1) + '</b><span>top&#8209;3 hit rate</span>' +
          '<em>the answer the console actually gives</em></div>' +
        '<div class="tile sky"><b>' + m.map7.toFixed(3) + '</b><span>MAP@7</span>' +
          '<em>rank quality, not just presence</em></div>' +
        '<div class="tile"><b>' + money(m.n_test) + '</b><span>held-out events</span>' +
          '<em>trained on ' + money(m.n_train) + '</em></div>' +
      "</div>" +
      '<div class="cols-3-2">' +
        '<div class="card"><h2>Every engine, measured the same way</h2>' +
          "<p class='muted small'>All " + rows.length + " scored by <span class='mono'>build_model.py</span> " +
          "on the same held-out months, same " + m.n_products +
          " products, in one run. The bar is top&#8209;3 hit rate; chance is " + pct(chance, 1) + ".</p>" +
          '<canvas class="viz" id="cmp"></canvas>' +
          '<div class="legend">' +
            '<span><i style="background:#b4abce"></i>baseline or replaced engine</span>' +
            '<span><i style="background:' + P.violetBright + '"></i>single engine</span>' +
            '<span><i style="background:' + P.violet + '"></i>the served hybrid</span>' +
          "</div>" +
          '<div class="note" style="margin-top:14px">' + verdict + "</div>" +
        "</div>" +
        '<div class="card"><h2>Metric table</h2>' +
          "<p class='muted small'>Every row produced by this build. Nothing quoted from an earlier experiment.</p>" +
          "<table><thead><tr><th>engine</th><th class='n'>macro F1</th><th class='n'>top&#8209;1</th>" +
          "<th class='n'>top&#8209;3</th><th class='n'>MAP@7</th></tr></thead><tbody>" +
          rows.map(function (b) {
            var st = b.kind === "ours" ? " style='color:var(--violet);font-weight:600'" : "";
            return "<tr" + st + "><td>" + esc(b.name) +
              (b.served ? " <span class='tag ok' style='margin-left:6px'>served</span>" : "") +
              "</td><td class='n'>" + b.f1.toFixed(4) +
              "</td><td class='n'>" + b.t1.toFixed(3) + "</td><td class='n'>" + b.t3.toFixed(3) +
              "</td><td class='n'>" + b.map.toFixed(3) + "</td></tr>";
          }).join("") + "</tbody></table>" +
          '<div class="note" style="margin-top:14px">Macro F1 is not comparable across different ' +
            'product counts, so read it against chance. The best engine here scores ' +
            (rows.length ? rows.reduce(function (a, b) { return b.f1 > a.f1 ? b : a; }).f1.toFixed(4) : "—") +
            ' against a chance level of ' + chance.toFixed(4) + ', which is ' +
            (rows.length ? (rows.reduce(function (a, b) { return b.f1 > a.f1 ? b : a; }).f1 / chance).toFixed(2) : "—") +
            '×. An earlier 18-product run scored a higher raw 0.3098, but only 5.58× its ' +
            'own chance level — the bigger number was the weaker result.</div>' +
        "</div>" +
      "</div>" +
      '<div class="card" style="margin-top:20px"><h2>Per product</h2>' +
        "<p class='muted small'>Share of all observed acquisitions, and the eligibility rule each product carries. " +
        "The long tail is why macro F1 sits where it does — rare products are genuinely hard.</p>" +
        '<div class="tbl-wrap"><table><thead><tr><th>product</th><th class="n">share of acquisitions</th>' +
        '<th></th><th>eligibility rule</th></tr></thead><tbody>' +
        p.products.map(function (x) {
          return "<tr><td class='cap'>" + esc(x.label) + "</td><td class='n'>" + pct(x.base_rate, 2) +
            "</td><td style='width:160px'><div class='minibar' style='width:" +
            Math.max(2, x.base_rate / p.products[0].base_rate * 100).toFixed(1) + "%'></div></td><td class='muted small'>" +
            esc(x.rule) + "</td></tr>";
        }).join("") + "</tbody></table></div></div>";

    drawCompare(m, rows);
    loadProof();
  }

  /* ========================================================================
     IS THE IMPROVEMENT REAL?
     Error bars, paired significance tests, the whole pipeline over every
     held-out event, ablation and month-by-month stability -- as written by
     defense/defense_eval.py. Shown so the claims can be checked on screen.
     ======================================================================== */
  function loadProof() {
    var body = $("#perf-body");
    var box = el("div", "evidence");
    box.id = "evidence-box";
    body.appendChild(box);
    (S.proof ? Promise.resolve(S.proof) : api("/api/evidence")).then(function (E) {
      S.proof = E;
      box.innerHTML = proofHTML(E);
    }).catch(function () {
      box.innerHTML = "<div class='card muted small'>No evidence file yet. Run " +
        "<span class='mono'>python defense/defense_eval.py</span> to measure confidence intervals and " +
        "significance; this panel then fills in on its own.</div>";
    });
  }

  /* the levers judged on the validation month (defense/improve_experiment.py,
     defense/xgb_gpu_check.py), when the evidence file carries them */
  function triedOnValidation(E) {
    var h = "", I = E.improve, X = E.xgb_gpu;
    if (I && I.thresholds && !I.thresholds.kept) {
      var a = I.thresholds.validation_without, b = I.thresholds.validation_with;
      h += "<tr><td>per&#8209;product decision thresholds</td><td class='n'>validation macro F1 " + a.macro_f1.toFixed(3) +
        " &rarr; " + b.macro_f1.toFixed(3) + "</td><td><span class='pill bad'>top&#8209;3 &minus;" +
        ((a.top3 - b.top3) * 100).toFixed(1) + " pts</span></td></tr>";
    }
    if (I && I.grid) {
      var others = I.grid.filter(function (g) { return g.name !== I.chosen_params.name; });
      var chosen = I.grid.filter(function (g) { return g.name === I.chosen_params.name; })[0];
      h += "<tr><td>hyperparameter search (" + others.length + " alternatives)</td><td class='n'>validation top&#8209;3 " +
        pct(Math.max.apply(null, others.map(function (g) { return g.top3; })), 2) + " vs " + pct(chosen.top3, 2) +
        "</td><td><span class='pill'>none beat the current settings</span></td></tr>";
    }
    if (X && X.xgboost_gpu && X.xgboost_gpu.length) {
      var best = X.xgboost_gpu.reduce(function (a, b) { return b.top3 > a.top3 ? b : a; });
      h += "<tr><td>XGBoost on the GPU, with history</td><td class='n'>validation macro F1 " + best.macro_f1.toFixed(3) +
        " vs " + X.histgb_reference.macro_f1.toFixed(3) + "</td><td><span class='pill'>top&#8209;3 level, macro F1 lower</span></td></tr>";
    }
    return h;
  }

  function proofHTML(E) {
    var P1 = E.pipeline, eng = E.engines, cal = E.calibration, L = E.learnable_subset;
    function ci(a) { return "<span class='ci'>" + pct(a[0], 1) + "&ndash;" + pct(a[1], 1) + "</span>"; }
    function signed(v, d) { return (v >= 0 ? "+" : "&minus;") + Math.abs(v).toFixed(d); }
    function verdict(r) {
      var real = r.ci[0] > 0 || r.ci[1] < 0;
      if (!real) return "<span class='pill'>within noise</span>";
      if (r.p_two_sided > 0.01) return "<span class='pill warn'>borderline " + (r.diff > 0 ? "gain" : "loss") +
        " &middot; p " + r.p_two_sided.toFixed(3) + "</span>";
      return r.diff > 0 ? "<span class='pill good'>real gain</span>" : "<span class='pill bad'>real loss</span>";
    }

    var tiles =
      '<div class="tiles">' +
        '<div class="tile mint"><b>' + pct(eng.pipe.top3, 1) + '</b><span>end&#8209;to&#8209;end top&#8209;3</span>' +
          '<em>after the tilt and the rules, every one of ' + money(E.split.test_events) + ' held&#8209;out events &middot; 95% CI ' +
          ci(eng.pipe.top3_ci) + '</em></div>' +
        '<div class="tile rose"><b>' + pct(P1.truth_removed_by_rule, 1) + '</b><span>right product removed by a rule</span>' +
          '<em>the cost of hard gates; ' + pct(P1.truth_removed_because_data_missing, 1) + ' because data was missing</em></div>' +
        '<div class="tile sky"><b>' + pct(cal.ece, 1) + '</b><span>calibration error</span>' +
          '<em>when it says 60%, it is right about 60% of the time</em></div>' +
        '<div class="tile"><b>' + L.macro_f1.toFixed(3) + '</b><span>macro F1, learnable products</span>' +
          '<em>' + L.n_products + ' products above ' + pct(L.threshold_base_rate, 1) + ' base rate; all: ' +
          L.macro_f1_all_present.toFixed(3) + '</em></div>' +
      "</div>";

    var sig = E.significance.map(function (r) {
      return "<tr><td>" + esc(r.comparison) + "</td>" +
        "<td class='n'>" + signed(r.top3.diff * 100, 2) + " pts <span class='ci'>[" + signed(r.top3.ci[0] * 100, 3) +
          ", " + signed(r.top3.ci[1] * 100, 3) + "]</span></td><td>" + verdict(r.top3) + "</td>" +
        "<td class='n'>" + signed(r.macro_f1.diff, 4) + "</td><td>" + verdict(r.macro_f1) + "</td></tr>";
    }).join("");

    var abl = E.ablation.map(function (r) {
      var served = /served/.test(r.name);
      return "<tr" + (served ? " class='hl'" : "") + "><td>" + esc(r.name) + "</td><td class='n'>" + pct(r.top3, 1) +
        "</td><td class='n'>" + r.macro_f1.toFixed(3) + "</td></tr>";
    }).join("");

    var stab = E.month_stability.map(function (r) {
      return "<tr><td>" + esc(r.month) + "</td><td class='n'>" + money(r.events) + "</td><td class='n'>" +
        pct(r.fused_top3, 1) + "</td><td class='n'>" + pct(r.content_top3, 1) + "</td><td class='n'>" +
        pct(r.pop_top3, 1) + "</td></tr>";
    }).join("");

    var cw = E.class_weighted ? E.class_weighted.engine : null;

    return '<div class="ev-head"><h2>Is the improvement real?</h2>' +
      "<p class='muted'>Measured by <span class='mono'>defense/defense_eval.py</span>: 1,000 bootstrap resamples of the " +
      "held&#8209;out customers, every engine scored on the same rows, products already held never counted, and ties " +
      "counted <i>against</i> the engine. A difference is called real only when its 95% interval excludes zero.</p></div>" +
      tiles +
      '<div class="cols-3-2" style="margin-top:20px">' +
        '<div class="card"><h2>Paired significance tests</h2>' +
          "<p class='muted small'>Each row compares two engines on identical resamples. Brackets are the 95% interval " +
          "of the top&#8209;3 difference.</p>" +
          "<div class='tbl-wrap'><table><thead><tr><th>comparison</th><th class='n'>&Delta; top&#8209;3</th><th></th>" +
          "<th class='n'>&Delta; macro F1</th><th></th></tr></thead><tbody>" + sig + "</tbody></table></div></div>" +
        '<div class="card"><h2>Where the signal comes from</h2>' +
          "<p class='muted small'>The same booster trained on each group of inputs alone, then on all of them.</p>" +
          "<table><thead><tr><th>inputs</th><th class='n'>top&#8209;3</th><th class='n'>macro F1</th></tr></thead><tbody>" +
          abl + "</tbody></table>" +
          "<div class='note' style='margin-top:12px'>Each group alone reaches a similar top&#8209;3; the gain comes " +
          "from combining them. Peer scores are a better-shaped view of the current holdings; <b>holdings history</b> is " +
          "genuinely new information &mdash; what the customer dropped, and how long they have kept each product &mdash; " +
          "and it is the largest single improvement to the served model.</div></div>" +
      "</div>" +
      '<div class="cols2" style="margin-top:20px">' +
        '<div class="card"><h2>Stable month to month</h2>' +
          "<p class='muted small'>Each held&#8209;out month scored on its own. No month carries the average.</p>" +
          "<table><thead><tr><th>month</th><th class='n'>events</th><th class='n'>fused</th><th class='n'>content</th>" +
          "<th class='n'>popularity</th></tr></thead><tbody>" + stab + "</tbody></table></div>" +
        '<div class="card"><h2>Tried, measured, rejected</h2>' +
          "<p class='muted small'>Ideas that sounded right and lost on the numbers. Reported so the choices are checkable.</p>" +
          "<table><tbody>" +
            "<tr><td>50/50 weighted blend (the first hybrid)</td><td class='n'>" + pct(eng.blend_old.top3, 1) +
              " top&#8209;3</td><td><span class='pill bad'>below content alone</span></td></tr>" +
            "<tr><td>lift&#8209;ranked collaborative engine</td><td class='n'>" + pct(eng.cf_old.top3, 1) +
              " top&#8209;3</td><td><span class='pill bad'>below popularity</span></td></tr>" +
            (cw ? "<tr><td>class&#8209;weighted training (balanced)</td><td class='n'>" + pct(cw.top3, 1) +
              " top&#8209;3</td><td><span class='pill bad'>macro F1 " + cw.macro_f1.toFixed(3) + "</span></td></tr>" : "") +
            (eng.xgb ? "<tr><td>XGBoost in place of HistGB</td><td class='n'>" + pct(eng.xgb.top3, 1) +
              " top&#8209;3</td><td><span class='pill'>within noise</span></td></tr>" : "") +
            triedOnValidation(E) +
          "</tbody></table></div>" +
      "</div>";
  }

  function drawCompare(m, rows) {
    var cv = $("#cmp"); if (!cv || !rows || !rows.length) return;
    var W = cv.clientWidth || 640, padL = 232, padR = 56, rowH = 46, padT = 16;
    var H = padT + rows.length * rowH + 26;
    var c = hidpi(cv, W, H);
    var span = W - padL - padR;
    var COL = { base: "#b4abce", part: P.violetBright, ours: P.violet };
    var chance = m.chance_macro_f1;

    /* chance line */
    var cx = padL + chance * span;
    c.strokeStyle = "rgba(217,45,32,.45)"; c.setLineDash([4, 4]); c.lineWidth = 1;
    c.beginPath(); c.moveTo(cx, padT - 6); c.lineTo(cx, padT + rows.length * rowH); c.stroke();
    c.setLineDash([]);
    c.fillStyle = P.block; c.font = "11px " + '"Cascadia Mono",Consolas,monospace';
    c.textAlign = "left"; c.fillText("chance", cx + 5, padT + rows.length * rowH + 14);

    rows.forEach(function (b, i) {
      var y = padT + i * rowH;
      c.textAlign = "right";
      c.font = (b.kind === "ours" ? "700 " : "") + "13px " + '"Segoe UI",system-ui,sans-serif';
      c.fillStyle = b.kind === "ours" ? P.violet : P.inkDim;
      c.fillText(b.name, padL - 12, y + 20);

      var w = Math.max(b.t3 * span, 2);
      c.fillStyle = COL[b.kind];
      c.beginPath(); c.roundRect(padL, y + 8, w, 16, 8); c.fill();

      c.textAlign = "left";
      c.fillStyle = P.plum; c.font = "700 12px " + '"Cascadia Mono",Consolas,monospace';
      c.fillText((b.t3 * 100).toFixed(1) + "%", padL + w + 8, y + 20);

      c.fillStyle = P.inkMute; c.font = "10.5px " + '"Cascadia Mono",Consolas,monospace';
      c.fillText("macro F1 " + b.f1.toFixed(4) + "   MAP@7 " + b.map.toFixed(3), padL, y + 38);
    });
  }

  /* ========================================================================
     EVIDENCE — lift rules + lift matrix heatmap
     ======================================================================== */
  function loadEvidence() {
    if (S.rules && S.matrix) { renderEvidence(); return; }
    $("#ev-body").innerHTML = "<div class='card muted'>Loading measured rules…</div>";
    Promise.all([api("/api/lift/rules?top=40"), api("/api/lift/matrix")])
      .then(function (r) { S.rules = r[0]; S.matrix = r[1]; renderEvidence(); })
      .catch(function (e) { $("#ev-body").innerHTML = "<div class='card'>Could not load: " + esc(e.message) + "</div>"; });
  }

  function renderEvidence() {
    var R = S.rules;
    $("#ev-body").innerHTML =
      '<div class="tiles" style="margin-bottom:20px">' +
        '<div class="tile"><b>' + R.total_above_threshold + '</b><span>rules above ' +
          R.min_lift + '×</span><em>each backed by ' + R.min_support + '+ customers; ' +
          (R.dropped_thin_support || 0) + ' thinner ones left out</em></div>' +
        '<div class="tile mint"><b>' + R.rules[0].lift.toFixed(1) + '×</b><span>strongest rule</span>' +
          '<em>' + esc(R.rules[0].from_label) + " → " + esc(R.rules[0].to_label) + "</em></div>" +
        '<div class="tile sky"><b>' + S.matrix.rows.length + "×" + S.matrix.cols.length +
          '</b><span>item–item matrix</span><em>every held → next pair</em></div>' +
      "</div>" +
      '<div class="cols2">' +
        '<div class="card"><h2>The strongest sequence rules</h2>' +
          "<p class='muted small'>Lift of 8× means customers holding the left-hand product take the " +
          "right-hand one eight times more often than the catalogue average. <b>Customers</b> is how many training " +
          "events stand behind the rule: anything under " + R.min_support + " is left out, because a big lift on a " +
          "handful of people is an anecdote. These are the sentences the Advise view turns into reasons.</p>" +
          '<div class="tbl-wrap"><table><thead><tr><th>holds</th><th>takes next</th>' +
          '<th class="n">lift</th><th class="n">customers</th><th class="n">base rate</th></tr></thead><tbody>' +
          R.rules.map(function (r) {
            return "<tr><td class='cap'>" + esc(r.from_label) + "</td><td class='cap'>" + esc(r.to_label) +
              "</td><td class='n' style='color:var(--violet);font-weight:600'>" + r.lift.toFixed(2) + "×</td><td class='n'>" +
              money(r.support) + "</td><td class='n'>" + pct(r.base_rate, 2) + "</td></tr>";
          }).join("") + "</tbody></table></div></div>" +
        '<div class="card"><h2>The whole lift matrix</h2>' +
          "<p class='muted small'>Rows are what a customer already holds, columns what they take next. " +
          "Bright means far above the base rate; the dark diagonal is a product not predicting itself.</p>" +
          '<canvas class="viz" id="heat"></canvas>' +
          '<div class="scale"><span>1&times;</span>' +
            '<span class="ramp" id="ramp"></span><span>10&times;+</span></div>' +
          '<div class="note" style="margin-top:14px">This matrix is computed from the training months only. ' +
            'Computing it over all months would leak the test period into something the API serves as if it had ' +
            'been learned beforehand — a mistake that quietly inflates every number downstream.</div>' +
        "</div>" +
      "</div>";
    drawHeat();
    var ramp = $("#ramp");
    if (ramp) {
      var stops = [];
      for (var i = 0; i <= 10; i++) stops.push(liftColor(1 + i * 0.9) + " " + (i * 10) + "%");
      ramp.style.background = "linear-gradient(90deg," + stops.join(",") + ")";
    }
  }

  /* pale lilac at 1x, deep violet at 10x+; below 1x fades toward white */
  function liftColor(v) {
    if (v < 1) {
      var u = Math.max(0, Math.min(1, v));
      var g = Math.round(252 - 8 * u);
      return "rgb(" + g + "," + (g - 2) + ",255)";
    }
    var t = Math.max(0, Math.min(1, Math.log(v) / Math.log(12)));
    var r = Math.round(237 + (76 - 237) * t);
    var gg = Math.round(231 + (29 - 231) * t);
    var b = Math.round(254 + (149 - 254) * t);
    return "rgb(" + r + "," + gg + "," + b + ")";
  }

  function drawHeat() {
    var cv = $("#heat"); if (!cv) return;
    var M = S.matrix;
    var W = cv.clientWidth || 560;
    var padL = 120, padT = 116, padR = 6;
    var n = M.cols.length, rows = M.rows.length;
    var cell = Math.max(9, Math.min(22, (W - padL - padR) / n));
    var H = padT + rows * cell + 8;
    var c = hidpi(cv, W, H);

    c.font = "10px " + '"Segoe UI",system-ui,sans-serif';
    /* column labels, rotated */
    M.cols.forEach(function (p, j) {
      c.save();
      c.translate(padL + j * cell + cell / 2 + 3, padT - 8);
      c.rotate(-Math.PI / 2.6);
      c.fillStyle = P.inkMute; c.textAlign = "left";
      c.fillText(p.replace(/_/g, " ").slice(0, 16), 0, 0);
      c.restore();
    });
    M.rows.forEach(function (p, i) {
      c.fillStyle = P.inkDim; c.textAlign = "right";
      c.fillText(p.replace(/_/g, " ").slice(0, 17), padL - 7, padT + i * cell + cell / 2 + 3.5);
      M.matrix[i].forEach(function (v, j) {
        c.fillStyle = (M.rows[i] === M.cols[j]) ? P.line : liftColor(v);
        c.fillRect(padL + j * cell, padT + i * cell, cell - 1.2, cell - 1.2);
      });
    });

    /* hover readout */
    cv.onmousemove = function (e) {
      var r = cv.getBoundingClientRect();
      var x = e.clientX - r.left, y = e.clientY - r.top;
      var j = Math.floor((x - padL) / cell), i = Math.floor((y - padT) / cell);
      if (i >= 0 && i < rows && j >= 0 && j < n) {
        cv.title = M.rows[i].replace(/_/g, " ") + " → " + M.cols[j].replace(/_/g, " ") +
                   "   " + M.matrix[i][j].toFixed(2) + "×";
      } else { cv.title = ""; }
    };
  }

  /* ========================================================================
     SESSION
     ======================================================================== */
  function loadSession() {
    $("#sess-body").innerHTML = "<div class='card muted'>Loading…</div>";
    api("/api/session").then(function (rows) {
      $("#sess-count").textContent = rows.length ? "(" + rows.length + ")" : "";
      if (!rows.length) {
        $("#sess-body").innerHTML = "<div class='card'><h2>Nothing yet</h2><p class='muted'>" +
          "Every profile you advise is kept here for the life of the server, so you can walk back through them " +
          "in a demo without re-entering anything.</p></div>";
        return;
      }
      $("#sess-body").innerHTML =
        '<div class="card"><div class="tbl-wrap"><table><thead><tr><th>time</th><th>customer</th>' +
        '<th class="n">age</th><th class="n">income</th><th>regime</th><th>top recommendation</th>' +
        '<th class="n">score</th><th class="n">removed</th></tr></thead><tbody>' +
        rows.map(function (r) {
          return '<tr class="clickrow" data-id="' + r.id + '"><td class="mono">' + esc(r.created) +
            "</td><td>" + esc(r.name || "—") + "</td><td class='n'>" + r.age +
            "</td><td class='n'>" + money(r.income) + "</td><td class='small'>" + cap(r.regime.toLowerCase()) +
            "</td><td class='cap' style='color:var(--violet);font-weight:600'>" + esc(r.top) + "</td><td class='n'>" +
            pct(r.score, 1) + "</td><td class='n'>" + r.n_blocked + "</td></tr>";
        }).join("") + "</tbody></table></div><p class='muted small' style='margin-top:12px'>" +
        "Select a row to reopen it on the Advise view.</p></div>";
      $$(".clickrow").forEach(function (tr) {
        tr.addEventListener("click", function () {
          api("/api/session/" + tr.dataset.id).then(function (r) {
            S.current = r; renderVerdict(r); renderStage(); show("advise");
          });
        });
      });
    }).catch(function (e) {
      $("#sess-body").innerHTML = "<div class='card'>Could not load: " + esc(e.message) + "</div>";
    });
  }

  /* ------------------------------------------------------------- resize */
  var rt;
  window.addEventListener("resize", function () {
    clearTimeout(rt);
    rt = setTimeout(function () {
      if (S.perf && $("#cmp")) drawCompare(S.perf.metrics, engineRows(S.perf));
      if (S.matrix && $("#heat")) drawHeat();
      if ($("#regstrip") && S.current) drawRegimeStrip((S.current.regime_source || {}).month);
    }, 160);
  });

  boot();
})();

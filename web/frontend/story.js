/* ==========================================================================
   SmartCross — the scroll story

   One real held-out customer walks into the bank, and the page follows them
   through the four engines as you scroll. Every number in the last two acts
   comes from /api/sample and /api/recommend for that customer; nothing in the
   pipeline panel is illustrative unless the API is unreachable, and then it
   says so on screen.

   Scroll position IS the timeline. Nothing plays on its own: the walk cycle,
   the doors, the camera and the bars are all functions of one number, p.
   ========================================================================== */
(function () {
  "use strict";

  var story = document.getElementById("story");
  if (!story) return;
  var svg = document.getElementById("story-svg");
  var caps = Array.prototype.slice.call(story.querySelectorAll(".story-cap .cap"));
  var bar = story.querySelector(".story-progress i");
  var NS = "http://www.w3.org/2000/svg";

  /* ------------------------------------------------------------ helpers */
  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function seg(p, a, b) { return clamp((p - a) / (b - a), 0, 1); }       // 0..1 inside [a,b]
  function ease(t) { return t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function $(s) { return svg.querySelector(s); }
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }
  function pretty(p) { return p.replace(/_/g, " "); }
  function money(v) { return v < 0 ? "not on file" : Math.round(v).toLocaleString("en-GB"); }

  /* -------------------------------------------------------- the person */
  /* Facing right, feet at (0,0), ~170 units tall. Limbs pivot at the hip
     and shoulder so the walk cycle is two rotations per side. */
  function person(id, female, grey) {
    var coat = "url(#st-coat)", coatD = "#4a2488", pants = "#2a1250",
        pantsD = "#1b0b38", skin = "#f0c9a8", skinD = "#dcae8a",
        hairC = grey ? "#cfc7dd" : "#2a1250";
    var body = female
      ? '<path d="M-17 -126 Q0 -134 17 -126 L25 -60 L-25 -60 Z" fill="' + coat + '"/>'
      : '<rect x="-18" y="-130" width="36" height="66" rx="14" fill="' + coat + '"/>';
    var hair = female
      ? '<path d="M-17 -149 Q-19 -172 2 -171 Q19 -170 17 -152 Q10 -161 -2 -160 Q-6 -150 -9 -138 Q-12 -124 -22 -118 Q-24 -136 -17 -149Z" fill="' + hairC + '"/>'
      : '<path d="M-16 -151 Q-16 -171 2 -170 Q18 -169 16 -154 Q9 -161 -3 -159 Q-9 -152 -16 -151Z" fill="' + hairC + '"/>';
    function leg(cls, fill) {
      return '<g class="' + cls + '"><rect x="-6.5" y="-70" width="13" height="64" rx="6.5" fill="' + fill + '"/>' +
             '<path d="M-7 -9 h17 a5 5 0 0 1 0 10 h-17 z" fill="#14082b"/></g>';
    }
    function arm(cls, fill, hand) {
      return '<g class="' + cls + '"><rect x="-5.5" y="-120" width="11" height="52" rx="5.5" fill="' + fill + '"/>' +
             '<circle cx="0" cy="-66" r="6" fill="' + hand + '"/></g>';
    }
    return '<g id="' + id + '">' +
      '<ellipse cx="2" cy="1" rx="28" ry="5" fill="#14082b" opacity=".22"/>' +
      '<g class="bob">' +
        arm("arm-b", coatD, skinD) + leg("leg-b", pantsD) + leg("leg-f", pants) + body +
        '<rect x="-5" y="-139" width="10" height="12" fill="' + skinD + '"/>' +
        '<circle cx="0" cy="-152" r="16" fill="' + skin + '"/>' + hair +
        '<circle cx="8" cy="-153" r="1.9" fill="#2a1250"/>' +
        '<path d="M6 -144 Q10 -142 13 -145" stroke="#2a1250" stroke-width="1.6" fill="none" stroke-linecap="round"/>' +
        arm("arm-f", coat, skin) +
      '</g></g>';
  }

  /* walk: phase from distance travelled, so scrolling literally walks them */
  function pose(g, x, y, scale, dist, moving) {
    if (!g) return;
    var ph = dist / 34, s = moving ? Math.sin(ph) : 0;
    g.setAttribute("transform", "translate(" + x.toFixed(1) + "," + y.toFixed(1) + ") scale(" + scale + ")");
    var q = function (c) { return g.querySelector("." + c); };
    q("leg-f").setAttribute("transform", "rotate(" + (s * 26).toFixed(1) + " 0 -68)");
    q("leg-b").setAttribute("transform", "rotate(" + (-s * 26).toFixed(1) + " 0 -68)");
    q("arm-f").setAttribute("transform", "rotate(" + (-s * 20).toFixed(1) + " 0 -116)");
    q("arm-b").setAttribute("transform", "rotate(" + (s * 20).toFixed(1) + " 0 -116)");
    q("bob").setAttribute("transform", "translate(0," + (moving ? -Math.abs(Math.sin(ph)) * 3.2 : 0).toFixed(1) + ")");
  }

  /* -------------------------------------------------------------- scenes */
  function columns() {
    var out = "", xs = [600, 668, 736, 944, 1012, 1080];
    xs.forEach(function (x) {
      out += '<rect x="' + (x - 15) + '" y="212" width="30" height="296" fill="#e6dffa"/>' +
             '<rect x="' + (x - 20) + '" y="204" width="40" height="10" rx="2" fill="#d9cdfb"/>' +
             '<rect x="' + (x - 20) + '" y="504" width="40" height="10" rx="2" fill="#d9cdfb"/>' +
             '<rect x="' + (x - 6) + '" y="220" width="4" height="280" fill="#d1c4f3" opacity=".7"/>';
    });
    return out;
  }
  function stars() {
    var out = "", r = 7;
    for (var i = 0; i < 46; i++) {
      r = (r * 9301 + 49297) % 233280;                  // deterministic, no Math.random
      var x = (r / 233280) * 1200;
      r = (r * 9301 + 49297) % 233280;
      var y = (r / 233280) * 230;
      out += '<circle cx="' + x.toFixed(0) + '" cy="' + y.toFixed(0) + '" r="' + (i % 5 ? 1.1 : 1.8) +
             '" fill="#efe9ff" opacity="' + (i % 3 ? .45 : .85) + '"/>';
    }
    return out;
  }
  function lamp(x) {
    return '<rect x="' + (x - 2) + '" y="380" width="4" height="180" fill="#14082b"/>' +
           '<circle cx="' + x + '" cy="378" r="9" fill="#fde68a"/>' +
           '<circle cx="' + x + '" cy="378" r="34" fill="url(#st-glow)"/>';
  }

  function build(female, grey) {
    svg.innerHTML =
      '<defs>' +
        '<linearGradient id="st-sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#14082b"/><stop offset=".6" stop-color="#361a66"/><stop offset="1" stop-color="#6d28d9"/></linearGradient>' +
        '<linearGradient id="st-coat" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#8b5cf6"/><stop offset="1" stop-color="#6d28d9"/></linearGradient>' +
        '<radialGradient id="st-glow"><stop offset="0" stop-color="#fde68a" stop-opacity=".55"/><stop offset="1" stop-color="#fde68a" stop-opacity="0"/></radialGradient>' +
        '<linearGradient id="st-wall" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#f3effe"/><stop offset="1" stop-color="#e6dffa"/></linearGradient>' +
        '<linearGradient id="st-light" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff7d6"/><stop offset="1" stop-color="#fde68a"/></linearGradient>' +
      '</defs>' +

      /* ---------- act 1: the street */
      '<g id="st-ext">' +
        '<rect x="-900" y="-700" width="3000" height="1300" fill="#14082b"/><rect x="-900" width="3000" height="675" fill="url(#st-sky)"/>' + stars() +
        '<circle cx="1050" cy="90" r="30" fill="#efe9ff" opacity=".9"/><circle cx="1062" cy="82" r="30" fill="#2a1250" opacity=".55"/>' +
        '<path d="M0 560 V330 h70 v-40 h60 v70 h50 V250 h80 v310 Z" fill="#2a1250" opacity=".85"/>' +
        '<path d="M380 560 V300 h90 v260 Z" fill="#2a1250" opacity=".7"/>' +
        lamp(130) + lamp(470) +
        /* the bank */
        '<polygon points="560,176 840,86 1120,176" fill="#efe9ff"/>' +
        '<polygon points="600,170 840,100 1080,170" fill="#e6dffa"/>' +
        '<circle cx="840" cy="146" r="17" fill="#6d28d9"/><path d="M830 150 L837 142 L842 147 L851 136" stroke="#fff" stroke-width="2.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/>' +
        '<rect x="550" y="176" width="580" height="30" fill="#d9cdfb"/>' +
        '<text x="840" y="198" text-anchor="middle" class="st-sign">SMARTCROSS BANK</text>' +
        '<rect x="570" y="206" width="540" height="304" fill="#cfc2f0"/>' +
        '<rect x="780" y="372" width="120" height="138" fill="url(#st-light)"/>' +
        '<g id="st-door-l"><rect x="780" y="372" width="60" height="138" fill="#4a2488"/><rect x="788" y="382" width="44" height="118" rx="3" fill="#6d28d9" opacity=".55"/><circle cx="832" cy="444" r="3" fill="#fde68a"/></g>' +
        '<g id="st-door-r"><rect x="840" y="372" width="60" height="138" fill="#4a2488"/><rect x="848" y="382" width="44" height="118" rx="3" fill="#6d28d9" opacity=".55"/><circle cx="848" cy="444" r="3" fill="#fde68a"/></g>' +
        '<path d="M774 372 Q840 318 906 372" fill="none" stroke="#efe9ff" stroke-width="8"/>' +
        columns() +
        '<rect x="560" y="510" width="560" height="14" fill="#efe9ff"/>' +
        '<rect x="545" y="524" width="590" height="14" fill="#e6dffa"/>' +
        '<rect x="530" y="538" width="620" height="22" fill="#d9cdfb"/>' +
        '<rect x="-900" y="560" width="3000" height="800" fill="#1b0b38"/>' +
        '<rect x="-900" y="560" width="3000" height="4" fill="#4a2488"/>' +
        '<g opacity=".35">' + [60, 220, 380, 1180].map(function (x) { return '<rect x="' + x + '" y="615" width="70" height="4" rx="2" fill="#b8a9e0"/>'; }).join("") + '</g>' +
        person("st-p1", female, grey) +
      '</g>' +

      /* ---------- act 2: the counter */
      '<g id="st-int" opacity="0">' +
        '<rect x="-900" y="-700" width="3000" height="1300" fill="#f3effe"/><rect x="-900" width="3000" height="675" fill="url(#st-wall)"/>' +
        [120, 360, 600].map(function (x) {
          return '<path d="M' + x + ' 360 V170 a90 90 0 0 1 180 0 V360 Z" fill="#fff" opacity=".75"/>' +
                 '<path d="M' + (x + 90) + ' 80 V360 M' + x + ' 230 H' + (x + 180) + '" stroke="#e6dffa" stroke-width="4"/>';
        }).join("") +
        '<text x="940" y="120" text-anchor="middle" class="st-wallsign">cross-sell desk</text>' +
        '<rect x="-900" y="540" width="3000" height="800" fill="#d9cdfb"/>' +
        [0, 1, 2, 3, 4, 5, 6].map(function (i) { return '<path d="M' + (i * 200 - 100) + ' 675 L' + (i * 170 + 40) + ' 540" stroke="#cfc2f0" stroke-width="2"/>'; }).join("") +
        '<rect x="640" y="430" width="500" height="112" rx="6" fill="#361a66"/>' +
        '<rect x="626" y="418" width="528" height="18" rx="5" fill="#4a2488"/>' +
        '<rect x="890" y="380" width="20" height="40" fill="#241046"/>' +
        '<rect x="830" y="406" width="140" height="12" rx="4" fill="#241046"/>' +
        '<rect id="st-mon" x="750" y="230" width="300" height="160" rx="12" fill="#241046"/>' +
        '<rect x="762" y="242" width="276" height="136" rx="6" fill="#14082b"/>' +
        '<text x="776" y="266" class="st-montitle">SmartCross · profile</text>' +
        '<g id="st-dots"></g>' +
        '<g id="st-chips"></g>' +
        person("st-p2", female, grey) +
      '</g>' +

      /* ---------- act 3 + 4: the engines */
      '<g id="st-pipe" opacity="0">' +
        '<rect x="-900" y="-700" width="3000" height="2100" fill="#14082b"/>' +
        '<g id="st-stages"></g>' +
        '<text id="st-who" x="60" y="66" class="st-who"></text>' +
        '<text id="st-whosub" x="60" y="90" class="st-whosub"></text>' +
        '<g id="st-bars"></g>' +
        '<text id="st-verdict" x="1140" y="640" text-anchor="end" class="st-verdict" opacity="0"></text>' +
      '</g>';

    pose($("#st-p1"), 60, 560, .74, 0, false);
    pose($("#st-p2"), 80, 560, 1.15, 0, false);
    buildDots();
  }

  /* the 35-feature vector: 24 holdings flags, then 11 behaviour/demographic */
  function buildDots() {
    var g = $("#st-dots"), out = "";
    for (var i = 0; i < 35; i++) {
      var c = i % 12, r = Math.floor(i / 12);
      out += '<circle cx="' + (790 + c * 20) + '" cy="' + (300 + r * 22) + '" r="6" class="st-dot" data-i="' + i + '"/>';
    }
    g.innerHTML = out;
  }

  /* -------------------------------------------------------------- data */
  var DATA = null;          // {sample, rec, rows, live}
  var STAGE_KEYS = ["content", "hybrid", "after_macro", "final"];
  var STAGE_NAMES = ["1 · What peers did", "2 · Content engine", "3 · Macro tilt", "4 · Eligibility"];

  function illustrative() {
    var names = ["pension plan", "credit card", "long term deposits", "funds", "e account", "mortgage", "securities", "loans"];
    var v = [[.30, .22, .12, .10, .09, .08, .05, .04], [.33, .25, .13, .08, .10, .06, .03, .02],
             [.31, .28, .12, .09, .09, .08, .03, .02], [.36, .32, .14, .10, .10, 0, .04, .03]];
    return {
      live: false,
      who: "Illustration — the API is offline",
      whosub: "Start the backend to replay this with a real held-out customer",
      regime: "RECOVERY",
      chips: ["age 34", "income 52,000", "holds current account", "holds payroll account", "7 months since last product"],
      female: Math.random() < .5, grey: false,
      rows: names.map(function (n, i) {
        return { label: n, v: [v[0][i], v[1][i], v[2][i], v[3][i]], tilt: [1, 1.15, 1, 1, 1, 1.2, 1.05, 1.05][i],
                 blocked: i === 5 ? "age 25-65, income at least 30,000, 24 months seniority" : null };
      }),
      verdict: null,
    };
  }

  function fromApi(s, rec) {
    var c = s.customer;
    var blocked = {};
    rec.blocked.forEach(function (b) { blocked[b.product] = b.rule; });
    var rows = rec.board.filter(function (b) { return !b.held; })
      .sort(function (a, b) { return b.after_macro - a.after_macro; })
      .slice(0, 8)
      .map(function (b) {
        return { label: b.label, product: b.product, v: [b.cf, b.hybrid, b.after_macro, b.final],
                 tilt: b.tilt, blocked: b.blocked ? (blocked[b.product] || "eligibility rule") : null };
      });
    var top = rec.recommendations.slice(0, 3).map(function (r) { return r.product; });
    var hit = top.indexOf(s.actual_next);
    return {
      live: true,
      who: c.name + " · a real record from the held-out months",
      whosub: "age " + c.age + " · income " + money(c.income) + " · " + c.holdings.length +
              " products held · regime " + rec.regime + (rec.regime_source && rec.regime_source.mode === "classified" ? " (classified from price series)" : ""),
      regime: rec.regime,
      chips: ["age " + c.age, "income " + money(c.income)]
        .concat(c.holdings.slice(0, 3).map(function (h) { return "holds " + pretty(h); }))
        .concat([c.months_since_last_acq >= 0 ? c.months_since_last_acq + " months since last product" : "tenure " + c.seniority_months + " months"]),
      female: !c.is_male, grey: c.age >= 62,
      rows: rows,
      top: top,
      verdict: hit >= 0
        ? "They actually took " + s.actual_label + " next — ranked #" + (hit + 1) + " here. ✓"
        : "They actually took " + s.actual_label + " next — not in this top three. Shown, not hidden.",
      verdictHit: hit >= 0,
    };
  }

  function load() {
    var api = function (path, opts) {
      return fetch(path, opts).then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); });
    };
    return api("/api/sample?min_holdings=2").then(function (s) {
      var body = Object.assign({}, s.customer, { regime: "AUTO", top_k: 3 });
      return api("/api/recommend", { method: "POST", headers: { "Content-Type": "application/json" },
                                     body: JSON.stringify(body) })
        .then(function (rec) { return fromApi(s, rec); });
    }).catch(function () { return illustrative(); });
  }

  function install(d) {
    DATA = d;
    build(d.female, d.grey);
    // chips
    $("#st-chips").innerHTML = d.chips.map(function (t, i) {
      var w = Math.max(110, t.length * 8.2 + 26);
      return '<g class="st-chip" data-i="' + i + '"><rect x="' + (-w / 2) + '" y="-15" width="' + w +
             '" height="30" rx="15"/><text x="0" y="5" text-anchor="middle">' + esc(t) + '</text></g>';
    }).join("");
    // stages
    $("#st-stages").innerHTML = STAGE_NAMES.map(function (n, i) {
      var x = 60 + i * 275;
      return '<g class="st-stage" data-i="' + i + '"><rect x="' + x + '" y="118" width="255" height="40" rx="20"/>' +
             '<text x="' + (x + 127) + '" y="143" text-anchor="middle">' + esc(n) + '</text></g>' +
             (i < 3 ? '<path d="M' + (x + 258) + ' 138 h14" class="st-arrow"/>' : "");
    }).join("");
    $("#st-who").textContent = d.who;
    $("#st-whosub").textContent = d.whosub;
    // bars
    $("#st-bars").innerHTML = d.rows.map(function (r, i) {
      var y = 196 + i * 50;
      return '<g class="st-row" data-i="' + i + '">' +
        '<text x="268" y="' + (y + 21) + '" text-anchor="end" class="st-lab">' + esc(r.label) + '</text>' +
        '<rect x="320" y="' + (y + 4) + '" width="760" height="26" rx="8" class="st-track"/>' +
        '<rect x="320" y="' + (y + 4) + '" width="0" height="26" rx="8" class="st-fill"/>' +
        '<text x="0" y="' + (y + 22) + '" class="st-val"></text>' +
        '<text x="330" y="' + (y + 22) + '" class="st-why" opacity="0"></text>' +
        '<g class="st-rank" opacity="0"><circle cx="296" cy="' + (y + 17) + '" r="0"/></g>' +
      '</g>';
    }).join("");
    render(lastP);
  }

  /* ------------------------------------------------------------ render */
  var lastP = 0;
  function render(p) {
    lastP = p;
    if (bar) bar.style.transform = "scaleX(" + p.toFixed(4) + ")";
    if (!DATA) return;

    /* act 1 — walk to the door, climb the steps, doors open, camera dives in */
    var walk = ease(seg(p, 0, .19));
    var x1 = lerp(60, 840, walk);
    var climb = clamp((x1 - 640) / 150, 0, 1);
    var y1 = 560 - climb * 46;
    var inside = seg(p, .19, .25);                       // stepping through the door
    pose($("#st-p1"), x1, y1, .74 - inside * .14, x1 - 60, walk > 0 && walk < 1);
    $("#st-p1").setAttribute("opacity", (1 - inside).toFixed(3));
    var open = ease(seg(p, .13, .2));
    $("#st-door-l").setAttribute("transform", "translate(780 0) scale(" + (1 - open * .82) + " 1) translate(-780 0)");
    $("#st-door-r").setAttribute("transform", "translate(900 0) scale(" + (1 - open * .82) + " 1) translate(-900 0)");
    var zoom = ease(seg(p, .21, .31));
    var s1 = 1 + zoom * 9;
    $("#st-ext").setAttribute("transform", "translate(840 450) scale(" + s1 + ") translate(-840 -450)");
    $("#st-ext").setAttribute("opacity", (1 - seg(p, .26, .31)).toFixed(3));

    /* act 2 — inside: walk to the counter, the record becomes a vector */
    var intIn = seg(p, .26, .31);
    var walk2 = ease(seg(p, .31, .44));
    var x2 = lerp(80, 520, walk2);
    pose($("#st-p2"), x2, 560, 1.15, x2 - 80, walk2 > 0 && walk2 < 1);
    var dive = ease(seg(p, .6, .68));                     // camera into the monitor
    var s2 = 1 + dive * 5.5;
    $("#st-int").setAttribute("transform", "translate(900 310) scale(" + s2 + ") translate(-900 -310)");
    $("#st-int").setAttribute("opacity", (intIn * (1 - seg(p, .64, .68))).toFixed(3));

    var chips = svg.querySelectorAll(".st-chip"), n = chips.length;
    for (var i = 0; i < n; i++) {
      var t0 = .45 + i * .022, t = ease(seg(p, t0, t0 + .06));
      var sx = 520 + (i % 2 ? 40 : -40), sy = 330 - i * 34;     // float up from the customer
      var cx = lerp(sx, 900, t), cy = lerp(sy, 312, t);
      var appear = seg(p, t0 - .02, t0);
      chips[i].setAttribute("transform", "translate(" + cx.toFixed(1) + "," + cy.toFixed(1) + ") scale(" + (1 - t * .75).toFixed(3) + ")");
      chips[i].setAttribute("opacity", (appear * (1 - seg(p, t0 + .05, t0 + .065))).toFixed(3));
    }
    var fill = seg(p, .46, .6), dots = svg.querySelectorAll(".st-dot");
    for (var k = 0; k < dots.length; k++) dots[k].classList.toggle("on", k / dots.length < fill);

    /* act 3 — the four engines on this customer's real scores */
    var pipeIn = seg(p, .65, .69);
    $("#st-pipe").setAttribute("opacity", pipeIn.toFixed(3));
    var u = clamp((p - .69) / .055, 0, 3.0);              // 0..3 across the four stages
    var si = Math.min(3, Math.floor(u + 1e-6)), sf = ease(u - Math.floor(u));
    if (u >= 3) { si = 3; sf = 0; }
    var stages = svg.querySelectorAll(".st-stage");
    for (var j = 0; j < stages.length; j++) {
      stages[j].classList.toggle("on", p >= .69 && j <= Math.round(u));
      stages[j].classList.toggle("now", p >= .69 && j === Math.min(3, Math.round(u)));
    }
    var vmax = 1e-9;
    DATA.rows.forEach(function (r) { r.v.forEach(function (v) { vmax = Math.max(vmax, v); }); });
    var rows = svg.querySelectorAll(".st-row");
    var stageNow = Math.min(3, Math.round(u));
    var showRanks = seg(p, .88, .92);
    var rankOf = {};
    if (DATA.top) DATA.top.forEach(function (prod, i) { rankOf[prod] = i + 1; });
    else {
      var order = DATA.rows.map(function (r, i) { return [r.v[3], i]; }).sort(function (a, b) { return b[0] - a[0]; });
      order.slice(0, 3).forEach(function (o, i) { rankOf["#" + o[1]] = i + 1; });
    }
    for (var r = 0; r < rows.length; r++) {
      var row = DATA.rows[r];
      var a = row.v[si], b = row.v[Math.min(3, si + 1)];
      var v = si >= 3 ? row.v[3] : lerp(a, b, sf);
      var w = Math.max(0, v / vmax * 760) * pipeIn;
      var el = rows[r];
      var fillEl = el.querySelector(".st-fill");
      fillEl.setAttribute("width", w.toFixed(1));
      var killed = row.blocked && u >= 2.5;
      el.classList.toggle("blocked", !!killed);
      var val = el.querySelector(".st-val");
      val.setAttribute("x", (320 + w + 10).toFixed(1));
      val.textContent = killed && v < 1e-6 ? "" : (v * 100).toFixed(1) + "%";
      var why = el.querySelector(".st-why");
      if (killed) { why.textContent = "removed — " + row.blocked; why.setAttribute("opacity", seg(u, 2.6, 3)); }
      else if (stageNow === 2 && Math.abs(row.tilt - 1) > .001) {
        why.textContent = (row.tilt > 1 ? "▲ ×" : "▼ ×") + row.tilt.toFixed(2) + " " + DATA.regime.toLowerCase();
        why.setAttribute("x", (320 + w + 70).toFixed(1));
        why.setAttribute("opacity", .9);
      } else why.setAttribute("opacity", 0);
      if (!killed) why.setAttribute("x", (320 + w + 70).toFixed(1));
      else why.setAttribute("x", "330");
      var rk = DATA.top ? rankOf[row.product] : rankOf["#" + r];
      var badge = el.querySelector(".st-rank");
      el.classList.toggle("top", !!rk && showRanks > 0);
      if (rk) {
        badge.setAttribute("opacity", showRanks.toFixed(3));
        badge.querySelector("circle").setAttribute("r", (13 * showRanks).toFixed(1));
        var bt = badge.querySelector("text");
        if (!bt) {
          bt = document.createElementNS(NS, "text");
          bt.setAttribute("text-anchor", "middle");
          bt.setAttribute("x", "296");
          bt.setAttribute("y", String(196 + r * 50 + 22));
          badge.appendChild(bt);
        }
        bt.textContent = rk;
      } else badge.setAttribute("opacity", 0);
    }

    /* act 4 — checked against what they actually did */
    var vEl = $("#st-verdict");
    if (DATA.verdict) {
      vEl.textContent = DATA.verdict;
      vEl.classList.toggle("miss", !DATA.verdictHit);
      vEl.setAttribute("opacity", seg(p, .92, .96).toFixed(3));
    }

    /* captions */
    var bounds = [0, .2, .31, .66, .7175, .7725, .8275];   // engine captions switch with the lit stage
    var c = 0;
    for (var q = 0; q < bounds.length; q++) if (p >= bounds[q]) c = q;
    caps.forEach(function (el, i) { el.classList.toggle("on", i === c); });
  }

  /* -------------------------------------------------------------- scroll */
  var ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(function () {
      ticking = false;
      var rect = story.getBoundingClientRect();
      var total = story.offsetHeight - window.innerHeight;
      if (total <= 0) return;
      render(clamp(-rect.top / total, 0, 1));
    });
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll);

  var skip = story.querySelector(".story-skip");
  if (skip) skip.addEventListener("click", function () {
    var target = document.getElementById("advise-start");
    if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
  });
  var again = story.querySelector(".story-again");
  if (again) again.addEventListener("click", function () {
    again.disabled = true;
    load().then(function (d) { install(d); onScroll(); again.disabled = false; });
  });

  install(illustrative());           // something to look at immediately
  load().then(function (d) { install(d); onScroll(); });
  onScroll();
})();

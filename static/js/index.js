(function () {
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const C = { c0: "#1f77b4", c1: "#ff7f0e", c2: "#2ca02c", c3: "#d62728" };
  const DASHDOT = "7 4 2 4";

  const clamp01 = (v) => Math.max(0, Math.min(1, v));
  const ease = (v) => (v < 0.5 ? 4 * v * v * v : 1 - Math.pow(-2 * v + 2, 3) / 2);

  // run `fn` once when `el` scrolls into view
  function onceVisible(el, fn, threshold = 0.25) {
    if (!el) return;
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) { io.disconnect(); fn(); }
    }, { threshold });
    io.observe(el);
  }

  // animate `tick(p)` for p in [0, 1] over `dur` ms; returns a cancel function
  function tween(dur, tick, done) {
    let raf = 0;
    const t0 = performance.now();
    const step = (now) => {
      const p = clamp01((now - t0) / dur);
      tick(p);
      if (p < 1) raf = requestAnimationFrame(step);
      else if (done) done();
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }

  function b64bytes(s) {
    const bin = atob(s);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  // ------------------------------------------------------------------ Fig. 1 teaser
  // All four panels build up together in three stages: marginal velocity, conditional / CFG
  // velocity (skipped where a panel has none), then the training trajectory for every panel.
  // Every layer is swept from t = 1 (right) to t = 0 (left). The figure stays once drawn.
  async function initTeaser() {
    const wrap = document.getElementById("fig1");
    const stage = wrap.querySelector(".fig1-stage");
    const steps = [...wrap.querySelectorAll(".fig1-steps li")];
    const replay = wrap.querySelector(".fig1-replay");
    try {
      const res = await fetch("static/img/fig1.svg");
      if (!res.ok) throw new Error(res.status);
      stage.innerHTML = await res.text();
    } catch (e) {
      stage.innerHTML = `<img src="${wrap.dataset.fallback}" alt="From consistency training to improved Self-Distillation">`;
      steps.forEach((s) => { s.classList.add("on"); s.style.setProperty("--p", 1); });
      return;
    }
    const svg = stage.querySelector("svg");
    const FADE = 500, SWEEP = 900, TRAJ = 1100, TRAJ_AT = 1800;
    const END = TRAJ_AT + TRAJ;
    const panels = [0, 1, 2, 3].map((p) => ({
      fade: [...svg.querySelectorAll(`.f1-p${p}.f1-dens, .f1-p${p}.f1-title`)],
      rect: (l) => svg.querySelector(`#rv-${p}-${l} rect`),
      timing: { marg: [300, SWEEP], cond: [1050, SWEEP], cfg: [1050, SWEEP], traj: [TRAJ_AT, TRAJ] },
      end: END,
    }));

    function setSweep(rect, v) {
      const x0 = +rect.dataset.x0, x1 = +rect.dataset.x1;
      const w = (x1 - x0 + 2) * v;
      rect.setAttribute("x", (x1 + 1 - w).toFixed(2));
      rect.setAttribute("width", w.toFixed(2));
    }

    function render(t) {
      panels.forEach((p, i) => {
        const a = clamp01(t / FADE);
        p.fade.forEach((e) => (e.style.opacity = a));
        for (const l in p.timing) {
          const [off, dur] = p.timing[l];
          setSweep(p.rect(l), ease(clamp01((t - off) / dur)));
        }
        const prog = clamp01(t / p.end);
        steps[i].classList.toggle("on", prog > 0);
        steps[i].style.setProperty("--p", prog.toFixed(3));
      });
    }

    let cancel = null;
    function play() {
      if (cancel) cancel();
      replay.classList.remove("show");
      cancel = tween(END, (p) => render(p * END), () => replay.classList.add("show"));
    }
    replay.addEventListener("click", play);
    if (REDUCED) { render(END); replay.classList.add("show"); return; }
    render(0);
    onceVisible(wrap, play, 0.3);
  }

  // ------------------------------------------------------------------ toy figures
  async function initToy() {
    let toy;
    try {
      toy = await (await fetch("static/data/toy.json")).json();
    } catch (e) {
      document.querySelectorAll(".toy-fig .toy-ctrl").forEach((c) => (c.textContent = "Serve this page over HTTP to load the toy animation."));
      return null;
    }
    const N = toy.bins * toy.bins;
    const lut = b64bytes(toy.lut);
    const runs = {};
    for (const k in toy.runs) runs[k] = b64bytes(toy.runs[k]);
    const gt = b64bytes(toy.gt);
    const F = toy.steps.length;
    const MAX = toy.steps[F - 1];

    function draw(canvas, src, f) {
      const ctx = canvas.getContext("2d");
      const img = ctx.createImageData(toy.bins, toy.bins);
      const i0 = Math.floor(f), i1 = Math.min(F - 1, i0 + 1), w = f - i0;
      for (let i = 0; i < N; i++) {
        const v = src.length === N ? src[i] : Math.round(src[i0 * N + i] * (1 - w) + src[i1 * N + i] * w);
        img.data[4 * i] = lut[3 * v];
        img.data[4 * i + 1] = lut[3 * v + 1];
        img.data[4 * i + 2] = lut[3 * v + 2];
        img.data[4 * i + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
    }

    const PLAY = 7000;  // ms for 0 -> MAX
    document.querySelectorAll(".toy-fig").forEach((fig) => {
      const names = fig.dataset.toyRuns.split(",");
      const canvases = [...fig.querySelectorAll("canvas")];
      const ctrl = fig.querySelector(".toy-ctrl");
      ctrl.innerHTML =
        `<button type="button" class="ctl-btn icon" aria-label="Play"><i class="fas fa-play" aria-hidden="true"></i></button>` +
        `<span class="lbl">Training steps</span>` +
        `<input type="range" min="0" max="${MAX}" step="1" value="0" aria-label="Training steps">` +
        `<output>0</output>`;
      const btn = ctrl.querySelector("button"), icon = btn.querySelector("i");
      const slider = ctrl.querySelector("input"), out = ctrl.querySelector("output");
      const listeners = [];
      fig._onStep = (fn) => listeners.push(fn);

      let step = 0, cancel = null;
      function render(s) {
        step = s;
        const f = (s / MAX) * (F - 1);
        names.forEach((n, i) => draw(canvases[i], n === "GT" ? gt : runs[n], f));
        slider.value = Math.round(s);
        out.textContent = Math.round(s);
        listeners.forEach((fn) => fn(s / MAX));
      }
      function setPlaying(on) {
        icon.className = on ? "fas fa-pause" : "fas fa-play";
        btn.setAttribute("aria-label", on ? "Pause" : "Play");
      }
      function pause() { if (cancel) { cancel(); cancel = null; } setPlaying(false); }
      function play() {
        if (step >= MAX) step = 0;
        const from = step, dur = PLAY * (1 - from / MAX);
        setPlaying(true);
        cancel = tween(dur, (p) => render(from + (MAX - from) * p), () => { cancel = null; setPlaying(false); });
      }
      btn.addEventListener("click", () => (cancel ? pause() : play()));
      slider.addEventListener("input", () => { pause(); render(+slider.value); });

      fig._render = render;
      fig._max = MAX;
      if (REDUCED) { render(MAX); return; }
      render(0);
      onceVisible(fig, play, 0.4);
    });
    return toy;
  }

  function initToyCharts(toy) {
    // L_ED over training steps (toy.tmlr.ipynb, C_plot): solid B=2048, dash-dot B=128
    const xs = [0, 200, 400, 600, 800, 1000];
    const color = { DT: C.c0, CT: C.c1, ED: C.c3 };
    const series = [];
    for (const k of ["DT", "ED", "CT"]) {
      for (const b of ["2048", "128"]) {
        series.push({ x: xs, y: toy.ed[`${k}_${b}`], color: color[k], width: b === "2048" ? 2.6 : 2.2,
          dash: b === "2048" ? null : DASHDOT, markers: 3.4 });
      }
    }
    const ed = Charts.lineChart(document.getElementById("chart-ed"), {
      width: 420, height: 290, aria: "L_ED over training steps",
      x: { domain: [0, 1000], ticks: [0, 200, 400, 600, 800, 1000], label: "Training Steps" },
      y: { domain: [0, 0.65], ticks: [0, 0.2, 0.4, 0.6], fmt: (v) => v.toFixed(1), label: "ED Loss" },
      series,
    });
    const legend = document.createElement("div");
    legend.className = "legend ed-legend";
    legend.innerHTML = ["DT", "ED", "CT"].map((k) => `<span><i style="background:${color[k]}"></i>${k}</span>`).join("");
    document.getElementById("chart-ed").after(legend);
    const fig = document.getElementById("toy-batch");
    if (fig._onStep && !REDUCED) { fig._onStep((p) => ed.setProgress(p)); ed.setProgress(0); }
    else ed.setProgress(1);

    // Gradient norms, toy setting (Ia_gradnorm): rolling window 40, mean ± std
    const names = { SD_2048: "SD", SGESD_2048: "SD w/SG (Boffi et al., 2025)", CT_2048: "CT", iSD_2048: "iSD (Ours)" };
    const cols = [C.c0, C.c1, C.c2, C.c3];
    const gx = toy.gradnorm_x;
    const gn = Charts.lineChart(document.getElementById("chart-gn-toy"), {
      width: 440, height: 300, aria: "Gradient norms in the toy setting",
      x: { domain: [0, 1000], ticks: [0, 250, 500, 750, 1000], label: "Training Steps" },
      y: { domain: [0, 0.5], ticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5], fmt: (v) => v.toFixed(1), label: "Gradient Norm" },
      series: Object.keys(names).map((k, i) => {
        const { mean, std } = toy.gradnorm[k];
        return { x: gx, y: mean, lo: mean.map((m, j) => (m == null ? null : m - std[j])),
          hi: mean.map((m, j) => (m == null ? null : m + std[j])), color: cols[i], width: 1.8 };
      }),
    });
    document.getElementById("legend-gn").innerHTML = Object.values(names)
      .map((n, i) => `<span><i style="background:${cols[i]}"></i>${n}</span>`).join("");
    return gn;
  }

  async function initImagenetGradnorm() {
    let d;
    try { d = await (await fetch("static/data/gradnorm_imagenet.json")).json(); } catch (e) { return null; }
    const cols = [C.c0, C.c1, C.c2, C.c3];
    const log = (a) => a.map((v) => (v == null || v <= 0 ? null : Math.log10(v)));
    return Charts.lineChart(document.getElementById("chart-gn-imagenet"), {
      width: 440, height: 300, aria: "Gradient norms in the ImageNet ablation setting",
      x: { domain: [0, 150000], ticks: [0, 50000, 100000, 150000], fmt: (v) => (v ? `${v / 1000}K` : "0"), label: "Training Steps" },
      y: { domain: [-3.5, -0.5], ticks: [-3, -2, -1], fmt: (v) => `1e${v}`, label: "Gradient Norm (Log-scale)" },
      series: ["SD", "SGESD", "CT", "iSD"].map((k, i) => ({
        x: d[k].x, y: log(d[k].mean), lo: log(d[k].lo), hi: log(d[k].hi), color: cols[i], width: 1.6,
      })),
    });
  }

  function animateDraw(el, charts, dur = 2600) {
    charts = charts.filter(Boolean);
    if (REDUCED) { charts.forEach((c) => c.setProgress(1)); return; }
    charts.forEach((c) => c.setProgress(0));
    onceVisible(el, () => tween(dur, (p) => charts.forEach((c) => c.setProgress(ease(p)))), 0.35);
  }

  // ------------------------------------------------------------------ time condition
  // Coefficients of L_ED under the flow map representation depend only on d = t - s:
  //   linear (alpha = 1 - t, sigma = t,     t in [0, 1]):    A'' = 0,       A' = 1,       A = d
  //   trig.  (alpha = cos t, sigma = sin t, t in [0, pi/2]): A'' = -sin d,  A' = cos d,   A = sin d
  function initTimeCondition() {
    const demo = document.getElementById("ts-demo");
    const eq = document.getElementById("ts-eq");
    const dIn = demo.querySelector(".ts-d"), dOut = demo.querySelector(".ts-dv");
    const btn = demo.querySelector(".ts-play"), icon = btn.querySelector("i");
    const bars = { a2: demo.querySelector("i.a2"), a1: demo.querySelector("i.a1"), a0: demo.querySelector("i.a0") };
    const vals = Object.fromEntries([...demo.querySelectorAll(".ts-val")].map((o) => [o.dataset.k, o]));
    const forms = Object.fromEntries([...demo.querySelectorAll(".ts-f")].map((o) => [o.dataset.k, o]));
    const GENERAL = String.raw`\mathcal L_\mathrm{ED} = \mathbb E\left[\left\|\htmlClass{tsk tsk-a2}{\color{#7a838e}A''_{t,s}x_t} + \htmlClass{tsk tsk-a1}{\color{#0070c0}A'_{t,s}(v^*_t - F_\theta)} - \htmlClass{tsk tsk-a0}{\color{#b4462b}A_{t,s}\frac{dF_\theta}{dt}}\right\|^2_2\right]`;
    const INTERP = {
      linear: {
        T: 1, forms: { a2: "0", a1: "1", a0: "t − s" },
        eq: String.raw`\mathbb E\left[\left\|\htmlClass{tsk tsk-a1}{\color{#0070c0}v^*_t - F_\theta} - \htmlClass{tsk tsk-a0}{\color{#b4462b}(t - s)\frac{dF_\theta}{dt}}\right\|^2_2\right]`,
        coef: (d) => ({ a2: 0, a1: 1, a0: d }),
      },
      trig: {
        T: Math.PI / 2, forms: { a2: "−sin(t−s)", a1: "cos(t−s)", a0: "sin(t−s)" },
        eq: String.raw`\mathbb E\left[\left\|\htmlClass{tsk tsk-a2}{\color{#7a838e}-\sin(t-s)\,x_t} + \htmlClass{tsk tsk-a1}{\color{#0070c0}\cos(t-s)(v^*_t - F_\theta)} - \htmlClass{tsk tsk-a0}{\color{#b4462b}\sin(t-s)\frac{dF_\theta}{dt}}\right\|^2_2\right]`,
        coef: (d) => ({ a2: -Math.sin(d), a1: Math.cos(d), a0: Math.sin(d) }),
      },
    };
    let interp = "linear", cancel = null;

    function render() {
      const I = INTERP[interp];
      const d = (+dIn.value / 1000) * I.T;
      const c = I.coef(d);
      dOut.textContent = d.toFixed(3);
      for (const k in c) {
        bars[k].style.width = `${Math.min(1, Math.abs(c[k])) * 100}%`;
        vals[k].textContent = (Math.abs(c[k]) < 5e-4 ? 0 : c[k]).toFixed(3);
        forms[k].textContent = I.forms[k];
        // emphasize each term of the equation by the magnitude of its coefficient
        const w = Math.min(1, Math.abs(c[k]));
        eq.querySelectorAll(`.tsk-${k}`).forEach((e) => {
          e.style.opacity = (0.25 + 0.75 * w).toFixed(3);
          e.style.fontSize = `${(0.8 + 0.2 * w).toFixed(3)}em`;  // shrinks, never exceeds the original size
        });
      }
    }
    function renderEq() {
      if (window.katex) katex.render(`${GENERAL} = ${INTERP[interp].eq}.`, eq,
        { displayMode: true, throwOnError: false, trust: (ctx) => ctx.command === "\\htmlClass" });
    }
    function setPlaying(on) {
      icon.className = on ? "fas fa-pause" : (+dIn.value >= 1000 ? "fas fa-rotate-right" : "fas fa-play");
      btn.setAttribute("aria-label", on ? "Pause" : +dIn.value >= 1000 ? "Replay" : "Play");
    }
    function pause() { if (cancel) { cancel(); cancel = null; } setPlaying(false); }
    function play() {
      if (+dIn.value >= 1000) dIn.value = 0;
      const from = +dIn.value;
      setPlaying(true);
      cancel = tween(3200 * (1 - from / 1000), (p) => { dIn.value = from + (1000 - from) * ease(p); render(); },
        () => { cancel = null; setPlaying(false); });
    }

    btn.addEventListener("click", () => (cancel ? pause() : play()));
    dIn.addEventListener("input", () => { pause(); render(); });
    demo.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
      interp = b.dataset.interp;
      demo.querySelectorAll(".seg button").forEach((x) => x.classList.toggle("on", x === b));
      demo.querySelectorAll(".ts-def").forEach((x) => (x.hidden = x.dataset.for !== interp));
      renderEq();
      render();
    }));
    renderEq();
    if (REDUCED) { dIn.value = 1000; render(); setPlaying(false); return; }
    render();
    onceVisible(demo, play, 0.5);
  }

  // ------------------------------------------------------------------ tabs / wipe / samples
  function retrigger(el) {
    el.classList.remove("in");
    void el.offsetWidth;
    el.classList.add("in");
  }

  function initWipe() {
    const els = document.querySelectorAll(".wipe");
    if (REDUCED) { els.forEach((e) => e.classList.add("in")); return; }
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
    }, { threshold: 0.2 });
    els.forEach((e) => io.observe(e));
  }

  function initTabs() {
    document.querySelectorAll("[data-tabs]").forEach((box) => {
      // only this box's own tabs: nested tab groups (landscapes, appendix) are handled separately
      const own = (e) => e.closest("[data-tabs]") === box;
      const buttons = [...box.querySelectorAll(".tab-list button")].filter(own);
      const panes = [...box.querySelectorAll(".tab-pane")].filter(own);
      const list = buttons.length ? buttons[0].closest(".tab-list") : null;
      const wipeVisible = () => panes.filter((p) => !p.hidden).forEach((p) =>
        p.querySelectorAll(".wipe").forEach(REDUCED ? (w) => w.classList.add("in") : retrigger));
      if (box.tagName === "DETAILS") box.addEventListener("toggle", () => box.open && wipeVisible());
      const show = (i, scroll) => {
        buttons.forEach((x, j) => x.setAttribute("aria-selected", i === j));
        panes.forEach((p, j) => (p.hidden = i !== j));
        panes[i].querySelectorAll(".wipe").forEach(REDUCED ? (w) => w.classList.add("in") : retrigger);
        if (scroll && list && box.classList.contains("sec-tabs")) {
          const top = box.getBoundingClientRect().top + window.scrollY - 56;
          if (window.scrollY > top) window.scrollTo({ top, behavior: REDUCED ? "auto" : "smooth" });
        }
      };
      buttons.forEach((b, i) => b.addEventListener("click", () => show(i, true)));
      box.querySelectorAll(".tab-next").forEach((n) => {
        if (own(n)) n.addEventListener("click", () => show(+n.dataset.next, true));
      });
    });
  }

  function initSamples() {
    document.querySelectorAll("[data-samples]").forEach((fig) => {
      const img = fig.querySelector(".sample-view img");
      const cap = fig.querySelector(".sample-cap");
      const buttons = [...fig.querySelectorAll(".sample-tabs button")];
      buttons.forEach((b) => b.addEventListener("click", () => {
        buttons.forEach((x) => x.setAttribute("aria-selected", x === b));
        img.classList.add("fading");
        const next = new Image();
        next.onload = () => {
          img.src = b.dataset.src;
          img.alt = `Two-step ImageNet samples from iSD-T (${b.textContent})`;
          img.classList.remove("fading");
          cap.innerHTML = b.dataset.caption;
          renderMath(cap);
        };
        next.src = b.dataset.src;
      }));
    });
  }

  // ------------------------------------------------------------------ table bar charts
  function initBars() {
    const ablation = [
      { label: "Consistency Training", fid: [120.49, "3.71"], is: [9.69, "0.47"] },
      { label: "+ Joint training w/ \\(\\mathcal L_\\text{CFM}\\)", fid: [96.28, "15.9"], is: [14.11, "4.15"] },
      { label: "+ Relaxing to \\(s&lt;t\\)", fid: [69.18, "2.84"], is: [21.45, "1.21"] },
      { label: "+ iSD", fid: [114.56, "2.80"], is: [11.04, "0.41"] },
      { label: "+ JVP approximation", fid: [108.70, "2.83"], is: [11.80, "0.36"], boldIsStd: true },
      { label: "+ iSD-T (\\(\\omega=5.0\\))", fid: [26.91, "0.50"], is: [74.70, "2.43"] },
      { label: "+ Trigonometric Interp.", fid: [14.62, "0.06"], is: [184.53, "0.77"], ours: true, boldFid: true, boldIs: true },
    ];
    const reprod = [
      { label: "MeanFlow (\\(\\omega = 3.0\\))", fid: [11.48, "1.68"], is: [167.60, "20.4"], boldFidMean: true },
      { label: "Shortcut Model (\\(\\omega = 1.5\\))", fid: [87.52, "24.3"], is: [14.94, "3.68"] },
      { label: "FACM (\\(\\omega = 1.75\\))", fid: [25.52, "1.73"], is: [65.42, "4.84"] },
      { label: "iSD-T (Ours, \\(\\omega = 5.0\\))", fid: [14.62, "0.06"], is: [184.53, "0.77"], ours: true, boldFidStd: true, boldIs: true },
    ];
    const rows = (data, key) => data.map((r) => ({
      label: r.label, mean: r[key][0], std: parseFloat(r[key][1]), stdText: r[key][1], ours: r.ours,
      boldMean: key === "fid" ? r.boldFid || r.boldFidMean : r.boldIs,
      boldStd: key === "fid" ? r.boldFid || r.boldFidStd : r.boldIs || r.boldIsStd,
    }));
    const charts = [
      [document.getElementById("ablation"), [
        Charts.barChart(document.getElementById("bars-ablation-fid"), rows(ablation, "fid"), { max: 140 }),
        Charts.barChart(document.getElementById("bars-ablation-is"), rows(ablation, "is"), { max: 200 }),
      ]],
      [document.getElementById("reprod"), [
        Charts.barChart(document.getElementById("bars-reprod-fid"), rows(reprod, "fid"), { max: 120 }),
        Charts.barChart(document.getElementById("bars-reprod-is"), rows(reprod, "is"), { max: 200 }),
      ]],
    ];
    for (const [fig, cs] of charts) {
      if (REDUCED) cs.forEach((c) => c.play(true));
      else onceVisible(fig, () => cs.forEach((c) => c.play(false)), 0.3);
    }
  }

  // ------------------------------------------------------------------ small pieces
  function initCountUp() {
    if (REDUCED) return;
    document.querySelectorAll(".countup").forEach((el) => {
      const to = parseFloat(el.dataset.to);
      const digits = (el.dataset.to.split(".")[1] || "").length;
      el.textContent = (0).toFixed(digits);
      onceVisible(el, () => tween(to > 100 ? 1800 : 1000, (p) => (el.textContent = (to * ease(p)).toFixed(digits))), 0.6);
    });
  }

  function initHeatTable() {
    const cells = [...document.querySelectorAll(".design-table td.heat")];
    const vals = cells.map((c) => parseFloat(c.textContent));
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const paint = () => cells.forEach((c, i) => {
      const a = 1 - (vals[i] - lo) / (hi - lo);
      c.style.backgroundColor = `rgba(0, 112, 192, ${(0.04 + 0.32 * a * a).toFixed(3)})`;
    });
    if (REDUCED) paint();
    else onceVisible(document.querySelector(".design-table"), paint, 0.4);
  }

  function initReveal() {
    const els = document.querySelectorAll(".reveal");
    if (REDUCED) { els.forEach((e) => e.classList.add("in")); return; }
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
    }, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
    els.forEach((e) => io.observe(e));
  }

  function initToc() {
    const links = [...document.querySelectorAll(".toc a")];
    const targets = links.map((a) => document.querySelector(a.getAttribute("href")));
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        links.forEach((a, i) => a.classList.toggle("active", targets[i] === e.target));
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    targets.forEach((t) => t && io.observe(t));
  }

  function initCopy() {
    document.querySelectorAll(".copy-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const text = document.querySelector(btn.dataset.copy).textContent;
        try { await navigator.clipboard.writeText(text); btn.querySelector("span").textContent = "Copied"; }
        catch (e) { btn.querySelector("span").textContent = "Select & copy"; }
        setTimeout(() => (btn.querySelector("span").textContent = "Copy"), 1600);
      });
    });
  }

  function renderMath(root = document.body) {
    if (window.renderMathInElement) {
      renderMathInElement(root, {
        delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }],
        throwOnError: false,
      });
    }
  }

  document.addEventListener("DOMContentLoaded", async () => {
    initBars();
    renderMath();
    initReveal();
    initWipe();
    initTabs();
    initSamples();
    initToc();
    initCopy();
    initCountUp();
    initHeatTable();
    initTimeCondition();
    initTeaser();
    const [toy, gnImagenet] = await Promise.all([initToy(), initImagenetGradnorm()]);
    const gnToy = toy ? initToyCharts(toy) : null;
    animateDraw(document.getElementById("gradnorm-fig"), [gnToy, gnImagenet]);
  });
})();

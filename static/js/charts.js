// Minimal SVG charts for the project page: line charts with ±σ bands that draw on with a
// clip rect, and horizontal mean±σ bar charts for the paper tables.
(function () {
  const NS = "http://www.w3.org/2000/svg";
  let uid = 0;

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function scale([d0, d1], [r0, r1]) {
    return (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
  }

  // index runs where every given array holds a finite value
  function runs(n, ...arrays) {
    const out = [];
    let cur = [];
    for (let i = 0; i < n; i++) {
      if (arrays.every((a) => a[i] != null && isFinite(a[i]))) cur.push(i);
      else if (cur.length) { out.push(cur); cur = []; }
    }
    if (cur.length) out.push(cur);
    return out;
  }

  function lineChart(container, cfg) {
    const W = cfg.width || 440, H = cfg.height || 300;
    const m = Object.assign({ top: 12, right: 14, bottom: 44, left: 52 }, cfg.margin || {});
    const x = scale(cfg.x.domain, [m.left, W - m.right]);
    const y = scale(cfg.y.domain, [H - m.bottom, m.top]);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": cfg.aria || "" }, container);

    const grid = el("g", { class: "grid" }, svg);
    for (const t of cfg.y.ticks) el("line", { x1: m.left, x2: W - m.right, y1: y(t), y2: y(t) }, grid);
    for (const t of cfg.x.ticks) el("line", { x1: x(t), x2: x(t), y1: m.top, y2: H - m.bottom }, grid);

    const axis = el("g", { class: "axis" }, svg);
    el("path", { d: `M${m.left},${m.top}V${H - m.bottom}H${W - m.right}`, fill: "none" }, axis);
    for (const t of cfg.y.ticks) {
      el("line", { x1: m.left - 4, x2: m.left, y1: y(t), y2: y(t) }, axis);
      const tx = el("text", { x: m.left - 7, y: y(t) + 3.5, "text-anchor": "end" }, axis);
      tx.textContent = cfg.y.fmt ? cfg.y.fmt(t) : t;
    }
    for (const t of cfg.x.ticks) {
      el("line", { x1: x(t), x2: x(t), y1: H - m.bottom, y2: H - m.bottom + 4 }, axis);
      const tx = el("text", { x: x(t), y: H - m.bottom + 16, "text-anchor": "middle" }, axis);
      tx.textContent = cfg.x.fmt ? cfg.x.fmt(t) : t;
    }
    el("text", { class: "ax-label", x: (m.left + W - m.right) / 2, y: H - 6, "text-anchor": "middle" }, svg).textContent = cfg.x.label;
    el("text", { class: "ax-label", transform: `translate(13,${(m.top + H - m.bottom) / 2}) rotate(-90)`, "text-anchor": "middle" }, svg).textContent = cfg.y.label;

    const id = `clip-${++uid}`;
    const clip = el("rect", { x: m.left, y: 0, width: 0, height: H }, el("clipPath", { id }, el("defs", {}, svg)));
    const body = el("g", { "clip-path": `url(#${id})` }, svg);
    const cy = (v) => y(Math.max(cfg.y.domain[0], Math.min(cfg.y.domain[1], v)));

    for (const s of cfg.series) {
      if (!(s.lo && s.hi)) continue;
      for (const idx of runs(s.x.length, s.lo, s.hi)) {
        const up = idx.map((i) => `${x(s.x[i]).toFixed(1)},${cy(s.hi[i]).toFixed(1)}`);
        const dn = idx.slice().reverse().map((i) => `${x(s.x[i]).toFixed(1)},${cy(s.lo[i]).toFixed(1)}`);
        el("path", { d: `M${up.join("L")}L${dn.join("L")}Z`, fill: s.color, opacity: 0.18 }, body);
      }
    }
    for (const s of cfg.series) {
      for (const idx of runs(s.x.length, s.y)) {
        el("path", {
          d: "M" + idx.map((i) => `${x(s.x[i]).toFixed(1)},${cy(s.y[i]).toFixed(1)}`).join("L"),
          fill: "none", stroke: s.color, "stroke-width": s.width || 2,
          "stroke-dasharray": s.dash || "none", "stroke-linejoin": "round", "stroke-linecap": "round",
        }, body);
        if (s.markers) for (const i of idx) el("circle", { cx: x(s.x[i]), cy: cy(s.y[i]), r: s.markers, fill: s.color }, body);
      }
    }

    const span = W - m.right - m.left;
    return {
      setProgress(p) {
        clip.setAttribute("width", Math.max(0, Math.min(1, p)) * span + (p >= 1 ? m.right + 2 : 0) + 6);
        clip.setAttribute("x", m.left - 6);
      },
    };
  }

  // rows: [{label, mean, std, stdText, ours, boldMean, boldStd}]
  function barChart(container, rows, cfg) {
    const head = document.createElement("div");
    head.className = "bars-head";
    head.textContent = container.dataset.metric || "";
    container.appendChild(head);
    const max = cfg.max;
    const fills = [];
    rows.forEach((r) => {
      const row = document.createElement("div");
      row.className = "bar-row" + (r.ours ? " ours" : "");
      const label = document.createElement("div");
      label.className = "bar-label";
      label.innerHTML = r.label;
      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      const err = document.createElement("div");
      err.className = "bar-err";
      const lo = Math.max(0, r.mean - r.std), hi = Math.min(max, r.mean + r.std);
      err.style.left = `${(lo / max) * 100}%`;
      err.style.width = `${((hi - lo) / max) * 100}%`;
      const val = document.createElement("div");
      val.className = "bar-val";
      val.style.left = `calc(${(hi / max) * 100}% + 8px)`;
      const m = r.boldMean ? `<b>${r.mean.toFixed(2)}</b>` : r.mean.toFixed(2);
      const s = r.boldStd ? `<b>±${r.stdText}</b>` : `±${r.stdText}`;
      val.innerHTML = `${m} <small>(${s})</small>`;
      track.append(fill, err, val);
      row.append(label, track);
      container.appendChild(row);
      fills.push([fill, (r.mean / max) * 100]);
    });
    return {
      play(instant) {
        container.classList.add("in");
        fills.forEach(([f, w], i) => {
          if (instant) f.style.transition = "none";
          else f.style.transitionDelay = `${i * 0.11}s`;
          f.style.width = `${w}%`;
        });
      },
    };
  }

  window.Charts = { lineChart, barChart };
})();

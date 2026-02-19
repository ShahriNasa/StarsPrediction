import React, { useMemo, useRef, useState, useEffect } from "react";




const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const PROPS = ["Rotation", "Gravity", "Temperature", "Mass"];
const DAYS = ["27", "97"];

const PROP_META = {
  Rotation: { emoji: "🌀", label: "Rotation Of Star", unit: "Days" },
  Gravity: { emoji: "🌍", label: "Gravity Of Star", unit: "cm/s²" },
  Temperature: { emoji: "🌡️", label: "Temperature Of Star", unit: "Kelvins" },
  Mass: { emoji: "⭐", label: "Mass Of Star", unit: "Solar Masses" },
};
const DIAGRAMS = {
  kiel: {
    title: "Kiel Diagram (log g vs Teff)",
    src: "/diagrams/kiel.jpg",
    // You MUST tune these to match your saved figure.
    // Teff axis on your plot: hotter left, cooler right.
    xMin: 3000,
    xMax: 12000,
    yMin: 0,
    yMax: 5.6,
    invertX: true,
    invertY: true,
    // plot area bounds inside the image (fractions of width/height)
    // Start with this, then adjust until marker matches grid intersections.
    plotRect: { l: 0.08, r: 0.96, t: 0.14, b: 0.90 },
  },
  hr: {
    title: "H-R Diagram (Luminosity vs Teff)",
    src: "/diagrams/hr.jpg",
    // ONLY if you compute luminosity
    xMin: 2500,
    xMax: 12000,
    // Many HR diagrams use log(L/Lsun). If your image is different, change it.
    yMin: -4,
    yMax: 6,
    invertX: true,
    invertY: false, // typically higher luminosity is "up"
    plotRect: { l: 0.10, r: 0.95, t: 0.10, b: 0.92 },
  }
};
// What each diagram needs
const DIAGRAM_REQUIREMENTS = {
  kiel: ["Temperature", "Gravity"],          // Teff + logg (or g -> logg)
  hr:   ["Temperature", "Gravity", "Mass"],  // need Mass to estimate luminosity via g=GM/R^2
};

// --- Categorization (match your Python logic) ---
// --- Categorization ---
const STAR_TYPES = ["White Dwarf", "Super Giant", "Red Giant", "Main Sequence", "Other"];

const STAR_TYPE_COLOR = {
  "Main Sequence": "yellow",
  "Red Giant": "red",
  "Super Giant": "magenta",
  "White Dwarf": "white",
  "Other": "grey",
};

// Fallback (pure logg-based like your original Python)
function categorizeByLogg(logg) {
  if (!Number.isFinite(logg)) return "Other";
  if (logg > 5.0) return "White Dwarf";
  if (logg < 3.0) return "Super Giant";
  if (logg < 3.5) return "Red Giant";
  if (logg >= 4.0 && logg <= 4.9) return "Main Sequence";
  return "Other";
}

// HR-based approximation (Teff + logL)
function categorizeByHR(teff, logL) {
  if (!Number.isFinite(teff) || !Number.isFinite(logL)) return null;

  // White dwarfs: hot but faint
  if (teff > 6000 && logL < -1) return "White Dwarf";

  // Supergiants: extremely luminous
  if (logL > 3.5) return "Super Giant";

  // Red giants: cool and luminous
  if (teff < 5500 && logL > 1) return "Red Giant";

  // Main sequence band (rough approximation)
  if (logL > -1.5 && logL < 2.0) return "Main Sequence";

  return "Other";
}

// Master classifier
function classifyStar({ teff, logg, logL }) {
  const hrClass = categorizeByHR(teff, logL);
  if (hrClass) return hrClass;

  return categorizeByLogg(logg);
}
function classificationModeForStar({ teff, logg, logL }) {
  // if HR classifier can run, it wins
  if (Number.isFinite(teff) && Number.isFinite(logL)) return "HR";
  if (Number.isFinite(logg)) return "LOGG";
  return "NONE";
}

function hasAll(set, arr) {
  for (const k of arr) if (!set.has(k)) return false;
  return true;
}
function Card({ title, children, style }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.06)",
      border: "1px solid rgba(255,255,255,0.10)",
      borderRadius: 16,
      padding: 16,
      ...style
    }}>
      <div style={{ fontSize: 14, opacity: 0.9, marginBottom: 10 }}>{title}</div>
      {children}
    </div>
  );
}
function legendTitleFromMode(mode) {
  if (mode === "HR") return "Star Type Legend (Teff + Luminosity)";
  if (mode === "LOGG") return "Star Type Legend (by log g)";
  if (mode === "MIXED") return "Star Type Legend (HR if possible, else log g)";
  return "Star Type Legend";
}
function downloadBlob(filename, contentType, text) {
  const blob = new Blob([text], { type: contentType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
function pickValue(items, property) {
  // prefer 97, then 27, else first
  const preferred = ["97", "27"];
  for (const d of preferred) {
    const hit = items.find(x => String(x.day) === d && x.property === property && Number.isFinite(Number(x.value)));
    if (hit) return Number(hit.value);
  }
  const any = items.find(x => x.property === property && Number.isFinite(Number(x.value)));
  return any ? Number(any.value) : null;
}

// Your backend returns Gravity in cm/s² (per your UI), but Kiel wants log g.
// If your model actually outputs logg already, then REMOVE this conversion.
function gravityToLoggMaybe(g_cgs) {
  // If value looks like logg (~0..6), keep it. If it looks like g (~1e2..1e6), convert.
  if (!Number.isFinite(g_cgs)) return null;
  if (g_cgs >= 0 && g_cgs <= 6.5) return g_cgs; // already logg
  if (g_cgs > 0) return Math.log10(g_cgs);
  return null;
}

function estimateLogLumFromMassLoggTeff(mass_solar, logg, teffK) {
  if (![mass_solar, logg, teffK].every(Number.isFinite)) return null;

  // constants
  const G = 6.67430e-11;
  const Msun = 1.98847e30;
  const Rsun = 6.95700e8;
  const Tsun = 5772;

  // logg is cgs, convert to SI
  const g_cgs = Math.pow(10, logg);
  const g_si = g_cgs / 100.0;

  const M = mass_solar * Msun;
  const R = Math.sqrt((G * M) / g_si); // meters

  const R_over = R / Rsun;
  const L_over = (R_over * R_over) * Math.pow(teffK / Tsun, 4);

  // log10(L/Lsun)
  return Math.log10(L_over);
}
function toCSV(rows) {
  const esc = (v) => {
    const s = String(v ?? "");
    if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
    return s;
  };
  const header = ["filename", "star_display", "day", "property", "value"];
  const lines = [header.join(",")];
  for (const r of rows) {
    lines.push([r.filename, r.starDisplay, r.day, r.property, r.value].map(esc).join(","));
  }
  return lines.join("\n");
}

async function runWithConcurrency(items, limit, worker, onProgress) {
  let i = 0;
  let done = 0;
  const results = new Array(items.length);

  const runners = new Array(Math.min(limit, items.length)).fill(0).map(async () => {
    while (true) {
      const idx = i++;
      if (idx >= items.length) break;
      results[idx] = await worker(items[idx], idx);
      done++;
      onProgress?.(done, items.length);
    }
  });

  await Promise.all(runners);
  return results;
}

/**
 * filename like: kplr007700670-2010355172524_llc.fits
 * display: Kepler 7700670 (Q2010355172524 • LLC)
 */
function starDisplayName(filename) {
  const base = filename.replace(/\.fits$/i, "");
  // Try parse common Kepler style
  const m = base.match(/^(kplr)(\d+)-(\d+)_([a-z]+)$/i);
  if (m) {
    const [, mission, idRaw, quarterRaw, cadenceRaw] = m;
    const id = String(parseInt(idRaw, 10)); // strips leading zeros
    const missionName = mission.toLowerCase() === "kplr" ? "Kepler" : mission.toUpperCase();
    const cadence = cadenceRaw.toLowerCase() === "llc" ? "LLC" : cadenceRaw.toUpperCase();
    return `${missionName} ${id} (Q${quarterRaw} • ${cadence})`;
  }
  // Fallback: shorten long names
  if (base.length > 28) return base.slice(0, 20) + "…" + base.slice(-7);
  return base;
}

function formatValue(property, value) {
  const meta = PROP_META[property] || { emoji: "✨", label: property, unit: "" };
  const v = Number(value);
  const num = Number.isFinite(v) ? v.toFixed(2) : String(value);
  const unit = meta.unit ? ` ${meta.unit}` : "";
  return `${meta.emoji} ${meta.label}: ${num}${unit}`;
}
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function dataToPixel({ x, y, xMin, xMax, yMin, yMax, invertX, invertY, plotRect, imgW, imgH }) {
  // plotRect is fractions: { l, r, t, b } in [0..1]
  const L = plotRect.l * imgW;
  const R = plotRect.r * imgW;
  const T = plotRect.t * imgH;
  const B = plotRect.b * imgH;

  const xn = (x - xMin) / (xMax - xMin);
  const yn = (y - yMin) / (yMax - yMin);

  const xx = invertX ? (1 - xn) : xn;
  const yy = invertY ? (1 - yn) : yn;

  return {
    px: L + clamp(xx, 0, 1) * (R - L),
    py: T + clamp(yy, 0, 1) * (B - T),
  };
}

function DiagramModal({ open, title, imgSrc, point, mapping, onClose, color = "rgba(0,255,255,1)" }) {
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const rafRef = useRef(null);

  useEffect(() => {
    if (!open) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const img = new Image();
    imgRef.current = img;
    img.crossOrigin = "anonymous"; // works for same-origin public files

    img.onload = () => {
      const ctx = canvas.getContext("2d");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;

      const draw = (t0) => {
        // flash: alpha oscillates
        const t = performance.now();
        const phase = Math.sin((t - t0) / 120); // speed
        const alpha = 0.35 + 0.65 * Math.abs(phase);

        // base
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);

        // marker
        if (point && Number.isFinite(point.x) && Number.isFinite(point.y)) {
          const { px, py } = dataToPixel({
            x: point.x,
            y: point.y,
            ...mapping,
            imgW: canvas.width,
            imgH: canvas.height,
          });

          // outer glow ring
          ctx.save();
          ctx.globalAlpha = alpha;
          ctx.lineWidth = Math.max(6, canvas.width * 0.002);
          ctx.strokeStyle = color;
          ctx.beginPath();
          ctx.arc(px, py, Math.max(18, canvas.width * 0.012), 0, Math.PI * 2);
          ctx.stroke();
          ctx.restore();

          // solid center dot (always visible)
          ctx.save();
          ctx.globalAlpha = 1;
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(px, py, Math.max(8, canvas.width * 0.005), 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
        }

        rafRef.current = requestAnimationFrame(() => draw(t0));
      };

      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      draw(performance.now());
    };

    img.src = imgSrc;

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [open, imgSrc, point, mapping]);

  function downloadMarked() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const a = document.createElement("a");
    a.download = `${title.replace(/\s+/g, "_").toLowerCase()}_marked.png`;
    a.href = canvas.toDataURL("image/png");
    a.click();
  }

  if (!open) return null;

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.72)",
        display: "grid",
        placeItems: "center",
        zIndex: 9999,
        padding: 18
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: "min(1100px, 96vw)",
          maxHeight: "92vh",
          overflow: "auto",
          borderRadius: 16,
          border: "1px solid rgba(255,255,255,0.15)",
          background: "rgba(20,20,20,0.98)",
          padding: 12
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "center", marginBottom: 10 }}>
          <div style={{ fontWeight: 900 }}>{title}</div>
          <div style={{ display: "flex", gap: 10 }}>
            <button
              onClick={downloadMarked}
              style={{
                padding: "8px 10px",
                borderRadius: 12,
                border: "1px solid rgba(255,255,255,0.15)",
                background: "rgba(255,255,255,0.08)",
                color: "inherit",
                cursor: "pointer",
                fontWeight: 900
              }}
            >
              ⬇️ Download marked
            </button>
            <button
              onClick={onClose}
              style={{
                padding: "8px 10px",
                borderRadius: 12,
                border: "1px solid rgba(255,255,255,0.15)",
                background: "rgba(255,255,255,0.08)",
                color: "inherit",
                cursor: "pointer",
                fontWeight: 900
              }}
            >
              ✕ Close
            </button>
          </div>
        </div>

        <canvas ref={canvasRef} style={{ width: "100%", height: "auto", display: "block", borderRadius: 12 }} />
      </div>
    </div>
  );
}

function DiagramThumb({ title, imgSrc, point, mapping, onOpen, color = "rgba(0,255,255,1)" }) {
  // lightweight preview using a plain <img> + absolute marker (good enough for thumbnail)
  const wrapRef = useRef(null);
  const [size, setSize] = useState({ w: 1, h: 1 });

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      setSize({ w: r.width, h: r.height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  let marker = null;
  if (point && Number.isFinite(point.x) && Number.isFinite(point.y)) {
    const { px, py } = dataToPixel({
      x: point.x,
      y: point.y,
      ...mapping,
      imgW: size.w,
      imgH: size.h,
    });

    marker = (
      <div
        style={{
          position: "absolute",
          left: px,
          top: py,
          width: 10,
          height: 10,
          transform: "translate(-50%, -50%)",
          borderRadius: 999,
          background: color,
          boxShadow: `0 0 14px ${color}`,
          pointerEvents: "none",
        }}
      />
    );
  }

  return (
    <div style={{ display: "grid", gap: 8 }}>
      <div style={{ fontSize: 12, opacity: 0.8 }}>{title}</div>
      <div
        ref={wrapRef}
        onClick={onOpen}
        style={{
          position: "relative",
          width: "100%",
          aspectRatio: "3 / 2",
          overflow: "hidden",
          borderRadius: 14,
          border: "1px solid rgba(255,255,255,0.12)",
          background: "rgba(255,255,255,0.04)",
          cursor: "pointer"
        }}
        title="Click to enlarge + flash marker"
      >
        <img
          src={imgSrc}
          alt={title}
          style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
        />
        {marker}
      </div>
    </div>
  );
}
export default function App() {
  const [page, setPage] = useState("SETUP"); // SETUP | RESULTS

  const [files, setFiles] = useState([]);
  const [properties, setProperties] = useState(() => new Set(["Rotation"]));
  const [days, setDays] = useState(() => new Set(["27"]));

  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");

  // rows: {filename, starDisplay, day, property, value}
  const [rows, setRows] = useState([]);
  const [errors, setErrors] = useState([]);

  const [runs, setRuns] = useState([]);
  const [runsBusy, setRunsBusy] = useState(false);
  const [runsErr, setRunsErr] = useState("");


  const [useHistory, setUseHistory] = useState(false);

  const [modalColor, setModalColor] = useState("rgba(0,255,255,1)");


  // multi-run selection
  const [selectedRunIds, setSelectedRunIds] = useState(() => new Set());

  // cache of run details: runId -> {run_id, stars:[...]}
  const [runCache, setRunCache] = useState(() => new Map());

  // selected stars across *all* runs: key = `${runId}:${star_file_id}`
  const [selectedStarKeys, setSelectedStarKeys] = useState(() => new Set());

  const [activeRunId, setActiveRunId] = useState(null);
  const [activeRun, setActiveRun] = useState(null); // full run payload {run_id, stars: [...]}
  const [selectedStarIds, setSelectedStarIds] = useState(() => new Set());


  const [progress, setProgress] = useState({ done: 0, total: 0, etaSec: null });
  const startRef = useRef(null);

  const canRun = useMemo(() => {
    return files.length > 0 && properties.size > 0 && days.size > 0 && !busy;
  }, [files, properties, days, busy]);

  const [modalOpen, setModalOpen] = useState(false);
  const [modalWhich, setModalWhich] = useState(null); // "kiel" | "hr"
  const [modalPoint, setModalPoint] = useState(null); // {x,y}
  const [modalTitle, setModalTitle] = useState("");
    function toggle(setter, value) {
      setter(prev => {
        const next = new Set(prev);
        if (next.has(value)) next.delete(value);
        else next.add(value);
        return next;
      });
    }

  const canShowKiel = useMemo(
    () => hasAll(properties, DIAGRAM_REQUIREMENTS.kiel),
    [properties]
  );

  const canShowHR = useMemo(
    () => hasAll(properties, DIAGRAM_REQUIREMENTS.hr),
    [properties]
  );

  function computeETA(done, total) {
    if (!startRef.current || done === 0) return null;
    const elapsed = (Date.now() - startRef.current) / 1000;
    const avgPerJob = elapsed / done;
    const remaining = total - done;
    return Math.max(0, Math.round(avgPerJob * remaining));
  }
function toggleRun(runId) {
  setSelectedRunIds(prev => {
    const next = new Set(prev);
    if (next.has(runId)) next.delete(runId);
    else next.add(runId);
    return next;
  });

  // prefetch the run details for smooth UX
  loadRun(runId).catch(err => setRunsErr(err?.message || String(err)));
}
function loadSelectedStarsToResults() {
  const picked = [];

  for (const s of starsFromSelectedRuns) {
    const k = starKey(s.run_id, s.star_file_id);
    if (selectedStarKeys.has(k)) picked.push(s);
  }

  const allRows = [];
  const allErrs = [];

  for (const s of picked) {
    for (const r of s.results) {
      allRows.push({
        filename: s.filename,
        starDisplay: s.star_display,
        day: r.day,
        property: r.property,
        value: Number(r.value),
        // optional: keep provenance
        runId: s.run_id,
        starFileId: s.star_file_id,
      });
    }
    for (const e of s.errors) {
      allErrs.push({
        filename: s.filename,
        starDisplay: s.star_display,
        day: e.day,
        property: e.property,
        error: e.error,
        runId: s.run_id,
        starFileId: s.star_file_id,
      });
    }
  }

  // sort
  allRows.sort((a, b) =>
    a.starDisplay.localeCompare(b.starDisplay) ||
    String(a.day).localeCompare(String(b.day)) ||
    a.property.localeCompare(b.property)
  );

  setRows(allRows);
  setErrors(allErrs);
  setStatus(`Loaded ${picked.length} star(s) from history.`);
  setPage("RESULTS");
}
const selectedRunsData = useMemo(() => {
  const arr = [];
  for (const runId of selectedRunIds) {
    const data = runCache.get(runId);
    if (data?.stars?.length) arr.push(data);
  }
  return arr;
}, [selectedRunIds, runCache]);

const starsFromSelectedRuns = useMemo(() => {
  const out = [];
  for (const run of selectedRunsData) {
    for (const s of run.stars || []) {
      out.push({
        run_id: run.run_id,
        created_at: run.created_at,
        star_file_id: s.star_file_id,
        filename: s.filename,
        star_display: s.star_display,
        results: s.results || [],
        errors: s.errors || [],
      });
    }
  }
  return out;
}, [selectedRunsData]);
function starKey(runId, starFileId) {
  return `${runId}:${starFileId}`;
}

function toggleStar(runId, starFileId) {
  const k = starKey(runId, starFileId);
  setSelectedStarKeys(prev => {
    const next = new Set(prev);
    if (next.has(k)) next.delete(k);
    else next.add(k);
    return next;
  });
}

function loadSelectedStarsFromRun() {
  if (!activeRun?.stars?.length) return;

  const pick = new Set(selectedStarIds);
  const pickedStars = activeRun.stars.filter(s => pick.has(s.star_file_id));

  const nextRows = [];
  const nextErrs = [];

  for (const s of pickedStars) {
    for (const r of (s.results || [])) {
      nextRows.push({
        filename: s.filename,
        starDisplay: s.star_display,
        day: r.day,
        property: r.property,
        value: Number(r.value),
      });
    }
    for (const e of (s.errors || [])) {
      nextErrs.push({
        filename: s.filename,
        starDisplay: s.star_display,
        day: e.day,
        property: e.property,
        error: e.error,
      });
    }
  }

  nextRows.sort((a, b) =>
    a.starDisplay.localeCompare(b.starDisplay) ||
    String(a.day).localeCompare(String(b.day)) ||
    a.property.localeCompare(b.property)
  );

  setRows(nextRows);
  setErrors(nextErrs);
  setStatus(`Loaded ${pickedStars.length} star(s) from run ${activeRun.run_id}`);
  setPage("RESULTS");
}

async function refreshRuns() {
  setRunsBusy(true);
  setRunsErr("");
  try {
    const resp = await fetch(`${API_BASE}/runs`);
    const data = await resp.json().catch(() => []);
    if (!resp.ok) throw new Error(data?.detail || "Failed to load runs");
    setRuns(Array.isArray(data) ? data : []);
  } catch (e) {
    setRunsErr(e?.message || String(e));
  } finally {
    setRunsBusy(false);
  }
}
async function deleteSelectedRuns() {
  if (selectedRunIds.size === 0) return;

  const ok = window.confirm(`Delete ${selectedRunIds.size} run(s)? This cannot be undone.`);
  if (!ok) return;

  try {
    setRunsBusy(true);
    setRunsErr("");

    const ids = Array.from(selectedRunIds);

    const resp = await fetch(`${API_BASE}/runs/bulk-delete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_ids: ids }),
    });

    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data?.detail || "Delete failed");

    // remove from UI state
    setSelectedRunIds(new Set());
    setSelectedStarKeys(new Set());

    setRunCache(prev => {
      const next = new Map(prev);
      for (const id of ids) next.delete(id);
      return next;
    });

    // refresh runs list
    await refreshRuns();
    setStatus(`Deleted ${ids.length} run(s).`);
  } catch (e) {
    setRunsErr(e?.message || String(e));
  } finally {
    setRunsBusy(false);
  }
}

async function deleteSelectedStars() {
  if (selectedStarKeys.size === 0) return;

  const ok = window.confirm(`Delete ${selectedStarKeys.size} star record(s) from history? This cannot be undone.`);
  if (!ok) return;

  // starKeys are "runId:starFileId" — convert to objects
  const stars = Array.from(selectedStarKeys).map(k => {
    const [run_id, star_file_id] = k.split(":");
    return { run_id: Number(run_id), star_file_id: Number(star_file_id) };
  });

  try {
    setRunsBusy(true);
    setRunsErr("");

    const resp = await fetch(`${API_BASE}/stars/bulk-delete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stars }),
    });

    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data?.detail || "Delete failed");

    // update caches: remove stars from cached runs
    setRunCache(prev => {
      const next = new Map(prev);
      for (const s of stars) {
        const run = next.get(s.run_id);
        if (!run?.stars) continue;
        run.stars = run.stars.filter(x => x.star_file_id !== s.star_file_id);
        next.set(s.run_id, { ...run });
      }
      return next;
    });

    setSelectedStarKeys(new Set());
    await refreshRuns();
    setStatus(`Deleted ${stars.length} star record(s).`);
  } catch (e) {
    setRunsErr(e?.message || String(e));
  } finally {
    setRunsBusy(false);
  }
}
useEffect(() => {
  if (useHistory) refreshRuns();
}, [useHistory]);
async function loadRun(runId) {
  // if cached, don’t refetch
  if (runCache.has(runId)) return;

  const resp = await fetch(`${API_BASE}/runs/${runId}`);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data?.detail || "Failed to load run");

  setRunCache(prev => {
    const next = new Map(prev);
    next.set(runId, data);
    return next;
  });
}


 async function predictOneFile(file, runId) {
  const form = new FormData();
  form.append("fits_file", file);
  form.append("run_id", runId);
  for (const p of Array.from(properties)) form.append("properties", p);
  for (const d of Array.from(days)) form.append("days", d);
    const resp = await fetch(`${API_BASE}/predict2`, { method: "POST", body: form });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data?.detail || "Request failed");

    const starDisplay = starDisplayName(file.name);

    const outRows = (data.results || []).map(r => ({
      filename: file.name,
      starDisplay,
      day: r.day,
      property: r.property,
      value: Number(r.value),
    }));

    const outErrs = (data.errors || []).map(e => ({
      filename: file.name,
      starDisplay,
      day: e.day,
      property: e.property,
      error: e.error,
    }));

    return { outRows, outErrs };
  }

  async function run() {
  if (!canRun) return;

  setBusy(true);
  setStatus("Creating run…");
  setRows([]);
  setErrors([]);
  setPage("SETUP");
  setProgress({ done: 0, total: files.length, etaSec: null });
  startRef.current = Date.now();

  const CONCURRENCY = 3;

  try {
    // 1) create run ONCE
    const runResp = await fetch(`${API_BASE}/runs`, { method: "POST" });
    const runData = await runResp.json().catch(() => ({}));

    if (!runResp.ok || !runData.run_id) {
      throw new Error(runData?.detail || "Failed to create run");
    }
    const runId = runData.run_id;

    // 2) now run predictions
    setStatus("Running predictions…");

    const results = await runWithConcurrency(
      files,
      CONCURRENCY,
      (f) => predictOneFile(f, runId),
      (done, total) => setProgress({ done, total, etaSec: computeETA(done, total) })
    );

    const allRows = [];
    const allErrs = [];
    for (const r of results) {
      if (!r) continue;
      allRows.push(...r.outRows);
      allErrs.push(...r.outErrs);
    }

    allRows.sort((a, b) =>
      a.starDisplay.localeCompare(b.starDisplay) ||
      String(a.day).localeCompare(String(b.day)) ||
      a.property.localeCompare(b.property)
    );

    setRows(allRows);
    setErrors(allErrs);
    setStatus("Done.");
    setPage("RESULTS");
  } catch (e) {
    setStatus(`Failed: ${e?.message || String(e)}`);
  } finally {
    setBusy(false);
  }
}


  function saveCSV() {
    if (rows.length === 0) return;
    downloadBlob("stellarscope_results.csv", "text/csv;charset=utf-8", toCSV(rows));
  }

  function saveJSON() {
    const payload = {
      files: files.map(f => f.name),
      properties: Array.from(properties),
      days: Array.from(days),
      rows,
      errors,
      generated_at: new Date().toISOString(),
    };
    downloadBlob("stellarscope_results.json", "application/json;charset=utf-8", JSON.stringify(payload, null, 2));
  }

  const prettyETA = progress.etaSec == null ? "" : `ETA ~ ${progress.etaSec}s`;

  // group rows by starDisplay
  const grouped = useMemo(() => {
    const m = new Map();
    for (const r of rows) {
      const key = `${r.filename}|${r.starDisplay}`;
      if (!m.has(key)) m.set(key, { filename: r.filename, starDisplay: r.starDisplay, items: [] });
      m.get(key).items.push(r);
    }
    return Array.from(m.values());
  }, [rows]);
  const legendMode = useMemo(() => {
      if (!grouped.length) return "NONE";

      let hr = 0, logg = 0, none = 0;

      for (const g of grouped) {
        const teff = pickValue(g.items, "Temperature");
        const grav = pickValue(g.items, "Gravity");
        const loggVal = gravityToLoggMaybe(grav);

        const mass = pickValue(g.items, "Mass");
        const logL = (Number.isFinite(mass) && Number.isFinite(loggVal) && Number.isFinite(teff))
          ? estimateLogLumFromMassLoggTeff(mass, loggVal, teff)
          : null;

        const canHR = Number.isFinite(teff) && Number.isFinite(logL);
        const canLogg = Number.isFinite(loggVal);

        if (canHR) hr++;
        else if (canLogg) logg++;
        else none++;
      }

      if (hr > 0 && logg > 0) return "MIXED";
      if (hr > 0) return "HR";
      if (logg > 0) return "LOGG";
      return "NONE";
    }, [grouped]);

  return (
    <div className="page">
      <div style={{ maxWidth: 1050, margin: "0 auto", padding: 24 }}>
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, marginBottom: 18 }}>
          <div>
            <div style={{ fontSize: 24, fontWeight: 800 }}>SpectroWave RCNN</div>
            <div style={{ opacity: 0.7, marginTop: 2 }}>
            FFT + DWT dual-domain preprocessing + RCNN modeling for stellar parameter prediction
            </div>
          </div>

          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            {page === "RESULTS" && (
              <button
                disabled={busy}
                onClick={() => setPage("SETUP")}
                style={{
                  padding: "10px 12px",
                  borderRadius: 12,
                  border: "1px solid rgba(255,255,255,0.15)",
                  background: "rgba(255,255,255,0.06)",
                  color: "inherit",
                  cursor: busy ? "not-allowed" : "pointer",
                  fontWeight: 700
                }}
              >
                ← New run
              </button>
            )}
          </div>
        </div>

        <div className="view">
        {/* SETUP PAGE CONTENT */}
        <div className={`viewPane ${page === "SETUP" ? "viewPane--visible" : "viewPane--hidden"}`}>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16, background:"transparent" }}>
              <Card title="Stars to Predict (FITS files)">
                <input
                  type="file"
                  accept=".fits"
                  multiple
                  disabled={busy}
                  onChange={(e) => {
                    const picked = Array.from(e.target.files || []);
                    if (picked.length === 0) return;

                    setFiles((prev) => {
                      const map = new Map(prev.map(f => [`${f.name}|${f.size}|${f.lastModified}`, f]));
                      for (const f of picked) map.set(`${f.name}|${f.size}|${f.lastModified}`, f);
                      return Array.from(map.values());
                    });

                    e.target.value = "";
                  }}
                />

                <div style={{ marginTop: 10, fontSize: 13, opacity: 0.85 }}>
                  {files.length ? `Selected: ${files.length} file(s)` : "No files selected"}
                </div>

                <div style={{ display: "flex", gap: 10, marginTop: 10 }}>
                  <button
                    disabled={busy || files.length === 0}
                    onClick={() => setFiles([])}
                    style={{
                      padding: "8px 10px",
                      borderRadius: 12,
                      border: "1px solid rgba(255,255,255,0.15)",
                      background: "rgba(255,255,255,0.08)",
                      color: "inherit",
                      cursor: busy || files.length === 0 ? "not-allowed" : "pointer",
                      fontWeight: 700
                    }}
                  >
                    Clear all
                  </button>
                </div>

                {files.length > 0 && (
                  <ul style={{ marginTop: 10, paddingLeft: 0, listStyle: "none", opacity: 0.92, maxHeight: 180, overflow: "auto" }}>
                    {files.map((f) => {
                      const key = `${f.name}|${f.size}|${f.lastModified}`;
                      return (
                        <li key={key} style={{ display: "flex", justifyContent: "space-between", gap: 10, padding: "6px 8px", borderRadius: 10, background: "rgba(255,255,255,0.04)", marginBottom: 6 }}>
                          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {starDisplayName(f.name)}
                            <span style={{ opacity: 0.65 }}> — {f.name}</span>
                          </span>
                          <button
                            disabled={busy}
                            onClick={() => setFiles(prev => prev.filter(x => `${x.name}|${x.size}|${x.lastModified}` !== key))}
                            style={{
                              borderRadius: 10,
                              padding: "4px 8px",
                              border: "1px solid rgba(255,255,255,0.15)",
                              background: "rgba(255,255,255,0.06)",
                              color: "inherit",
                              cursor: busy ? "not-allowed" : "pointer",
                              fontWeight: 700
                            }}
                          >
                            Remove
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </Card>

              <Card>
                <button
                  onClick={run}
                  disabled={!canRun}
                  style={{
                    width: "100%",
                    padding: "12px 12px",
                    borderRadius: 12,
                    border: "1px solid rgba(255,255,255,0.15)",
                    background: canRun ? "rgba(120,170,255,0.20)" : "rgba(255,255,255,0.08)",
                    color: "inherit",
                    fontWeight: 800,
                    cursor: canRun ? "pointer" : "not-allowed",
                  }}
                >
                  {busy ? "Running…" : "Predict"}
                </button>

                <div style={{ marginTop: 10, fontSize: 13, opacity: 0.85 }}>
                  {status}{" "}
                  {busy && (
                    <span>
                      • {progress.done}/{progress.total} • {prettyETA}
                    </span>
                  )}
                </div>

                <div style={{ marginTop: 14, opacity: 0.8, fontSize: 13, lineHeight: 1.4 }}>
                  Output units:
                  <div>🌀 Rotation: Days</div>
                  <div>🌍 Gravity: cm/s²</div>
                  <div>🌡️ Temperature: Kelvins</div>
                  <div>⭐ Mass: Solar Masses</div>
                </div>
              </Card>

              <Card title="Star Features to Predict">
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                  {PROPS.map(p => (
                    <label key={p} style={{ display: "flex", alignItems: "center", gap: 8, padding: 8, borderRadius: 12, background: "rgba(255,255,255,0.04)" }}>
                      <input type="checkbox" checked={properties.has(p)} disabled={busy} onChange={() => toggle(setProperties, p)} />
                      <span>{PROP_META[p]?.emoji} {p}
                        {(p === "Temperature" || p === "Gravity") && (
                          <span style={{ marginLeft: 8, fontSize: 11, opacity: 0.75 }}>(Kiel)</span>
                        )}
                        {p === "Mass" && (
                          <span style={{ marginLeft: 8, fontSize: 11, opacity: 0.75 }}>(H-R)</span>
                        )}
                      </span>
                    </label>
                  ))}
                </div>
              </Card>

              <Card title="Input Days (how many past days of data to use?)">
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {DAYS.map(d => (
                    <label key={d} style={{ display: "flex", alignItems: "center", gap: 8, padding: 8, borderRadius: 12, background: "rgba(255,255,255,0.04)" }}>
                      <input type="checkbox" checked={days.has(d)} disabled={busy} onChange={() => toggle(setDays, d)} />
                      <span>{d} days</span>
                    </label>
                  ))}
                </div>
              </Card>

            
            <Card title="Historical Prediction Runs">
                <label style={{ display:"flex", gap:8, alignItems:"center", marginBottom:12 }}>
                    <input
                    type="checkbox"
                    checked={useHistory}
                    onChange={(e) => setUseHistory(e.target.checked)}
                    />
                    <span>Use saved predictions (database)</span>
                </label>

                {!useHistory ? (
                    <div style={{ opacity: 0.8 }}>
                    Disabled. Turn it on to browse saved runs and load old results.
                    </div>
                ) : (
                    <>
                    {/* refresh + errors */}
                    <div style={{ display: "flex", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
                        <button onClick={refreshRuns} disabled={runsBusy} style={{
                                                        padding: "8px 10px",
                                                        borderRadius: 12,
                                                        border: "1px solid rgba(255,255,255,0.15)",
                                                        background: "rgba(255,255,255,0.08)",
                                                        color: "inherit",
                                                        cursor: runsBusy ? "not-allowed" : "pointer",
                                                        fontWeight: 800
                                                    }}>
                        ↻ Refresh
                        </button>
                        <button
                          onClick={deleteSelectedRuns}
                          disabled={runsBusy || selectedRunIds.size === 0}
                          style={{
                            padding: "8px 10px",
                            borderRadius: 12,
                            border: "1px solid rgba(255,255,255,0.15)",
                            background: selectedStarKeys.size ? "rgba(255,90,90,0.22)" : "rgba(255,255,255,0.08)",
                            color: "inherit",
                            cursor: (!runsBusy && selectedStarKeys.size) ? "pointer" : "not-allowed",
                            fontWeight: 900
                          }}
                        >
                          🗑️ Delete selected runs
                        </button>

                        <button
                          onClick={deleteSelectedStars}
                          disabled={runsBusy || selectedStarKeys.size === 0}
                          style={{
                            padding: "8px 10px",
                            borderRadius: 12,
                            border: "1px solid rgba(255,255,255,0.15)",
                            background: selectedStarKeys.size ? "rgba(255,90,90,0.22)" : "rgba(255,255,255,0.08)",
                            color: "inherit",
                            cursor: (!runsBusy && selectedStarKeys.size) ? "pointer" : "not-allowed",
                            fontWeight: 900
                          }}
                        >
                          🗑️ Delete selected stars
                        </button>
                        {runsErr && <span style={{ opacity: 0.85 }}>⚠️ {runsErr}</span>}
                        
                    </div>
                    

                    {runsBusy && <div style={{ opacity: 0.8 }}>Loading…</div>}
                    {!runsBusy && runs.length === 0 && <div style={{ opacity: 0.8 }}>No saved runs yet.</div>}

                    {!runsBusy && runs.length > 0 && (
                        <div style={{ display: "grid", gap: 8, maxHeight: 180, overflow: "auto" }}>
                        {runs.map(r => {
                            const checked = selectedRunIds.has(r.run_id);
                            return (
                            <label
                                key={r.run_id}
                                style={{
                                display: "flex",
                                gap: 10,
                                alignItems: "center",
                                padding: "10px 10px",
                                borderRadius: 12,
                                border: "1px solid rgba(255,255,255,0.12)",
                                background: checked ? "rgba(120,170,255,0.18)" : "rgba(255,255,255,0.06)",
                                cursor: "pointer"
                                }}
                            >
                                <input
                                type="checkbox"
                                checked={checked}
                                onChange={() => toggleRun(r.run_id)}
                                />
                                <div style={{ overflow:"hidden" }}>
                                <div style={{ fontWeight: 800 }}>
                                    Run #{r.run_id} • {new Date(r.created_at).toLocaleString()}
                                </div>
                                {r.note ? <div style={{ opacity: 0.75 }}>{r.note}</div> : null}
                                </div>
                            </label>
                            );
                        })}
                        </div>
                    )}

                    {selectedRunIds.size > 0 && (
                        <div style={{ marginTop: 12 }}>
                        <div style={{ opacity: 0.85, marginBottom: 8 }}>
                            Stars in selected runs: <b>{starsFromSelectedRuns.length}</b>
                        </div>

                        <div style={{ display: "grid", gap: 6, maxHeight: 180, overflow: "auto" }}>
                            {starsFromSelectedRuns.map(s => {
                            const k = `${s.run_id}:${s.star_file_id}`;
                            const checked = selectedStarKeys.has(k);
                            return (
                                <label
                                key={k}
                                style={{
                                    display: "flex",
                                    alignItems: "center",
                                    gap: 8,
                                    padding: "8px 10px",
                                    borderRadius: 12,
                                    background: "rgba(255,255,255,0.04)"
                                }}
                                >
                                <input
                                    type="checkbox"
                                    checked={checked}
                                    onChange={() => toggleStar(s.run_id, s.star_file_id)}
                                />
                                <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                    <b>{s.star_display}</b>
                                    <span style={{ opacity: 0.65 }}> — {s.filename}</span>
                                    <span style={{ opacity: 0.55 }}> • Run #{s.run_id}</span>
                                </span>
                                </label>
                            );
                            })}
                        </div>

                        <div style={{ display: "flex", gap: 10, marginTop: 10, flexWrap: "wrap" }}>
                            <button
                            onClick={loadSelectedStarsToResults}
                            disabled={selectedStarKeys.size === 0}
                            style={{
                                        padding: "10px 12px",
                                        borderRadius: 12,
                                        border: "1px solid rgba(255,255,255,0.15)",
                                        background: selectedStarKeys.size ? "rgba(120,170,255,0.20)" : "rgba(255,255,255,0.08)",
                                        color: "inherit",
                                        cursor: selectedStarKeys.size ? "pointer" : "not-allowed",
                                        fontWeight: 900
                                    }}
                            >
                            Load selected → Results
                            </button>

                            <button
                            onClick={() => setSelectedStarKeys(new Set())}
                            disabled={selectedStarKeys.size === 0}
                            style={{
                                    padding: "10px 12px",
                                    borderRadius: 12,
                                    border: "1px solid rgba(255,255,255,0.15)",
                                    background: "rgba(255,255,255,0.08)",
                                    color: "inherit",
                                    cursor: selectedStarKeys.size ? "pointer" : "not-allowed",
                                    fontWeight: 800
                                }}
                            >
                            Clear selection
                            </button>
                        </div>
                        </div>
                    )}
                    </>
                )}
                </Card>
                <Card title="Diagrams (requirements)">
                  <div style={{ fontSize: 13, opacity: 0.85, lineHeight: 1.5 }}>
                    <div style={{ fontWeight: 800, marginBottom: 6 }}>To show diagrams in Results:</div>
                    <div>• Kiel diagram needs: <b>Temperature</b> + <b>Gravity</b></div>
                    <div>• Classical H-R diagram needs: <b>Temperature</b> + <b>Gravity</b> + <b>Mass</b></div>
                    <div style={{ marginTop: 8, opacity: 0.8 }}>
                      Current selection:{" "}
                      <span style={{ opacity: canShowKiel ? 1 : 0.7 }}>
                        Kiel {canShowKiel ? "✅" : "❌"}
                      </span>
                      {" • "}
                      <span style={{ opacity: canShowHR ? 1 : 0.7 }}>
                        H-R {canShowHR ? "✅" : "❌"}
                      </span>
                    </div>
                  </div>
                </Card>
            </div>

            {/* show errors quickly if run failed */}
            {errors.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <Card title="Errors">
                  <ul style={{ margin: 0, paddingLeft: 18 }}>
                    {errors.map((e, idx) => (
                      <li key={idx} style={{ marginBottom: 6, opacity: 0.9 }}>
                        <b>{e.starDisplay}</b> — Day {e.day}, {e.property}: {e.error}
                      </li>
                    ))}
                  </ul>
                </Card>
              </div>
            )}
        </div>  {/* close SETUP pane */}
        {/* RESULTS PAGE CONTENT */}
        <div className={`viewPane ${page === "RESULTS" ? "viewPane--visible" : "viewPane--hidden"}`}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 12, marginBottom: 12 }}>
              <div style={{ opacity: 0.85 }}>
                Showing results for <b>{grouped.length}</b> star(s).
              </div>
              <Card title={legendTitleFromMode(legendMode)}>
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                  {STAR_TYPES.map(t => (
                    <div
                      key={t}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        padding: "6px 10px",
                        borderRadius: 999,
                        background: "rgba(255,255,255,0.06)",
                        border: "1px solid rgba(255,255,255,0.10)"
                      }}
                    >
                      <span style={{ width: 10, height: 10, borderRadius: 999, background: STAR_TYPE_COLOR[t] }} />
                      <span style={{ opacity: 0.9 }}>{t}</span>
                    </div>
                  ))}
                </div>

                <div style={{ marginTop: 8, fontSize: 12, opacity: 0.75 }}>
                  {legendMode === "HR" && "Classification uses Teff + Luminosity (HR plane)."}
                  {legendMode === "LOGG" && "Classification uses log g only (fallback)."}
                  {legendMode === "MIXED" && "Some stars use HR; others fall back to log g."}
                  {legendMode === "NONE" && "No usable Teff/logL/logg available."}
                </div>
              </Card>
              <div style={{ display: "flex", gap: 10 }}>
                <button
                  onClick={saveCSV}
                  disabled={rows.length === 0 || busy}
                  style={{
                    padding: "10px 12px",
                    borderRadius: 12,
                    border: "1px solid rgba(255,255,255,0.15)",
                    background: "rgba(255,255,255,0.08)",
                    color: "inherit",
                    cursor: rows.length && !busy ? "pointer" : "not-allowed",
                    fontWeight: 800,
                  }}
                >
                  ⬇️ Save CSV
                </button>
                <button
                  onClick={saveJSON}
                  disabled={(rows.length === 0 && errors.length === 0) || busy}
                  style={{
                    padding: "10px 12px",
                    borderRadius: 12,
                    border: "1px solid rgba(255,255,255,0.15)",
                    background: "rgba(255,255,255,0.08)",
                    color: "inherit",
                    cursor: (rows.length || errors.length) && !busy ? "pointer" : "not-allowed",
                    fontWeight: 800,
                  }}
                >
                  ⬇️ Save JSON
                </button>
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 16 }}>
              {grouped.map((g) => {
                const teff = pickValue(g.items, "Temperature");
                const grav = pickValue(g.items, "Gravity");
                const logg = gravityToLoggMaybe(grav);

                const mass = pickValue(g.items, "Mass");
                const logL = (Number.isFinite(mass) && Number.isFinite(logg) && Number.isFinite(teff))
                  ? estimateLogLumFromMassLoggTeff(mass, logg, teff)
                  : null;

                const starCanKiel = Number.isFinite(teff) && Number.isFinite(logg);
                const starCanHR = Number.isFinite(teff) && Number.isFinite(logL);
                const category = classifyStar({ teff, logg, logL });
                const colorName = STAR_TYPE_COLOR[category] || "grey";
                const markerColor = colorName; // using raw CSS color strings per your list
                
                return (
                  <Card key={`${g.filename}|${g.starDisplay}`} title={`⭐ ${g.starDisplay} — ${category}`} style={{ border: `1px solid ${markerColor}` }}>
                    {/* results list */}
                    <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 6, fontSize: 15 }}>
                      {g.items.map((it, idx) => (
                        <div key={idx} style={{ padding: "8px 10px", borderRadius: 12, background: "rgba(255,255,255,0.04)" }}>
                          <div style={{ opacity: 0.75, fontSize: 12, marginBottom: 2 }}>
                            Input Days: {it.day}
                          </div>
                          <div style={{ fontWeight: 700 }}>
                            {formatValue(it.property, it.value)}
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* diagrams section */}
                    <div style={{ marginTop: 12 }}>
                      <div style={{ fontSize: 13, opacity: 0.85, marginBottom: 8, fontWeight: 800 }}>
                        Diagrams
                      </div>

                      {(!canShowKiel || !canShowHR) && (
                        <div style={{ opacity: 0.75, fontSize: 13, lineHeight: 1.4 }}>
                          Diagram availability depends on selected predictions:
                          <div style={{ marginTop: 6 }}>
                            • Kiel: <b>Temperature</b> + <b>Gravity</b><br/>
                            • H-R: <b>Temperature</b> + <b>Gravity</b> + <b>Mass</b>
                          </div>
                        </div>
                      )}

                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                        {/* Kiel thumb */}
                        {canShowKiel ? (
                          starCanKiel ? (
                            <DiagramThumb
                              title="Kiel Diagram"
                              imgSrc={DIAGRAMS.kiel.src}
                              mapping={DIAGRAMS.kiel}
                              point={{ x: teff, y: logg }}
                              color={markerColor}
                              onOpen={() => {
                                setModalWhich("kiel");
                                setModalTitle(`⭐ ${g.starDisplay} — Kiel — ${category}`);
                                setModalPoint({ x: teff, y: logg });
                                setModalColor(markerColor);
                                setModalOpen(true);
                              }}
                            />
                          ) : (
                            <div style={{ borderRadius: 14, border: "1px dashed rgba(255,255,255,0.18)", padding: 12, opacity: 0.8 }}>
                              <div style={{ fontWeight: 900, marginBottom: 6 }}>Kiel Diagram</div>
                              <div style={{ fontSize: 13, lineHeight: 1.35 }}>
                                Missing usable values for this star (Temperature/Gravity).
                              </div>
                            </div>
                          )
                        ) : (
                          <div style={{ borderRadius: 14, border: "1px dashed rgba(255,255,255,0.18)", padding: 12, opacity: 0.8 }}>
                            <div style={{ fontWeight: 900, marginBottom: 6 }}>Kiel Diagram</div>
                            <div style={{ fontSize: 13, lineHeight: 1.35 }}>
                              Disabled (select <b>Temperature</b> + <b>Gravity</b> in Setup).
                            </div>
                          </div>
                        )}

                        {/* HR thumb */}
                        {canShowHR ? (
                          starCanHR ? (
                            <DiagramThumb
                              title="H-R Diagram"
                              imgSrc={DIAGRAMS.hr.src}
                              mapping={DIAGRAMS.hr}
                              point={{ x: teff, y: logL }}
                              onOpen={() => {
                                setModalWhich("hr");
                                setModalTitle(`⭐ ${g.starDisplay} — H-R — ${category}`);
                                setModalPoint({ x: teff, y: logL });
                                setModalColor(markerColor);
                                setModalOpen(true);
                              }}
                            />
                          ) : (
                            <div style={{ borderRadius: 14, border: "1px dashed rgba(255,255,255,0.18)", padding: 12, opacity: 0.8 }}>
                              <div style={{ fontWeight: 900, marginBottom: 6 }}>H-R Diagram</div>
                              <div style={{ fontSize: 13, lineHeight: 1.35 }}>
                                Not available (need Mass + Gravity + Temperature to estimate luminosity).
                              </div>
                            </div>
                          )
                        ) : (
                          <div style={{ borderRadius: 14, border: "1px dashed rgba(255,255,255,0.18)", padding: 12, opacity: 0.8 }}>
                            <div style={{ fontWeight: 900, marginBottom: 6 }}>H-R Diagram</div>
                            <div style={{ fontSize: 13, lineHeight: 1.35 }}>
                              Disabled (select <b>Temperature</b> + <b>Gravity</b> + <b>Mass</b> in Setup).
                            </div>
                          </div>
                        )}
                      </div>
                      <DiagramModal
                        open={modalOpen}
                        title={modalTitle || "Diagram"}
                        imgSrc={modalWhich ? DIAGRAMS[modalWhich].src : ""}
                        mapping={modalWhich ? DIAGRAMS[modalWhich] : DIAGRAMS.kiel}
                        point={modalPoint}
                        color={modalColor}
                        onClose={() => setModalOpen(false)}
                      />
                      {/* plotted numeric recap (only if enabled) */}
                      {(starCanKiel || starCanHR) && (
                        <div style={{ marginTop: 10, opacity: 0.8, fontSize: 13 }}>
                          {starCanKiel && (
                            <div>
                              Kiel point: Teff={teff.toFixed(0)} K • log g={logg.toFixed(3)}
                            </div>
                          )}
                          {starCanHR && (
                            <div>
                              H-R point: Teff={teff.toFixed(0)} K • log(L/L☉)={logL.toFixed(3)}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </Card>
                  
                );
              })}

              {errors.length > 0 && (
                <Card title="Errors">
                  <ul style={{ margin: 0, paddingLeft: 18 }}>
                    {errors.map((e, idx) => (
                      <li key={idx} style={{ marginBottom: 6, opacity: 0.9 }}>
                        <b>{e.starDisplay}</b> — Day {e.day}, {e.property}: {e.error}
                      </li>
                    ))}
                  </ul>
                </Card>
              )}
            </div>
    {/* close RESULTS pane */}
      </div> 
      {/* close view */} 
    </div> 
    {/* container */}   
    </div> 
    {/* page */}
    </div>   
);      
    
}

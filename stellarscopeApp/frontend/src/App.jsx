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

function Card({ title, children }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.06)",
      border: "1px solid rgba(255,255,255,0.10)",
      borderRadius: 16,
      padding: 16
    }}>
      <div style={{ fontSize: 14, opacity: 0.9, marginBottom: 10 }}>{title}</div>
      {children}
    </div>
  );
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

  function toggle(setter, value) {
    setter(prev => {
      const next = new Set(prev);
      if (next.has(value)) next.delete(value);
      else next.add(value);
      return next;
    });
  }

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
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
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
                      <span>{PROP_META[p]?.emoji} {p}</span>
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
                                        background: selectedStarIds.size ? "rgba(120,170,255,0.20)" : "rgba(255,255,255,0.08)",
                                        color: "inherit",
                                        cursor: selectedStarIds.size ? "pointer" : "not-allowed",
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
                                    cursor: selectedStarIds.size ? "pointer" : "not-allowed",
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
              {grouped.map((g) => (
                <Card key={g.filename} title={`⭐ ${g.starDisplay}`}>
                  {/* show results as readable lines instead of a boring table */}
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
                </Card>
              ))}

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

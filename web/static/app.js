const show = (v) => (v === null || v === undefined || v === "" ? "unknown" : typeof v === "number" ? v.toFixed(1) : String(v));
const percent = (v) => (v === null || v === undefined ? "unknown" : (v * 100).toFixed(1) + "%");

const LOG_LIMIT = 100;  // requests kept in the page buffer, matching RequestLog.max_entries

const COLUMNS = [
  "model", "time", "status", "prompt", "reused", "output",
  "prefill_tps", "decode_tps", "cache_hit_rate", "duration",
];

function toggleDetail(row, open) {
  const detail = detailFor(row);
  if (!detail) return null;
  detail.hidden = !open;
  row.setAttribute("aria-expanded", String(open));
  return detail;
}

function detailRow(entry, open) {
  const tr = document.createElement("tr");
  tr.className = "detail";
  tr.dataset.time = entry.time;  // pairs with the request row and survives a refresh
  tr.hidden = !open;

  const td = document.createElement("td");
  td.colSpan = COLUMNS.length;

  for (const [label, value] of [["Prompt", entry.prompt], ["Output", entry.output]]) {
    const block = document.createElement("div");
    block.className = "detail-block";
    const name = document.createElement("span");
    name.className = "detail-label";
    name.textContent = label;
    const text = document.createElement("pre");
    text.textContent = value || "unknown";
    block.append(name, text);
    td.appendChild(block);
  }

  const meta = document.createElement("p");
  meta.className = "note";
  meta.textContent = `route ${entry.route_reason || "unknown"} · outcome ${entry.outcome} · ttft ${show(entry.ttft_ms)} ms${entry.truncated ? " · text capped" : ""}`;
  td.appendChild(meta);

  tr.appendChild(td);
  return tr;
}

function detailFor(row) {
  const detail = row.nextElementSibling;
  return detail && detail.classList.contains("detail") ? detail : null;
}

async function refreshLog() {
  const tbody = document.getElementById("log");
  if (!tbody) return;

  const limit = Number(tbody.dataset.limit) || LOG_LIMIT;
  // The rebuild replaces every row, so the open state has to be carried across by entry time.
  const open = new Set([...tbody.querySelectorAll("tr.detail")].filter((tr) => !tr.hidden).map((tr) => tr.dataset.time));
  // Focus is on a row element that is about to be discarded, so remember which row it was.
  const focused = document.activeElement && document.activeElement.classList.contains("req") ? document.activeElement.dataset.time : null;

  const data = await fetch(`/api/log?limit=${limit}`).then((r) => r.json());
  tbody.innerHTML = "";

  for (const e of data.entries) {
    const tr = document.createElement("tr");
    tr.className = "req";
    tr.dataset.time = e.time;
    tr.tabIndex = 0;
    tr.setAttribute("aria-expanded", String(open.has(e.time)));

    const cells = [
      e.model, e.time, e.status, e.prompt, show(e.reused), e.output,
      show(e.prefill_tps), show(e.decode_tps), percent(e.cache_hit_rate), `${show(e.duration_ms)} ms`,
    ];
    cells.forEach((value, i) => {
      const td = document.createElement("td");
      if (i === 3 || i === 5) td.className = "clip";
      if (i === 2) td.className = `status ${e.outcome}`;
      td.textContent = value || "unknown";
      tr.appendChild(td);
    });

    tbody.append(tr, detailRow(e, open.has(e.time)));
  }

  if (focused) {
    const row = tbody.querySelector(`tr.req[data-time="${focused}"]`);
    if (row) row.focus();  // keep the keyboard user's place across the rebuild
  }
}

function matches(tr, needle) {
  return (
    tr.dataset.key.toLowerCase().includes(needle) ||
    tr.dataset.provider.toLowerCase().includes(needle) ||
    tr.textContent.toLowerCase().includes(needle)
  );
}

function applyFilter() {
  const input = document.getElementById("filter");
  if (!input) return;

  const needle = input.value.trim().toLowerCase();
  const rows = [...document.querySelectorAll("tr[data-key]")];
  const visible = rows.filter((tr) => !needle || matches(tr, needle));
  rows.forEach((tr) => {
    const hide = !visible.includes(tr);
    tr.hidden = hide;
    const detail = detailFor(tr);
    if (detail) detail.hidden = hide;  // a detail row never outlives its request row
  });

  document.querySelectorAll("section.group").forEach((group) => {
    group.hidden = needle !== "" && ![...group.querySelectorAll("tr[data-key]")].some((tr) => !tr.hidden);
  });

  const count = document.getElementById("filter-count");
  if (count) count.textContent = needle ? `${visible.length} of ${rows.length} rows` : `${rows.length} rows`;
}

function sortRows(th) {
  const table = th.closest("table");
  const tbody = table.querySelector("tbody");
  const rows = [...tbody.querySelectorAll("tr[data-key]")];
  const pairs = rows.map((tr) => [tr, detailFor(tr)]);
  const column = [...th.parentNode.children].indexOf(th);
  const dir = th.dataset.dir === "asc" ? "desc" : "asc";

  table.querySelectorAll("th").forEach((header) => {
    if (header !== th) delete header.dataset.dir;
  });
  th.dataset.dir = dir;
  table.querySelectorAll("th[data-sort]").forEach((header) => header.removeAttribute("aria-sort"));
  th.setAttribute("aria-sort", dir === "asc" ? "ascending" : "descending");

  const cell = (tr) => (tr.children[column] ? tr.children[column].textContent.trim() : "");
  const number = (text) => {
    // Strict parse: "2026-10-23 09:14:02" must stay text so timestamps sort chronologically.
    const value = Number(text.replace("%", ""));
    return text && Number.isFinite(value) ? value : null;
  };

  const compare = (a, b) => {
    const x = cell(a);
    const y = cell(b);
    const nx = number(x);
    const ny = number(y);
    if (nx === null && ny === null) return x.localeCompare(y) * (dir === "asc" ? 1 : -1);
    if (nx === null) return 1;  // unmeasured values always sort last
    if (ny === null) return -1;
    return (nx - ny) * (dir === "asc" ? 1 : -1);
  };

  pairs.sort((a, b) => compare(a[0], b[0]));
  pairs.forEach(([tr, detail]) => {
    tbody.appendChild(tr);
    if (detail) tbody.appendChild(detail);
  });
}

async function retest(button) {
  const tr = button.closest("tr[data-key]");
  const status = tr.querySelector(".retest-status");
  button.disabled = true;
  if (status) status.textContent = "running…";

  try {
    const res = await fetch("/api/retest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_key: tr.dataset.key }),
    });
    const data = await res.json();

    if (!res.ok || !data.attempt) {
      if (status) status.textContent = (data.error && data.error.message) || `retest failed: ${res.status}`;
      return;
    }

    const attempt = data.attempt;
    const detail = `${attempt.outcome}${attempt.error_class && attempt.error_class !== "ok" ? ` · ${attempt.error_class}` : ""} · ${attempt.latency_ms ?? "unknown"} ms`;
    if (status) status.textContent = `${attempt.run_at} · ${detail}`;

    // The config row shows the most recent attempt, so it reflects the new one.
    const last = tr.querySelector(".last-result");
    if (last) {
      last.className = `status last-result ${attempt.outcome}`;
      last.textContent = detail;
    }
  } catch (err) {
    if (status) status.textContent = `retest failed: ${err.message}`;
  } finally {
    button.disabled = false;
  }
}

async function saveConfig() {
  const status = document.getElementById("save-status");
  const providers = [...document.querySelectorAll("section.group")].map((group) => ({
    provider: group.dataset.provider,
    api_key: group.querySelector(".provider-key").value.trim(),
    tpm: Number(group.querySelector(".tpm").value) || 0,
  }));
  const models = [...document.querySelectorAll("tr[data-key]")].map((tr) => ({
    model_key: tr.dataset.key,
    host: tr.querySelector(".host").value.trim(),
    keyword: tr.querySelector(".keyword").value.trim(),
    api_key: tr.querySelector(".key").value.trim(),
    selected: tr.querySelector(".sel").checked,
  }));

  try {
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        api_host: document.getElementById("api_host").value.trim(),
        api_key: document.getElementById("api_key").value.trim(),
        providers,
        models,
      }),
    });
    const data = await res.json();

    if (!res.ok) {
      if (status) status.textContent = `save failed: ${(data.error && data.error.message) || res.status}`;
      return;
    }

    // Keys are runtime-only, so the source only changes for this process.
    document.querySelectorAll("section.group").forEach((group) => {
      const providerKey = group.querySelector(".provider-key").value.trim();
      group.querySelectorAll("tr[data-key]").forEach((tr) => {
        const source = tr.querySelector(".source");
        if (!source) return;
        if (tr.querySelector(".key").value.trim()) source.textContent = "runtime";
        else if (providerKey) source.textContent = "provider";
      });
    });

    if (status) status.textContent = `saved ${data.saved ? data.models : 0} models`;
  } catch (err) {
    if (status) status.textContent = `save failed: ${err.message}`;
  }
}

const logBody = document.getElementById("log");
if (logBody) {
  refreshLog();
  setInterval(refreshLog, 5000);

  const toggleFrom = (event) => {
    const row = event.target.closest("tr.req");
    const detail = row && detailFor(row);
    if (!detail) return;
    toggleDetail(row, detail.hidden);
  };

  logBody.addEventListener("click", toggleFrom);
  // Rows are focusable, so the same toggle has to work from the keyboard.
  logBody.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    toggleFrom(event);
  });
}

const saveButton = document.getElementById("save");
if (saveButton) saveButton.addEventListener("click", saveConfig);

const filterInput = document.getElementById("filter");
if (filterInput) {
  filterInput.addEventListener("input", applyFilter);
  applyFilter();
}

document.querySelectorAll("th[data-sort]").forEach((th) => {
  th.tabIndex = 0;
  const activate = () => sortRows(th);
  th.addEventListener("click", activate);
  // A header is only sortable from the keyboard if it can be focused and activated.
  th.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    activate();
  });
});

document.querySelectorAll("button.retest").forEach((button) => button.addEventListener("click", () => retest(button)));

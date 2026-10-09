"use strict";

const ui = Object.fromEntries([
  "browser", "status", "error", "remote-root", "local-root", "parent", "refresh",
  "current", "entries", "selected", "enter", "confirmation", "confirmation-title",
  "source", "destination", "file-count", "collision-warning", "conflicts",
  "skipped-section", "skipped", "cancel", "confirm",
].map(id => [id, document.getElementById(id)]));
let config = null;
let listing = null;
let selected = null;
let plan = null;
let busy = false;

function controls() {
  ui.status.setAttribute("data-busy", String(busy));
  ui.browser.disabled = busy || plan !== null;
  ui.parent.disabled = !listing || listing.path === config.initial_path;
  ui.enter.disabled = !selected || busy || plan !== null;
  ui.confirm.disabled = busy || !plan;
  ui.cancel.disabled = busy;
}

async function request(url, body) {
  const response = await fetch(url, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  let data;
  try { data = await response.json(); }
  catch { throw new Error("サーバーの応答を読み取れませんでした。"); }
  if (!response.ok) throw new Error(data.error?.message || `処理に失敗しました（${response.status}）。`);
  return data;
}

function showError(error) {
  ui.error.textContent = error.message;
  ui.error.hidden = false;
}

function choose(path) {
  if (busy || plan) return;
  selected = path;
  ui.selected.textContent = path;
  controls();
}

function directoryRow(name, path, canOpen) {
  const row = document.createElement("div");
  row.className = "entry";
  const label = document.createElement("label");
  const radio = document.createElement("input");
  radio.type = "radio";
  radio.name = "directory";
  radio.value = path;
  radio.addEventListener("change", () => choose(path));
  const text = document.createElement("span");
  text.textContent = name;
  label.append(radio, text);
  row.append(label);
  if (canOpen) {
    const open = document.createElement("button");
    open.type = "button";
    open.textContent = "開く";
    open.setAttribute("aria-label", `${name}を開く`);
    open.addEventListener("click", () => navigate(path));
    row.append(open);
  }
  return row;
}

async function navigate(path) {
  if (busy || plan) return;
  busy = true;
  selected = null;
  listing = null;
  ui.selected.textContent = "未選択";
  const loading = document.createElement("p");
  loading.className = "loading";
  loading.textContent = "ディレクトリの中身を読み込み中…";
  ui.entries.replaceChildren(loading);
  ui.current.textContent = path || "";
  ui.error.hidden = true;
  ui.status.textContent = "ディレクトリを読み込んでいます…";
  controls();
  try {
    if (!config) {
      config = await request("/api/config");
      ui["remote-root"].textContent = config.remote_root;
      ui["local-root"].textContent = config.local_destination_absolute;
    }
    const target = path ?? config.initial_path;
    const within = p => p === config.initial_path ||
      (config.initial_path === "." ? !p.startsWith("../") : p.startsWith(`${config.initial_path}/`));
    if (!within(target)) throw new Error("取得対象の範囲外です。");
    listing = await request(`/api/directories?${new URLSearchParams({ path: target })}`);
    ui.current.textContent = listing.path;
    ui.entries.replaceChildren();
    ui.entries.append(directoryRow("このディレクトリを選択", listing.path, false));
    for (const entry of listing.entries) {
      if (entry.type === "directory" && within(entry.path)) {
        ui.entries.append(directoryRow(entry.name, entry.path, true));
      } else {
        const row = document.createElement("div");
        row.className = "entry";
        const text = document.createElement("span");
        const kind = { file: "ファイル", symlink: "リンク・取得対象外", other: "取得対象外" };
        text.textContent = `${entry.name}（${kind[entry.type] || "取得対象外"}）`;
        row.append(text);
        ui.entries.append(row);
      }
    }
    ui.status.textContent = listing.entries.length ? "ディレクトリを選択してください。" : "空のディレクトリです。";
  } catch (error) {
    listing = null;
    ui.entries.replaceChildren();
    showError(error);
    ui.status.textContent = "再読み込みでやり直せます。";
  } finally {
    busy = false;
    controls();
  }
}

function showList(element, items) {
  element.replaceChildren(...items.map(item => {
    const li = document.createElement("li");
    li.textContent = item;
    return li;
  }));
}

ui.enter.addEventListener("click", async () => {
  if (busy || plan || !selected) return;
  busy = true;
  ui.error.hidden = true;
  ui.status.textContent = "取得内容と既存ファイルを確認しています…";
  controls();
  try {
    plan = await request("/api/download/preview", { path: selected });
    ui.source.textContent = plan.path;
    ui.destination.textContent = plan.destination;
    ui["file-count"].textContent = `${plan.file_count}件`;
    showList(ui.conflicts, plan.conflicts);
    showList(ui.skipped, plan.skipped);
    ui["collision-warning"].hidden = plan.conflicts.length === 0;
    ui["skipped-section"].hidden = plan.skipped.length === 0;
    ui.confirm.textContent = plan.conflicts.length ? "上書きする" : "取得する";
    ui.confirmation.hidden = false;
    ui["confirmation-title"].focus();
    ui.status.textContent = "内容を確認してから取得を開始してください。";
  } catch (error) {
    plan = null;
    showError(error);
    ui.status.textContent = "取得は開始していません。";
  } finally {
    busy = false;
    controls();
  }
});

ui.cancel.addEventListener("click", () => {
  if (busy) return;
  plan = null;
  ui.confirmation.hidden = true;
  ui.status.textContent = "取得は開始していません。選択を変更できます。";
  controls();
  ui.enter.focus();
});

ui.confirm.addEventListener("click", async () => {
  if (busy || !plan) return;
  busy = true;
  ui.error.hidden = true;
  ui.status.textContent = "転送中です。完了までお待ちください。";
  controls();
  try {
    const result = await request("/api/download", {
      token: plan.token, overwrite: plan.conflicts.length > 0,
    });
    ui.status.textContent = `取得が完了しました。保存先：${result.destination}`;
  } catch (error) {
    showError(error);
    ui.status.textContent = "完了を確認できませんでした。転送済みファイルは残ります。通信が切れた場合は転送が続いている可能性があります。状態を確認してからEnterで再確認してください。";
  } finally {
    plan = null;
    busy = false;
    ui.confirmation.hidden = true;
    controls();
    ui.enter.focus();
  }
});

ui.parent.addEventListener("click", () => {
  if (listing && listing.path !== config.initial_path) return navigate(listing.parent);
});
ui.refresh.addEventListener("click", () => navigate(listing?.path));
navigate();

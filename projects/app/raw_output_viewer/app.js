"use strict";

const ui = Object.fromEntries([
  "group", "parameter", "fold", "mode-filter", "label-filter", "samples",
  "sample-count", "selection", "status", "error", "warnings", "plots", "foraging", "rumination", "overlay",
  "fold-navigation", "fold-buttons", "judgment",
].map(id => [id, document.getElementById(id)]));
const labels = { foraging: "採食", rumination: "反芻", train: "訓練", test: "テスト" };
let samples = [];
let selected = null;
let conditionVersion = 0;
let plotVersion = 0;
let catalog = [];
let sampleVersion = 0;

async function request(path, parameters = {}) {
  const response = await fetch(`${path}?${new URLSearchParams(parameters)}`);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error?.message || "データの取得に失敗しました。");
  return body;
}

function showError(error) {
  ui.error.textContent = error.message || "通信に失敗しました。";
  ui.error.hidden = false;
  ui.status.textContent = "";
}

function clearPlot() {
  sampleVersion++;
  selected = null;
  ui["fold-navigation"].hidden = true;
  ui["fold-buttons"].replaceChildren();
  clearImages();
  ui.selection.textContent = "データを選択してください";
}

function clearImages() {
  plotVersion++;
  ui.judgment.hidden = true;
  ui.judgment.textContent = "";
  ui.judgment.className = "";
  ui.plots.hidden = true;
  ui.foraging.removeAttribute("src");
  ui.rumination.removeAttribute("src");
  ui.overlay.removeAttribute("src");
  ui.warnings.replaceChildren();
  ui.warnings.hidden = true;
  ui.error.hidden = true;
}

function setOptions(select, values) {
  select.replaceChildren(...values.map(value => new Option(String(value), String(value))));
  select.value = values.length ? String(values[0]) : "";
  select.disabled = values.length === 0;
}

function conditions() {
  return { group: ui.group.value, parameter: ui.parameter.value, fold: ui.fold.value };
}

function visibleSamples() {
  return samples.filter(sample =>
    (!ui["mode-filter"].value || sample.mode === ui["mode-filter"].value) &&
    (!ui["label-filter"].value || sample.true_label === ui["label-filter"].value));
}

function renderList() {
  const visible = visibleSamples();
  if (selected && !visible.includes(selected)) {
    clearPlot();
    ui.status.textContent = "データを選択してください。";
  }
  ui.samples.replaceChildren();
  ui["sample-count"].textContent = `${visible.length} / ${samples.length}件`;
  for (const sample of visible) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sample";
    button.setAttribute("aria-pressed", String(sample === selected));
    const id = document.createElement("span");
    id.className = "sample-id";
    id.textContent = sample.id;
    button.append(id);
    for (const value of [sample.mode, sample.true_label]) {
      const badge = document.createElement("span");
      badge.className = `badge ${value}`;
      badge.textContent = labels[value];
      button.append(badge);
    }
    button.addEventListener("click", () => selectSample(sample, button));
    ui.samples.append(button);
  }
  if (!visible.length) {
    const empty = document.createElement("p");
    empty.textContent = samples.length ? "該当するデータはありません。" : "条件を選択してください。";
    ui.samples.append(empty);
  }
}

async function loadConditions(level) {
  const version = ++conditionVersion;
  clearPlot();
  samples = [];
  renderList();
  if (level === 0) setOptions(ui.parameter, []);
  if (level <= 1) setOptions(ui.group, []);
  if (level <= 2) setOptions(ui.fold, []);
  ui.status.textContent = "読み込んでいます…";
  try {
    if (level === 0) {
      const body = await request("/api/conditions");
      if (version !== conditionVersion) return;
      const entries = await Promise.all(body.groups.map(async group => ({
        group, parameters: (await request("/api/conditions", { group })).parameters,
      })));
      if (version !== conditionVersion) return;
      catalog = entries;
      setOptions(ui.parameter, [...new Set(catalog.flatMap(entry => entry.parameters))].sort());
    }
    if (level <= 1) {
      setOptions(ui.group, catalog.filter(entry => entry.parameters.includes(ui.parameter.value))
        .map(entry => entry.group).sort());
    }
    if (!ui.parameter.value || !ui.group.value) {
      ui.status.textContent = "表示できる条件がありません。";
      return;
    }
    if (level <= 2) {
      const body = await request("/api/conditions", {
        group: ui.group.value, parameter: ui.parameter.value,
      });
      if (version !== conditionVersion) return;
      setOptions(ui.fold, body.folds);
    }
    if (ui.fold.value === "") {
      ui.status.textContent = "表示できるFoldがありません。";
      return;
    }
    const body = await request("/api/samples", conditions());
    if (version !== conditionVersion) return;
    samples = body.samples;
    renderList();
    ui.status.textContent = "左の一覧からデータを選択してください。";
  } catch (error) {
    if (version === conditionVersion) showError(error);
  }
}

async function selectSample(sample, button) {
  clearPlot();
  selected = sample;
  const version = sampleVersion;
  for (const item of ui.samples.children) {
    if (item.tagName === "BUTTON") item.setAttribute("aria-pressed", String(item === button));
  }
  ui.selection.textContent = `${sample.id} · Fold一覧を読み込んでいます…`;
  ui.status.textContent = "同じデータのFoldを確認しています…";
  try {
    const body = await request("/api/sample-folds", {
      group: ui.group.value, parameter: ui.parameter.value, id: sample.id,
    });
    if (version !== sampleVersion) return;
    renderFoldButtons(sample, body.folds);
    await showSamplePlot(sample, Number(ui.fold.value), sample.mode);
  } catch (error) {
    if (version === sampleVersion) showError(error);
  }
}

function renderJudgment(judgment) {
  if (judgment?.status === "available") {
    ui.judgment.textContent = `多数決判定：${labels[judgment.pred_label]}　真値：${labels[judgment.true_label]}　［${judgment.is_correct ? "正答" : "誤判定"}］`;
    ui.judgment.className = judgment.is_correct ? "correct" : "incorrect";
  } else {
    const messages = {
      disabled: "判定結果の参照先が未設定です。",
      not_found: "対応する判定結果がありません。",
      invalid_data: "判定結果を表示できません。",
    };
    ui.judgment.textContent = messages[judgment?.status] || "判定情報がありません。サーバーを再起動してください。";
    if (judgment?.status === "invalid_data" && judgment.message) {
      ui.judgment.textContent += ` ${judgment.message}`;
    }
    ui.judgment.className = "unavailable";
  }
  ui.judgment.hidden = false;
}

function renderFoldButtons(sample, folds) {
  ui["fold-buttons"].replaceChildren();
  for (const entry of folds) {
    for (const targetMode of entry.modes.length ? entry.modes : [null]) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = `Fold ${entry.fold}・${targetMode ? labels[targetMode] : "データなし"}`;
      button.disabled = targetMode === null;
      button.value = `${entry.fold}:${targetMode}`;
      if (targetMode) button.addEventListener("click", () => showSamplePlot(sample, entry.fold, targetMode));
      ui["fold-buttons"].append(button);
    }
  }
  ui["fold-navigation"].hidden = false;
}

async function showSamplePlot(sample, fold, mode) {
  if (sample !== selected) return;
  clearImages();
  const version = plotVersion;
  ui.selection.textContent = `${sample.id} · 表示中：Group ${ui.group.value} · Fold ${fold} · ${labels[mode]}`;
  for (const button of ui["fold-buttons"].children) {
    button.setAttribute("aria-pressed", String(button.value === `${fold}:${mode}`));
  }
  ui.status.textContent = "グラフを生成しています…";
  try {
    const body = await request("/api/plots", { ...conditions(), fold, mode, id: sample.id });
    if (version !== plotVersion) return;
    ui.selection.textContent += ` · 真値：${labels[body.true_label]}`;
    renderJudgment(body.judgment);
    for (const node of ["foraging", "rumination", "overlay"]) {
      ui[node].src = `data:image/png;base64,${body.images[node]}`;
    }
    for (const warning of body.warnings) {
      const message = document.createElement("p");
      message.textContent = warning.message;
      ui.warnings.append(message);
    }
    ui.warnings.hidden = body.warnings.length === 0;
    ui.plots.hidden = false;
    ui.status.textContent = `${body.length}時刻 · 縦軸 −1.5〜2.5`;
  } catch (error) {
    if (version === plotVersion) showError(error);
  }
}

ui.parameter.addEventListener("change", () => loadConditions(1));
ui.group.addEventListener("change", () => loadConditions(2));
ui.fold.addEventListener("change", () => loadConditions(3));
ui["mode-filter"].addEventListener("change", renderList);
ui["label-filter"].addEventListener("change", renderList);
loadConditions(0);

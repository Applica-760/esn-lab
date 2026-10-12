const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");

class Element {
  constructor(tagName = "DIV") {
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.events = {};
    this.attributes = {};
    this.value = "";
    this.textContent = "";
  }
  replaceChildren(...children) {
    this.children = children;
    if (this.tagName === "SELECT") this.value = children[0]?.value || "";
  }
  append(child) { this.children.push(child); }
  add(child) { this.append(child); }
  addEventListener(name, callback) { this.events[name] = callback; }
  setAttribute(name, value) { this.attributes[name] = value; }
  removeAttribute(name) { delete this[name]; }
}

async function setup(empty = false) {
  const nodes = {};
  const pending = [];
  const context = vm.createContext({
    URLSearchParams,
    Option: function (text, value) { return { text, value }; },
    document: {
      getElementById(id) {
        const select = ["group", "parameter", "fold", "mode-filter", "label-filter"].includes(id);
        return nodes[id] ||= new Element(select ? "SELECT" : "DIV");
      },
      createElement: tag => new Element(tag),
    },
    fetch: url => new Promise(resolve => pending.push({
      url,
      respond(body, ok = true) { resolve({ ok, json: async () => body }); },
    })),
  });
  vm.runInContext(readFileSync(join(__dirname, "../app/raw_output_viewer/app.js"), "utf8"), context);
  const settle = () => new Promise(resolve => setImmediate(resolve));
  pending.shift().respond({ groups: ["a", "b"] });
  await settle();
  pending.shift().respond({ parameters: empty ? [] : ["p", "q"] });
  pending.shift().respond({ parameters: empty ? [] : ["p"] });
  await settle();
  async function change(id, value, body) {
    nodes[id].value = value;
    const done = nodes[id].events.change();
    if (body) {
      for (const response of (Array.isArray(body) ? body : [body])) {
        pending.shift().respond(response);
        await settle();
      }
    }
    await done;
  }
  if (!empty) {
    pending.shift().respond({ folds: [0, 1] });
    await settle();
    pending.shift().respond({ samples: [
      { id: "same", mode: "train", true_label: "foraging", length: 4 },
      { id: "same", mode: "test", true_label: "rumination", length: 5 },
    ] });
    await settle();
  }
  return { nodes, pending, change, settle };
}

function plot(marker, warnings = [], judgment = { status: "disabled" }) {
  return { true_label: "foraging", length: 4, images: { foraging: marker, rumination: marker, overlay: marker }, warnings, judgment };
}

const folds = { folds: [
  { fold: 0, modes: ["train", "test"] },
  { fold: 1, modes: ["train"] },
  { fold: 2, modes: [] },
] };

async function beginPlot(view, index = 0) {
  const done = view.nodes.samples.children[index].events.click();
  const request = view.pending.shift();
  assert.match(request.url, /^\/api\/sample-folds\?/);
  request.respond(folds);
  await view.settle();
  return { done, request: view.pending.shift() };
}

test("fold zero, badges and filters retain train/test identity", async () => {
  const view = await setup();
  const { nodes, change } = view;
  assert.equal(nodes.parameter.value, "p");
  assert.equal(nodes.group.value, "a");
  assert.equal(nodes.fold.value, "0");
  assert.equal(nodes.samples.children.length, 2);
  assert.equal(nodes.samples.children[0].children[1].textContent, "訓練");
  await change("mode-filter", "test");
  assert.equal(nodes.samples.children.length, 1);
  const { done, request } = await beginPlot(view);
  const url = new URL(request.url, "http://localhost");
  assert.equal(url.searchParams.get("fold"), "0");
  assert.equal(url.searchParams.get("mode"), "test");
  request.respond(plot("png", [{ message: "範囲外です" }]));
  await done;
  assert.equal(nodes.warnings.hidden, false);
  assert.equal(nodes.plots.hidden, false);
  await change("label-filter", "foraging");
  assert.equal(nodes.plots.hidden, true);
  assert.equal(nodes["sample-count"].textContent, "0 / 2件");
});

test("an older image response cannot overwrite the latest selection", async () => {
  const view = await setup();
  const { nodes } = view;
  const first = await beginPlot(view);
  const second = await beginPlot(view, 1);
  second.request.respond(plot("new"));
  await second.done;
  first.request.respond(plot("old"));
  await first.done;
  assert.equal(nodes.foraging.src, "data:image/png;base64,new");
  assert.equal(nodes.overlay.src, "data:image/png;base64,new");
  assert.match(nodes.selection.textContent, /テスト/);
});

test("changing conditions discards pending images and stale condition lists", async () => {
  const view = await setup();
  const { nodes, pending, settle } = view;
  const image = await beginPlot(view);
  nodes.group.value = "b";
  const older = nodes.group.events.change();
  nodes.group.value = "a";
  const newer = nodes.group.events.change();
  pending[1].respond({ folds: [0] });
  await settle();
  pending[2].respond({ samples: [] });
  await newer;
  pending[0].respond({ folds: [9] });
  await older;
  image.request.respond(plot("old"));
  await image.done;
  assert.equal(nodes.plots.hidden, true);
  assert.equal(nodes.foraging.src, undefined);
  assert.equal(nodes.overlay.src, undefined);
  assert.equal(nodes.fold.value, "0");
  assert.equal(pending.length, 3);
  assert.equal(nodes["fold-navigation"].hidden, true);
  assert.equal(nodes.judgment.hidden, true);
  assert.equal(nodes.judgment.textContent, "");
});

test("parameter changes select only compatible groups and automatically load the first fold", async () => {
  const { nodes, pending, change } = await setup();
  await change("group", "b", [{ folds: [1] }, { samples: [] }]);
  assert.equal(nodes.fold.value, "1");
  const done = change("parameter", "q", [{ folds: [0] }, { samples: [] }]);
  assert.equal(nodes.group.value, "a");
  await done;
  assert.deepEqual(nodes.group.children.map(option => option.value), ["a"]);
  assert.equal(nodes.fold.value, "0");
  assert.equal(pending.length, 0);
});

test("empty parameters leave dependent controls disabled without further requests", async () => {
  const { nodes, pending } = await setup(true);
  assert.equal(nodes.parameter.disabled, true);
  assert.equal(nodes.group.disabled, true);
  assert.equal(nodes.fold.disabled, true);
  assert.equal(pending.length, 0);
  assert.match(nodes.status.textContent, /表示できる条件がありません/);
});

test("API errors show a message and leave old images hidden", async () => {
  const view = await setup();
  const { nodes } = view;
  const { done, request } = await beginPlot(view);
  request.respond({ error: { message: "データが見つかりません。" } }, false);
  await done;
  assert.equal(nodes.error.hidden, false);
  assert.equal(nodes.error.textContent, "データが見つかりません。");
  assert.equal(nodes.plots.hidden, true);
  assert.equal(nodes.judgment.hidden, true);
});

for (const [judgment, message, className] of [
  [{ status: "available", pred_label: "foraging", true_label: "foraging", is_correct: true }, "多数決判定：採食　真値：採食　［正答］", "correct"],
  [{ status: "available", pred_label: "rumination", true_label: "foraging", is_correct: false }, "多数決判定：反芻　真値：採食　［誤判定］", "incorrect"],
  [{ status: "disabled" }, "判定結果の参照先が未設定です。", "unavailable"],
  [{ status: "not_found" }, "対応する判定結果がありません。", "unavailable"],
  [{ status: "invalid_data", message: "真値が一致しません。" }, "判定結果を表示できません。 真値が一致しません。", "unavailable"],
]) {
  test(`judgment ${judgment.status}/${className} is displayed separately from images`, async () => {
    const view = await setup();
    const { done, request } = await beginPlot(view);
    request.respond(plot("png", [], judgment));
    await done;
    assert.equal(view.nodes.judgment.textContent, message);
    assert.equal(view.nodes.judgment.className, className);
    assert.equal(view.nodes.judgment.hidden, false);
    assert.equal(view.nodes.plots.hidden, false);
  });
}

test("judgment follows the latest fold response and clears on ID changes and failures", async () => {
  const view = await setup();
  const { nodes, pending } = view;
  const correct = { status: "available", pred_label: "foraging", true_label: "foraging", is_correct: true };
  const incorrect = { ...correct, pred_label: "rumination", is_correct: false };
  const first = await beginPlot(view);
  first.request.respond(plot("initial", [], correct));
  await first.done;
  const older = nodes["fold-buttons"].children[2].events.click();
  const olderRequest = pending.shift();
  assert.equal(nodes.judgment.hidden, true);
  assert.equal(nodes.judgment.textContent, "");
  const newer = nodes["fold-buttons"].children[1].events.click();
  pending.shift().respond(plot("latest", [], incorrect));
  await newer;
  olderRequest.respond(plot("old", [], correct));
  await older;
  assert.equal(nodes.judgment.className, "incorrect");
  assert.match(nodes.judgment.textContent, /誤判定/);
  const selection = await beginPlot(view, 1);
  assert.equal(nodes.judgment.hidden, true);
  assert.equal(nodes.judgment.textContent, "");
  selection.request.respond({ error: { message: "描画失敗" } }, false);
  await selection.done;
  assert.equal(nodes.judgment.hidden, true);
  assert.equal(nodes.error.textContent, "描画失敗");
});

test("fold switches preserve button elements, selection state and list conditions", async () => {
  const view = await setup();
  const { nodes, pending } = view;
  const initial = await beginPlot(view, 1);
  initial.request.respond(plot("initial"));
  await initial.done;
  const list = nodes.samples.children;
  assert.deepEqual(nodes["fold-buttons"].children.map(button => button.textContent), [
    "Fold 0・訓練", "Fold 0・テスト", "Fold 1・訓練", "Fold 2・データなし",
  ]);
  assert.equal(nodes["fold-buttons"].children[1].attributes["aria-pressed"], "true");
  assert.equal(nodes["fold-buttons"].children[3].disabled, true);
  const buttons = [...nodes["fold-buttons"].children];
  const older = nodes["fold-buttons"].children[2].events.click();
  for (const [index, button] of buttons.entries()) {
    assert.equal(nodes["fold-buttons"].children[index], button);
  }
  assert.equal(buttons[1].attributes["aria-pressed"], "false");
  assert.equal(buttons[2].attributes["aria-pressed"], "true");
  assert.equal(nodes.plots.hidden, true);
  assert.equal(nodes.overlay.src, undefined);
  const request = pending.shift();
  const url = new URL(request.url, "http://localhost");
  assert.deepEqual(Object.fromEntries(url.searchParams), {
    group: "a", parameter: "p", fold: "1", mode: "train", id: "same",
  });
  const newer = nodes["fold-buttons"].children[0].events.click();
  pending.shift().respond(plot("latest"));
  await newer;
  request.respond(plot("old"));
  await older;
  for (const [index, button] of buttons.entries()) {
    assert.equal(nodes["fold-buttons"].children[index], button);
  }
  assert.equal(nodes.overlay.src, "data:image/png;base64,latest");
  assert.match(nodes.selection.textContent, /Fold 0 · 訓練/);
  assert.equal(nodes["fold-buttons"].children[0].attributes["aria-pressed"], "true");
  assert.equal(nodes.fold.value, "0");
  assert.equal(nodes.group.value, "a");
  assert.equal(nodes.parameter.value, "p");
  assert.equal(nodes.samples.children, list);
  assert.equal(list[1].attributes["aria-pressed"], "true");
});

test("stale fold lists cannot replace a different ID or survive condition changes", async () => {
  const view = await setup();
  const { nodes, pending, change } = view;
  await change("fold", "1", { samples: [
    { id: "first", mode: "test", true_label: "foraging" },
    { id: "second", mode: "test", true_label: "rumination" },
  ] });
  const older = nodes.samples.children[0].events.click();
  const stale = pending.shift();
  const current = await beginPlot(view, 1);
  assert.equal(new URL(current.request.url, "http://localhost").searchParams.get("id"), "second");
  current.request.respond(plot("current"));
  await current.done;
  stale.respond({ folds: [{ fold: 9, modes: ["test"] }] });
  await older;
  assert.match(nodes.selection.textContent, /^second/);
  assert.equal(nodes["fold-buttons"].children.length, 4);
  const unfinished = nodes.samples.children[0].events.click();
  const late = pending.shift();
  await change("parameter", "q", [{ folds: [0] }, { samples: [] }]);
  late.respond(folds);
  await unfinished;
  assert.equal(nodes["fold-navigation"].hidden, true);
  assert.equal(nodes.plots.hidden, true);
  assert.equal(pending.length, 0);
});

test("fold-list failure clears old buttons and can be retried by selecting the ID", async () => {
  const view = await setup();
  const { nodes, pending } = view;
  const initial = await beginPlot(view);
  initial.request.respond(plot("initial"));
  await initial.done;
  const failed = nodes.samples.children[1].events.click();
  pending.shift().respond({ error: { message: "Fold一覧の取得に失敗" } }, false);
  await failed;
  assert.equal(nodes["fold-navigation"].hidden, true);
  assert.equal(nodes["fold-buttons"].children.length, 0);
  assert.equal(nodes.plots.hidden, true);
  assert.equal(nodes.error.textContent, "Fold一覧の取得に失敗");
  const retry = await beginPlot(view, 1);
  retry.request.respond(plot("retry"));
  await retry.done;
  assert.equal(nodes.error.hidden, true);
  assert.equal(nodes["fold-navigation"].hidden, false);
  assert.equal(nodes.overlay.src, "data:image/png;base64,retry");
});

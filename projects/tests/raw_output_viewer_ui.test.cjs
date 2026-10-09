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

function plot(marker, warnings = []) {
  return { length: 4, images: { foraging: marker, rumination: marker }, warnings };
}

test("fold zero, badges and filters retain train/test identity", async () => {
  const { nodes, pending, change } = await setup();
  assert.equal(nodes.parameter.value, "p");
  assert.equal(nodes.group.value, "a");
  assert.equal(nodes.fold.value, "0");
  assert.equal(nodes.samples.children.length, 2);
  assert.equal(nodes.samples.children[0].children[1].textContent, "訓練");
  await change("mode-filter", "test");
  assert.equal(nodes.samples.children.length, 1);
  const done = nodes.samples.children[0].events.click();
  const request = pending.shift();
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
  const { nodes, pending } = await setup();
  const first = nodes.samples.children[0].events.click();
  const second = nodes.samples.children[1].events.click();
  pending[1].respond(plot("new"));
  await second;
  pending[0].respond(plot("old"));
  await first;
  assert.equal(nodes.foraging.src, "data:image/png;base64,new");
  assert.match(nodes.selection.textContent, /テスト/);
});

test("changing conditions discards pending images and stale condition lists", async () => {
  const { nodes, pending, settle } = await setup();
  const image = nodes.samples.children[0].events.click();
  nodes.group.value = "b";
  const older = nodes.group.events.change();
  nodes.group.value = "a";
  const newer = nodes.group.events.change();
  pending[2].respond({ folds: [0] });
  await settle();
  pending[3].respond({ samples: [] });
  await newer;
  pending[1].respond({ folds: [9] });
  await older;
  pending[0].respond(plot("old"));
  await image;
  assert.equal(nodes.plots.hidden, true);
  assert.equal(nodes.foraging.src, undefined);
  assert.equal(nodes.fold.value, "0");
  assert.equal(pending.length, 4);
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
  const { nodes, pending } = await setup();
  const done = nodes.samples.children[0].events.click();
  pending.shift().respond({ error: { message: "データが見つかりません。" } }, false);
  await done;
  assert.equal(nodes.error.hidden, false);
  assert.equal(nodes.error.textContent, "データが見つかりません。");
  assert.equal(nodes.plots.hidden, true);
});

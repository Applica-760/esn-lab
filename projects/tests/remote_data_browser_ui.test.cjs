const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");

class Element {
  constructor() { this.children = []; this.events = {}; this.attributes = {}; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener(name, callback) { this.events[name] = callback; }
  setAttribute(name, value) { this.attributes[name] = value; }
  focus() { this.focused = true; }
}

const frontend = join(__dirname, "../app/remote_data_browser");
const settle = () => new Promise(resolve => setImmediate(resolve));
const listing = {
  path: "outputs", parent: ".", entries: [
    { name: "experiments", path: "outputs/experiments", type: "directory" },
    { name: "<img src=x onerror=alert(1)>", path: "outputs/file", type: "file" },
    { name: "link", path: "outputs/link", type: "symlink" },
  ],
};

async function setup(failConfig = false) {
  const nodes = Object.fromEntries([...readFileSync(join(frontend, "index.html"), "utf8")
    .matchAll(/id="([^"]+)"/g)].map(match => [match[1], new Element()]));
  const pending = [];
  const context = vm.createContext({
    URLSearchParams,
    document: { getElementById: id => nodes[id], createElement: () => new Element() },
    fetch: (url, options) => new Promise((resolve, reject) => pending.push({
      url, options, reject,
      respond(body, ok = true) { resolve({ ok, status: ok ? 200 : 409, json: async () => body }); },
    })),
  });
  vm.runInContext(readFileSync(join(frontend, "app.js"), "utf8"), context);
  if (failConfig) {
    pending.shift().reject(new Error("接続できません"));
    await settle();
    return { nodes, pending };
  }
  pending.shift().respond({ initial_path: "outputs", remote_root: "/remote", local_destination_absolute: "/local/downloads" });
  await settle();
  pending.shift().respond(listing);
  await settle();
  return { nodes, pending };
}

function choose(nodes) { nodes.entries.children[1].children[0].children[0].events.change(); }

async function preview(nodes, pending, conflicts = []) {
  choose(nodes);
  const done = nodes.enter.events.click();
  assert.equal(nodes.status.attributes["data-busy"], "true");
  assert.match(nodes.status.textContent, /確認しています/);
  const request = pending.shift();
  assert.equal(request.url, "/api/download/preview");
  assert.deepEqual(JSON.parse(request.options.body), { path: "outputs/experiments" });
  request.respond({ path: "outputs/experiments", destination: "/local/downloads/experiments", token: "token", conflicts, skipped: ["experiments/link"], file_count: 2 });
  await done;
  assert.equal(nodes.status.attributes["data-busy"], "false");
}

test("selection and navigation never transfer; Enter opens a separate confirmation", async () => {
  const { nodes, pending } = await setup();
  assert.equal(nodes.parent.disabled, true);
  assert.equal(nodes.enter.disabled, true);
  choose(nodes);
  assert.equal(nodes.selected.textContent, "outputs/experiments");
  assert.equal(pending.length, 0);
  assert.equal(nodes.entries.children[2].children[0].textContent, "<img src=x onerror=alert(1)>（ファイル）");
  assert.equal(nodes.entries.children[3].children.length, 1);
  await preview(nodes, pending);
  assert.equal(pending.length, 0);
  assert.equal(nodes.confirmation.hidden, false);
  assert.equal(nodes.browser.disabled, true);
  assert.equal(nodes.confirm.textContent, "取得する");
  assert.equal(nodes.destination.textContent, "/local/downloads/experiments");
  assert.equal(nodes["skipped-section"].hidden, false);
  nodes.cancel.events.click();
  assert.equal(pending.length, 0);
  assert.equal(nodes.confirmation.hidden, true);
  assert.equal(nodes.browser.disabled, false);
  await nodes.confirm.events.click();
  assert.equal(pending.length, 0);
});

test("opening a directory clears selection and root cannot be escaped", async () => {
  const { nodes, pending } = await setup();
  await nodes.parent.events.click();
  assert.equal(pending.length, 0);
  choose(nodes);
  const done = nodes.entries.children[1].children[1].events.click();
  assert.equal(nodes.status.attributes["data-busy"], "true");
  assert.match(nodes.entries.children[0].textContent, /読み込み中/);
  assert.equal(nodes.enter.disabled, true);
  assert.match(pending[0].url, /path=outputs%2Fexperiments/);
  pending.shift().respond({ path: "outputs/experiments", parent: "outputs", entries: [] });
  await done;
  assert.equal(nodes.status.attributes["data-busy"], "false");
  assert.equal(nodes.parent.disabled, false);
  assert.equal(nodes.selected.textContent, "未選択");
  assert.equal(nodes.entries.children.length, 1);
  nodes.entries.children[0].children[0].children[0].events.change();
  assert.equal(nodes.selected.textContent, "outputs/experiments");
});

for (const conflicts of [[], ["experiments/result"]]) {
  test(`only explicit confirmation starts transfer: conflicts=${conflicts.length}`, async () => {
    const { nodes, pending } = await setup();
    await preview(nodes, pending, conflicts);
    assert.equal(nodes["collision-warning"].hidden, !conflicts.length);
    assert.equal(nodes.confirm.textContent, conflicts.length ? "上書きする" : "取得する");
    if (conflicts.length) assert.equal(nodes.conflicts.children[0].textContent, conflicts[0]);
    const done = nodes.confirm.events.click();
    assert.equal(nodes.status.attributes["data-busy"], "true");
    assert.equal(nodes.confirm.disabled, true);
    assert.equal(nodes.cancel.disabled, true);
    assert.equal(nodes.browser.disabled, true);
    await nodes.confirm.events.click();
    nodes.cancel.events.click();
    await nodes.refresh.events.click();
    assert.equal(pending.length, 1);
    const request = pending.shift();
    assert.equal(request.url, "/api/download");
    assert.deepEqual(JSON.parse(request.options.body), { token: "token", overwrite: conflicts.length > 0 });
    request.respond({ status: "completed", destination: "/local/downloads/experiments" });
    await done;
    assert.equal(nodes.status.attributes["data-busy"], "false");
    assert.match(nodes.status.textContent, /取得が完了/);
    assert.equal(nodes.confirmation.hidden, true);
    assert.equal(nodes.browser.disabled, false);
  });
}

test("failed preview never transfers and failed transfer requires another preview", async () => {
  const { nodes, pending } = await setup();
  choose(nodes);
  let done = nodes.enter.events.click();
  pending.shift().respond({ error: { message: "読み取り失敗" } }, false);
  await done;
  assert.equal(nodes.error.textContent, "読み取り失敗");
  assert.equal(nodes.enter.disabled, false);
  await nodes.confirm.events.click();
  assert.equal(pending.length, 0);
  await preview(nodes, pending);
  done = nodes.confirm.events.click();
  pending.shift().reject(new Error("通信切断"));
  await done;
  assert.equal(nodes.error.textContent, "通信切断");
  assert.equal(nodes.status.attributes["data-busy"], "false");
  assert.match(nodes.status.textContent, /転送が続いている可能性/);
  assert.equal(nodes.confirmation.hidden, true);
  await nodes.confirm.events.click();
  assert.equal(pending.length, 0);
  await preview(nodes, pending);
  assert.equal(nodes.confirmation.hidden, false);
});

test("configuration failure can be retried", async () => {
  const { nodes, pending } = await setup(true);
  assert.equal(nodes.error.textContent, "接続できません");
  assert.equal(nodes.status.attributes["data-busy"], "false");
  assert.equal(nodes.entries.children.length, 0);
  assert.equal(nodes.browser.disabled, false);
  const done = nodes.refresh.events.click();
  assert.equal(pending[0].url, "/api/config");
  pending.shift().respond({ initial_path: "outputs", remote_root: "/remote", local_destination_absolute: "/local" });
  await settle();
  pending.shift().respond(listing);
  await done;
  assert.equal(nodes.error.hidden, true);
});

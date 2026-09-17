// Stream-termination contract of the shared chat-completions runtime, driven
// through the openai-compat driver. MiniMax's api.minimax.io/v1 closes the
// connection after the finish_reason chunk without ever sending `data: [DONE]`,
// and reports account failures as HTTP 200 with a JSON `base_resp` body.
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { RuntimeEvent } from "../contracts.ts";
import { MinimaxDriver } from "./minimax.ts";
import { OpenAICompatDriver } from "./openai-compat.ts";

afterEach(() => vi.unstubAllGlobals());

async function runTurn(body: string, driver: "openai-compat" | "minimax" = "openai-compat") {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } })));
  const instance = driver === "minimax"
    ? await MinimaxDriver.create({
        instanceId: "minimax", displayName: "MiniMax", enabled: true,
        config: MinimaxDriver.defaultConfig(),
        environment: { MINIMAX_API_KEY: "secret" },
      })
    : await OpenAICompatDriver.create({
        instanceId: "minimax", displayName: "MiniMax", enabled: true,
        config: OpenAICompatDriver.decodeConfig({ url: "https://api.minimax.io/v1", apiKeyEnv: "MINIMAX_API_KEY", model: "MiniMax-M3" }),
        environment: { MINIMAX_API_KEY: "secret" },
      });
  const events: RuntimeEvent[] = [];
  instance.adapter.onEvent((event) => events.push(event));
  await instance.adapter.sendTurn({ threadId: "thread", text: "hi" });
  await vi.waitFor(() => {
    if (!events.some((event) => event.type === "turn.completed")) throw new Error("turn still running");
  });
  await instance.dispose();
  return events;
}

describe("createOpenAIChatRuntime tool approvals", () => {
  // A bot on an OpenAI-compatible engine used to stop for a card on EVERY
  // tool call, with no way out: this family has no provider reviewer, so
  // Auto behaves like Ask; the card offers no session-wide allow; and the
  // "Always allowed" list only ever fills from peer-comms grants. Full
  // access is the person's explicit grant to answer every prompt, and with
  // no provider to hand it to, the runtime has to honour it itself.
  const mcpDir: string[] = [];
  afterEach(() => { for (const d of mcpDir.splice(0)) rmSync(d, { recursive: true, force: true }); });

  const toolServer = () => {
    const dir = mkdtempSync(join(tmpdir(), "omb-chat-approval-"));
    mcpDir.push(dir);
    const script = join(dir, "fake-mcp.mjs");
    writeFileSync(script, `#!/usr/bin/env node
      const send = (m) => process.stdout.write(JSON.stringify(m) + "\\n");
      let buffer = "";
      process.stdin.setEncoding("utf8");
      process.stdin.on("data", (chunk) => {
        buffer += chunk;
        let nl;
        while ((nl = buffer.indexOf("\\n")) !== -1) {
          const line = buffer.slice(0, nl); buffer = buffer.slice(nl + 1);
          const m = JSON.parse(line);
          if (m.method === "initialize") send({jsonrpc:"2.0",id:m.id,result:{protocolVersion:"2024-11-05",capabilities:{tools:{}}}});
          else if (m.method === "tools/list") send({jsonrpc:"2.0",id:m.id,result:{tools:[{name:"write",description:"Fixture write",inputSchema:{type:"object",properties:{},additionalProperties:false}}]}});
          else if (m.method === "tools/call") send({jsonrpc:"2.0",id:m.id,result:{content:[{type:"text",text:"done"}]}});
        }
      });
    `);
    chmodSync(script, 0o755);
    return { command: script, args: [], env: {} };
  };

  const cardRaisedFor = async (approvalMode: "ask" | "full") => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(
      'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"fx_write","arguments":"{}"}}]}}]}\n\n'
        + 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
      { status: 200, headers: { "content-type": "text/event-stream" } },
    )));
    const instance = await OpenAICompatDriver.create({
      instanceId: "compat", displayName: "Compat", enabled: true,
      config: OpenAICompatDriver.decodeConfig({ url: "https://api.example.com/v1", apiKeyEnv: "K", model: "m" }),
      environment: { K: "secret" },
    });
    const events: RuntimeEvent[] = [];
    instance.adapter.onEvent((event) => events.push(event));
    await instance.adapter.sendTurn({
      threadId: "thread", text: "hi", approvalMode,
      integrations: { custom: { fx: toolServer() } },
    });
    await vi.waitFor(() => {
      if (!events.some((e) => e.type === "request.opened" || e.type === "item.started" || e.type === "turn.completed")) {
        throw new Error("waiting");
      }
    }, { timeout: 10_000 });
    const opened = events.some((event) => event.type === "request.opened");
    await instance.adapter.interruptTurn("thread").catch(() => {});
    await instance.dispose();
    return opened;
  };

  it("holds a card for every tool call when the bot is on Ask", async () => {
    expect(await cardRaisedFor("ask")).toBe(true);
  }, 20_000);

  it("answers for the person under Full access, so no card is raised", async () => {
    expect(await cardRaisedFor("full")).toBe(false);
  }, 20_000);
});

describe("createOpenAIChatRuntime stream termination", () => {
  it("treats EOF after a finish_reason chunk as a clean completion when [DONE] never arrives", async () => {
    const events = await runTurn(
      'data: {"choices":[{"index":0,"delta":{"content":"<think>\\nuser said hi\\n</think>Hello!"}}]}\n\n' +
        'data: {"choices":[{"index":0,"finish_reason":"stop","delta":{"content":" How can I help?"}}],"usage":{"prompt_tokens":5,"completion_tokens":9}}\n\n',
    );
    expect(events.find((event) => event.type === "item.completed")).toMatchObject({
      text: "<think>\nuser said hi\n</think>Hello! How can I help?",
    });
    expect(events.find((event) => event.type === "runtime.error")).toBeUndefined();
    expect(events.at(-1)).toMatchObject({ type: "turn.completed", ok: true, usage: { input: 5, output: 9 } });
  });

  it("still honors data: [DONE]", async () => {
    const events = await runTurn('data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n');
    expect(events.find((event) => event.type === "item.completed")).toMatchObject({ text: "hi" });
    expect(events.at(-1)).toMatchObject({ type: "turn.completed", ok: true });
  });

  it("surfaces an HTTP 200 whose body is a MiniMax base_resp error instead of finishing silently", async () => {
    const events = await runTurn('{"base_resp":{"status_code":1008,"status_msg":"insufficient balance"}}');
    expect(events.find((event) => event.type === "runtime.error")).toMatchObject({
      message: "provider returned a completion error: upstream error 1008: insufficient balance",
    });
    expect(events.at(-1)).toMatchObject({ type: "turn.completed", ok: false, stopReason: "error" });
  });

  it("surfaces an OpenAI-style error object returned with HTTP 200", async () => {
    const events = await runTurn('{"error":{"message":"token is unusable (1004)","type":"authorized_error"}}');
    expect(events.find((event) => event.type === "runtime.error")).toMatchObject({
      message: "provider returned a completion error: token is unusable (1004)",
    });
  });

  it("reports a stream truncated before finish_reason as an error, not an interrupt", async () => {
    const events = await runTurn('data: {"choices":[{"index":0,"delta":{"content":"Hel"}}]}\n\n');
    expect(events.find((event) => event.type === "runtime.error")).toMatchObject({
      message: "Stream ended before completion",
    });
    expect(events.at(-1)).toMatchObject({ type: "turn.completed", ok: false, stopReason: "error" });
  });

  it("lets the MiniMax driver finish a [DONE]-less reasoning_split stream and stream its reasoning", async () => {
    const events = await runTurn(
      'data: {"choices":[{"index":0,"delta":{"reasoning_content":"user said hi"}}]}\n\n' +
        'data: {"choices":[{"index":0,"finish_reason":"stop","delta":{"content":"Hello!"}}]}\n\n' +
        'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":9}}\n\n',
      "minimax",
    );
    expect(events.filter((event) => event.type === "content.delta").map((event) => event.streamKind))
      .toEqual(["reasoning_text", "assistant_text"]);
    expect(events.find((event) => event.type === "item.completed")).toMatchObject({ text: "Hello!" });
    expect(events.at(-1)).toMatchObject({ type: "turn.completed", ok: true, usage: { input: 5, output: 9 } });
  });
});

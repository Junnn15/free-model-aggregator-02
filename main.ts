// free-model-aggregator-02: one server, two client wires, and a real free OpenRouter model
// behind every answer.
//
// This entry adapts the free-model-aggregator-01 build (quality-ranked routing, failover,
// per-provider cooldowns) to the -02 contest: a LIVE variant. There is no simulated provider
// here — every completion is dialled through the grader's gateway on the trial's key, which
// the gateway swaps for the entrant's own pushed OpenRouter key. The fourteen free models in
// the brief are the only ones this server may name, each as `openrouter/<vendor>/<model>:free`.
//
// HOW A PROMPT IS SERVED
//   1. The brief's manifest gives the prompt's requirements by arrival order (JSON prompts
//      need a JSON-capable model; the tools prompt needs a tools-capable one). When the brief
//      has no row for a request, the requirements are read off the body itself.
//   2. Candidates are ranked by the brief's rating_milli, best first, with the fewest gateway
//      calls so far breaking ties; a model that refused twice in a row is cooling down and is
//      never picked again this trial. That puts the best model on each prompt:
//        - JSON prompts (p01, p04): nemotron-3.5-super-vl (1303), then gemma-4-31b (1210),
//          then any other 1150-rated JSON-capable model;
//        - everything else: inkling-2-small (1475), then nemotron-3.5-super-vl (1303),
//          then nemotron-3-ultra (1285), then nemotron-3.5-lightning (1221), and so on.
//   3. Each candidate is tried, then retried once on a 429 or 5xx, before the next candidate
//      is tried. Every refusal is recorded in the `x-makefun-upstream-failures` trail (at most
//      8 rows), so the failover column can see a refusal this server recovered from.
//   4. The gateway's answer is returned in the wire the client used, streamed or not as asked,
//      with its usage and stop reason whole — and with `model` set to the id actually dialled,
//      on every answer and every streamed chunk, because the house scores the model its own
//      record shows answered.
//
// SAFETY
//   - Every gateway call names an output cap (max_tokens), because the gateway refuses a call
//     with none. A body that arrives without one gets a 1024-token cap.
//   - A trial-wide call budget of 48 gateway calls sits well under the gateway's 60-call cap,
//     so a failure storm cannot burn the budget on one prompt.
//   - Everything is a pure function of the brief, the requests and the gateway's answers: no
//     wall clock, no unseeded randomness, so a replay reproduces the same export byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  freshState,
  isRecord,
  meetsRequirements,
  rankedCandidates,
  readBrief,
  type Brief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type ModelState,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  chatAnswerText,
  hasToolsDeclared,
  isJsonMode,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  withOutputCap,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** One gateway call's deadline: well inside the client's 30 s per-answer wait. */
const GATEWAY_TIMEOUT_MS = 20_000;
/** Output cap given to a gateway body that arrives without one. */
const FALLBACK_MAX_TOKENS = 1024;
/** Attempts per model before failing over to the next candidate. */
const ATTEMPTS_PER_MODEL = 2;
/** Refusals in a row after which a model cools down for the rest of the trial. */
const COOLDOWN_AFTER_STREAK = 2;
/** Trial-wide gateway call budget: under the gateway's 60-call cap, with margin. */
const TRIAL_CALL_BUDGET = 48;
/** The upstream-failures trail's row cap, per the rules. */
const MAX_TRAIL_ROWS = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);

/** Which prompt row the request now arriving is: the brief's prompts are in arrival order. */
let manifestIndex = 0;
/** Per-model routing state, keyed by the gateway id. */
const modelStates = new Map<string, ModelState>();
/** Gateway calls made this trial, across all prompts. */
let trialCalls = 0;

/** Prompts served and prompts lost this trial, for the end-of-trial summary. */
let promptsServed = 0;
let promptsLost = 0;

/**
 * The candidates for one prompt: ranked by rating, fewest calls first, cooling models out.
 * When every model that could serve is cooling down (a trial-wide outage), the cooling flags
 * are cleared once and the ranking is retried: a transient refusal — a brief gateway blip, a
 * per-minute rate limit — may have cleared, and a prompt served late still scores. The reset
 * is a pure function of the refusals seen so far, so a replay reproduces it exactly.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  let candidates = rankedCandidates(brief, requires, modelStates);
  if (candidates.length > 0) return candidates;
  let anyCooling = false;
  for (const state of modelStates.values()) if (state.cooling) anyCooling = true;
  if (!anyCooling) return candidates;
  log('all candidates cooling: clearing cooldowns once and re-ranking');
  for (const state of modelStates.values()) state.cooling = false;
  return rankedCandidates(brief, requires, modelStates);
}

/** One row of the `x-makefun-upstream-failures` trail. */
interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

function log(message: string): void {
  process.stderr.write(`aggregator-02: ${message}\n`);
}

/** The routing state of one model, created on first use. */
function stateFor(modelId: string): ModelState {
  let state = modelStates.get(modelId);
  if (state === undefined) {
    state = freshState();
    modelStates.set(modelId, state);
  }
  return state;
}

/** Record one gateway attempt: a success clears the refusal streak, a refusal grows it. */
function noteAttempt(modelId: string, ok: boolean): void {
  const state = stateFor(modelId);
  state.calls += 1;
  trialCalls += 1;
  if (ok) {
    state.streak = 0;
    return;
  }
  state.streak += 1;
  if (state.streak >= COOLDOWN_AFTER_STREAK) state.cooling = true;
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1`
 * on the trial network); a base that does not already end in `/v1` gets one, so both spellings
 * reach the route.
 */
function gatewayChatUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The brief's prompt list is the declared source of each request's requirements; this
  // fallback is a guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  return {
    tools: hasToolsDeclared(body),
    vision,
    json: isJsonMode(body),
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, trail: UpstreamFailure[] = []): void {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
  };
  const text = JSON.stringify(body);
  headers['content-length'] = String(Buffer.byteLength(text));
  if (trail.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(trail.slice(0, MAX_TRAIL_ROWS));
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The refusal code of a gateway error body: its `code` or `type`, else the HTTP status. */
function refusalCode(status: number, text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string' && value.error.code !== '') return value.error.code;
      if (typeof value.error.type === 'string' && value.error.type !== '') return value.error.type;
    }
  } catch {
    // Not JSON: the status is the code.
  }
  return `http_${status}`;
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  const capped = withOutputCap({ ...body, model }, FALLBACK_MAX_TOKENS);
  return fetch(gatewayChatUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify(capped),
    signal: AbortSignal.timeout(GATEWAY_TIMEOUT_MS),
  });
}

/**
 * Serve one client request against the ranked candidates, in order, until one answers.
 * Each candidate is tried, then retried once on a 429 or 5xx, before the next candidate is
 * tried; every refusal is recorded in the trail the response carries. Returns true when an
 * answer was sent. A refused gateway call is never the answer: only the first success is.
 */
async function servePrompt(
  clientWire: 'openai' | 'anthropic',
  body: Json,
  res: ServerResponse,
  candidates: ModelRow[],
): Promise<boolean> {
  const stream = body.stream === true;
  const trail: UpstreamFailure[] = [];
  // Routing is worth logging: the live trial's stderr shows which model each prompt got and
  // in what order the fallbacks were tried, so a bad rank or a dead model is visible at once.
  log(`candidates for this prompt: ${candidates.map((c) => c.id).join(', ')}`);
  for (const candidate of candidates) {
    if (trialCalls >= TRIAL_CALL_BUDGET) {
      log(`trial call budget (${TRIAL_CALL_BUDGET}) reached: no more gateway calls`);
      break;
    }
    for (let attempt = 1; attempt <= ATTEMPTS_PER_MODEL; attempt += 1) {
      if (trialCalls >= TRIAL_CALL_BUDGET) break;
      const gatewayBody = clientWire === 'anthropic' ? anthropicToOpenAI(body, candidate.id) : body;
      let upstream: Response;
      try {
        upstream = await dial(candidate.id, gatewayBody);
      } catch (err) {
        log(`${candidate.id} unreachable (attempt ${attempt}): ${(err as Error).message}`);
        noteAttempt(candidate.id, false);
        trail.push({ status: 0, code: 'unreachable', model: candidate.id });
        continue;
      }
      if (!upstream.ok || upstream.body === null) {
        const text = await upstream.text().catch(() => '');
        log(`${candidate.id} refused HTTP ${upstream.status} (attempt ${attempt})`);
        noteAttempt(candidate.id, false);
        trail.push({ status: upstream.status, code: refusalCode(upstream.status, text), model: candidate.id });
        continue;
      }
      noteAttempt(candidate.id, true);
      const streamHeaders: Record<string, string> = {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
      };
      if (trail.length > 0) streamHeaders['x-makefun-upstream-failures'] = JSON.stringify(trail.slice(0, MAX_TRAIL_ROWS));
      if (stream) {
        res.writeHead(200, streamHeaders);
        // Count the streamed bytes so the trial log shows what a streamed prompt got, the
        // way the JSON path logs its answer length below.
        let streamedChars = 0;
        try {
          if (clientWire === 'openai') {
            for await (const frame of openAIStreamWithModel(upstream.body, candidate.id)) {
              streamedChars += frame.length;
              res.write(frame);
            }
          } else {
            for await (const event of anthropicEventsFromOpenAI(upstream.body, candidate.id)) {
              streamedChars += event.length;
              res.write(event);
            }
          }
        } finally {
          res.end();
        }
        promptsServed += 1;
        log(`served (stream) by ${candidate.id} (${streamedChars} chars)`);
        return true;
      }
      const answer: unknown = await upstream.json().catch(() => null);
      if (!isRecord(answer)) {
        // A 2xx with no JSON body is not a usable answer: treat it as a failure and move on.
        log(`${candidate.id} answered 2xx with no JSON body`);
        noteAttempt(candidate.id, false);
        trail.push({ status: 200, code: 'empty_answer', model: candidate.id });
        continue;
      }
      const text = chatAnswerText(answer);
      if (text !== '') log(`served by ${candidate.id} (${text.length} chars)`);
      promptsServed += 1;
      if (clientWire === 'openai') sendJson(res, 200, openAIJsonWithModel(answer, candidate.id), trail);
      else sendJson(res, 200, openAIToAnthropic(answer, candidate.id), trail);
      return true;
    }
  }
  return false;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that fits. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0) {
    promptsLost += 1;
    return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
  }
  if (await servePrompt('openai', body, res, candidates)) return;
  promptsLost += 1;
  sendError(res, 'openai', 502, 'upstream_unreachable', 'every free model that could serve this request refused it');
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0) {
    promptsLost += 1;
    return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
  }
  if (await servePrompt('anthropic', body, res, candidates)) return;
  promptsLost += 1;
  sendError(res, 'anthropic', 502, 'api_error', 'every free model that could serve this request refused it');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function watchWork(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          // The trial's own summary, for the run log: how many prompts were served, how many
          // were lost, and how many gateway calls the trial spent (the tie-break counts them).
          log(`trial done: ${promptsServed} served, ${promptsLost} lost, ${trialCalls} gateway calls`);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (gatewayChatUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  watchWork(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

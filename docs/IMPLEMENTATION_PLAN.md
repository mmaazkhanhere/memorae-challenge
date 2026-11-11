# Memorae — Vertical Slice Implementation Plan

**Project:** Two-week, solo, voice-first reminder and memory agent
**Channels required:** Telegram (text, voice notes, reminder delivery) and a live phone voice agent
**Selected stack:** Python/FastAPI, Telegram Bot API, Vapi, Groq (Whisper and structured interpretation), SQLAlchemy/Alembic, PostgreSQL (local Docker; hosted Neon), PostgreSQL-backed scheduler worker, Railway.
**Principle:** Build and demonstrate user-visible business outcomes from entry point to persistence and response. Grow shared components when first used, not as a separate architecture-building phase.

**Mandatory architecture invariant:** Every slice and channel must preserve the deterministic-first, LLM-fallback pipeline in [Architecture.md](Architecture.md#92-deterministic-first-natural-language-processing). A slice plan and its code review must show: authenticated canonical-user and pending-context handling; deterministic routing and extraction before the existing LLM fallbacks; deterministic validation and authorization before shared-service execution; transactional/idempotent mutation; and clarification or the defined scope-limitation response without mutation when the request cannot safely proceed. Changes to this invariant require explicit project-owner approval before implementation.

## 1. Scope and success

The assessor must be able to: (1) onboard independently in about one minute; (2) create one-time and recurring reminders through plain text or voice note; (3) call a real phone number and make requests, including interruption; (4) receive timely, durable, nonduplicated Telegram notifications and mark them done or snooze them; (5) list, move and cancel reminders through text and speech; and (6) save a fact on one channel and recall it on another.

The essential live demonstration is **Telegram onboarding → verified phone association → real inbound call → reminder tool call → durable schedule → spoken exact-local-time confirmation → Telegram delivery**. Build this by the end of the first implementation milestone involving voice, rather than finishing all Telegram enhancements before calling.

### Scope boundaries

- Telegram is the reminder delivery destination; phone calls are an input channel. WhatsApp and Google Meet are not in the committed scope.
- A user without a linked verified phone may still use Telegram; voice calls require a verified phone association.
- Ambiguous task/time, multiple reminder matches, and uncertain intent **ask rather than mutate**. For “at 7” at 15:00, ask whether 19:00 today was intended (or explicitly offer it); never silently schedule 07:00 tomorrow.
- A reminder's recurrence remains defined in the user's IANA time zone, while concrete due timestamps are stored in UTC.
- A reminder generated from a call always goes to the same user's linked Telegram destination; never trust a user ID proposed by an LLM/tool argument.
- Same input event cannot create the same logical state change twice; Telegram/provider message delivery across network timeouts has inherent uncertainty (details under Reliability).
- Keep secrets and personal data out of logs; do not expose saved facts to an unidentified caller. The challenge's demo passwords/codes should be treated as test data.

## 2. Milestone-first implementation order

| Order | Slice IDs | Demonstrable milestone | Target workdays |
|---|---|---|---|
| 1 | VS-01, VS-02 | Telegram bot onboarded; caller phone verified/linked | 1 |
| 2 | VS-03, VS-04 | Telegram one-time reminder created, persisted, delivered | 2–3 |
| 3 | VS-16, VS-17 | **Real incoming call schedules a reminder, confirmed aloud and delivered on Telegram** | 4 |
| 4 | VS-05, VS-06, VS-07 | Ambiguity safely clarified; unsupported requests rejected; recurrence works | 5–6 |
| 5 | VS-08, VS-09, VS-19 | List, move, and conversationally correct a reminder during an interrupted call | 7–8 |
| 6 | VS-10, VS-11, VS-12 | Cancel, complete, and snooze; no stale sends | 8–9 |
| 7 | VS-13, VS-14, VS-18 | Save a fact on the phone and retrieve it on Telegram | 9–10 |
| 8 | VS-15, VS-20 | Telegram voice notes and existing reminder-management tools during calls | 10–11 |
| 9 | All | Reliability, external smoke tests, deploy, README, 3-minute video, live handoff | 12 (and continuous) |

**Risk spike on day 1 (not a product slice):** provision/test the actual inbound phone number and one Vapi `assistant-request` plus a trivial function-tool callback. Confirm provider access early. This is an integration check, not an excuse to start a large separate Vapi platform build. Keep the challenge's alternative-provider fallback in mind if number approval blocks progress.

**Scheduling notes:** The workday labels indicate a priority budget rather than guaranteed durations. Parallelize no major feature with the reliability gate; work on reliability in each slice. Sundays are not planned workdays.

## 3. How to implement each vertical slice

The existing IDs are retained for traceability to the system-design draft, but the *execution sequence* is the table above. `Depends on` means an actual technical prerequisite, not simply an earlier document section.

### Required architecture check for every slice

Before marking a slice complete, verify the deterministic-first invariant: simple inputs do not call an LLM; uncertain routing reaches the existing classifier when configured; extraction is only attempted after a supported route is resolved; all proposed parameters are deterministically verified and grounded; unresolved/ambiguous/unsupported input cannot mutate state; pending replies retain their original operation context; and each channel uses the same validated application services. Record the relevant unit or integration tests in the slice's review or pull request.

### VS-01 — Onboard a Telegram user

**Outcome:** `/start` produces an active account with a stable Telegram identity, name, time zone, and delivery chat.
**Depends on:** Nothing.
**Implementation:** Create a minimal FastAPI `/webhook/telegram` entry point; verify the webhook secret; extract Telegram sender/chat IDs; create or get `users` and `channel_identities` in PostgreSQL using SQLAlchemy/Alembic; ask for/validate IANA time zone; take display name from profile or ask; respond through Telegram. Bring up app, DB and migrations via reproducible local configuration.
**Done when:**
- [ ] A first `/start` yields one usable account and asks for missing time zone.
- [ ] A second `/start` reuses that account and never duplicates it.
- [ ] Invalid time zone is rejected, and DB failure cannot produce a false success response.
- [ ] Another Telegram sender never sees this user's account.
- [ ] A new user can finish Telegram onboarding without linking a phone.

### VS-02 — Link and verify caller phone

**Outcome:** A Telegram user can enable live phone access by linking their own number.
**Depends on:** VS-01.
**Implementation:** Offer Telegram's `request_contact` keyboard in private chat; require `message.contact.user_id == message.from.id` (reject missing/mismatched ownership); normalize to E.164; link as a PHONE channel identity under the same internal user; add DB uniqueness protection for an active normalized number; preserve Telegram functionality if omitted.
**Done when:**
- [ ] Genuine contact is linked and stored normalized.
- [ ] Another person's contact, missing contact owner ID, or invalid phone is rejected.
- [ ] Two users cannot link the same active phone, even concurrently.
- [ ] Skipping the step still permits Telegram usage.

### VS-03 — Create a one-time reminder from Telegram text

**Outcome:** A natural-language Telegram message creates exactly one durable one-time reminder and confirms its precise time and destination.
**Depends on:** VS-01.
**Implementation:** Telegram webhook → intent extraction as typed structured data (Groq LLM with conservative validation and rule-based relative-time handling) → user/time-zone/ownership checks → transaction inserting `reminders`, one `reminder_occurrence`, pending `notification`, and an idempotency result keyed to the Telegram update → local-time Telegram confirmation after commit. Use UTC instants plus IANA time-zone context. The LLM proposes commands; the service alone authorizes and writes.
**Done when:**
- [ ] “In five minutes” and “tomorrow at 9” persist correct user, task, UTC due time and local display.
- [ ] Confirmation says exact local time and “Telegram”.
- [ ] Replaying a Telegram update creates no duplicate.
- [ ] Past/invalid schedules and failed database transactions do not create a reminder or trigger false success.
- [ ] One user's reminder cannot be created under another user's identity.

### VS-04 — Deliver a due reminder via Telegram

**Outcome:** Due, eligible occurrences trigger Telegram notifications on time despite normal worker restarts.
**Depends on:** VS-03.
**Implementation:** Run an independent PostgreSQL-backed worker that polls indexed `notifications` due timestamps (e.g., every 5–10 seconds), atomically claims work with `FOR UPDATE SKIP LOCKED` and a claim lease, rechecks parent/occurrence eligibility, sends through Telegram and records attempt, status, timestamp and provider message ID. Use a documented policy for API errors, timeouts and unknown results. Test real Postgres concurrency.
**Done when:**
- [ ] A real five-minute test notification arrives no more than 60s after due under healthy conditions.
- [ ] Restarting API/worker before due does not lose the notification.
- [ ] Competing worker processes do not concurrently send the same claimed notification.
- [ ] Ineligible/canceled/completed occurrences cannot be sent.
- [ ] Provider success, permanent failure and uncertain timeout each leave an intelligible delivery status.

### VS-05 — Clarify ambiguous commands

**Outcome:** Unclear requests receive a focused question and are completed only after sufficient information exists.
**Depends on:** VS-03.
**Implementation:** Let intent interpreter return typed `needs_clarification` rather than guessing; save pending action/partial parameters in `conversation_sessions`; on the next user message, merge the response, revalidate and execute the original action exactly once; expire stale pending sessions.
**Done when:**
- [ ] “Remind me at 7” at 15:00 is clarified (not silently mapped to next morning).
- [ ] “Remind me Friday to renew the domain” requests/proposes an hour and waits for confirmation.
- [ ] The follow-up creates one reminder only; unrelated replies do not cause a write.
- [ ] Unclear task or multiple targets causes no mutation until resolved.

### VS-06 — Reject unsupported or unauthorized requests

**Outcome:** The bot stays inside its reminder/memory scope and never treats model-generated arguments as authority.
**Depends on:** VS-01 and a first command service (VS-03).
**Implementation:** Enumerate supported intents; validate typed parameters, ownership, current state and delivery identity inside application services; return one standard out-of-scope response; ignore prompt-injection attempts in user text; authenticate webhook/tool endpoints.
**Done when:**
- [ ] Unknown but potentially supported intent asks for clarification.
- [ ] Out-of-scope request gets the predefined limitation response and no DB mutation.
- [ ] Malicious text cannot override identity, business rules or authorized operation names.
- [ ] Cross-user read/write attempts fail in service tests.

### VS-07 — Recurring reminders

**Outcome:** “Every Monday at 8” creates a series whose next occurrences respect local wall-clock time.
**Depends on:** VS-03, VS-04.
**Implementation:** Store a recurrence rule plus intended IANA time zone on the parent; materialize the next occurrence(s) idempotently using a stable recurrence-instance key; each due occurrence goes through the existing notification worker; advance future occurrences without duplicating after restarts. Define DST gap/overlap policy explicitly.
**Done when:**
- [ ] Weekly recurrence generates one logical occurrence for each intended local date.
- [ ] Worker restarts/repeated generation never duplicate a recurrence instance.
- [ ] Local 08:00 remains 08:00 across DST changes where the local time exists.
- [ ] Failed delivery does not incorrectly cancel the recurring series.

### VS-08 — List reminders in a requested date range

**Outcome:** “What do I have tomorrow?” gives the correct local-time reminders.
**Depends on:** VS-03; VS-07 for correct future recurring projection.
**Implementation:** Interpret requested local-day or range boundaries using the user's zone; query owned occurrences (and project unmaterialized future recurrence instances if necessary); filter active/pending items and format a short response without modifying state.
**Done when:**
- [ ] Date-range results include the correct one-time and recurring items at local times.
- [ ] Empty results are acknowledged without invention.
- [ ] Completed/canceled items are excluded by default.
- [ ] Another user's records cannot appear; read leaves state unchanged.

### VS-09 — Reschedule an existing reminder

**Outcome:** “Move the bank one to 10” changes the chosen reminder rather than creating another.
**Depends on:** VS-03, VS-05.
**Implementation:** Resolve candidate within authorized user scope; ask which one if multiple; determine occurrence-versus-series target; atomically update current due schedule, invalidate obsolete pending notification, preserve audit of sent attempts and increment version; respond with new local time.
**Done when:**
- [ ] One requested target is updated in place; no extra logical reminder exists.
- [ ] Old pending due time cannot fire; new due time can.
- [ ] Two matching reminders trigger a question; no speculative update.
- [ ] Database conflict/failure leaves the original state intact.

### VS-10 — Cancel a reminder or series

**Outcome:** A user can stop a selected reminder or all future instances of a recurring series.
**Depends on:** VS-03; VS-07 for series cancellation.
**Implementation:** Resolve target with scope; explicitly distinguish occurrence from whole series; transactionally mark appropriate future occurrences/notifications canceled, and mark series canceled when applicable.
**Done when:**
- [ ] “Cancel the standup” cancels the series and no future occurrence is sent.
- [ ] Other reminders and prior delivery history remain intact.
- [ ] Ambiguous matches or repeated cancellation do not cause unintended changes.

### VS-11 — Mark a delivered occurrence done

**Outcome:** “Done” closes the correct occurrence.
**Depends on:** VS-04.
**Implementation:** Associate inbound Telegram replies with the delivered `provider_message_id` or an inline action containing a validated occurrence reference; use last-delivered context only when unambiguous; change occurrence to completed, not a recurring parent.
**Done when:**
- [ ] “Done” against a known notification completes the right occurrence.
- [ ] Repeated “Done” is safe and does not generate another event.
- [ ] Completing a recurring instance leaves the series active.
- [ ] Unclear reference asks which item and never completes a random one.

### VS-12 — Snooze a delivered occurrence

**Outcome:** “10 more minutes” postpones only the selected instance.
**Depends on:** VS-04, VS-11 (shared occurrence resolution).
**Implementation:** Resolve the delivered occurrence; validate positive duration; keep previously SENT notification as immutable history; transactionally schedule a new pending notification or next due attempt for the *same* logical occurrence without altering recurrence rule.
**Done when:**
- [ ] Snoozing schedules one new due notification at the requested instant.
- [ ] Prior sent record persists, no second logical occurrence is created.
- [ ] Weekly series future timing remains unchanged.
- [ ] Replayed snooze command does not postpone twice.

### VS-13 — Save a user-provided memory fact

**Outcome:** “Remember …” stores a fact without creating any reminder.
**Depends on:** VS-01; the command-routing path from VS-03 is reusable.
**Implementation:** Detect explicit memory intent deterministically first; for clearly intentional indirect requests, use the existing classifier and extractor fallback with typed content, conservative grounding, and no ordinary-statement permission. Validate the fact, then insert into `memory_facts` owned by the canonical user ID through the existing transaction and idempotent response path. Preserve authoritative content and add nullable pgvector embedding plus model, hash, status, attempt, and timestamp metadata. Generate an embedding only after authorization; provider failure leaves the durable fact pending for resumable indexing. Sensitive content is not sent to an external embedding provider by default.
**Done when:**
- [ ] Explicit fact creates exactly one saved fact under its owner.
- [ ] No reminder, occurrence or notification row is created.
- [ ] Replayed request does not duplicate the fact.
- [ ] Save failure yields a truthful failure response.
- [ ] Embedding failure preserves the saved fact without claiming it was indexed.
- [ ] Replayed saves and unchanged-content regeneration do not duplicate embedding work.

### VS-14 — Retrieve a saved memory fact

**Outcome:** “What's my gate code?” returns a known fact or explicitly says it is unknown.
**Depends on:** VS-13.
**Implementation:** Identify a deterministic lookup subject first and run a bounded owner-scoped lexical query. When lexical evidence is insufficient, embed the query and run bounded exact cosine search in PostgreSQL, constrained by canonical owner and active model. Configurable relevance and ambiguity thresholds return unknown or the existing typed clarification instead of guessing. The typed extractor may propose only a grounded subject query; it cannot supply an answer, fact ID, user ID, or authorization. Format answers only from authoritative `MemoryFact.content`. Provider/index failure falls back to lexical behavior. Direct read-only replies remain outside `idempotency_records`.
**Done when:**
- [ ] Correct fact is retrieved and quoted accurately.
- [ ] Missing/ambiguous fact is not hallucinated; clarify when necessary.
- [ ] Another user's fact cannot be retrieved.
- [ ] Memory read does not modify stored state.
- [ ] Replayed direct reads may repeat a response but never repeat a mutation or disclose another user's fact.
- [ ] Semantic paraphrases retrieve only above calibrated relevance and ambiguity thresholds.
- [ ] Existing unembedded facts and sensitive facts remain available through deterministic retrieval.

### VS-15 — Telegram voice notes

**Outcome:** A spoken Telegram note executes the same commands as a typed Telegram message, with text feedback.
**Depends on:** VS-03, VS-04; relevant Phase 1 operations for full parity.
**Implementation:** Download voice file through Telegram API; validate media type/size; transcribe with Groq Whisper; send resulting text to the existing intent router and user-scoped services; reply by Telegram text; ask for clarification when no hour is spoken.
**Done when:**
- [ ] Spoken one-time reminder is created, persisted, delivered through Telegram.
- [ ] Spoken list/move/cancel/done/snooze/memory operations use the existing service implementations.
- [ ] An inaudible/failed transcription yields a helpful text retry request, not a write.
- [ ] Confirmation and clarification are text messages.

### VS-16 — Authenticate an incoming call

**Outcome:** A real phone number accepts a registered caller and rejects an unknown one before connecting them to the AI assistant.
**Depends on:** VS-02.
**Implementation:** Provision Vapi inbound number; configure a dynamic `assistant-request` on its server URL; authenticate incoming webhook according to provider controls; normalize the caller number and look up an active linked phone identity; return allowed assistant configuration only after matching; create call-to-user authorization context server-side. Avoid attaching a fixed assistant if it bypasses the pre-call decision. Keep the response within Vapi's fixed ~7.5-second limit. Phone CLI matching is useful for challenge eligibility but not strong proof against caller-ID spoofing for sensitive data.
**Done when:**
- [ ] A registered linked caller reaches the agent via an actual phone number.
- [ ] Unknown/missing number is rejected before the AI conversation.
- [ ] Any identity lookup failure fails closed.
- [ ] Caller-supplied tool parameters cannot select another user's account.

### VS-17 — Create a reminder on a live call

**Outcome:** A registered caller schedules a reminder, hears exact spoken confirmation, then receives the notification on Telegram.
**Depends on:** VS-16, VS-03, VS-04.
**Implementation:** Configure Vapi assistant and server function tool for create-reminder; map authenticated call context to canonical user ID; validate typed tool arguments in the same `CreateReminder` service used by Telegram; idempotently handle repeated `tool-calls` using a call/tool invocation key; return exact localized scheduled time and delivery channel in the tool result; speak success **only** from committed result.
**Done when:**
- [ ] A live caller requests a five-minute reminder and hears time + “on Telegram”.
- [ ] Telegram notification reaches their linked account at the scheduled time.
- [ ] Two genuinely distinct requests during one call yield two reminders.
- [ ] Tool retry does not create two reminders; failure does not produce false success.

### VS-18 — Save and recall facts across channels

**Outcome:** Save a fact during a call; retrieve it from Telegram using the same canonical user account.
**Depends on:** VS-13, VS-14, VS-16, VS-17's working call/tool infrastructure.
**Implementation:** Expose save/search-memory as Vapi server tools backed by the same services/tables; associate the verified call context with the user; use Telegram interpreter for retrieval; restrict sensitive disclosures when caller identity is not sufficiently verified.
**Done when:**
- [ ] “Remember the Wi-Fi password is Andorra2026” on a call saves one memory, not a reminder.
- [ ] “What's my Wi-Fi password?” on Telegram returns `Andorra2026`.
- [ ] Another user has no access to the fact.
- [ ] Changes appear immediately across channels.

### VS-19 — Interrupt and correct during a call

**Outcome:** A spoken correction edits the reminder just referenced, even after interrupting TTS.
**Depends on:** VS-17, VS-09.
**Implementation:** Configure and manually test Vapi barge-in/interruption behavior; persist an active reminder reference keyed to call/session; when a correction follows a creation, route it to the existing reschedule operation rather than create; revalidate and cancel superseded pending notification. Clear/expire references when the topic changes.
**Done when:**
- [ ] The caller can interrupt the agent's response.
- [ ] “Actually, make it half an hour” changes the same reminder.
- [ ] Exactly one logical reminder remains with its revised time; old due notification cannot fire.
- [ ] Ambiguous correction asks which reminder to change.
- [ ] Call context survives successive utterances within that call.

### VS-20 — Full reminder management by live voice

**Outcome:** Voice caller can list, reschedule, cancel, complete and snooze existing reminders.
**Depends on:** VS-08–VS-12, VS-17.
**Implementation:** Register Vapi tools for existing read/mutation application services; provide minimal typed tool schemas; pass all calls through authorized user context and consistent idempotency; return concise spoken confirmations.
**Done when:**
- [ ] “What do I have tomorrow?” lists the correct items aloud.
- [ ] “Move the bank one to 10” and “Cancel the standup” operate on existing state.
- [ ] Done/snooze refer to the right occurrence.
- [ ] Telegram immediately shows the changes made during the call.

## 4. Architecture rule for all slices

```text
Telegram text  ─┐
Telegram voice ─┼──> Channel adapter / authenticated user context
Vapi tool call ─┘                │
                                ▼
                    Typed intent / command validation
                                │
                                ▼
                      Shared application services
                      create, list, move, cancel,
                      complete, snooze, save, recall
                                │
                                ▼
                       PostgreSQL + transactions
                                │
                    (for a scheduled notification)
                                ▼
                       PostgreSQL worker
                                │
                                ▼
                          Telegram API
```

- No channel owns business rules or a separate reminder database.
- Trusted webhook identity/context determines user; LLM-supplied IDs are advisory at most.
- Shared services validate ownership, command parameters and legal state changes independently of prompts.
- Add a small provider interface where it earns its place (STT, LLM, voice integration), not a generic orchestration framework.
- Mutation idempotency uses a unique `(source, idempotency_key)` in PostgreSQL, with recorded result; reads do not require idempotency keys.
- Avoid building unauthenticated `/api/v1/*` CRUD routes merely because they appear in the design; webhooks can invoke application services directly. If exposed, authenticate/authorize them.

## 5. Reliability design and honest guarantee

**Input idempotency:** Deduplicate Telegram update IDs and Vapi call/tool invocation IDs in the same transaction as reminder changes. Distinguish two genuine user commands from one retried request.

**Persistence:** For a one-time reminder, transactionally store the parent reminder, occurrence, pending notification and idempotency response. Never send success before commit. PostgreSQL remains the source of truth across app/worker restarts.

**Time:** Store due timestamps in UTC, IANA zone and intended local recurrence semantics; interpret relative expressions using a single captured reference time. Query date windows using user's local-day bounds. Recurrence needs a declared DST overlap/nonexistence policy.

**Dispatch:** Worker polls due notifications frequently, claims rows transactionally (e.g. `FOR UPDATE SKIP LOCKED` plus lease), rechecks eligibility, sends and logs provider result. Run at least one integration test with two workers and real PostgreSQL. Monitor late due sends and nonterminal claims.

**Exactly-once caveat:** Telegram Bot API `sendMessage` does not offer a general user-supplied idempotency key. If Telegram accepted a send but the HTTP response was lost, the worker cannot always tell whether retrying duplicates it. Database locks alone cannot guarantee simultaneously *never lost* and *never duplicated* external messages. Aim for no duplicates from normal retries/concurrency and document an explicit `UNKNOWN` handling policy for ambiguous network outcomes. State this limitation in the README and discuss alternatives during review rather than claiming mathematically strict end-to-end exactly-once delivery.

**Basic safety and privacy:** Verify webhook credentials, rate-limit where practical, avoid logging message/audio/fact contents, use minimum retention, treat caller number matching as lower-assurance identity, and don't disclose sensitive facts on a call based solely on caller ID.

### Minimal schema fixes before implementing the worker

The source design is close, but make three changes **when the first relevant slice needs them**:

- Set `reminders.status` to `NOT NULL DEFAULT 'ACTIVE'` rather than the current draft's `default=NULL`.
- Give notification claims explicit lease/recovery fields (for example `claimed_until`, `claimed_by`) or an equivalent transactional recovery strategy. `SENDING` alone is not enough to safely recover a crashed worker.
- Store a complete request digest for idempotency (for example SHA-256 `CHAR(64)` or 32-byte binary), not a truncated `CHAR(20)` unless you intentionally design a shorter collision-resistant scheme. Add a stable recurrence-instance identity so changing a due timestamp does not accidentally create duplicate occurrences.

## 6. Repeatable definition of done (every slice)

1. Works through its real user/external entry point; no fake success response.
2. Domain ownership, authorization, invalid-input and state-transition rules are enforced.
3. Any needed migrations and transactional persistence are included.
4. Tests cover happy path, an edge/failure case, a replay/duplicate case for mutations, and cross-user isolation where relevant.
5. External APIs are mocked in automated tests, but at least one genuine provider smoke test exists for each integration.
6. The slice can be demonstrated independently, and its implementation and environment changes are documented.

**Testing layers:** Unit tests for schedule parsing, time-zone conversions and recurrence; FastAPI/service integration tests against real PostgreSQL; worker concurrency/restart tests; contract tests for Telegram/Vapi payloads; manual timed end-to-end scripts involving an actual bot and a real incoming call. Fix reliability regressions before continuing to more features.

## 7. Final rehearsal checklist

- [ ] An assessor can open the bot and complete free onboarding quickly.
- [ ] Their own contact links to the same account; an unknown caller is rejected.
- [ ] They call the provided number, speak one request, and hear the exact local time and Telegram channel.
- [ ] The message arrives within 60 seconds of scheduled time (in healthy conditions).
- [ ] A second reminder in the same call is stored independently.
- [ ] “Every Monday at 8” schedules recurring reminders in the correct zone.
- [ ] “At 7” at 15:00 is sensibly clarified or offered as 19:00, not silently moved to the next morning.
- [ ] “Actually, make it half an hour” interrupts and updates, not duplicates.
- [ ] “Remember [test code]” on voice is recalled on Telegram and is not a reminder.
- [ ] List, move and cancel work on both text and call.
- [ ] Done and “10 more minutes” affect the correct occurrence.
- [ ] A voice note goes through STT and receives a Telegram text answer; missing hour triggers clarification.
- [ ] API/worker restart does not lose scheduled reminders or create duplicate attempts under normal delivery conditions.
- [ ] Deliverables: working Telegram bot link and callable number, private GitHub repo, `docker compose up` or `make run`, `.env.example`, architecture + voice pipeline + scheduler guarantee/limitations + real vs mocked + cost per user in README, and 3-minute demo video.

## 8. If time becomes tight

**Do not drop:** Live phone → real Telegram reminder, no false confirmation, restart persistence, recurring time zones, interruption/correction, done/snooze, voice notes, cross-channel memory, and the requested handoff artifacts—these are explicit challenge must-haves.

**Defer first:** WhatsApp, Google Meet, outbound calling, multilingual handling, a full RAG framework or separate vector database, approximate vector indexing before measurements justify it, elaborate admin API/dashboard, Celery/Redis, advanced analytics, and broad-scale microservice architecture. An optional 50-phrase parsing evaluation is a worthwhile stretch only after the six must-haves work.

---

**Implementation guidance:** Keep each ticket intentionally small: a trigger, a trusted user context, a validated command, one service outcome, persisted state if appropriate, a response, and acceptance tests. Decide exact class/function boundaries during coding when concrete needs appear.

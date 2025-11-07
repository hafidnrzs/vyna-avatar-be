# Task - Perform the RPC and document testing steps

## Summary

Implement an RPC-capable function in the agent that uses helper utilities in `src/illustration_helpers.py` to build an illustration/component payload and send it to the frontend via LiveKit RPC. Then provide a short, clear doc that frontend developers can use to test the RPC end-to-end.

## Why this task

- Enables the agent to create and display UI components/illustrations in the frontend by sending structured RPC messages.
- Provides reproducible test instructions so FE devs can verify the integration without guessing prompt wording or payload schema.

## What to read first

- `src/illustration_helpers.py` — inspect available helpers, their arguments, and the JSON/dict payload schema they build.
- `src/agent.py` — integrate the helper into a `@function_tool` function and ensure correct imports and typings.

## Concrete implementation steps

1. Inspect `src/illustration_helpers.py` and identify the helper function(s) used to create payloads (note exact function name, parameters, and returned structure).

2. Add a `@function_tool` function to `src/agent.py` (example name: `create_illustration`) with a signature similar to:

   async def create_illustration(self, context: RunContext[UserData], content: str, style: Optional[str] = None)

   Implementation details:

   - Call the helper from `illustration_helpers.py` to produce a JSON-serializable payload (a dict).
   - Ensure `context.userdata` is present and update `UserData` if you need to track created items.
   - Find the target remote participant in the room (use the first remote participant as done elsewhere in the repo).
   - Serialize the payload to JSON and call:

await room.local_participant.perform_rpc(
destination_identity=participant.identity,
method="client.showIllustration",
payload=json_payload,
)

- Return a human-readable success message (or a descriptive error string on failure).

3. Add or update imports in `src/agent.py` as required (e.g., `import json`, and the exact helper import from `illustration_helpers.py`).

4. If the helper requires pre-downloaded assets or external model files, ensure `download-files` covers them.

5. Create `docs/RPC_TESTING.md` describing how to trigger and verify the RPC (minimum content described below).

## Acceptance criteria

- The new function exists in `src/agent.py` and imports `illustration_helpers` correctly.
- The agent starts in dev/console mode without import/type errors.
- Triggering the function (via an appropriate user prompt or direct invocation) sends an RPC using `perform_rpc` and does not raise an exception.
  -- `docs/RPC_TESTING.md` is present and contains clear prompts, payload schema, and verification steps.

## Example user prompts (what a user should send to trigger the function)

- "Create an illustration: show a red ball with the caption 'Test card' and display it to me."
- "Please create a component that shows the text 'Hello from agent' and show it to the user."

## Expected RPC method and payload shape (FE expectations)

-- RPC method: `client.showIllustration`
-- Example payload (JSON):

```json
{
  "state": "show",
  "image_url": "https://example.com/image.png"
}
```

## FE dev notes (how to verify)

- Ensure the FE registers an RPC handler for the method used by the agent (e.g., `client.showIllustration`).
- When the FE receives the RPC, it should parse the payload (expecting `state` and optional `image_url`) and render/update the illustration UI by calling `updateIllustration(state, image_url)`.
- Log the payload in the FE console to confirm keys and values match the expected schema.

## Commands to run locally (PowerShell)

1. Download models/files (once):

```powershell
uv run src/agent.py download-files
```

2. (Optional) Start token server if used by your FE:

```powershell
uv run src/server.py
```

3. Start agent:

```powershell
uv run src/agent.py dev
```

5. In the frontend (or via a connected client that receives RPCs), send one of the example prompts above to the agent. The agent should respond by calling the function and performing the RPC to the FE.

## Docs file to create: `docs/RPC_TESTING.md` (minimum sections)

- Purpose and quick summary
- Exact example prompt(s) to send to the agent
- RPC method name the agent will call
- JSON payload schema (keys + types)
- How to start agent & token server (commands above)
- How to confirm receipt in the FE (console logs + expected UI change)

## Optional follow-ups

- Add unit tests that mock the room/participant and assert `perform_rpc` is called with expected payload.
- Add a small e2e harness (agent + mocked FE client) that asserts end-to-end behavior.

If you want, I can implement the code changes and create `docs/RPC_TESTING.md` now — tell me to proceed and I will apply the edits and attempt a local run to validate.

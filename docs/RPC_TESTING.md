# RPC Testing Guide - Illustration Display

## Purpose

This guide provides step-by-step instructions for testing the LiveKit RPC integration that allows the AI agent to display and control illustrations in the frontend. The agent can show images/diagrams to users during conversations and hide them when needed.

## What This Does

The agent has two function tools available:

- `show_illustration` - Displays an image to the user with an optional description
- `hide_illustration` - Hides the currently displayed image

These tools use LiveKit RPC to communicate with the frontend via the `client.showIllustration` method.

## RPC Method Details

**Method Name:** `client.showIllustration`

**Payload Schema (JSON):**

To show an illustration:

```json
{
  "state": "show",
  "image_url": "https://example.com/image.png"
}
```

To hide an illustration:

```json
{
  "state": "hidden"
}
```

**Response Schema:**

```json
{
  "ok": true
}
```

Or on error:

```json
{
  "ok": false,
  "error": "Error message here"
}
```

## Prerequisites

1. Environment variables configured in `.env.local`:

   - `LIVEKIT_URL`
   - `LIVEKIT_API_KEY`
   - `LIVEKIT_API_SECRET`
   - `OPENAI_API_KEY`
   - `ELEVEN_API_KEY`

2. Dependencies installed:

```powershell
uv sync
```

3. Models downloaded (first time only):

```powershell
uv run src/agent.py download-files
```

## Running the Backend

### Step 1: Start the Token Server

In one terminal:

```powershell
cd d:\aegislabs\poc-vyna-avatar\vyna-avatar-be
uv run src/server.py
```

The token server should start on `http://localhost:10001`

### Step 2: Start the Agent

In a separate terminal:

```powershell
cd d:\aegislabs\poc-vyna-avatar\vyna-avatar-be
uv run src/agent.py dev
```

The agent will wait for a client to connect.

## Testing from the Frontend

### Frontend Requirements

The frontend must:

1. Register the RPC handler `client.showIllustration` (already implemented in `IllustrationRpcHandler.tsx`)
2. Have the `useIllustration` hook available for state management
3. Call `updateIllustration(state, image_url)` when receiving RPC messages

### Example Prompts to Test

Once the frontend is connected to the agent, send these voice/text prompts:

#### Test 1: Show an Illustration

**Prompt:**

```
"Can you show me an illustration of a red ball?"
```

or

```
"Show me a diagram of the Pythagorean theorem"
```

or

```
"Display an image of a triangle to help me understand"
```

**Expected Behavior:**

- The agent should call the `show_illustration` function tool
- An RPC message is sent to the frontend with method `client.showIllustration`
- The frontend receives the payload and displays the image
- The agent responds with something like "I've displayed the illustration to you."

#### Test 2: Hide the Illustration

**Prompt:**

```
"Hide the illustration"
```

or

```
"Clear the image"
```

or

```
"Remove the diagram"
```

**Expected Behavior:**

- The agent should call the `hide_illustration` function tool
- An RPC message is sent with state "hidden"
- The frontend hides the currently displayed image
- The agent responds with "I've hidden the illustration."

#### Test 3: Show with Specific URL (Advanced)

If you want to test with a specific image URL, you can ask:

```
"Show me this image: https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Pythagorean_theorem_abc.svg/300px-Pythagorean_theorem_abc.svg.png"
```

## Verification Steps

### Backend Console Logs

You should see logs like:

```
[Illustration] Show result: {'ok': True}
INFO - Successfully showed illustration: https://example.com/image.png
```

or

```
[Illustration] Hide result: {'ok': True}
INFO - Successfully hid illustration
```

### Frontend Console Logs

In the browser console, you should see:

```
[RPC] Registering client.showIllustration method
[RPC] Received client.showIllustration from: <agent_identity>
[RPC] Illustration updated: { state: 'show', imageUrl: 'https://...' }
```

### Visual Confirmation

- When showing: The illustration component should become visible and display the image
- When hiding: The illustration component should become invisible or clear

## Quick Fix Checklist (If You See Errors)

If you encounter errors like "overflow when subtracting durations", "ConnectionResetError", or "DuplexClosed":

1. ✅ **Start frontend FIRST, agent SECOND**

   - Start token server: `uv run src/server.py`
   - Start frontend and wait for it to fully load
   - Check browser console for: `[RPC] Registering client.showIllustration method`
   - Only then start the agent: `uv run src/agent.py dev`

2. ✅ **Wait for connection**

   - After starting the agent, wait 2-3 seconds for the room to stabilize
   - Look for "remote participant joined" in agent logs

3. ✅ **Use clear prompts**

   - Say: "Show me a picture of a triangle"
   - Or: "Display an illustration of a circle"

4. ✅ **If error occurs, restart the agent**
   - Stop with Ctrl+C
   - Restart: `uv run src/agent.py dev`

## Troubleshooting

### Issue: "overflow when subtracting durations" / Rust panic / ConnectionResetError

**Symptoms:**

```
thread 'tokio-runtime-worker' panicked at /rustc/.../time.rs:1126:31:
overflow when subtracting durations
ConnectionResetError: [WinError 64] The specified network name is no longer available
```

**Cause:** This is a known issue on Windows when:

1. The frontend is not connected or the RPC handler is not registered
2. The RPC times out waiting for a response
3. Timeout values are too long causing duration overflow in the Rust runtime

**Solution:**

1. **Ensure frontend is connected FIRST** before sending prompts to the agent

   - Start the frontend and wait for it to fully connect to the LiveKit room
   - Look for "[RPC] Registering client.showIllustration method" in the browser console
   - Only then send prompts to trigger the illustration functions

2. **The timeout has been reduced to 2 seconds** (from the default 5 seconds) in the latest code

   - This should prevent the duration overflow issue
   - If you still see the error, the frontend likely isn't connected

3. **Verify frontend RPC registration:**

   ```javascript
   // In browser console, check:
   console.log("Room connected:", room.state);
   // Should show "connected"
   ```

4. **Restart the agent** after the error to recover:
   ```powershell
   # Stop the agent (Ctrl+C)
   # Then restart:
   uv run src/agent.py dev
   ```

### Issue: "Cannot show illustration: no participants found"

**Cause:** The frontend hasn't connected yet or disconnected.

**Solution:**

- Ensure the frontend is running and connected to the same LiveKit room
- Check that the token server is accessible from the frontend
- Verify the frontend obtained a valid connection token
- **Wait 2-3 seconds after the frontend connects before sending prompts**

### Issue: RPC timeout or no response

**Cause:** The frontend hasn't registered the RPC handler.

**Solution:**

- Check that `IllustrationRpcHandler` component is rendered in the frontend
- Verify the RPC method name matches exactly: `client.showIllustration`
- Check browser console for registration logs

### Issue: Agent doesn't call the function

**Cause:** The prompt might not be clear enough for the LLM.

**Solution:**

- Use more explicit prompts like "Show me an illustration of..."
- Make sure the agent instructions include guidance about using illustrations
- Check that the function tools are properly registered (no import errors in agent startup)

## Advanced Testing

### Testing with curl (Backend Only)

You can test the illustration helpers directly without the frontend:

```python
# In Python console or test script
from src.illustration_helpers import show_illustration, hide_illustration
from livekit import rtc

# Assuming you have a connected room and participant
result = await show_illustration(
    room=room,
    participant_identity="user_identity",
    image_url="https://example.com/test.png"
)
print(result)
```

### Integration Test

Create a test that:

1. Starts the agent
2. Connects a mock client that registers `client.showIllustration`
3. Sends a prompt to trigger the function
4. Asserts the RPC was called with correct payload

## Expected Function Call Flow

1. User sends prompt: "Show me a diagram"
2. Agent LLM decides to use `show_illustration` function tool
3. Function tool:
   - Validates room and participant availability
   - Calls `show_illustration` helper from `illustration_helpers.py`
   - Helper serializes payload: `{"state": "show", "image_url": "..."}`
   - Helper calls `room.local_participant.perform_rpc(method="client.showIllustration", ...)`
4. Frontend receives RPC:
   - `IllustrationRpcHandler` handles the message
   - Parses payload
   - Calls `updateIllustration(state, image_url)`
   - Returns `{"ok": true}`
5. Agent receives response and confirms to user

## Summary

- **RPC Method:** `client.showIllustration`
- **Payload Keys:** `state` ("show" | "hidden"), `image_url` (optional, required when showing)
- **Agent Functions:** `show_illustration(image_url, description)`, `hide_illustration()`
- **Frontend Handler:** `IllustrationRpcHandler.tsx` with `useIllustration` hook
- **Test Prompts:** "Show me an illustration...", "Hide the illustration"

For more details on the frontend implementation, see `vyna-avatar-fe/docs/ILLUSTRATION_RPC_TESTING.md`.

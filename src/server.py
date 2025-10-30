import os
import random
from datetime import timedelta
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from livekit import api
from pydantic import BaseModel

load_dotenv(".env.local")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
API_KEY = os.getenv("LIVEKIT_API_KEY")
API_SECRET = os.getenv("LIVEKIT_API_SECRET")

if LIVEKIT_URL is None:
    raise ValueError("LIVEKIT_URL is not defined")
if API_KEY is None:
    raise ValueError("LIVEKIT_API_KEY is not defined")
if API_SECRET is None:
    raise ValueError("LIVEKIT_API_SECRET is not defined")


class ConnectionDetails(BaseModel):
    serverUrl: str  # noqa: N815
    roomName: str  # noqa: N815
    participantName: str  # noqa: N815
    participantToken: str  # noqa: N815


def create_participant_token(
    identity: str,
    name: str,
    room_name: str,
    agent_name: Optional[str] = None,
) -> str:
    """Create a participant token for LiveKit."""
    at = (
        api.AccessToken(API_KEY, API_SECRET)
        .with_identity(identity)
        .with_name(name)
        .with_ttl(timedelta(minutes=15))
    )
    grants = api.VideoGrants(
        room=room_name,
        room_join=True,
        can_publish=True,
        can_publish_data=True,
        can_subscribe=True,
    )
    at.with_grants(grants)

    if agent_name is not None:
        room_config = api.RoomConfiguration(
            agents=[api.RoomAgentDispatch(agent_name=agent_name)]
        )
        at.with_room_config(room_config)

    return at.to_jwt()


@app.post("/api/connection-details")
async def connection_details(request: Request):
    """
    Return Livekit connection details to the client.
    """
    try:
        # Parse agent configuration from request body
        body = await request.json()
        agent_name: Optional[str] = (
            body.get("room_config", {}).get("agents", [{}])[0].get("agent_name")
            if body.get("room_config", {}).get("agents")
            else None
        )

        # Generate participant token
        participant_name = "user"
        participant_identity = f"voice_assistant_user_{random.randint(0, 10_000)}"
        room_name = f"voice_assistant_room_{random.randint(0, 10_000)}"

        participant_token = create_participant_token(
            identity=participant_identity,
            name=participant_name,
            room_name=room_name,
            agent_name=agent_name,
        )

        # Return connection details
        data = ConnectionDetails(
            serverUrl=LIVEKIT_URL,
            roomName=room_name,
            participantName=participant_name,
            participantToken=participant_token,
        )

        return JSONResponse(
            content=data.model_dump(),
            headers={"Cache-Control": "no-store"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/")
async def root():
    return {"message": "LiveKit token server is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

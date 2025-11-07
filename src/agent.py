import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    metrics,
)
from livekit.plugins import elevenlabs, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Available illustrations that can be displayed to users
AVAILABLE_ILLUSTRATIONS = {
    "pythagoras": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Pythagorean.svg/512px-Pythagorean.svg.png",
        "description": "Pythagorean theorem diagram showing a² + b² = c²",
        "topics": ["mathematics", "geometry", "pythagoras", "triangle", "theorem"],
    },
    "trigonometry": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Trigonometry_triangle.svg/800px-Trigonometry_triangle.svg.png",
        "description": "A trigonometry triangle is a right-angled triangle used as the fundamental scaffold for defining sine, cosine, and tangent",
        "topics": ["mathematics", "geometry", "trigonometry", "triangle"],
    },
    # Add more illustrations here as they become available
}


@dataclass
class UserInfo:
    """Class to represent a user information"""

    id: str
    name: str
    age: int | None


@dataclass
class Component:
    """Class to represent a FE component"""

    id: str
    content: str
    is_showed: bool = False


@dataclass
class UserData:
    """Class to store user data during a session"""

    ctx: Optional[JobContext] = None
    name: str = field(default_factory=str)
    age: int = field(default_factory=int)
    components: list[Component] = field(default_factory=list)

    def set_user_info(self, name: str, age: int) -> UserInfo:
        """Set user information"""
        user_info = UserInfo(id=str(uuid.uuid4()), name=name, age=age)
        self.name = name
        self.age = age
        return user_info

    def get_user_info(self) -> Optional[UserInfo]:
        """Get the user information (name and age)"""
        if self.name and (self.age is not None):
            return UserInfo(id=str(uuid.uuid4()), name=self.name, age=self.age)
        return None

    def add_component(self, content: str) -> Component:
        """Add a new component to the collection"""
        component = Component(id=str(uuid.uuid4()), content=content)
        self.components.append(component)
        return component

    def get_component(self, action_id: str) -> Optional[Component]:
        """Get a component by ID"""
        for component in self.components:
            if component.id == action_id:
                return component
        return None

    def toggle_component(self, action_id: str) -> Optional[Component]:
        """Toggle display of the component by ID"""
        component = self.get_component(action_id)
        if component:
            component.is_showed = not component.is_showed
            return component
        return None


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.

            Use the lookup_weather function if the user asked about the current weather.

            When user ask who they are, use the function get_user_data.
            And when user introduce their name and age, use the function set_user_data.

            When teaching or explaining visual concepts, use show_illustration to display relevant images or diagrams to help the user understand better.
            Available illustrations: "pythagoras" (for Pythagorean theorem, geometry, triangles).
            You can use hide_illustration when you want to clear the display or move to a different topic.
            """,
        )

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        """Use this tool to look up current weather information in the given location.

        If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.

        Args:
            location: The location to look up weather information for (e.g. city name)
        """

        logger.info(f"Looking up weather for {location}")

        return "sunny with a temperature of 70 degrees."

    @function_tool
    async def set_user_data(self, context: RunContext[UserData], name: str, age: int):
        """Store the user's name and age in this session

        Args:
            name: Name of the user
            age: Age of the user
        """
        userdata = context.userdata
        userdata.set_user_info(name, age)

        return (
            f"Okay, now I will remember your name is {name} and you are {age} year old."
        )

    @function_tool
    async def get_user_data(self, context: RunContext[UserData]):
        """Get the current session user name and age"""
        userdata = context.userdata
        user_info = userdata.get_user_info()

        if user_info:
            return f"Your name: {user_info.name} and your age: {user_info.age}"
        return "I don't know your name. Please introduce your name and your age"

    @function_tool
    async def create_component(self, context: RunContext[UserData], content: str):
        """Create a component that store text and display it to the user

        Args:
            content: The text that want to be displayed
        """
        userdata = context.userdata
        component = userdata.add_component(content)

        # Get the room from the userdata
        if not userdata.ctx or not userdata.ctx.room:
            return "Created a component, but couldn't access the room to send it"
        room = userdata.ctx.room

        # Get the first participant in the room (should be the client)
        participants = room.remote_participants
        if not participants:
            return "Created a component, but no participants found to send it to"

        # Get the first participant from the dictionary of remote participant
        participant = next(iter(participants.values()), None)
        if not participant:
            return "Created a component, but couldn't get the first participant"
        payload = {
            "action": "show",
            "id": component.id,
            "content": component.content,
            "index": len(userdata.components) - 1,
        }

        # Make sure payload is properly serialized
        json_payload = json.dumps(payload)
        logger.info(f"Sending component payload: {json_payload}")
        await room.local_participant.perform_rpc(
            destination_identity=participant.identity,
            method="client.component",
            payload=json_payload,
        )

        return f"I've created a component with the content: {content}"

    @function_tool
    async def toggle_component(self, context: RunContext[UserData], component_id: str):
        """Toggle display of the component (show/hide)

        Args:
            component_id: The ID of the component to be toggled
        """
        userdata = context.userdata
        component = userdata.toggle_component(component_id)

        if not component:
            return f"Component with ID {component_id} not found"

        # Get the room from the userdata
        if not userdata.ctx or not userdata.ctx.room:
            return "Toggled the component, but couldn't access the room to send it"
        room = userdata.ctx.room

        # Get the first participant in the room (should be the client)
        participants = room.remote_participants
        if not participants:
            return "Toggled the component, but no participants found to send it to"

        # Get the first participant from the dictionary of remote participants
        participant = next(iter(participants.values()), None)
        if not participant:
            return "Toggled the component, but couldn't get the first participant."
        payload = {"action": "toggle", "id": component.id}

        # Make sure payload is properly serialized
        json_payload = json.dumps(payload)
        logger.info(f"Send toggle component payload: {json_payload}")
        await room.local_participant.perform_rpc(
            destination_identity=participant.identity,
            method="client.component",
            payload=json_payload,
        )

        return f"I've toggled the component to {'show' if component.is_showed else 'hide'} the component"

    @function_tool
    async def show_illustration(
        self, context: RunContext[UserData], illustration_key: str
    ):
        """Show an illustration/image to the user. Use this when you want to display visual aids, diagrams, or educational images.

        Available illustrations:
        - "pythagoras": Pythagorean theorem diagram (a² + b² = c²) - use for geometry, triangles, mathematics
        - "trigonometry": Trigonometry - sin, cosine, and tangent

        Args:
            illustration_key: The key of the illustration to display (e.g., "pythagoras")
        """
        userdata = context.userdata

        # Validate illustration key
        if illustration_key not in AVAILABLE_ILLUSTRATIONS:
            available_keys = ", ".join(AVAILABLE_ILLUSTRATIONS.keys())
            return f"I don't have an illustration called '{illustration_key}'. Available illustrations are: {available_keys}"

        illustration = AVAILABLE_ILLUSTRATIONS[illustration_key]
        image_url = illustration["url"]
        description = illustration["description"]

        # Get the room from the userdata
        if not userdata.ctx or not userdata.ctx.room:
            return "Cannot show illustration: couldn't access the room"
        room = userdata.ctx.room

        # Get the first participant in the room (should be the client)
        participants = room.remote_participants
        if not participants:
            return "Cannot show illustration: no participants found in the room"

        # Get the first participant from the dictionary of remote participants
        participant = next(iter(participants.values()), None)
        if not participant:
            return "Cannot show illustration: couldn't get the first participant"

        # Prepare and send RPC to show the illustration
        try:
            payload = json.dumps({"state": "show", "image_url": image_url})
            logger.info(f"Sending show illustration payload: {payload}")

            # Wrap RPC call with asyncio timeout to catch errors before Rust panic
            result = await asyncio.wait_for(
                room.local_participant.perform_rpc(
                    destination_identity=participant.identity,
                    method="client.showIllustration",
                    payload=payload,
                ),
                timeout=4.0,  # Python timeout slightly longer than RPC timeout
            )

            response = json.loads(result)
            logger.info(f"[Illustration] Show result: {response}")

            if response.get("ok"):
                desc_msg = f" showing {description}" if description else ""
                return f"I've displayed the illustration{desc_msg} to you."
            else:
                error = response.get("error", "Unknown error")
                return f"I tried to show the illustration but encountered an error: {error}"

        except asyncio.TimeoutError:
            logger.error("Show illustration timed out - frontend may not be ready")
            return "The illustration request timed out. Please make sure the frontend is connected and try again."
        except Exception as e:
            logger.error(f"Failed to show illustration: {e!s}")
            return "I encountered an error while trying to show the illustration. The frontend may not be ready to receive it."

    @function_tool
    async def hide_illustration(self, context: RunContext[UserData]):
        """Hide the currently displayed illustration from the user. Use this when you want to clear the visual display.

        No arguments required.
        """
        userdata = context.userdata

        # Get the room from the userdata
        if not userdata.ctx or not userdata.ctx.room:
            return "Cannot hide illustration: couldn't access the room"
        room = userdata.ctx.room

        # Get the first participant in the room (should be the client)
        participants = room.remote_participants
        if not participants:
            return "Cannot hide illustration: no participants found in the room"

        # Get the first participant from the dictionary of remote participants
        participant = next(iter(participants.values()), None)
        if not participant:
            return "Cannot hide illustration: couldn't get the first participant"

        # Prepare and send RPC to hide the illustration
        try:
            payload = json.dumps({"state": "hidden"})
            logger.info(f"Sending hide illustration payload: {payload}")

            # Wrap RPC call with asyncio timeout to catch errors before Rust panic
            result = await asyncio.wait_for(
                room.local_participant.perform_rpc(
                    destination_identity=participant.identity,
                    method="client.showIllustration",
                    payload=payload,
                ),
                timeout=4.0,  # Python timeout slightly longer than RPC timeout
            )

            response = json.loads(result)
            logger.info(f"[Illustration] Hide result: {response}")

            if response.get("ok"):
                return "I've hidden the illustration."
            else:
                error = response.get("error", "Unknown error")
                return f"I tried to hide the illustration but encountered an error: {error}"

        except asyncio.TimeoutError:
            logger.error("Hide illustration timed out - frontend may not be ready")
            return "The hide illustration request timed out. Please make sure the frontend is connected."
        except Exception as e:
            logger.error(f"Failed to hide illustration: {e!s}")
            return "I encountered an error while trying to hide the illustration. The frontend may not be ready."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    userdata = UserData(ctx=ctx)
    session = AgentSession[UserData](
        userdata=userdata,
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=openai.STT(model="gpt-4o-mini-transcribe", detect_language=True),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=openai.LLM(model="gpt-4.1-mini", temperature=0.4),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=elevenlabs.TTS(
            model="eleven_multilingual_v2",
            voice_id="iWydkXKoiVtvdn4vLKp9",
            language="id",
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # Register RPC method for listening button from FE
    async def handle_toggle_component(rpc_data):
        try:
            logger.info(f"Received toggle component payload: {rpc_data}")

            # Extract the payload from the RpcInvocationData object
            payload_str = rpc_data.payload
            logger.info(f"Extracted quiz submission string: {payload_str}")

            # Parse the JSON payload
            payload_data = json.loads(payload_str)
            logger.info(f"Parsed clicked button data: {payload_data}")

            action_id = payload_data.get("id")

            if action_id:
                component = userdata.toggle_component(action_id)
                if component:
                    logger.info(
                        f"Toggled component {action_id}, is_showed: {component.is_showed}"
                    )
                    # Send a message to the user via the agent
                    session.generate_reply(
                        instructions=(
                            "Say to the user that they successfully toggle the component"
                        )
                    )
                else:
                    logger.error(f"Component with ID {action_id} not found")
            else:
                logger.error("No action ID found in payload")

            return "success"
        except Exception as e:
            logger.error(f"Error handling button click: {e}")
            return f"error: {str(e)}"

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the userW
    await ctx.connect()

    # Register RPC methods - The method names need to match exactly what the client is calling
    logger.info("Registering RPC methods")
    ctx.room.local_participant.register_rpc_method(
        "agent.toggleComponent", handle_toggle_component
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

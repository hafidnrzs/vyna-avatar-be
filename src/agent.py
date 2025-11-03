import logging
import uuid

from dataclasses import dataclass, field
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
)
from typing import Optional
from livekit.agents import function_tool, Agent, RunContext
from livekit.plugins import elevenlabs, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

@dataclass
class UserInfo:
    """Class to represent a user information"""
    id: str
    name: str
    age: int | None

@dataclass
class UserData:
    """Class to store user data during a session"""
    ctx: Optional[JobContext] = None
    name: str = field(default_factory=str)
    age: int = field(default_factory=int)

    def set_user_info(self, name: str, age: int) -> UserInfo:
        """Set user information"""
        user_info = UserInfo(
            id=str(uuid.uuid4()),
            name=name,
            age=age
        )
        self.name = name
        self.age = age
        return user_info

    def get_user_info(self) -> Optional[UserInfo]:
        """Get the user information (name and age)"""
        if self.name and (self.age is not None):
            return UserInfo(id=str(uuid.uuid4()), name=self.name, age=self.age)
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

        return f"Okay, now I will remember your name is {name} and you are {age} year old."
    
    @function_tool
    async def get_user_data(self, context: RunContext[UserData]):
        """Get the current session user name and age"""
        userdata = context.userdata
        user_info = userdata.get_user_info()

        if user_info:
            return f"Your name: {user_info.name} and your age: {user_info.age}"
        return "I don't know your name. Please introduce your name and your age"

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

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

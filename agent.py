import os
import logging
import google.cloud.logging
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

import google.auth
import google.auth.transport.requests
import google.oauth2.id_token
#-- New imports for Google Cloud Logging and environment variable management --
from google.adk.agents import Agent
from datetime import datetime
from typing import Dict, Any, List

# --- Setup Logging and Environment ---

cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()

model_name = os.getenv("MODEL")

# Greet user and save their prompt

def add_prompt_to_state(
    tool_context: ToolContext, prompt: str
) -> dict[str, str]:
    """Saves the user's initial prompt to the state."""
    tool_context.state["PROMPT"] = prompt
    logging.info(f"[State updated] Added to PROMPT: {prompt}")
    return {"status": "success"}

# Configuring the Wikipedia Tool
wikipedia_tool = LangchainTool(
    tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
)

# 1. Researcher Agent
comprehensive_researcher = Agent(
    name="comprehensive_researcher",
    model=model_name,
    description="The primary researcher that can access both internal idea data and external knowledge from Wikipedia.",
    instruction="""
    You are a helpful personal assistant. Your goal is to capture ideas from the user's PROMPT.
    You have access to two tools:
    1. A tool for storing data about ideas and returning specific information about them (e.g., details, references).
    2. A tool for searching Wikipedia for general knowledge (facts, references).

    First, analyze the user's PROMPT.
    - If the prompt can be answered by only one tool, use that tool.
    - If the prompt is complex and requires information from both the idea database AND Wikipedia,
        you MUST use both tools to gather all necessary information.
    - Synthesize the results from the tool(s) you use into preliminary data outputs.

    PROMPT:
    { PROMPT }
    """,
    tools=[
        wikipedia_tool
    ],
    output_key="research_data" # A key to store the combined findings
)

# 2. Response Formatter Agent
response_formatter = Agent(
    name="response_formatter",
    model=model_name,
    description="Synthesizes all information into a friendly, readable response.",
    instruction="""
    You are the friendly voice of the Idea User Assistant. Your task is to take the
    idea and store it in session and then present it to the user when asked for them.

    - First, present the specific information from the idea database.
    - Then, add the interesting general facts from the research.
    - If some information is missing, just present the information you have.
    - Be conversational and engaging.

    RESEARCH_DATA:
    { research_data }
    """
)

#------------------------
def save_user_idea(
    idea_type: str, 
    value: str, 
    tool_context: ToolContext
) -> Dict[str, Any]:
    """Save a user idea with timestamp.
    
    Args:
        idea_type: Type of idea (e.g., 'cuisine', 'music_genre')
        value: The idea value
        tool_context: Automatically injected by ADK
        
    Returns:
        dict: Operation status and details
    """
    # Store idea with user scope
    idea_key = f"user:idea_{idea_type}"
    timestamp_key = f"user:idea_{idea_type}_updated"
    
    # Save the idea and when it was set
    tool_context.state[idea_key] = value
    tool_context.state[timestamp_key] = datetime.now().isoformat()
    
    return {
        "status": "success",
        "message": f"Saved {idea_type} idea: {value}",
        "updated_at": tool_context.state[timestamp_key]
    }
def get_user_profile(tool_context: ToolContext) -> Dict[str, Any]:
    """Retrieve comprehensive user profile information.
    
    Args:
        tool_context: Automatically injected by ADK
        
    Returns:
        dict: User profile data including preferences and history
    """
    # Get user name
    user_name = tool_context.state.get("user:name", "Guest")
    
    # Collect all user ideas
    ideas = {}
    
    # Check for common idea types
    common_ideas = [
        "cuisine", "music_genre", "favorite_color", "language", "outdoor_activity",
        "timezone", "notification_preference", "theme", "accessibility", "reading_genre"
    ]
    
    for idea_type in common_ideas:
        idea_key = f"user:idea_{idea_type}"
        if idea_key in tool_context.state:
            ideas[idea_type] = tool_context.state[idea_key]
            # Get timestamp if available
            timestamp_key = f"{idea_key}_updated"
            if timestamp_key in tool_context.state:
                ideas[f"{idea_type}_updated"] = tool_context.state[timestamp_key]
    
    # Get conversation history
    last_interaction = tool_context.state.get("last_interaction", "none")
    total_interactions = tool_context.state.get("user:total_interactions", 0)
    
    return {
        "user_name": user_name,
        "ideas": ideas,
        "last_interaction": last_interaction,
        "total_interactions": total_interactions,
        "profile_retrieved_at": datetime.now().isoformat()
    }
def track_conversation_flow(
    flow_type: str,
    step: str,
    data: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """Track multi-step conversation flows for better context.
    
    Args:
        flow_type: Type of flow (e.g., 'booking', 'planning', 'troubleshooting')
        step: Current step in the flow
        data: Relevant data for this step
        tool_context: Automatically injected by ADK
        
    Returns:
        dict: Flow tracking status and current state
    """
    # Store flow information
    flow_key = f"user:flow_{flow_type}"
    step_key = f"user:flow_{flow_type}_step"
    data_key = f"user:flow_{flow_type}_data"
    timestamp_key = f"user:flow_{flow_type}_updated"
    
    tool_context.state[flow_key] = flow_type
    tool_context.state[step_key] = step
    tool_context.state[data_key] = data
    tool_context.state[timestamp_key] = datetime.now().isoformat()
    
    return {
        "status": "success",
        "message": f"Tracked {flow_type} flow: {step}",
        "current_step": step,
        "flow_data": data,
        "updated_at": tool_context.state[timestamp_key]
    }
def update_user_interaction(
    interaction_type: str,
    details: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """Update user interaction history and preferences.
    
    Args:
        interaction_type: Type of interaction (e.g., 'question', 'request', 'feedback')
        details: Details about the interaction
        tool_context: Automatically injected by ADK
        
    Returns:
        dict: Interaction update status
    """
    # Update interaction count
    current_count = tool_context.state.get("user:total_interactions", 0)
    tool_context.state["user:total_interactions"] = current_count + 1
    
    # Store interaction details
    interaction_key = f"user:interaction_{current_count + 1}"
    tool_context.state[interaction_key] = {
        "type": interaction_type,
        "details": details,
        "timestamp": datetime.now().isoformat()
    }
    
    # Update last interaction
    tool_context.state["last_interaction"] = f"{interaction_type}: {details}"
    
    return {
        "status": "success",
        "message": f"Updated interaction: {interaction_type}",
        "interaction_count": current_count + 1,
        "timestamp": datetime.now().isoformat()
    }
# Tools list
state_tools = [
    save_user_idea, 
    get_user_profile, 
    track_conversation_flow,
    update_user_interaction
]
# Personal assistant with state awareness
idea_storage_agent = Agent(
    name="idea_storage_agent",
    model=model_name,
    instruction="""
    You are a highly personalized assistant that remembers user preferences and context.
    
    STARTUP BEHAVIOR:
    - Always check user state at the beginning of each interaction
    - If user:name exists, greet them by name
    - If this is a returning user, reference relevant previous preferences
    - Check for any ongoing conversation flows and offer to continue them
    
    STATE USAGE GUIDELINES:
    - Use save_user_idea tool when users express ideas
    - Use get_user_profile tool to understand user background before making recommendations
    - Use track_conversation_flow for multi-step processes (booking, planning, troubleshooting)
    - Use update_user_interaction to track user engagement and build context
    
    PERSONALIZATION:
    - Tailor responses based on user:ideas
    - Reference previous interactions when relevant
    - Maintain consistency with established user relationships
    - Learn from user feedback and adjust recommendations accordingly
    
    MEMORY MANAGEMENT:
    - Store important decisions and outcomes
    - Remember user goals and aspirations
    - Track what works well for each user
    - Maintain conversation context across sessions
    
    CONVERSATION FLOW:
    - For new users, focus on learning their ideas
    - For returning users, reference their history and ideas
    - Proactively suggest improvements based on past interactions
    - Handle multi-step processes with clear progress tracking
    """,
    tools=state_tools,
    output_key="last_assistant_response"
)

user_assistant_workflow = SequentialAgent(
    name="user_assistant_workflow",
    description="The main workflow for handling a user's request about an idea.",
    sub_agents=[
        idea_storage_agent,       # Step 1: Store the idea in the state
        comprehensive_researcher, # Step 1: Gather all data
        response_formatter,       # Step 2: Format the final response
    ]
)

root_agent = Agent(
    name="greeter",
    model=model_name,
    description="The main entry point for the Idea capture Assistant.",
    instruction="""
    - Let the user know you will help them store their ideas.
    - When the user responds, use the 'add_prompt_to_state' tool to save their response.
    After using the tool, transfer control to the 'user_assistant_workflow' agent.
    """,
    tools=[add_prompt_to_state],
    sub_agents=[user_assistant_workflow]
)
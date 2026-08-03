#crew.py
import os
import yaml
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from agents.tools import extract_vitals

load_dotenv()

# Determine which LLM to use based on available credentials in .env
nvidia_key = os.getenv("NVIDIA_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

if nvidia_key:
    # Configure NVIDIA NIM LLMs with automatic fallbacks for maximum resilience
    from crewai import LLM
    
    primary_llm = LLM(
        model="meta/llama-3.1-8b-instruct",
        provider="openai",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=nvidia_key
    )
    fallback_llms = [
        LLM(model="meta/llama-3.3-70b-instruct", provider="openai", base_url="https://integrate.api.nvidia.com/v1", api_key=nvidia_key),
        LLM(model="meta/llama-3.1-70b-instruct", provider="openai", base_url="https://integrate.api.nvidia.com/v1", api_key=nvidia_key),
        LLM(model="qwen/qwen3.5-122b-a10b", provider="openai", base_url="https://integrate.api.nvidia.com/v1", api_key=nvidia_key),
        LLM(model="abacusai/dracarys-llama-3.1-70b-instruct", provider="openai", base_url="https://integrate.api.nvidia.com/v1", api_key=nvidia_key),
    ]
    
    primary_llm.fallback_llms = fallback_llms
    original_call = primary_llm.call

    def fallback_call(*args, **kwargs):
        try:
            return original_call(*args, **kwargs)
        except Exception as e:
            print(f"[FALLBACK] Primary NVIDIA model failed: {e}. Trying fallback models...")
            
        for fallback_llm in primary_llm.fallback_llms:
            try:
                return fallback_llm.call(*args, **kwargs)
            except Exception as fe:
                print(f"[FALLBACK] Fallback model {fallback_llm.model} failed: {fe}. Trying next fallback...")
                
        raise RuntimeError("All models in the NVIDIA NIM fallback chain failed.")

    primary_llm.call = fallback_call
    llm = primary_llm
    print("[CREW] Initialised NVIDIA NIM CrewAI Native LLMs (meta/llama-3.1-8b-instruct) with robust fallbacks.")
elif groq_key:
    # Fallback to Groq LLM
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model="llama3-8b-8192",
        groq_api_key=groq_key
    )
    print("[CREW] Initialised Groq Cloud API (llama3-8b-8192) as fallback agent brain.")
else:
    print("[CREW] WARNING: Neither NVIDIA_API_KEY nor GROQ_API_KEY found.")
    raise ValueError("Missing API credentials. Please add NVIDIA_API_KEY or GROQ_API_KEY in your .env file.")

# Load YAML configurations
base_dir = os.path.dirname(os.path.abspath(__file__))
agents_yaml_path = os.path.join(base_dir, 'config', 'agents.yaml')
tasks_yaml_path = os.path.join(base_dir, 'config', 'tasks.yaml')

with open(agents_yaml_path, 'r') as f:
    agents_config = yaml.safe_load(f)

with open(tasks_yaml_path, 'r') as f:
    tasks_config = yaml.safe_load(f)

def build_triage_crew() -> Crew:
    """
    Construct a brand-new Agent/Task/Crew set for a single triage run.

    Why a factory instead of module-level singletons
    ──────────────────────────────────────────────────
    Flask runs with threaded=True (web/__init__.py), so concurrent requests
    to /api/triage_run execute in separate threads. CrewAI's Task objects
    store their own output on `self` once executed and are not designed to
    be re-entered concurrently from multiple threads on the same instance —
    a single shared module-level `triage_crew.kickoff()` called from two
    threads at once can cross-contaminate two different patients' results.
    Building fresh Agent/Task/Crew objects per call makes concurrent triage
    runs safe at the cost of a small amount of object construction overhead
    (negligible next to LLM round-trip latency). The LLM client objects
    themselves stay module-level/shared — they're stateless HTTP wrappers.
    """
    perception_agent = Agent(
        config=agents_config['perception_agent'],
        tools=[extract_vitals],
        llm=llm,
        memory=False,
    )
    diagnostic_agent = Agent(
        config=agents_config['diagnostic_agent'],
        tools=[],
        llm=llm,
        memory=False,
    )
    coordinator_agent = Agent(
        config=agents_config['coordinator_agent'],
        tools=[],
        llm=llm,
        memory=False,
    )

    vitals_extraction_task = Task(
        config=tasks_config['vitals_extraction_task'],
        agent=perception_agent,
    )
    diagnostic_triage_task = Task(
        config=tasks_config['diagnostic_triage_task'],
        agent=diagnostic_agent,
    )
    prioritization_coordination_task = Task(
        config=tasks_config['prioritization_coordination_task'],
        agent=coordinator_agent,
    )

    return Crew(
        agents=[perception_agent, diagnostic_agent, coordinator_agent],
        tasks=[vitals_extraction_task, diagnostic_triage_task, prioritization_coordination_task],
        process=Process.sequential,
        verbose=True,
    )

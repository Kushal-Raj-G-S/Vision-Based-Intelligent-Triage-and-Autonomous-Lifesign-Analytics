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
    from crewai import LLM
    from agents import llm_pool

    # NVIDIA's hosted models come and go per API key (see agents/llm_pool.py for the
    # measured history: 410 retirements, per-account 404s, sudden 40 s timeouts —
    # all within hours). So no model is hardcoded as "the" model. The pool probes
    # the candidates in the background, ranks the live ones (quality tier, then
    # latency), and this file just walks that ranking on every call.
    #
    # max_tokens=2000: reasoning-style models spend part of their budget on hidden
    # chain-of-thought before the visible answer; at 500 the 3-agent crew task was
    # cut off mid-reasoning and that unfinished reasoning leaked out as the "raw
    # decision trace". timeout/max_retries=0: a stalled model must fail over to the
    # next one promptly instead of silently retrying for ~45 s.
    llm_pool.start_background_probe()

    _worker_cache = {}

    def _worker(model):
        if model not in _worker_cache:
            _worker_cache[model] = LLM(
                model=model, provider="openai",
                base_url=llm_pool.NIM_BASE, api_key=nvidia_key,
                max_tokens=2000, timeout=30, max_retries=0,
            )
        return _worker_cache[model]

    _DEAD_MARKERS = ("404", "410", "not found", "gone", "end of life", "timed out",
                     "timeout", "502", "503", "500", "connection")

    # The object CrewAI holds. Only its .call is replaced; the real calls go through
    # separate worker objects, so there is no recursion when a model is also first in line.
    llm = LLM(model=llm_pool.CANDIDATES[0][0], provider="openai",
              base_url=llm_pool.NIM_BASE, api_key=nvidia_key, max_tokens=2000)

    def fallback_call(*args, **kwargs):
        llm_pool.wait_ready(30)   # first call after startup: use the real ranking
        for model in llm_pool.ranked():
            try:
                return _worker(model).call(*args, **kwargs)
            except Exception as e:
                msg = str(e)
                dead = llm_pool.is_dead_error(e)
                if dead:
                    llm_pool.mark_dead(model)
                print(f"[FALLBACK] {model} failed ({type(e).__name__}: {msg[:110]})"
                      f"{' — skipped for ' + str(llm_pool.DEAD_TTL_S) + 's' if dead else ''}; trying next.")
        raise RuntimeError("All models in the NVIDIA NIM fallback chain failed.")

    llm.call = fallback_call
    print("[CREW] Initialised self-healing NVIDIA NIM model pool (see agents/llm_pool.py).")
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
        verbose=False,
    )

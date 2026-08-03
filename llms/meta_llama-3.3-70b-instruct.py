import os
import time
from pathlib import Path

from openai import OpenAI


BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "meta/llama-3.3-70b-instruct"
DEFAULT_QUESTION = "Explain one practical way to improve coding interview performance in 30 days."


def _read_env_file_value(key: str) -> str:
    env_file = Path(".env")
    if not env_file.exists():
        return ""

    try:
        lines = env_file.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    except Exception:
        return ""

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        left, right = line.split("=", 1)
        if left.strip() == key:
            return right.strip().strip('"').strip("'")

    return ""


def _load_api_key() -> str:
    return os.getenv("NVIDIA_API_KEY", "").strip() or _read_env_file_value("NVIDIA_API_KEY")


def _load_question() -> str:
    from_env = os.getenv("QUESTION", "").strip() or _read_env_file_value("QUESTION")
    return from_env or DEFAULT_QUESTION


def _save_run_log(model: str, question: str, full_output: str, ttft_seconds: float | None, total_seconds: float) -> Path:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    model_slug = model.replace("/", "_").replace(":", "_")
    log_path = logs_dir / f"{model_slug}_{ts}.txt"

    ttft_text = f"{ttft_seconds:.3f} sec" if ttft_seconds is not None else "not available"
    content = (
        f"Model: {model}\n"
        f"Question: {question}\n"
        f"Time to first token: {ttft_text}\n"
        f"Total generation time: {total_seconds:.3f} sec ({(total_seconds / 60):.3f} min)\n"
        "\n"
        "Output:\n"
        f"{full_output}\n"
    )
    log_path.write_text(content, encoding="utf-8")
    return log_path


def main() -> None:
    api_key = _load_api_key()
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not found in environment or .env")

    question = _load_question()

    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    print(f"Model: {MODEL}")
    print(f"Question: {question}")

    start_time = time.perf_counter()
    first_token_time = None
    full_output_parts: list[str] = []

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        stream=True,
    )

    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue

        content = chunk.choices[0].delta.content
        if content is not None:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            print(content, end="", flush=True)
            full_output_parts.append(content)

    print()

    total_seconds = time.perf_counter() - start_time
    total_minutes = total_seconds / 60
    ttft_seconds = (first_token_time - start_time) if first_token_time is not None else None

    if ttft_seconds is not None:
        print(f"Time to first token: {ttft_seconds:.3f} sec")
    else:
        print("Time to first token: not available")

    print(f"Total generation time: {total_seconds:.3f} sec ({total_minutes:.3f} min)")

    log_path = _save_run_log(MODEL, question, "".join(full_output_parts), ttft_seconds, total_seconds)
    print(f"Saved run log: {log_path}")


if __name__ == "__main__":
    main()

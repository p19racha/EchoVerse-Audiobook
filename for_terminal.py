#!/usr/bin/env python3

import argparse
import json
import os
import sys
import textwrap
import datetime
import shutil
import subprocess
from pathlib import Path

import requests

try:
    from gtts import gTTS
    _HAS_GTTS = True
except Exception:
    _HAS_GTTS = False

try:
    from pydub import AudioSegment
    _HAS_PYDUB = True
except Exception:
    _HAS_PYDUB = False


DEFAULT_TONES = [
    "Neutral",
    "Suspenseful",
    "Inspiring",
    "Joyful",
    "Calm",
    "Dramatic",
    "Motivational",
    "Humorous",
    "Serious",
    "Urgent",
    "Formal",
    "Casual",
    "Friendly",
    "Authoritative",
    "Romantic",
    "Cinematic",
    "Narrative",
    "Empathetic",
]

def pick_tone_interactive(tones):
    print("\nSelect a tone:")
    for i, t in enumerate(tones, 1):
        print(f"  {i}. {t}")
    print("  0. Custom")

    while True:
        try:
            sel = int(input("\nEnter number: ").strip())
        except ValueError:
            print("Please enter a number.")
            continue

        if sel == 0:
            return input("Enter your custom tone: ").strip()
        if 1 <= sel <= len(tones):
            return tones[sel - 1]
        print("Invalid selection, try again.")


def read_text_interactive():
    print("\nPaste your text. End input with a blank line (press Enter twice):")
    lines = []
    while True:
        line = sys.stdin.readline()
        if not line:
            break  # EOF
        if line.strip() == "" and len(lines) > 0:
            break
        lines.append(line)
    return "".join(lines).strip()


def ollama_models(base_url: str):
    """Return list of installed Ollama models (by name:tag) via /api/tags."""
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        models = []
        for m in data.get("models", []):
            if "name" in m:
                models.append(m["name"])
        return models
    except requests.RequestException:
        return []


def ensure_model_present(model: str, base_url: str):
    models = ollama_models(base_url)
    if model not in models:
        msg = (
            f"Ollama model '{model}' not found at {base_url}.\n"
            f"To fix:\n"
            f"  1) Ensure Ollama is running.\n"
            f"  2) Pull the model:\n"
            f"     ollama pull {model}\n"
            f"  3) Retry this script (or pass --model to use another installed model).\n"
            f"Installed models I can see: {', '.join(models) if models else '(none detected)'}"
        )
        raise RuntimeError(msg)


def rewrite_with_ollama(
    text: str,
    tone: str,
    model: str = "gemma3:4b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    """
    Uses Ollama /api/generate with stream=False to rewrite text in a given tone.
    """
    ensure_model_present(model, base_url)

    url = f"{base_url.rstrip('/')}/api/generate"
    prompt = textwrap.dedent(f"""
    You are a writing assistant.

    Task: Rewrite the user's text in a **{tone}** tone.
    Rules:
    - Preserve the original meaning and key facts.
    - Keep it clear and natural.
    - Maintain the original language (do NOT translate).
    - Use an appropriate register for the tone.
    - Output ONLY the rewritten text—no preface, no quotes, no explanations.

    User text:
    ---
    {text}
    ---
    """).strip()

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    try:
        r = requests.post(url, json=payload, timeout=120)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to reach Ollama at {url}. Error: {e}")

    if r.status_code != 200:
        # Add a friendlier hint for model-not-found
        if r.status_code == 404 and "not found" in r.text.lower():
            raise RuntimeError(
                f"Ollama says the model '{model}' is not found.\n"
                f"Run:  ollama pull {model}\n"
                f"Raw response: {r.text}"
            )
        raise RuntimeError(f"Ollama returned HTTP {r.status_code}: {r.text}")

    data = r.json()
    if "response" not in data:
        raise RuntimeError(f"Unexpected Ollama response: {json.dumps(data)[:500]}")

    return data["response"].strip()


def ensure_outputs_dir() -> Path:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_text(text: str, tone: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ensure_outputs_dir()
    safe_tone = "".join(c for c in tone if c.isalnum() or c in ("-", "_")).strip("_")
    path = out_dir / f"rewritten_{safe_tone}_{ts}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def tts_with_gtts(text: str, lang: str, mp3_path: Path):
    if not _HAS_GTTS:
        raise RuntimeError("gTTS not installed. Install with: pip install gTTS")
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(str(mp3_path))


def run_cmd(cmd: list):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def tts_with_piper(text: str, model_path: Path, out_mp3_path: Path):
    """
    Uses Piper CLI to synthesize a WAV and then converts to MP3.
    Requires 'piper' installed and a .onnx voice model.
    """
    if shutil.which("piper") is None:
        raise RuntimeError("Piper CLI not found in PATH. See: https://github.com/rhasspy/piper")

    out_dir = out_mp3_path.parent
    wav_path = out_dir / (out_mp3_path.stem + ".wav")

    # Piper synth
    cmd = ["piper", "--model", str(model_path), "--output_file", str(wav_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate(input=text)
    if proc.returncode != 0:
        raise RuntimeError(f"Piper failed (rc={proc.returncode}). Stderr:\n{stderr}")

    # Convert WAV -> MP3
    if shutil.which("ffmpeg"):
        rc, out, err = run_cmd(["ffmpeg", "-y", "-i", str(wav_path), str(out_mp3_path)])
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed to convert WAV->MP3.\n{err}")
        wav_path.unlink(missing_ok=True)
    else:
        if not _HAS_PYDUB:
            raise RuntimeError("ffmpeg not found and pydub not installed. Install one to get MP3. "
                               f"WAV file at: {wav_path}")
        audio = AudioSegment.from_wav(str(wav_path))
        audio.export(str(out_mp3_path), format="mp3")
        wav_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite text to a chosen tone using Ollama (gemma3:4b by default) and convert to speech (MP3)."
    )
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "gemma3:4b"),
                        help="Ollama model name (default: gemma3:4b). Example: gemma3:4b, qwen2.5:7b, mistral, etc.")
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"),
                        help="Base URL for Ollama (default: http://localhost:11434)")
    parser.add_argument("--tone", help="Tone to use (if omitted, you'll pick interactively)")
    parser.add_argument("--lang", default="en", help="gTTS language code (default: en). Example: te, hi, en, etc.")
    parser.add_argument("--input-file", help="Path to a text file to rewrite (else paste interactively).")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=512, help="LLM max tokens to generate (default: 512)")
    parser.add_argument("--piper-model", help="Path to a Piper .onnx voice model for offline TTS (optional).")
    parser.add_argument("--out-prefix", default="speech", help="Filename prefix for saved MP3 (default: speech)")
    args = parser.parse_args()

    # Gather text
    if args.input_file:
        src = Path(args.input_file)
        if not src.exists():
            print(f"Input file not found: {src}", file=sys.stderr)
            sys.exit(1)
        input_text = src.read_text(encoding="utf-8").strip()
    else:
        input_text = read_text_interactive()
    if not input_text:
        print("No input text provided.", file=sys.stderr)
        sys.exit(1)

    # Pick tone
    tone = args.tone or pick_tone_interactive(DEFAULT_TONES)

    print(f"\n→ Rewriting with Ollama model '{args.model}' in tone: {tone}")
    try:
        rewritten = rewrite_with_ollama(
            input_text,
            tone=tone,
            model=args.model,
            base_url=args.ollama_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
    except Exception as e:
        print(f"\n[ERROR] LLM rewrite failed: {e}", file=sys.stderr)
        sys.exit(2)

    # Save rewritten text
    text_path = save_text(rewritten, tone)
    print(f"✓ Rewritten text saved to: {text_path}")

    # TTS
    out_dir = ensure_outputs_dir()
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_tone = "".join(c for c in tone if c.isalnum() or c in ("-", "_")).strip("_")
    mp3_path = out_dir / f"{args.out_prefix}_{safe_tone}_{ts}.mp3"

    try:
        if args.piper_model:
            print("→ Using Piper (offline TTS).")
            tts_with_piper(rewritten, Path(args.piper_model), mp3_path)
        else:
            print("→ Using gTTS (simple). To use Piper offline, pass --piper-model /path/to/model.onnx")
            tts_with_gtts(rewritten, args.lang, mp3_path)
        print(f"✓ Audio saved to: {mp3_path}")
    except Exception as e:
        print(f"\n[ERROR] TTS failed: {e}", file=sys.stderr)
        sys.exit(3)

    print("\nDone.")


if __name__ == "__main__":
    main()

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import whisper


# Keep the Whisper model loaded while this program runs.
_WHISPER_MODEL = None


def convert_to_wav(input_file: Path, output_file: Path) -> None:
    """Convert audio/video input to mono 22050 Hz WAV using FFmpeg."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg and add its bin folder to PATH. "
            "Then restart VS Code. See the setup instructions."
        )

    command = [
        "ffmpeg", "-y", "-i", str(input_file),
        "-vn", "-ac", "1", "-ar", "22050",
        str(output_file), "-loglevel", "error",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg could not read this file.\n"
            + (result.stderr.strip() or "Unknown FFmpeg error.")
        )


def analyze_speech(audio_file, duration_seconds=None, language="en"):
    global _WHISPER_MODEL

    audio_path = Path(audio_file).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio/video file not found: {audio_path}")

    with tempfile.TemporaryDirectory(prefix="speech_analysis_") as temp_dir:
        wav_path = Path(temp_dir) / "speech.wav"
        convert_to_wav(audio_path, wav_path)

        # Load audio for duration and pitch analysis.
        y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
        actual_duration = len(y) / sr

        duration = (
            float(duration_seconds)
            if duration_seconds is not None and duration_seconds > 0
            else actual_duration
        )

        if _WHISPER_MODEL is None:
            print("Loading Whisper 'small' model (first run downloads it)...")
            _WHISPER_MODEL = whisper.load_model("small")

        result = _WHISPER_MODEL.transcribe(
            str(wav_path),
            language=language,
            word_timestamps=True,
            fp16=False,
            initial_prompt=(
                "Transcribe verbatim, preserving filler words, "
                "hesitations and repetitions."
            ),
            condition_on_previous_text=False,
            temperature=0,
        )

    transcript = result["text"].strip()
    words = re.findall(r"\b[\w']+\b", transcript)
    total_words = len(words)

    # Filler counts (same list and counting logic as the Colab code).
    patterns = {
        "um": r"\bum+\b",
        "uh": r"\buh+\b",
        "erm": r"\berm+\b",
        "hmm": r"\bhm+\b",
        "you know": r"\byou\s+know\b",
        "i mean": r"\bi\s+mean\b",
        "basically": r"\bbasically\b",
        "actually": r"\bactually\b",
        "literally": r"\bliterally\b",
        "like": r"\blike\b",
    }

    filler_counts = {
        name: len(re.findall(pattern, transcript.lower()))
        for name, pattern in patterns.items()
    }
    total_fillers = sum(filler_counts.values())

    # Most frequent filler; include ties, matching the original logic.
    max_count = max(filler_counts.values(), default=0)
    most_fillers = [
        {"word": word, "count": count}
        for word, count in filler_counts.items()
        if count == max_count and count > 0
    ]

    # Collect word-level timestamps from Whisper.
    word_data = []
    for segment in result["segments"]:
        for word in segment.get("words", []):
            word_data.append({
                "word": word["word"].strip(),
                "start": float(word["start"]),
                "end": float(word["end"]),
            })

    # Count pauses of 2 seconds or longer.
    pauses = []
    for prev, curr in zip(word_data, word_data[1:]):
        gap = curr["start"] - prev["end"]
        if gap >= 2.0:
            pauses.append({
                "start": round(prev["end"], 2),
                "end": round(curr["start"], 2),
                "duration": round(gap, 2),
            })

    # Group timestamped words into sentences using ending punctuation.
    groups = []
    current = []
    for item in word_data:
        current.append(item)
        if re.search(r'[.!?…]+["\'”’)\]]*$', item["word"]):
            groups.append(current)
            current = []

    if current:
        groups.append(current)

    # Estimate fundamental frequency (pitch).
    f0, _, _ = librosa.pyin(
        y,
        fmin=65,
        fmax=500,
        sr=sr,
        frame_length=2048,
        hop_length=256,
    )
    pitch_times = librosa.times_like(f0, sr=sr, hop_length=256)

    sentence_pitch = []
    for group in groups:
        start = group[0]["start"]
        end = group[-1]["end"]

        sentence = " ".join(w["word"] for w in group)
        sentence = re.sub(r"\s+([,.;:!?…])", r"\1", sentence)

        mask = (
            (pitch_times >= start)
            & (pitch_times < end)
            & np.isfinite(f0)
        )
        pitch_values = f0[mask]
        pitch = (
            round(float(np.median(pitch_values)), 2)
            if len(pitch_values)
            else None
        )

        sentence_pitch.append({
            "sentence": sentence,
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(end - start, 2),
            "pitch_hz": pitch,
        })

    return {
        "transcript": transcript,
        "duration": round(duration, 2),
        "words_per_minute": round(
            total_words / (duration / 60), 2
        ) if duration > 0 else 0,
        "total_filler_words": total_fillers,
        "most_filler_words": most_fillers,
        "number_of_pauses": len(pauses),
        "sentence_pitch": sentence_pitch,
    }


def main():
    print("=== Speech Analysis ===")
    file_input = input(
        "Enter the path to your audio/video file "
        "(e.g. my_speech.webm): "
    ).strip().strip('"').strip("'")

    language = input("Language code [en]: ").strip() or "en"

    try:
        report = analyze_speech(file_input, language=language)
        print("\n=== Analysis Results ===")
        print(json.dumps(report, indent=2, ensure_ascii=False))

        output_path = Path("speech_analysis_report.json")
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nReport saved to: {output_path.resolve()}")

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

#Run this 
# python -m pip install SpeechRecognition


import argparse
import json
import re
import sys
import wave
import audioop
from pathlib import Path
from statistics import median

import speech_recognition as sr


DEFAULT_RECORDING = "audio_withfillers.wav"

FILLER_PATTERNS = {
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


def load_wav(filename):
    """Read a PCM WAV file using Python's built-in wave module."""
    with wave.open(filename, "rb") as audio:
        channels = audio.getnchannels()
        width = audio.getsampwidth()
        sample_rate = audio.getframerate()
        frames = audio.readframes(audio.getnframes())

    if width != 2:
        raise ValueError(
            "Use a 16-bit PCM WAV file. Convert the recording to "
            "16-bit PCM WAV and try again."
        )

    if channels == 2:
        frames = audioop.tomono(frames, 2, 0.5, 0.5)
    elif channels != 1:
        raise ValueError("Only mono or stereo WAV files are supported.")

    duration = len(frames) / (2 * sample_rate)

    if duration <= 0:
        raise ValueError("The audio file is empty.")

    return frames, sample_rate, duration


def transcribe_audio(filename, language="en-US"):
    """Transcribe using Google's online speech-recognition service."""
    recognizer = sr.Recognizer()

    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)

    try:
        return recognizer.recognize_google(
            audio, language=language
        ).strip()
    except sr.UnknownValueError:
        raise ValueError("Speech could not be recognized.")
    except sr.RequestError as exc:
        raise RuntimeError(
            f"Speech-recognition service unavailable: {exc}"
        )


def count_fillers(transcript):
    text = transcript.lower()

    counts = {
        name: len(re.findall(pattern, text))
        for name, pattern in FILLER_PATTERNS.items()
    }

    total = sum(counts.values())
    maximum = max(counts.values(), default=0)

    most_frequent = [
        {"word": word, "count": count}
        for word, count in counts.items()
        if count == maximum and count > 0
    ]

    return total, most_frequent, counts


def detect_pauses(pcm, sample_rate, threshold=2.0):
    """Estimate pauses from quiet audio regions lasting >= threshold."""
    window_ms = 100
    window_bytes = int(sample_rate * 2 * window_ms / 1000)
    windows = []

    for offset in range(0, len(pcm), window_bytes):
        chunk = pcm[offset:offset + window_bytes]
        if len(chunk) < 2:
            continue

        rms = audioop.rms(chunk, 2)
        start = offset / (sample_rate * 2)
        end = min(
            (offset + len(chunk)) / (sample_rate * 2),
            len(pcm) / (sample_rate * 2),
        )
        windows.append((start, end, rms))

    if not windows:
        return []

    # Relative threshold helps adapt to different microphone levels.
    peak = max(item[2] for item in windows)
    silence_level = max(100, peak * 0.035)

    pauses = []
    silence_start = None
    silence_end = None

    for start, end, rms in windows:
        if rms <= silence_level:
            if silence_start is None:
                silence_start = start
            silence_end = end
        elif silence_start is not None:
            gap = silence_end - silence_start
            if gap >= threshold:
                pauses.append({
                    "start": round(silence_start, 2),
                    "end": round(silence_end, 2),
                    "duration": round(gap, 2),
                })
            silence_start = None
            silence_end = None

    if silence_start is not None:
        gap = silence_end - silence_start
        if gap >= threshold:
            pauses.append({
                "start": round(silence_start, 2),
                "end": round(silence_end, 2),
                "duration": round(gap, 2),
            })

    return pauses


def estimate_pitch(pcm, sample_rate, start, end):
    """Estimate median pitch with simple autocorrelation."""
    # Analyze a short section near the middle of the sentence.
    middle = (start + end) / 2
    center = int(middle * sample_rate)
    radius = int(0.04 * sample_rate)

    raw = pcm[max(0, (center - radius) * 2):
              min(len(pcm), (center + radius) * 2)]

    if len(raw) < 400:
        return None

    samples = [
        int.from_bytes(raw[i:i + 2], "little", signed=True)
        for i in range(0, len(raw) - 1, 2)
    ]

    if not samples:
        return None

    # Remove DC offset.
    average = sum(samples) / len(samples)
    samples = [value - average for value in samples]

    if max(abs(value) for value in samples) < 150:
        return None

    min_lag = max(1, int(sample_rate / 500))
    max_lag = min(len(samples) // 2, int(sample_rate / 65))

    correlations = []
    for lag in range(min_lag, max_lag + 1):
        score = sum(
            samples[i] * samples[i + lag]
            for i in range(len(samples) - lag)
        )
        correlations.append((score, lag))

    if not correlations:
        return None

    _, best_lag = max(correlations)
    return round(sample_rate / best_lag, 2)


def analyze_speech(filename, language="en-US"):
    path = Path(filename)

    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {filename}")

    pcm, sample_rate, duration = load_wav(str(path))
    transcript = transcribe_audio(str(path), language)

    words = re.findall(r"\b[\w']+\b", transcript)
    total_words = len(words)

    total_fillers, most_fillers, filler_counts = count_fillers(
        transcript
    )

    pauses = detect_pauses(pcm, sample_rate)

    # Sentence timing is estimated from word proportions because
    # Google's basic transcription endpoint does not return word times.
    sentence_texts = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", transcript)
        if part.strip()
    ]

    sentence_pitch = []
    cursor = 0.0
    total_sentence_words = sum(
        len(re.findall(r"\b[\w']+\b", sentence))
        for sentence in sentence_texts
    )

    for sentence in sentence_texts:
        sentence_words = len(
            re.findall(r"\b[\w']+\b", sentence)
        )

        if total_sentence_words:
            sentence_duration = (
                duration * sentence_words / total_sentence_words
            )
        else:
            sentence_duration = 0.0

        start = cursor
        end = min(duration, cursor + sentence_duration)
        cursor = end

        sentence_pitch.append({
            "sentence": sentence,
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(end - start, 2),
            "pitch_hz": estimate_pitch(
                pcm, sample_rate, start, end
            ),
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
    parser = argparse.ArgumentParser(
        description="Analyze speech from a WAV recording."
    )
    parser.add_argument(
        "--file", default=DEFAULT_RECORDING,
        help="Input WAV file (default: audio_withfillers.wav)"
    )
    parser.add_argument(
        "--language", default="en-US",
        help="Recognition language (default: en-US)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional JSON report output path"
    )
    args = parser.parse_args()

    try:
        report = analyze_speech(args.file, args.language)
        output = json.dumps(report, indent=2, ensure_ascii=False)
        print(output)

        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            print(f"\nReport saved to {args.output}")

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

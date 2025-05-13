#!/usr/bin/env python3
"""
Voice Assistant: Real-Time Voice Chat

This app runs on a Raspberry Pi (or Linux desktop) and creates a low-latency, full-duplex voice interaction
with an AI character. It uses local speech recognition
(Vosk), local text-to-speech synthesis (Piper), and a locally hosted large language model via Ollama.

Key Features:
- Wake-free, continuous voice recognition with real-time transcription
- LLM-driven responses streamed from a selected local model (e.g., LLaMA, Qwen, Gemma)
- Audio response synthesis with a gruff custom voice using ONNX-based Piper models
- Optional noise mixing and filtering via SoX
- System volume control via ALSA
- Modular and responsive design suitable for low-latency, character-driven agents

Ideal for embedded voice AI demos, cosplay companions, or standalone AI characters.

Copyright: M15.ai
License: MIT
"""

import os
import json
import queue
import threading
import time
import wave
import io
import re
import subprocess
from vosk import Model, KaldiRecognizer
import pyaudio
import requests
from pydub import AudioSegment
import soxr
import numpy as np

# ------------------- TIMING UTILITY -------------------
class Timer:
    def __init__(self, label):
        self.label = label
        self.enabled = True
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            elapsed_ms = (time.time() - self.start) * 1000
            print(f"[Timing] {self.label}: {elapsed_ms:.0f} ms")
    def disable(self):
        self.enabled = False

# ------------------- FUNCTIONS -------------------

def get_input_device_index(preferred_name="Shure MVX2U"):
    pa = pyaudio.PyAudio()
    index = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if preferred_name.lower() in info['name'].lower() and info['maxInputChannels'] > 0:
            print(f"[Debug] Selected input device {i}: {info['name']}")
            print(f"[Debug] Device sample rate: {info['defaultSampleRate']} Hz")
            index = i
            break
    pa.terminate()
    if index is None:
        print("[Warning] Preferred mic not found. Falling back to default.")
    return index

def get_output_device_index(preferred_name):
    pa = pyaudio.PyAudio()
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if preferred_name.lower() in info['name'].lower() and info['maxOutputChannels'] > 0:
            print(f"[Debug] Selected output device {i}: {info['name']}")
            return i
    print("[Warning] Preferred output device not found. Using default index 0.")
    return 0

def parse_card_number(device_str):
    """
    Extract ALSA card number from string like 'plughw:3,0'
    """
    try:
        return int(device_str.split(":")[1].split(",")[0])
    except Exception as e:
        print(f"[Warning] Could not parse card number from {device_str}: {e}")
        return 0  # fallback

def list_input_devices():
    pa = pyaudio.PyAudio()
    print("[Debug] Available input devices:")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  {i}: {info['name']} ({int(info['defaultSampleRate'])} Hz, {info['maxInputChannels']}ch)")
    pa.terminate()

def resample_audio(data, orig_rate=48000, target_rate=16000):
    # Convert byte string to numpy array
    audio_np = np.frombuffer(data, dtype=np.int16)
    # Resample using soxr
    resampled_np = soxr.resample(audio_np, orig_rate, target_rate)
    # Convert back to bytes
    return resampled_np.astype(np.int16).tobytes()

def set_output_volume(volume_level, card_id=3):
    """
    Set output volume using ALSA 'Speaker' control on specified card.
    volume_level: 1–10 (user scale)
    card_id: ALSA card number (from aplay -l)
    """
    percent = max(1, min(volume_level, 10)) * 10  # map to 10–100%
    try:
        subprocess.run(
            ['amixer', '-c', str(card_id), 'sset', 'Speaker', f'{percent}%'],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"[Debug] Volume set to {percent}% on card {card_id}")
    except Exception as e:
        print(f"[Warning] Volume control failed on card {card_id}: {e}")

# ------------------- PATHS -------------------

CONFIG_PATH = os.path.expanduser("va_config.json")
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'vosk-model')
CHAT_URL = 'http://localhost:11434/api/chat'

# ------------------- CONFIG FILE LOADING -------------------

DEFAULT_CONFIG = {
    "volume": 9,
    "mic_name": "Plantronics",
    "audio_output_device": "Plantronics",
    "model_name": "qwen2.5:0.5b",
    "voice": "en_US-kathleen-low.onnx",
    "enable_audio_processing": False,
    "history_length": 4,
    "system_prompt": "You are a helpful assistant."
}

def load_config():
    # Load config from system file or fall back to defaults
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                user_config = json.load(f)
            return {**DEFAULT_CONFIG, **user_config}  # merge with defaults
        except Exception as e:
            print(f"[Warning] Failed to load system config: {e}")

    print("[Debug] Using default config.")

    return DEFAULT_CONFIG

config = load_config()

# Apply loaded config values
VOLUME = config["volume"]
MIC_NAME = config["mic_name"]
AUDIO_OUTPUT_DEVICE = config["audio_output_device"]
AUDIO_OUTPUT_DEVICE_INDEX = get_output_device_index(config["audio_output_device"])
OUTPUT_CARD = parse_card_number(AUDIO_OUTPUT_DEVICE)
MODEL_NAME = config["model_name"]
VOICE_MODEL = os.path.join("voices", config["voice"])
ENABLE_AUDIO_PROCESSING = config["enable_audio_processing"]
HISTORY_LENGTH = config["history_length"]

# Set system volume
set_output_volume(VOLUME, OUTPUT_CARD)

# Setup messages with system prompt
messages = [{"role": "system", "content": config["system_prompt"]}]

list_input_devices()
RATE = 48000
CHUNK = 1024
CHANNELS = 1
mic_enabled = True
DEVICE_INDEX = get_input_device_index()

# SOUND EFFECTS
NOISE_LEVEL = '0.04'
BANDPASS_HIGHPASS = '300'
BANDPASS_LOWPASS = '800'

# ------------------- VOICE MODEL -------------------

VOICE_MODELS_DIR = os.path.join(BASE_DIR, 'voices')
if not os.path.isdir(VOICE_MODELS_DIR):
    os.makedirs(VOICE_MODELS_DIR)

VOICE_MODEL = os.path.join(VOICE_MODELS_DIR, config["voice"])

print('[Debug] Available Piper voices:')
for f in os.listdir(VOICE_MODELS_DIR):
    if f.endswith('.onnx'):
        print('  ', f)
print(f'[Debug] Using VOICE_MODEL: {VOICE_MODEL}')
print(f"[Debug] Config loaded: model={MODEL_NAME}, voice={config['voice']}, vol={VOLUME}, mic={MIC_NAME}")

# ------------------- CONVERSATION STATE -------------------

audio_queue = queue.Queue()

# Audio callback form Shure
def audio_callback(in_data, frame_count, time_info, status):
    global mic_enabled
    if not mic_enabled:
        return (None, pyaudio.paContinue)
    resampled_data = resample_audio(in_data, orig_rate=48000, target_rate=16000)
    audio_queue.put(resampled_data)
    return (None, pyaudio.paContinue)

# ------------------- STREAM SETUP -------------------

def start_stream():
    pa = pyaudio.PyAudio()

    stream = pa.open(
        rate=RATE,
        format=pyaudio.paInt16,
        channels=CHANNELS,
        input=True,
        input_device_index=DEVICE_INDEX,
        frames_per_buffer=CHUNK,
        stream_callback=audio_callback
    )
    stream.start_stream()
    print(f'[Debug] Stream @ {RATE}Hz')
    return pa, stream

# ------------------- QUERY OLLAMA CHAT ENDPOINT -------------------

def query_ollama():
    payload = {
        "model": MODEL_NAME,
        "messages": [messages[0]] + messages[-HISTORY_LENGTH:],  # force system prompt at top
        "stream": False}

    with Timer("Inference"):  # measure inference latency
        resp = requests.post(CHAT_URL, json=payload)
    #print(f'[Debug] Ollama status: {resp.status_code}')
    data = resp.json()
    # Extract assistant message
    reply = ''
    if 'message' in data and 'content' in data['message']:
        reply = data['message']['content'].strip()
    #print('[Debug] Reply:', reply)
    return reply

# ------------------- TTS & DEGRADATION -------------------

import tempfile

def play_response(text):
    import io
    import tempfile

    # Mute the mic during playback to avoid feedback loop
    global mic_enabled
    mic_enabled = False  # 🔇 mute mic

    # clean the response
    clean = re.sub(r"[\*]+", '', text)                # remove asterisks
    clean = re.sub(r"\(.*?\)", '', clean)             # remove (stage directions)
    clean = re.sub(r"<.*?>", '', clean)               # remove HTML-style tags
    clean = clean.replace('\n', ' ').strip()          # normalize newlines
    clean = re.sub(r'\s+', ' ', clean)                # collapse whitespace
    clean = re.sub(r'[\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]+', '', clean)  # remove emojis

    piper_path = os.path.join(BASE_DIR, 'bin', 'piper', 'piper')

    # 1. Generate Piper raw PCM
    with Timer("Piper inference"):
        piper_proc = subprocess.Popen(
            [piper_path, '--model', VOICE_MODEL, '--output_raw'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        tts_pcm, _ = piper_proc.communicate(input=clean.encode())

    if ENABLE_AUDIO_PROCESSING:
        # SoX timing consolidation
        sox_start = time.time()

        # 2. Convert raw PCM to WAV
        pcm_to_wav = subprocess.Popen(
            ['sox', '-t', 'raw', '-r', '16000', '-c', str(CHANNELS), '-b', '16',
            '-e', 'signed-integer', '-', '-t', 'wav', '-'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        tts_wav_16k, _ = pcm_to_wav.communicate(input=tts_pcm)

        # 3. Estimate duration
        duration_sec = len(tts_pcm) / (RATE * 2)

        # 4. Generate white noise WAV bytes
        noise_bytes = subprocess.check_output([
            'sox', '-n',
            '-r', '16000',
            '-c', str(CHANNELS),
            '-b', '16',
            '-e', 'signed-integer',
            '-t', 'wav', '-',
            'synth', str(duration_sec),
            'whitenoise', 'vol', NOISE_LEVEL
        ], stderr=subprocess.DEVNULL)

        # 5. Write both to temp files & mix
        with tempfile.NamedTemporaryFile(suffix='.wav') as tts_file, tempfile.NamedTemporaryFile(suffix='.wav') as noise_file:
            tts_file.write(tts_wav_16k)
            noise_file.write(noise_bytes)
            tts_file.flush()
            noise_file.flush()
            mixer = subprocess.Popen(
                ['sox', '-m', tts_file.name, noise_file.name, '-t', 'wav', '-'],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            mixed_bytes, _ = mixer.communicate()

        # 6. Apply filter
        filter_proc = subprocess.Popen(
            #['sox', '-t', 'wav', '-', '-t', 'wav', '-', 'highpass', BANDPASS_HIGHPASS, 'lowpass', BANDPASS_LOWPASS],
            ['sox', '-t', 'wav', '-', '-r', '48000', '-t', 'wav', '-',
             'highpass', BANDPASS_HIGHPASS, 'lowpass', BANDPASS_LOWPASS],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        final_bytes, _ = filter_proc.communicate(input=mixed_bytes)

        sox_elapsed = (time.time() - sox_start) * 1000
        print(f"[Timing] SoX (total): {int(sox_elapsed)} ms")
    
    else:
        # No FX: just convert raw PCM to WAV
        pcm_to_wav = subprocess.Popen(
            ['sox', '-t', 'raw', '-r', '16000', '-c', str(CHANNELS), '-b', '16',
             '-e', 'signed-integer', '-', '-t', 'wav', '-'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        tts_wav_16k, _ = pcm_to_wav.communicate(input=tts_pcm)

        resample_proc = subprocess.Popen(
            ['sox', '-t', 'wav', '-', '-r', '48000', '-t', 'wav', '-'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        final_bytes, _ = resample_proc.communicate(input=tts_wav_16k)

    # 7. Playback
    with Timer("Playback"):
        try:
            wf = wave.open(io.BytesIO(final_bytes), 'rb')


            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pa.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                output_device_index=AUDIO_OUTPUT_DEVICE_INDEX
            )

            data = wf.readframes(CHUNK)
            while data:
                stream.write(data)
                data = wf.readframes(CHUNK)

            stream.stop_stream()
            stream.close()
            pa.terminate()
            wf.close()

        except wave.Error as e:
            print(f"[Error] Could not open final WAV: {e}")
        
        finally:
            mic_enabled = True      # 🔊 unmute mic
            time.sleep(0.3)         # optional: small cooldown


# ------------------- PROCESSING LOOP -------------------

def processing_loop():
    model = Model(MODEL_PATH)
    #rec = KaldiRecognizer(model, RATE)
    rec = KaldiRecognizer(model, 16000)
    MAX_DEBUG_LEN = 200  # optional: limit length of debug output
    LOW_EFFORT_UTTERANCES = {"huh", "uh", "um", "erm", "hmm", "he's", "but"}

    while True:
        data = audio_queue.get()

        if rec.AcceptWaveform(data):
            start = time.time()
            r = json.loads(rec.Result())
            elapsed_ms = int((time.time() - start) * 1000)

            user = r.get('text', '').strip()
            if user:
                print(f"[Timing] STT parse: {elapsed_ms} ms")
                print("User:", user)

                if user.lower().strip(".,!? ") in LOW_EFFORT_UTTERANCES:
                    print("[Debug] Ignored low-effort utterance.")
                    rec = KaldiRecognizer(model, 16000)
                    continue  # Skip LLM response + TTS for accidental noise

                messages.append({"role": "user", "content": user})
                # Generate assistant response
                resp_text = query_ollama()
                if resp_text:
                    # Clean debug print (remove newlines and carriage returns)
                    clean_debug_text = resp_text.replace('\n', ' ').replace('\r', ' ')
                    if len(clean_debug_text) > MAX_DEBUG_LEN:
                        clean_debug_text = clean_debug_text[:MAX_DEBUG_LEN] + '...'

                    print('Assistant:', clean_debug_text)
                    messages.append({"role": "assistant", "content": clean_debug_text})

                    # TTS generation + playback
                    play_response(resp_text)
                else:
                    print('[Debug] Empty response, skipping TTS.')

                # Reset recognizer after each full interaction
                rec = KaldiRecognizer(model, 16000)

# ------------------- MAIN -------------------

if __name__ == '__main__':
    pa, stream = start_stream()
    t = threading.Thread(target=processing_loop, daemon=True)
    t.start()
    try:
        while stream.is_active():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stream.stop_stream(); stream.close(); pa.terminate()

# Local Voice Assistant (Offline, Real-Time AI)

**Lightweight, low-latency voice assistant running fully offline on a Raspberry Pi or Linux machine.**  
Powered by PyAudio, Vosk STT, Piper TTS, and local LLMs via Ollama.

![badge](https://img.shields.io/badge/Offline-Voice%20AI-blue)
![badge](https://img.shields.io/badge/Audio-PyAudio-yellow)
![badge](https://img.shields.io/badge/TTS-Piper-orange)
![badge](https://img.shields.io/badge/LLM-Gemma2%20%7C%20Qwen-success)

---

## 🎯 Features

- 🎙️ **Microphone Input** using PyAudio
- 🔊 **Real-Time Transcription** with [Vosk](https://alphacephei.com/vosk/)
- 🧠 **LLM-Powered Responses** using [Ollama](https://ollama.com) with models like `gemma2:2b`, `qwen2.5:0.5b`
- 🗣️ **Natural Voice Output** via [Piper TTS](https://github.com/rhasspy/piper)
- 🎛️ Optional **Noise & Filter FX** using SoX for realism
- 🔧 ALSA **Volume Control**
- 🧩 Modular Python code ready for customization

---

## 🛠 Requirements

- Raspberry Pi 5 or Linux desktop
- Python 3.9+
- PyAudio, NumPy, requests, soxr, pydub, vosk
- SoX + ALSA utilities
- Ollama with one or more small LLMs (e.g., Gemma or Qwen)
- Piper TTS with ONNX voice models

Install dependencies:

```
pip install pyaudio requests soxr numpy pydub vosk
sudo apt install sox alsa-utils
```

## ⚙️ JSON Configuration

Place a config file at va_config.json:

```
{
  "volume": 8,
  "mic_name": "Plantronics",
  "audio_output_device": "Plantronics",
  "model_name": "gemma2:2b",
  "voice": "en_US-kathleen-low.onnx",
  "enable_audio_processing": false,
  "history_length": 6,
  "system_prompt": "You are a helpful assistant."
}
```

Note: if the configuration file is not found, defaults withing the main python app will be used:

```
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
```

### 🔁 What `history_length` Means

The `history_length` setting controls how many previous exchanges (user + assistant messages) are included when generating each new reply.

- A value of `6` means the model receives the last 6 exchanges, plus the system prompt.
- This allows the assistant to maintain **short-term memory** for more coherent conversations.
- Setting it lower (e.g., `2`) increases speed and memory efficiency.

### ✅ `requirements.txt`

```
pyaudio
vosk
soxr
numpy
requests
pydub
```

If you plan to run this on a Raspberry Pi, you may also need:

```
soundfile  # for pydub compatibility on some distros
```

## 🐍 Install with Virtual Environment

```
# 1. Clone the repo

git clone https://github.com/your-username/voice-assistant-local.git
cd voice-assistant-local

# 2. Create and activate a virtual environment

python3 -m venv env
source env/bin/activate

# 3. Install dependencies

pip install -r requirements.txt

# 4. Install SoX and ALSA utilities (if not already installed)

sudo apt install sox alsa-utils

# 5. (Optional) Test PyAudio installation

python -m pip install --upgrade pip setuptools wheel
```

> 💡 If you get errors installing PyAudio on Raspberry Pi, try:
>
> ```
> sudo apt install portaudio19-dev
> pip install pyaudio
> ```

## 🆕 🔧 Piper Installation (Binary)

Piper is a standalone text-to-speech engine used by this assistant. It's **not a Python package**, so it must be installed manually.

#### ✅ Install Piper

1. Download the appropriate Piper binary from:
    👉 https://github.com/rhasspy/piper/releases

   For Ubuntu Linux, download:
    `piper_linux_x86_64.tar.gz`

2. Extract it:

   ```
   tar -xvzf piper_linux_x86_64.tar.gz
   ```

3. Move the binary into your project directory:

   ```
   mkdir -p bin/piper
   mv piper bin/piper/
   chmod +x bin/piper/piper
   ```

4. ✅ Done! The script will automatically call it from `bin/piper/piper`.

## 📂 Directory Example

```
voice_assistant.py
va_config.json
requirements.txt
bin/
└── piper/
    └── piper        ← (binary)
voices/
└── en_US-kathleen-low.onnx
└── en_US-kathleen-low.onnx.json
```



## 🔌 Finding Your USB Microphone & Speaker

To configure the correct audio devices, use these commands on your Raspberry Pi or Linux terminal:

1. List Microphones (Input Devices)

```
python3 -m pip install pyaudio
python3 -c "import pyaudio; p = pyaudio.PyAudio(); \
[print(f'{i}: {p.get_device_info_by_index(i)}') for i in range(p.get_device_count())]"
```

Look for your microphone name (e.g., Plantronics) and use that as mic_name.
2. List Speakers (Output Devices)

```
aplay -l
```

Example output:

```
card 3: Device [USB PnP Sound Device], device 0: USB Audio [USB Audio]
```

Use this info to set your audio_output_device to something like:

```
"audio_output_device": "USB PnP"
```

## 🔧 Ollama Installation (Required)

Ollama is a local model runner for LLMs. You need to install it separately (outside of Python).

#### 💻 Install Ollama

On **Linux (x86 or ARM)**:

```
curl -fsSL https://ollama.com/install.sh | sh
```

Or follow detailed instructions:
 👉 https://ollama.com/download

Then start the daemon:

```
ollama serve
```

#### 📥 Download the Models

After Ollama is installed and running, open a terminal and run:

##### ✅ For Gemma 2B:

```
ollama run gemma2:2b
```

#####  For Qwen 0.5B:

```
ollama run qwen2.5:0.5b
```

This will automatically download and start the models. You only need to run this once per model.

##### ⚠️ Reminder

> Ollama is **not a Python package** — it is a background service.
>  Do **not** add it to `requirements.txt`. Just make sure it’s installed and running before launching the assistant.

## 🎤 Installing Piper Voice Models

To enable speech synthesis, you'll need to download a **voice model (.onnx)** and its matching **config (.json)** file.

#### ✅ Steps:

1. Visit the official Piper voices list:
    📄 https://github.com/rhasspy/piper/blob/master/VOICES.md

2. Choose a voice you like (e.g., `en_US-lessac-medium` or `en_US-amy-low`).

3. Download **both** files for your chosen voice:

   - `voice.onnx`
   - `config.json`

4. If you wish, you can rename the ONNX file and config file using the same base name. For example:

   ```
   amy-low.onnx
   amy-low.json
   ```

5. Place both files in a directory called `voices/` next to your script. 
   Example Directory Structure:

   ```
   voice_assistant.py
   voices/
   ├── amy-low.onnx
   └── amy-low.json
   ```

6. Update your `config.json`:

   ```
   "voice": "amy-low.onnx"
   ```

> ⚠️ Make sure both `.onnx` and `.json` are present in the `voices/` folder with matching names (excluding the extension).

## 🧪 **Performance Report**

The script prints out debug timing for the STT, LLM, and TTS parts of the pipeline. I asked ChatGPT4 to analyze some of the results i obtained.

**System:** Ubuntu laptop, Intel Core i5
 **Model:** `qwen2.5:0.5b` (local via Ollama)
 **TTS:** `piper` with `en_US-kathleen-low.onnx`
 **Audio:** Plantronics USB headset

------

### 📊 **Timing Metrics (avg)**

| Stage          | Metric (ms)   | Notes                                   |
| -------------- | ------------- | --------------------------------------- |
| STT Parse      | 4.5 ms avg    | Vosk transcribes near-instantly         |
| LLM Inference  | ~2,200 ms avg | Ranges from ~1s (short queries) to 5s   |
| TTS Generation | ~1,040 ms avg | Piper ONNX performs well on CPU         |
| Audio Playback | ~7,250 ms avg | Reflects actual audio length, not delay |

### ✅ Observations

- **STT speed is excellent** — under 10 ms consistently.
- **LLM inference is snappy** for a 0.5b model running locally. Your best response came in under 1.1 sec.
- **TTS is consistent and fast** — Kathleen-low voice is fully synthesized in ~800–1600 ms.
- **Playback timing matches response length** — no lag, just actual audio time.
- End-to-end round trip time from speaking to hearing a reply is about **8–10 seconds**, including speech and playback time.

## 💡 Use Cases

- ​    Offline smart assistants

- ​    Wearable or embedded AI demos

- ​    Voice-controlled kiosks

- ​    Character-based roleplay agents


## 📄 License

MIT © 2024 M15.ai
import sounddevice as sd
import numpy as np
import soundfile as sf
from collections import deque
from faster_whisper import WhisperModel
from piper import PiperVoice
import wave
from scipy.signal import resample_poly
import ollama
from pathlib import Path

sd.default.device = (2, None)
sample_rate = 48000
block_size = 1024

threshold = 0.01
silence_duration = 2.0

voice_buffer = []
recent_rms = deque(maxlen=int(silence_duration * sample_rate / 1024))

SOURCE_PATH = Path(__file__).resolve().parent
CANTO_PATH = str(Path(SOURCE_PATH / "canto-model"))
VOICE_PATH = str(Path(SOURCE_PATH / "piper/voices/en_US-amy-medium.onnx"))

model = WhisperModel(CANTO_PATH, device="cpu", compute_type="float32")

voice = PiperVoice.load(VOICE_PATH)

def translation(instruction, input_text):
    prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    messages = [{ "role": "user", "content": prompt }]

    try:
        # Send the POST request and capture the response
        response = ollama.chat(model="7shi/llama-translate:8b-q4_K_M", messages=messages)
        # print(response)
    except ollama.ResponseError as e:
        # if the request was failed
        print("Error:", e.error)
        return None

    # Extract the 'content' field from the response
    response_content = response["message"]["content"].strip()

    return response_content


def rms(indata):
    return np.sqrt(np.mean(indata**2))

def save_audio(data):
    print("saving")
    audio = np.concatenate(data, axis=0)
    sf.write("output.wav", audio, sample_rate)
    print("saved")
    segments, info = model.transcribe(
        "output.wav",
        task="translate"
    )

    text = " ".join(seg.text for seg in segments)
    translated_line = translation(f"Translate Chinese to English.", text)
    print(text)
    print(translated_line)
    with wave.open("test.wav", "wb") as wf:
        voice.synthesize_wav(translated_line, wf)
    
    data, fs = sf.read("test.wav", dtype='float32')

    if fs != sample_rate:
        gcd = np.gcd(fs, sample_rate)
        up = sample_rate // gcd
        down = fs // gcd
        resampled = resample_poly(data, up, down)

    sd.play(resampled, sample_rate, device=1)
    sd.wait()

def callback(indata, frames, time, status):
    if status:
        print(status)

    current_rms = rms(indata)
    recent_rms.append(current_rms)

    if current_rms > threshold:
        voice_buffer.append(indata.copy())
    else:
        if len(recent_rms) == recent_rms.maxlen and max(recent_rms) < threshold:
            if voice_buffer:
                save_audio(voice_buffer)
                voice_buffer.clear()

with sd.InputStream(
        samplerate=48000,
        channels=1,
        blocksize=block_size,
        callback=callback
    ):
        try:
            print("ready")
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("ending")
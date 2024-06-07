# %pip install git+https://github.com/huggingface/transformers.git

import os
import torch
import librosa 
from huggingface_hub import HfApi, login
from datasets import load_dataset, Audio
from transformers import Wav2Vec2ForCTC, AutoProcessor

# Load the token from environment variable
hf_token = "Hugging_Face_Token"
login(token=hf_token)
api = HfApi(token=hf_token)

# Now you can use the API client to interact with Hugging Face Hub

# !huggingface-cli login

# ================ English =======================
stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="test", streaming=True)
stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
en_sample = next(iter(stream_data))["audio"]["array"]


# ================ French ========================
stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "fr", split="test", streaming=True)
stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
fr_sample = next(iter(stream_data))["audio"]["array"]

model_id = "facebook/mms-1b-all"    # STT Model

processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

inputs = processor(en_sample, sampling_rate=16_000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).logits

ids = torch.argmax(outputs, dim=-1)[0]
transcription = processor.decode(ids)

audio_path= "kurdish_02.wav"    # Audio Media File
sampling_rate=16000
audio, _ = librosa.load(audio_path, sr=sampling_rate)

# Kurdish Language Model
processor.tokenizer.set_target_lang("kmr-script_latin")
model.load_adapter("kmr-script_latin")

inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).logits

ids = torch.argmax(outputs, dim=-1)[0]
transcription = processor.decode(ids)

# Final Kurdish speech to Text
print(transcription)
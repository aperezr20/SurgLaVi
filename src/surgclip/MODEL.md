# SurgCLIP

Surgical dual-encoder video-language model.

## Installation

```bash
pip install surgclip
```

## Quickstart

### Video clip — from a frame path (loads neighbors automatically) -> RECOMMENDED

```python
import torch
import surgclip
from surgclip import VideoPreprocessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, tokenizer = surgclip.load("SurgCLIP-B", device=device)
labels = [
    "Prepares for surgery by inserting trocars into the patient's abdominal cavity",
    "Employs grasper and hook during calot triangle dissection, manipulating gallbladder to reveal hepatic triangle, cystic duct and cystic artery",
    "Utilizes clipper to secure cystic duct and artery, followed by precise dissection using scissors",
    "Utilizes a hook to dissect the connective tissue during the dissection phase, separating gallbladder from the liver",
    "Secures the removed gallbladder in the specimen bag during the packaging phase of the procedure",
    "Employs suction and irrigation techniques to maintain a clear surgical field during the clean and coagulation phase, simultaneously coagulating bleeding vessels",
    "Handles the specimen bag during the retraction",
]
tokens = surgclip.tokenize(labels, tokenizer, device=device)

# Offline: window centered on the anchor frame
proc = VideoPreprocessor(num_frames=16 , sample_rate=1, mode="centered")
video = proc("./cholec80/frames/video01/video01_000843.png").to(device)

# Online: anchor frame is the last in the window
proc = VideoPreprocessor(num_frames=16 , sample_rate=1, mode="online")
video = proc("./cholec80/frames/video01/video01_000843.png").to(device)

with torch.no_grad():
    logits, _ = model(video, tokens)
    probs = logits.softmax(dim=-1).cpu().numpy()


print("Phase probs:", probs)
max_prob = logits.argmax(dim=-1).cpu().numpy()
pred = [labels[i] for i in max_prob]
print("Prediction:", pred)
```


### Video clip — from a list of frames

```python
from surgclip import VideoPreprocessor
from PIL import Image

proc = VideoPreprocessor(num_frames=16, sample_type="uniform")
frames = [
    Image.open("./cholec80/frames/video01/video01_000842.png"), 
    Image.open("./cholec80/frames/video01/video01_000843.png"), ...]

video = proc(frames).to(device)  # (1, 16, 3, 224, 224)

with torch.no_grad():
    logits, _ = model(video, tokens)
    probs = logits.softmax(dim=-1).cpu().numpy()

print("Phase probs:", probs)
```

### Single image 
##### Single image inference is supported, but we highly recommend using video input for better performance

```python
from PIL import Image
img = preprocess(Image.open("./cholec80/frames/video01/video01_000843.png")).unsqueeze(0).unsqueeze(0).to(device)
tokens = surgclip.tokenize(labels, tokenizer, device=device)

with torch.no_grad():
    logits, _ = model(img, tokens)
    probs = logits.softmax(dim=-1).cpu().numpy()

print("Phase probs:", probs)
```

### Feature extraction

```python
import torch.nn.functional as F

with torch.no_grad():
    _, pooled_vision = model.encode_vision(video)   # (B, 768)
    _, pooled_text = model.encode_text(tokens)    # (B, 768)

    sim_v2t, sim_t2v = model.get_sim(
        model.vision_proj(pooled_vision),
        model.text_proj(pooled_text),
        temp=model.temp,
    )
```

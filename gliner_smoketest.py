import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("1) starting")
import torch
print("2) torch", torch.__version__, "cuda?", torch.cuda.is_available())

print("3) importing gliner")
from gliner import GLiNER
print("4) gliner imported")

print("5) loading model")
m = GLiNER.from_pretrained("urchade/gliner_base")
print("6) model loaded")

text = "Julio Gonzalo, Teresa Solfas, University of Bologna, Bologna Italy, julio@unibo.it"
labels = ["person", "organization", "location", "city", "country", "gpe"]
print("7) predicting")
out = m.predict_entities(text, labels)
print("8) done", out[:3])

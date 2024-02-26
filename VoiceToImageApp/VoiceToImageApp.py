from diffusers import DiffusionPipeline
import speech_recognition as sr
from googletrans import Translator
import torch


r=sr.Recognizer()
with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source)
    print("Sizi dinliyoruz..")
    data=r.record(source,duration=5)
    text=r.recognize_google(data,language="tr",show_all= False)
    print(text)


translator=Translator()
result=translator.translate(text,src="tr",dest="en")
print(result.text)


device = "cpu"
model_id = "runwayml/stable-diffusion-v1-5"


ldm = DiffusionPipeline.from_pretrained(model_id)
ldm = ldm.to(device)


prompt = result.text
image = ldm([prompt]).images[0]

image.save("sonuc.png")
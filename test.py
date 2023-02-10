import whisper

model = whisper.load_model("medium")
result = model.transcribe("hello.mp3")
print(result["text"])
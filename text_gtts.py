from gtts import gTTS

# Read the text from 'my_voice.txt'
with open('my_voice.txt', 'r') as file:
    text = file.read()

# Generate speech from the text
tts = gTTS(text, lang="en")

# Save the generated speech to an MP3 file
tts.save("voice_sample.mp3")

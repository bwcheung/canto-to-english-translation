# canto-to-english-translation
A personal project using a raspberry pi 5 and local LLMs to translate spoken Cantonese into English audio. The goal of the project was to learn how to use local LLMs and playing around with the raspberry pi 5.

Translation flow:

1. fine tuned whisper model for Contanese speech to text: https://huggingface.co/alvanlii/whisper-small-cantonese
2. Cantonese text to English text translation: https://ollama.com/7shi/llama-translate
3. English text to English speech: https://huggingface.co/rhasspy/piper-voices

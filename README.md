# Custom-AI-Agent-with-Memory
A smart voice + text AI assistant that remembers your conversations — built with React, FastAPI, LangChain, ChromaDB, and Hugging Face models.

This project is a context-aware AI assistant that can communicate through both text and voice, understand what you say using Whisper, generate intelligent responses using Hugging Face (LLaMA 3.1), and store important information persistently in a vector database (ChromaDB).

Every message you send — whether typed or spoken — is embedded into high-dimensional vectors using Sentence Transformers, allowing the assistant to recall relevant memories from previous conversations. This gives your assistant a human-like ability to remember past details and respond more personally over time.

The app’s frontend is built with React, using plain CSS for a clean, responsive, and modern design. On the backend, FastAPI handles the API routes for chat, memory, and voice transcription. The integration of LangChain makes the memory retrieval and prompt construction modular and powerful.

It’s essentially your own ChatGPT-style assistant, but one that remembers who you are.

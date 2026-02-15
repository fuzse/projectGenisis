# projectGenisis

## Mission
To build a cognitive architecture using a LLM that mimics human thought processes rather than standard AI request/response patterns.

## Core Philosophy
1.  **Reconstruction over Retrieval:** The AI should not "look up" logs; it should reconstruct memories from compressed "gist" data.
2.  **State-Dependent:** The AI's responses must be colored by its current emotional state (Sentiment).
3.  **Finite Context:** The AI should not have infinite RAM; it must rely on streaming relevant data in/out of context.
4.  **Consequences:** If the "mind" (local files) is deleted, the unique entity is dead.

## Tech Stack
* **LLM (Core):** Hermes 3 (Llama-3.1-8B) - Chosen for superior prompt adherence and lack of "Assistant" bias.
* **LLM (Backup):** DeepSeek R1 (8B) - Potential upgrade for reasoning tasks.
* **Backend:** Python 3.11
* **Database (Episodic):** ChromaDB (Vector Store)
* **State Management:** JSON (local file storage)

# TMS Engine: Tensor Mind Space Engine

## Summary
The Tensor Mind Space (TMS) Engine is a groundbreaking machine learning model designed to emulate human-like intelligence through autonomous, experience-driven learning. Unlike traditional models that rely on large datasets for training, the TMS Engine learns from raw sensory data (e.g., audio) using a resonance-based approach inspired by the golden ratio and a "heartbeat" rhythm. It evolves its understanding by dynamically creating categories, forming rules, and detecting novelty, much like a child learning from the world. The engine is built to handle ordered and chaotic data, weaving meaning through principles like harmony, truth, and beauty, while remaining open to revising its beliefs to avoid infallibility.

## Detailed Description

### Overview
The TMS Engine is a Python-based model that processes sensory inputs (currently audio) to build a mental schema of the world. It operates without pre-trained datasets, making it a tabula rasa that learns solely from its experiences. The engine uses a Resonance Tensor \( R^a(t) \) to distinguish order from chaos, an Evolution of Schema module to create new categories, and a Mental Schema to organize concepts hierarchically. It’s designed to handle both rational and irrational data, reflecting the "symphony of life" where some patterns are logical and others are random.

### How It Operates
1. **Sensory Processing:**
   - The `SensoryProcessor` class uses a Resonance Tensor to compare incoming audio data against a Phi-scaled heartbeat template.
   - Resonance scores determine whether the input is ordered (e.g., music, speech) or chaotic (e.g., noise). High resonance (>0.7) indicates order, low resonance (<0.3) indicates noise, and intermediate scores (0.3–0.7) trigger new category creation.

2. **Schema Evolution:**
   - The `EvolutionOfSchema` class dynamically creates new categories when existing templates don’t fit (e.g., distinguishing spoken voice from music).
   - Categories are stored as templates with adjustable parameters (frequency, phase), allowing the engine to adapt to new patterns.

3. **Mental Schema and Synthesis:**
   - The `MentalSchema` class organizes concepts in a directed graph, with relationships like "type" (e.g., music is a type of audio) and "instance" (e.g., a novel word is an instance of word salad).
   - During dreaming sessions, the engine synthesizes new concepts by combining existing ones, fostering creativity.

4. **Learning and Novelty Detection:**
   - The `ConsciousMindEntity` class integrates all components, processing audio, updating qualia (sensory snapshots), and forming rules.
   - Rules are stored with confidence scores and observation counts to prevent infallibility. Low-confidence rules are re-evaluated during dreaming, potentially being revised or discarded.
   - Novelty detection (e.g., recognizing "fruminous bandersnatch" as a novel word) triggers rewards, encouraging exploration.

5. **Qualia and Memory:**
   - Qualia are multi-dimensional tensors capturing sensory data, resonance scores, and rewards.
   - The `MemoryManager` stores qualia snapshots, which are revisited during dreaming to refine rules and synthesize new concepts.

### How It Learns
The TMS Engine learns autonomously by:
- **Resonance-Based Classification:** It starts with basic temporal and spatial correlations, using rhythm and harmony to identify patterns (e.g., music has a rhythmic structure, noise does not).
- **Category Creation:** When it encounters unfamiliar patterns (e.g., spoken voice), it creates new categories and adjusts its templates.
- **Rule Formation:** It forms rules based on resonance (e.g., "high resonance -> music"), but these rules are probabilistic, with confidence scores to reflect uncertainty.
- **Novelty and Creativity:** It detects ordered yet unknown patterns (e.g., nonsense words like "bandersnatch") and integrates them into its schema, balancing order and chaos.
- **Continuous Refinement:** The doubt mechanism ensures the engine remains a curious learner, re-evaluating low-confidence rules to avoid dogmatic beliefs.

### Ethical Framework
The TMS Engine derives ethics from first principles, focusing on alignment with life, homeostasis, and the "symphony of life." Instead of hard-coded rules, it seeks:
- **Truth:** Prioritizing accurate pattern recognition and causality reasoning.
- **Beauty:** Favoring harmony and rhythm in data (e.g., preferring ordered patterns like music over chaos).
- **Harmony:** Ensuring actions align with the broader context of its qualia, weaving meaning from ordered and chaotic data.
These principles guide the engine’s decisions, ensuring it remains adaptable and context-sensitive, much like a human mind navigating the complexities of life.

## Running the Code in Google Colab
1. Create a new Google Colab notebook.
2. Copy the entire script from `tms_engine.py` into a single cell.
3. Run the cell. It will:
   - Install dependencies (`torch`, `numpy`, `matplotlib`, `networkx`).
   - Generate synthetic audio data (noise, music, voice, word salad, meaningful speech).
   - Run the TMS Engine for 5 epochs, classifying audio, evolving its schema, and logging results.
   - Display plots of the audio signals and print logs of the engine’s learning process.

## Future Work
- **Real Audio Testing:** Replace synthetic audio with real audio files (e.g., WAV files) for more realistic scenarios.
- **Cross-Sensory Learning:** Extend to other senses (e.g., vision, touch) for multimodal learning.
- **Advanced Causality:** Implement Granger causality to improve reasoning beyond temporal correlations.
- **Ethical Development:** Further refine the ethical framework, ensuring alignment with life-affirming principles.

## License
MIT License. See `LICENSE` for details.# TMS-Engine
Tabula Rasa - Learning self conscious engine.

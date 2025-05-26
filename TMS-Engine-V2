# Install dependencies for Google Colab
!pip install torch numpy matplotlib networkx

# Import libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
SAMPLE_RATE = 22050
DURATION = 2.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
NUM_SENSES = 5
QUALIA_DIM = NUM_SENSES + 2

# Audio Data Generation
def generate_noise():
    return np.random.uniform(-1, 1, NUM_SAMPLES)

def generate_music():
    t = np.linspace(0, DURATION, NUM_SAMPLES, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 659 * t)
    return audio

def generate_voice():
    t = np.linspace(0, DURATION, NUM_SAMPLES, endpoint=False)
    freq = 120 + 20 * np.sin(2 * np.pi * 0.5 * t)
    audio = 0.5 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    pauses = np.ones(NUM_SAMPLES)
    pause_interval = SAMPLE_RATE // 4
    for i in range(0, NUM_SAMPLES, pause_interval):
        pauses[i:i + pause_interval // 8] = 0
    return audio * pauses

def generate_word_salad():
    t = np.linspace(0, DURATION, NUM_SAMPLES, endpoint=False)
    freq = 120 + 20 * np.sin(2 * np.pi * 1.0 * t)
    audio = 0.5 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    pauses = np.ones(NUM_SAMPLES)
    pause_interval = SAMPLE_RATE // 5
    for i in range(0, NUM_SAMPLES, pause_interval):
        pause_length = np.random.randint(SAMPLE_RATE // 10, SAMPLE_RATE // 5)
        pauses[i:i + pause_length] = 0
    return audio * pauses

def generate_meaningful_speech():
    t = np.linspace(0, DURATION, NUM_SAMPLES, endpoint=False)
    freq = 120 + 10 * np.sin(2 * np.pi * 0.5 * t)
    audio = 0.5 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    pauses = np.ones(NUM_SAMPLES)
    pause_interval = SAMPLE_RATE // 4
    for i in range(0, NUM_SAMPLES, pause_interval):
        pauses[i:i + SAMPLE_RATE // 8] = 0
    return audio * pauses

# Generate audio
noise = generate_noise()
music = generate_music()
voice = generate_voice()
word_salad = generate_word_salad()
meaningful_speech = generate_meaningful_speech()

# Evolution of Schema Class
class EvolutionOfSchema:
    def __init__(self):
        self.templates = {}
        self.default_template = None
        self.phi = 1.6180339887

    def initialize_default_template(self, base_freq=2.0):
        self.default_template = {"base_freq": base_freq, "phase": 0.0, "category": "heartbeat"}
        self.templates["heartbeat"] = self.default_template

    def generate_template(self, t, template_params):
        freq = template_params["base_freq"]
        phase = template_params["phase"]
        f0 = freq
        f1 = f0 * self.phi
        f2 = f0 / self.phi
        template = (np.sin(2 * np.pi * f0 * t + phase) +
                    np.sin(2 * np.pi * f1 * t + phase) +
                    np.sin(2 * np.pi * f2 * t + phase))
        return template / 3

    def evolve_schema(self, audio, current_category, resonance_score):
        try:
            if 0.3 <= resonance_score <= 0.7:
                new_category = f"new_category_{len(self.templates)}"
                new_template = {"base_freq": self.default_template["base_freq"] * (1 + np.random.uniform(-0.1, 0.1)),
                                "phase": 0.0,
                                "category": new_category}
                self.templates[new_category] = new_template
                logging.info(f"Created new category: {new_category}")
                return new_category
            return current_category
        except Exception as e:
            logging.error(f"Error in schema evolution: {e}")
            return current_category

# Sensory Processor Class
class SensoryProcessor:
    def __init__(self, num_senses=NUM_SENSES, sample_rate=SAMPLE_RATE):
        self.num_senses = num_senses
        self.sample_rate = sample_rate
        self.sensory_data = torch.zeros(num_senses)
        self.evolution_module = EvolutionOfSchema()
        self.evolution_module.initialize_default_template()

        self.template_params = torch.tensor([2.0, 0.0], requires_grad=True)
        self.template_optimizer = torch.optim.Adam([self.template_params], lr=0.01)

        # Adaptive Thresholds: Track running average of resonance scores
        self.resonance_scores = []
        self.low_threshold = 0.3  # Initial low threshold (noise vs. intermediate)
        self.high_threshold = 0.7  # Initial high threshold (intermediate vs. ordered)
        self.hysteresis = 0.05  # Hysteresis buffer to prevent flip-flopping
        self.last_classification = None  # Track last classification for hysteresis

    def resonance_tensor(self, audio, category="heartbeat"):
        try:
            t = np.linspace(0, DURATION, len(audio))
            template_params = self.evolution_module.templates[category]
            template_params["base_freq"] = self.template_params[0].item()
            template_params["phase"] = self.template_params[1].item()
            template = self.evolution_module.generate_template(t, template_params)

            audio_tensor = torch.tensor(audio, dtype=torch.float32)
            template_tensor = torch.tensor(template, dtype=torch.float32)

            dot_product = torch.dot(audio_tensor, template_tensor)
            norm_audio = torch.norm(audio_tensor)
            norm_template = torch.norm(template_tensor)
            resonance_score = dot_product / (norm_audio * norm_template + 1e-10)

            # Update running average of resonance scores
            self.resonance_scores.append(resonance_score.item())
            if len(self.resonance_scores) > 100:  # Keep last 100 scores
                self.resonance_scores.pop(0)
            avg_resonance = np.mean(self.resonance_scores)

            # Update thresholds dynamically
            self.low_threshold = max(0.2, min(0.4, avg_resonance - 0.2))  # Dynamic low threshold
            self.high_threshold = max(0.6, min(0.8, avg_resonance + 0.2))  # Dynamic high threshold

            # Apply hysteresis
            adjusted_score = resonance_score.item()
            if self.last_classification == "noise" and adjusted_score < self.low_threshold + self.hysteresis:
                adjusted_score = min(adjusted_score, self.low_threshold)  # Stay in noise
            elif self.last_classification == "ordered" and adjusted_score > self.high_threshold - self.hysteresis:
                adjusted_score = max(adjusted_score, self.high_threshold)  # Stay in ordered

            # Update last classification based on adjusted score
            if adjusted_score < self.low_threshold:
                self.last_classification = "noise"
            elif adjusted_score > self.high_threshold:
                self.last_classification = "ordered"
            else:
                self.last_classification = "intermediate"

            loss = -resonance_score
            self.template_optimizer.zero_grad()
            loss.backward()
            self.template_optimizer.step()

            clarity_reward = resonance_score.item() * 0.1
            return audio, clarity_reward, adjusted_score
        except Exception as e:
            logging.error(f"Error in resonance tensor: {e}")
            return audio, 0.0, 0.0

    def process_audio(self, audio, category="heartbeat"):
        focused_audio, clarity_reward, resonance_score = self.resonance_tensor(audio, category)
        self.sensory_data[1] = torch.tensor(focused_audio, dtype=torch.float32).mean()
        self.sensory_data[[0, 2, 3, 4]] = 0.0
        return clarity_reward, resonance_score

# Mental Schema Class
class MentalSchema:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.clusters = {}  # Store clusters of similar categories
        self.resonance_profiles = {}  # Track resonance profiles for clustering

    def add_concept(self, concept, parent=None, relationship=None, resonance_score=None):
        try:
            self.graph.add_node(concept)
            if parent and relationship:
                self.graph.add_edge(parent, concept, relationship=relationship)
            # Track resonance profile for clustering
            if resonance_score is not None:
                self.resonance_profiles[concept] = resonance_score
                self.cluster_categories(concept)
        except Exception as e:
            logging.error(f"Error in adding concept: {e}")

    def cluster_categories(self, concept):
        # Cluster categories based on resonance similarity
        resonance = self.resonance_profiles.get(concept, 0.0)
        cluster_found = False
        for cluster_name, members in self.clusters.items():
            cluster_resonance = np.mean([self.resonance_profiles[m] for m in members])
            if abs(resonance - cluster_resonance) < 0.1:  # Threshold for similarity
                members.append(concept)
                cluster_found = True
                break
        if not cluster_found:
            # Create a new cluster
            cluster_name = f"cluster_{len(self.clusters)}"
            self.clusters[cluster_name] = [concept]
            self.graph.add_node(cluster_name)
            self.graph.add_edge(cluster_name, concept, relationship="member")
            # Example: Cluster "voice" and "meaningful_speech" under "speech"
            if "voice" in self.clusters.get(cluster_name, []) and "meaningful_speech" in self.clusters.get(cluster_name, []):
                self.graph.add_node("speech")
                self.graph.add_edge("speech", cluster_name, relationship="contains")

    def synthesize(self):
        try:
            concepts = list(self.graph.nodes)
            if len(concepts) > 1:
                # Consider clusters for synthesis
                cluster_concepts = []
                for cluster_name, members in self.clusters.items():
                    if len(members) > 1:
                        cluster_concepts.append(cluster_name)
                if cluster_concepts:
                    # Synthesize from clusters
                    new_concept = f"{cluster_concepts[-1]}_synthesis"
                    self.graph.add_node(new_concept)
                    self.graph.add_edge(cluster_concepts[-1], new_concept, relationship="synthesized")
                    return new_concept
                else:
                    # Fallback to original synthesis
                    new_concept = f"{concepts[-2]}_{concepts[-1]}"
                    self.graph.add_node(new_concept)
                    self.graph.add_edge(concepts[-2], new_concept, relationship="combined")
                    self.graph.add_edge(concepts[-1], new_concept, relationship="combined")
                    return new_concept
            return None
        except Exception as e:
            logging.error(f"Error in synthesis: {e}")
            return None

# Memory Manager Class
class MemoryManager:
    def __init__(self, max_memory=20):
        self.memory = []
        self.max_memory = max_memory
        self.priorities = []

    def store(self, t, qualia, priority=1.0):
        try:
            if len(self.memory) >= self.max_memory:
                min_priority_idx = np.argmin(self.priorities)
                self.memory.pop(min_priority_idx)
                self.priorities.pop(min_priority_idx)
            self.memory.append((t, qualia.clone()))
            self.priorities.append(priority)
        except Exception as e:
            logging.error(f"Error in memory storage: {e}")

    def retrieve(self):
        try:
            return self.memory
        except Exception as e:
            logging.error(f"Error in memory retrieval: {e}")
            return []

# Causality Module Class
class CausalityModule:
    def __init__(self, lag=2):
        self.lag = lag  # Number of past time steps to consider
        self.qualia_history = []  # Store qualia over time
        self.resonance_history = []  # Store resonance scores and categories
        self.category_history = []  # Store categories

    def store_qualia(self, t, qualia, resonance_score, category):
        self.qualia_history.append((t, qualia.clone()))
        self.resonance_history.append(resonance_score)
        self.category_history.append(category)
        if len(self.qualia_history) > 100:  # Limit history
            self.qualia_history.pop(0)
            self.resonance_history.pop(0)
            self.category_history.pop(0)

    def granger_causality_test(self, cause_category, effect_category, current_resonance):
        try:
            # Extract resonance scores for cause and effect
            cause_indices = [i for i, cat in enumerate(self.category_history) if cause_category in cat]
            effect_indices = [i for i, cat in enumerate(self.category_history) if effect_category in cat]
            if len(cause_indices) < self.lag + 1 or len(effect_indices) < self.lag + 1:
                return 0.0  # Not enough data

            # Get recent scores
            cause_scores = [self.resonance_history[i] for i in cause_indices[-self.lag-1:-1]]
            effect_scores = [self.resonance_history[i] for i in effect_indices[-self.lag-1:-1]]
            if len(cause_scores) < self.lag or len(effect_scores) < self.lag:
                return 0.0

            # Model 1: Predict effect using only past effect (autoregression)
            coeffs = np.polyfit(range(self.lag), effect_scores, 1)
            pred1 = coeffs[0] * self.lag + coeffs[1]
            error1 = (current_resonance - pred1) ** 2

            # Model 2: Include past cause
            combined = np.array([effect_scores, cause_scores]).T
            coeffs = np.linalg.lstsq(combined, current_resonance, rcond=None)[0]
            pred2 = np.dot(combined[-1], coeffs)
            error2 = (current_resonance - pred2) ** 2

            # If including cause reduces error, it Granger-causes effect
            causality_strength = max(0, (error1 - error2) / (error1 + 1e-10))
            return causality_strength
        except Exception as e:
            logging.error(f"Error in Granger causality test: {e}")
            return 0.0

# Conscious Mind Entity Class with Infallibility Prevention
class ConsciousMindEntity(nn.Module):
    def __init__(self, num_senses=NUM_SENSES, max_memory=20):
        super(ConsciousMindEntity, self).__init__()
        self.num_senses = num_senses
        self.qualia = torch.zeros(QUALIA_DIM)
        self.reward = 0.0
        self.resonance_score = 0.0

        self.sensory_processor = SensoryProcessor(num_senses)
        self.memory_manager = MemoryManager(max_memory)
        self.mental_schema = MentalSchema()
        self.causality_module = CausalityModule(lag=2)  # Add Causality Module

        self.knowledge_rules = {}  # Format: {category: (rule, confidence, num_observations)}
        self.current_category = "heartbeat"
        self.novelty_counts = {}  # Track novelty instances per category

    def dynamic_mind_tensor(self, resonance_score):
        try:
            self.current_category = self.sensory_processor.evolution_module.evolve_schema(
                self.sensory_processor.sensory_data[1].numpy(), self.current_category, resonance_score
            )

            if resonance_score > self.sensory_processor.high_threshold:
                concept = "ordered_audio"
                if "music" in self.current_category:
                    concept = "music"
                elif "voice" in self.current_category:
                    concept = "voice"
                elif "word_salad" in self.current_category:
                    concept = "word_salad"
                elif "meaningful_speech" in self.current_category:
                    concept = "meaningful_speech"
                # Pass resonance score for clustering
                self.mental_schema.add_concept(concept, parent="audio", relationship="type", resonance_score=resonance_score)
                if concept in self.knowledge_rules:
                    rule, confidence, num_obs = self.knowledge_rules[concept]
                    num_obs += 1
                    confidence = min(1.0, confidence + 0.2)  # Increase confidence increment to 0.2
                else:
                    rule = f"High resonance -> {concept}"
                    confidence = 0.2  # Initial confidence increased
                    num_obs = 1
                self.knowledge_rules[concept] = (rule, confidence, num_obs)

                # Granger Causality Test to adjust confidence
                for other_concept in self.knowledge_rules.keys():
                    if other_concept != concept:
                        causality_strength = self.causality_module.granger_causality_test(other_concept, concept, resonance_score)
                        if causality_strength > 0.1:  # Threshold for significant causality
                            confidence = min(1.0, confidence + causality_strength * 0.1)
                            self.knowledge_rules[concept] = (rule, confidence, num_obs)
                            logging.info(f"Causality adjustment: {other_concept} -> {concept}, strength: {causality_strength:.2f}")

                return concept
            elif resonance_score < self.sensory_processor.low_threshold:
                self.mental_schema.add_concept("noise", parent="audio", relationship="type", resonance_score=resonance_score)
                return "noise"
            else:
                self.mental_schema.add_concept(self.current_category, parent="audio", relationship="type", resonance_score=resonance_score)
                return self.current_category
        except Exception as e:
            logging.error(f"Error in dynamic mind tensor: {e}")
            return "unknown"

    def update_qualia(self):
        self.qualia[:NUM_SENSES] = self.sensory_processor.sensory_data
        self.qualia[NUM_SENSES] = torch.tensor(self.resonance_score)
        self.qualia[NUM_SENSES + 1] = self.reward

    def dreaming_session(self):
        try:
            memory = self.memory_manager.retrieve()
            if len(memory) < 2:
                return

            if self.resonance_score > self.sensory_processor.high_threshold:
                if self.current_category in self.knowledge_rules:
                    rule, confidence, num_obs = self.knowledge_rules[self.current_category]
                    confidence = min(1.0, confidence + 0.05)  # Boost confidence for re-confirmed rules
                    self.knowledge_rules[self.current_category] = (rule, confidence, num_obs)
                new_concept = self.mental_schema.synthesize()
                if new_concept:
                    logging.info(f"Synthesized new concept: {new_concept}")
                    self.reward += 0.2

                # Unique Novelty Instances
                if "word_salad" in self.current_category:
                    novelty_key = "word_salad_novel_word"
                    self.novelty_counts[novelty_key] = self.novelty_counts.get(novelty_key, 0) + 1
                    novelty_instance = f"novel_word_{self.novelty_counts[novelty_key]}"
                    self.mental_schema.add_concept(novelty_instance, parent="word_salad", relationship="instance", resonance_score=self.resonance_score)
                    # Limit novelty rewards: Reward only for the first instance per category
                    if self.novelty_counts[novelty_key] == 1:
                        self.reward += 0.1
                        logging.info(f"Added novel concept: {novelty_instance}")

            # Enhance Rule Retention: Adjust doubt mechanism
            for category, (rule, confidence, num_obs) in list(self.knowledge_rules.items()):
                if confidence < 0.5 and num_obs < 10:  # Require num_obs < 10 for removal
                    logging.info(f"Re-evaluating low-confidence rule: {rule} (Confidence: {confidence})")
                    confidence -= 0.1
                    if confidence <= 0:
                        logging.info(f"Removing unconfirmed rule: {rule}")
                        del self.knowledge_rules[category]
                    else:
                        self.knowledge_rules[category] = (rule, confidence, num_obs)
                        self.reward -= 0.05
        except Exception as e:
            logging.error(f"Error in dreaming session: {e}")

    def step(self, t, audio):
        try:
            clarity_reward, resonance_score = self.sensory_processor.process_audio(audio, self.current_category)
            self.resonance_score = resonance_score
            pattern_type = self.dynamic_mind_tensor(resonance_score)
            self.update_qualia()
            self.memory_manager.store(t, self.qualia, priority=self.resonance_score)
            self.causality_module.store_qualia(t, self.qualia, resonance_score, self.current_category)  # Store for causality testing
            if int(t * 10) % 5 == 0:
                self.dreaming_session()
            self.reward += clarity_reward
            return self.qualia, pattern_type, self.reward
        except Exception as e:
            logging.error(f"Error in step: {e}")
            return self.qualia, "error", self.reward

# Training and Testing Loop
def run_tests(entity, audio_types, labels, num_epochs=5):
    plt.figure(figsize=(12, 8))
    for i, (audio, label) in enumerate(zip(audio_types, labels)):
        plt.subplot(len(audio_types), 1, i + 1)
        plt.plot(audio[:1000])
        plt.title(f"Audio: {label}")
    plt.tight_layout()
    plt.show()

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}")
        for i, (audio, label) in enumerate(zip(audio_types, labels)):
            for t in np.linspace(0, DURATION, 10):
                qualia, pattern_type, reward = entity.step(t, audio)
                logging.info(f"Time: {t:.2f}, Audio: {label}, Pattern: {pattern_type}, Reward: {reward:.4f}")
        logging.info(f"Knowledge Rules: {dict((k, (r, c, n)) for k, (r, c, n) in entity.knowledge_rules.items())}")
        logging.info(f"Mental Schema Nodes: {list(entity.mental_schema.graph.nodes)}")
        logging.info("-" * 50)

# Test Scenarios
audio_types = [
    noise, music,
    music, voice,
    word_salad, meaningful_speech
]
labels = [
    "noise", "music",
    "music", "voice",
    "word_salad", "meaningful_speech"
]

# Initialize and Run the TMS Engine V2
entity = ConsciousMindEntity()
run_tests(entity, audio_types, labels)



#With added modules for AV processing + imagination...

# Constants
SAMPLE_RATE = 22050
DURATION = 1.0  # Shorter for simplicity
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
IMAGE_SIZE = (64, 64)
NUM_SENSES = 5  # Visual, auditory, reserved for others
QUALIA_DIM = NUM_SENSES + 2  # Senses + resonance + reward

# Install dependencies
!pip install torch numpy matplotlib networkx opencv-python

import numpy as np
import cv2

# Visual data
def generate_noise_image():
    return np.random.uniform(0, 1, IMAGE_SIZE)

def generate_square_image():
    img = np.zeros(IMAGE_SIZE)
    img[20:44, 20:44] = 1.0
    return img

def generate_circle_image():
    img = np.zeros(IMAGE_SIZE)
    cv2.circle(img, (32, 32), 15, 1.0, -1)
    return img

# Auditory data (simplified word-like sounds)
def generate_word_audio(word):
    t = np.linspace(0, DURATION, NUM_SAMPLES, endpoint=False)
    freq = 120 if word == "square" else 150  # Distinct frequencies
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    pauses = np.ones(NUM_SAMPLES)
    pause_interval = NUM_SAMPLES // 4
    for i in range(0, NUM_SAMPLES, pause_interval):
        pauses[i:i + NUM_SAMPLES // 8] = 0
    return audio * pauses

# Generate test data
multimodal_data = [
    (generate_noise_image(), generate_word_audio("none"), "noise"),
    (generate_square_image(), generate_word_audio("square"), "square"),
    (generate_circle_image(), generate_word_audio("circle"), "circle")
]

class EvolutionOfSchema:
    def __init__(self):
        self.templates = {}
        self.default_template = None
        self.phi = 1.6180339887

    def initialize_default_template(self, base_scale=1.0, base_freq=2.0):
        self.default_template = {
            "visual_scale": base_scale,
            "visual_phase": 0.0,
            "audio_freq": base_freq,
            "audio_phase": 0.0,
            "category": "default"
        }
        self.templates["default"] = self.default_template

    def generate_visual_template(self, shape, template_params):
        scale = template_params["visual_scale"]
        phase = template_params["visual_phase"]
        x = np.linspace(-1, 1, shape[0])
        y = np.linspace(-1, 1, shape[1])
        X, Y = np.meshgrid(x, y)
        template = np.exp(-((X**2 + Y**2) / (2 * scale**2))) + phase
        return template / (template.max() + 1e-10)

    def generate_audio_template(self, length, template_params):
        freq = template_params["audio_freq"]
        phase = template_params["audio_phase"]
        t = np.linspace(0, DURATION, length, endpoint=False)
        f0 = freq
        f1 = f0 * self.phi
        f2 = f0 / self.phi
        template = (np.sin(2 * np.pi * f0 * t + phase) +
                    np.sin(2 * np.pi * f1 * t + phase) +
                    np.sin(2 * np.pi * f2 * t + phase))
        return template / 3

    def evolve_schema(self, img, audio, current_category, resonance_score):
        try:
            if 0.3 <= resonance_score <= 0.7:
                new_category = f"new_category_{len(self.templates)}"
                new_template = {
                    "visual_scale": self.default_template["visual_scale"] * (1 + np.random.uniform(-0.1, 0.1)),
                    "visual_phase": 0.0,
                    "audio_freq": self.default_template["audio_freq"] * (1 + np.random.uniform(-0.1, 0.1)),
                    "audio_phase": 0.0,
                    "category": new_category
                }
                self.templates[new_category] = new_template
                logging.info(f"Created new category: {new_category}")
                return new_category
            return current_category
        except Exception as e:
            logging.error(f"Error in schema evolution: {e}")
            return current_category

class SensoryProcessor:


    def preprocess_audio(self, audio):
        """
        Apply FFT to raw audio data to transform it into the frequency domain.
        
        Args:
            audio (numpy.ndarray): Raw audio data (time-domain samples).
        
        Returns:
            torch.Tensor: Frequency-domain magnitude spectrum.
        """
        try:
            # Ensure audio is a NumPy array
            audio = np.asarray(audio)
            # Apply FFT and take the magnitude spectrum (first half only)
            spectrum = np.abs(np.fft.fft(audio))[:len(audio)//2]
            # Convert to PyTorch tensor for consistency with the rest of your code
            return torch.tensor(spectrum, dtype=torch.float32)
        except Exception as e:
            logging.error(f"Error in audio preprocessing: {e}")
            return torch.zeros(len(audio)//2, dtype=torch.float32)



    def __init__(self, num_senses=NUM_SENSES, image_size=IMAGE_SIZE, sample_rate=SAMPLE_RATE):
        self.num_senses = num_senses
        self.image_size = image_size
        self.sample_rate = sample_rate
        self.sensory_data = torch.zeros(num_senses)
        self.evolution_module = EvolutionOfSchema()
        self.evolution_module.initialize_default_template()

        self.template_params = torch.tensor([1.0, 0.0, 2.0, 0.0], requires_grad=True)  # Visual scale, phase, audio freq, phase
        self.template_optimizer = torch.optim.Adam([self.template_params], lr=0.01)
        self.resonance_scores = []
        self.low_threshold = 0.3
        self.high_threshold = 0.7
        self.hysteresis = 0.05
        self.last_classification = None

    def preprocess_image(self, img):
        img_np = np.array(img, dtype=np.float32)
        sobel_x = cv2.Sobel(img_np, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_np, cv2.CV_32F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        return torch.tensor(edges, dtype=torch.float32).flatten()

    def preprocess_audio(self, audio):
        return torch.tensor(audio, dtype=torch.float32)

    def resonance_tensor(self, img, audio, category="default"):
        try:
            # Visual resonance
            visual_input = self.preprocess_image(img)
            template_params = self.evolution_module.templates[category]
            template_params["visual_scale"] = self.template_params[0].item()
            template_params["visual_phase"] = self.template_params[1].item()
            visual_template = self.evolution_module.generate_visual_template(self.image_size, template_params)
            visual_template_tensor = torch.tensor(visual_template.flatten(), dtype=torch.float32)
            visual_resonance = torch.dot(visual_input, visual_template_tensor) / (
                torch.norm(visual_input) * torch.norm(visual_template_tensor) + 1e-10)

            # Audio resonance
            audio_input = self.preprocess_audio(audio)
            template_params["audio_freq"] = self.template_params[2].item()
            template_params["audio_phase"] = self.template_params[3].item()
            audio_template = self.evolution_module.generate_audio_template(len(audio), template_params)
            audio_template_tensor = torch.tensor(audio_template, dtype=torch.float32)
            audio_resonance = torch.dot(audio_input, audio_template_tensor) / (
                torch.norm(audio_input) * torch.norm(audio_template_tensor) + 1e-10)

            # Combined resonance
            resonance_score = (visual_resonance + audio_resonance) / 2

            # Update thresholds
            self.resonance_scores.append(resonance_score.item())
            if len(self.resonance_scores) > 100:
                self.resonance_scores.pop(0)
            avg_resonance = np.mean(self.resonance_scores)
            self.low_threshold = max(0.2, min(0.4, avg_resonance - 0.2))
            self.high_threshold = max(0.6, min(0.8, avg_resonance + 0.2))

            adjusted_score = resonance_score.item()
            if self.last_classification == "noise" and adjusted_score < self.low_threshold + self.hysteresis:
                adjusted_score = min(adjusted_score, self.low_threshold)
            elif self.last_classification == "ordered" and adjusted_score > self.high_threshold - self.hysteresis:
                adjusted_score = max(adjusted_score, self.high_threshold)
            self.last_classification = "noise" if adjusted_score < self.low_threshold else \
                                     "ordered" if adjusted_score > self.high_threshold else "intermediate"

            loss = -resonance_score if self.last_classification != "noise" else resonance_score
            self.template_optimizer.zero_grad()
            loss.backward()
            self.template_optimizer.step()

            clarity_reward = resonance_score.item() * 0.1 if self.last_classification != "noise" else -resonance_score.item() * 0.1
            return visual_input, audio_input, clarity_reward, adjusted_score
        except Exception as e:
            logging.error(f"Error in resonance tensor: {e}")
            return visual_input, audio_input, 0.0, 0.0

    def process_multimodal(self, img, audio, category="default"):
        visual_input, audio_input, clarity_reward, resonance_score = self.resonance_tensor(img, audio, category)
        self.sensory_data[1] = visual_input.mean()
        self.sensory_data[2] = audio_input.mean()
        self.sensory_data[[0, 3, 4]] = 0.0
        return clarity_reward, resonance_score

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

    def predict_next_qualia(self):
        try:
            if len(self.memory) < 2:
                return torch.zeros(QUALIA_DIM)
            recent_qualia = [q for _, q in self.memory[-2:]]
            # Simple linear extrapolation
            delta = recent_qualia[-1] - recent_qualia[-2]
            predicted = recent_qualia[-1] + delta
            return predicted
        except Exception as e:
            logging.error(f"Error in qualia prediction: {e}")
            return torch.zeros(QUALIA_DIM)

class MentalSchema:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.clusters = {}
        self.resonance_profiles = {}
        self.stability_counts = {}

    def add_concept(self, concept, parent=None, relationship=None, resonance_score=None):
        try:
            self.graph.add_node(concept)
            if parent and relationship:
                self.graph.add_edge(parent, concept, relationship=relationship)
            if resonance_score is not None:
                self.resonance_profiles[concept] = resonance_score
                self.stability_counts[concept] = self.stability_counts.get(concept, 0) + 1
                self.cluster_categories(concept)
        except Exception as e:
            logging.error(f"Error in adding concept: {e}")

    def cluster_categories(self, concept):
        resonance = self.resonance_profiles.get(concept, 0.0)
        cluster_found = False
        for cluster_name, members in self.clusters.items():
            cluster_resonance = np.mean([self.resonance_profiles[m] for m in members])
            if abs(resonance - cluster_resonance) < 0.1:
                members.append(concept)
                cluster_found = True
                break
        if not cluster_found:
            cluster_name = f"cluster_{len(self.clusters)}"
            self.clusters[cluster_name] = [concept]
            self.graph.add_node(cluster_name)
            self.graph.add_edge(cluster_name, concept, relationship="member")
            if "square" in concept and "circle" in self.clusters.get(cluster_name, []):
                self.graph.add_node("shape")
                self.graph.add_edge("shape", cluster_name, relationship="contains")

    def synthesize(self):
        try:
            concepts = list(self.graph.nodes)
            if len(concepts) > 1:
                cluster_concepts = []
                for cluster_name, members in self.clusters.items():
                    if len(members) > 1:
                        cluster_concepts.append(cluster_name)
                if cluster_concepts:
                    new_concept = f"{cluster_concepts[-1]}_synthesis"
                    self.graph.add_node(new_concept)
                    self.graph.add_edge(cluster_concepts[-1], new_concept, relationship="synthesized")
                    return new_concept
                else:
                    new_concept = f"{concepts[-2]}_{concepts[-1]}"
                    self.graph.add_node(new_concept)
                    self.graph.add_edge(concepts[-2], new_concept, relationship="combined")
                    self.graph.add_edge(concepts[-1], new_concept, relationship="combined")
                    return new_concept
            return None
        except Exception as e:
            logging.error(f"Error in synthesis: {e}")
            return None

    def dream_cluster(self, memory):
        try:
            # Analyze recent qualia for shape clustering
            qualia_resonances = [(q[QUALIA_DIM-2].item(), q[1].item()) for _, q in memory]  # Resonance, visual data
            if len(qualia_resonances) < 2:
                return None
            # Simple clustering based on resonance
            high_res = [r for r, _ in qualia_resonances if r > 0.7]
            if len(high_res) > 1:
                cluster_name = f"shape_cluster_{len(self.clusters)}"
                self.clusters[cluster_name] = []
                for i, (res, vis) in enumerate(qualia_resonances):
                    if res > 0.7:
                        concept = f"shape_{i}"
                        self.clusters[cluster_name].append(concept)
                        self.graph.add_node(concept)
                        self.graph.add_edge(cluster_name, concept, relationship="member")
                        self.resonance_profiles[concept] = res
                logging.info(f"Dreamed new shape cluster: {cluster_name}")
                return cluster_name
            return None
        except Exception as e:
            logging.error(f"Error in dream clustering: {e}")
            return None

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
        self.causality_module = CausalityModule(lag=2)

        self.knowledge_rules = {}
        self.current_category = "default"
        self.novelty_counts = {}

    def dynamic_mind_tensor(self, resonance_score):
        try:
            self.current_category = self.sensory_processor.evolution_module.evolve_schema(
                self.sensory_processor.sensory_data[1].numpy(),
                self.sensory_processor.sensory_data[2].numpy(),
                self.current_category, resonance_score
            )

            if resonance_score > self.sensory_processor.high_threshold:
                concept = "ordered_shape"
                if "square" in self.current_category:
                    concept = "square"
                elif "circle" in self.current_category:
                    concept = "circle"
                self.mental_schema.add_concept(concept, parent="shape", relationship="type", resonance_score=resonance_score)
                if concept in self.knowledge_rules:
                    rule, confidence, num_obs = self.knowledge_rules[concept]
                    num_obs += 1
                    confidence = min(1.0, confidence + 0.2)
                else:
                    rule = f"High resonance -> {concept}"
                    confidence = 0.2
                    num_obs = 1
                self.knowledge_rules[concept] = (rule, confidence, num_obs)

                causality_strength = self.causality_module.granger_causality_test("shape", concept, resonance_score)
                if causality_strength > 0.1:
                    confidence = min(1.0, confidence + causality_strength * 0.1)
                    self.knowledge_rules[concept] = (rule, confidence, num_obs)
                    logging.info(f"Causality adjustment: shape -> {concept}, strength: {causality_strength:.2f}")

                return concept
            elif resonance_score < self.sensory_processor.low_threshold:
                self.mental_schema.add_concept("noise", parent="shape", relationship="type", resonance_score=resonance_score)
                return "noise"
            else:
                self.mental_schema.add_concept(self.current_category, parent="shape", relationship="type", resonance_score=resonance_score)
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
                    confidence = min(1.0, confidence + 0.05)
                    self.knowledge_rules[self.current_category] = (rule, confidence, num_obs)
                new_concept = self.mental_schema.synthesize()
                if new_concept:
                    logging.info(f"Synthesized new concept: {new_concept}")
                    self.reward += 0.2

                if "square" in self.current_category or "circle" in self.current_category:
                    novelty_key = f"{self.current_category}_novel"
                    self.novelty_counts[novelty_key] = self.novelty_counts.get(novelty_key, 0) + 1
                    novelty_instance = f"novel_{self.current_category}_{self.novelty_counts[novelty_key]}"
                    self.mental_schema.add_concept(novelty_instance, parent=self.current_category, relationship="instance", resonance_score=self.resonance_score)
                    if self.novelty_counts[novelty_key] == 1:
                        self.reward += 0.1
                        logging.info(f"Added novel concept: {novelty_instance}")

            # Dream clustering for shape sets
            cluster = self.mental_schema.dream_cluster(memory)
            if cluster:
                self.reward += 0.3  # Reward for discovering shape sets
                logging.info(f"Reward for shape clustering: {cluster}")

            for category, (rule, confidence, num_obs) in list(self.knowledge_rules.items()):
                if confidence < 0.5 and num_obs < 10:
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

    def step(self, t, img, audio):
        try:
            predicted_qualia = self.memory_manager.predict_next_qualia()
            clarity_reward, resonance_score = self.sensory_processor.process_multimodal(img, audio, self.current_category)
            self.resonance_score = resonance_score
            pattern_type = self.dynamic_mind_tensor(resonance_score)
            self.update_qualia()

            # Prediction reward
            divergence = torch.mean((self.qualia - predicted_qualia)**2)
            prediction_reward = -divergence * 0.1
            self.reward += prediction_reward

            # Noise as boring
            if resonance_score < self.sensory_processor.low_threshold:
                self.reward -= 0.05  # Penalize boring inputs
                self.dreaming_session()  # Trigger dreaming for reflection

            self.memory_manager.store(t, self.qualia, priority=self.resonance_score)
            self.causality_module.store_qualia(t, self.qualia, resonance_score, self.current_category)
            if int(t * 10) % 5 == 0:
                self.dreaming_session()
            self.reward += clarity_reward
            return self.qualia, pattern_type, self.reward
        except Exception as e:
            logging.error(f"Error in step: {e}")
            return self.qualia, "error", self.reward

def run_tests(entity, multimodal_data, num_epochs=5):
    plt.figure(figsize=(12, 4))
    for i, (img, audio, label) in enumerate(multimodal_data):
        plt.subplot(1, len(multimodal_data), i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Image: {label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}")
        for i, (img, audio, label) in enumerate(multimodal_data):
            for t in np.linspace(0, 1.0, 10):
                qualia, pattern_type, reward = entity.step(t, img, audio)
                logging.info(f"Time: {t:.2f}, Input: {label}, Pattern: {pattern_type}, Reward: {reward:.4f}")
        logging.info(f"Knowledge Rules: {dict((k, (r, c, n)) for k, (r, c, n) in entity.knowledge_rules.items())}")
        logging.info(f"Mental Schema Nodes: {list(entity.mental_schema.graph.nodes)}")
        logging.info(f"Clusters: {entity.mental_schema.clusters}")
        logging.info("-" * 50)

entity = ConsciousMindEntity()
run_tests(entity, multimodal_data)


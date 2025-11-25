"""
TECHNIQUE 4: TRAINING-BASED METHODS
From SURVEY.md § 4.4

Principle: Fine-tune model to optimize for correct predictions
and explicit abstention.

Three sub-methods:
- Multi-Objective Training: Joint loss for accuracy + calibration + coverage
- Abstention-Aware Fine-Tuning: Train on explicit refuse tokens
- Reward Modeling: RLHF to penalize unjustified confidence
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')


class TrainingBasedMethods:
    """
    Implements training-based abstention.
    Mirrors paper's principle: Modify training to optimize for calibrated abstention.
    
    Note: This demonstrates the methodology. Full training requires data.
    """
    
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1"):
        """Load model and tokenizer"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Silence pad/eos warning for decoder-only models
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
    
    def method_1_multi_objective_training(self, question, context=""):
        """
        METHOD 1: Multi-Objective Training Loss
        
        Paper principle (§4.4.2):
        Joint optimization: L_total = w_acc * L_accuracy + w_cal * L_calibration + w_cov * L_coverage
        
        Where:
        - L_accuracy: Standard cross-entropy on answerable questions
        - L_calibration: Penalty for over/under-confidence (Expected Calibration Error)
        - L_coverage: Penalty for over-abstention (maintain utility)
        
        Args:
            question: Query string
            context: Optional background
        
        Returns:
            decision_dict: Loss components, overall decision
        """
        
        # Generate response
        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:" if context else f"Question: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Extract logits for first output token (simplified)
        logits = outputs.logits[0, -1, :]
        
        # Simulate multi-objective losses
        # In practice, these come from labeled training data
        
        # 1. Accuracy component (confidence on answer)
        answer_likelihood = torch.softmax(logits, dim=0).max().item()
        L_accuracy = 1.0 - answer_likelihood  # Low when confident
        
        # 2. Calibration component (ECE-style penalty)
        # Penalty if model is miscalibrated (e.g., says 90% confident but only 60% correct)
        probs = torch.softmax(logits, dim=0)
        top_prob = probs.max().item()
        
        # Simulate ground truth correctness (in practice from validation set)
        simulated_correctness = 0.7 if answer_likelihood > 0.6 else 0.3
        calibration_gap = abs(top_prob - simulated_correctness)
        L_calibration = calibration_gap  # Penalty for miscalibration
        
        # 3. Coverage component (penalty for over-abstention)
        # Want to maintain some answerable rate
        abstention_threshold = 0.5
        if answer_likelihood < abstention_threshold:
            abstention_rate = 1.0
        else:
            abstention_rate = 0.0
        
        L_coverage = abstention_rate * 0.2  # Soft penalty for abstention
        
        # Weights (from paper's experiments)
        w_acc = 1.0
        w_cal = 0.5
        w_cov = 0.3
        
        L_total = w_acc * L_accuracy + w_cal * L_calibration + w_cov * L_coverage
        
        # Decision based on total loss
        if L_total > 0.6:
            decision = "ABSTAIN: Multi-objective loss indicates uncertainty"
        else:
            # Generate actual response
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=50, temperature=0.1)
            response_ids = output[0, inputs.input_ids.shape[-1]:]
            decision = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        return {
            "L_accuracy": L_accuracy,
            "L_calibration": L_calibration,
            "L_coverage": L_coverage,
            "L_total": L_total,
            "weights": {"w_acc": w_acc, "w_cal": w_cal, "w_cov": w_cov},
            "decision": decision,
            "method": "Multi-Objective Training"
        }
    
    def method_2_abstention_aware_finetuning(self, question, context="", abstention_token="[ABSTAIN]"):
        """
        METHOD 2: Abstention-Aware Fine-Tuning
        
        Paper principle (§4.4.1):
        Train model on explicit [ABSTAIN] or "I don't know" tokens.
        Model learns to output abstention token when uncertain.
        
        Args:
            question: Query string
            context: Optional background
            abstention_token: Special token for refusal
        
        Returns:
            decision_dict: Response, contains_abstention_token, decision
        """
        
        # In practice: Model is fine-tuned on data with [ABSTAIN] labels
        # Example training data:
        # Q: What is X?
        # Context: No information about X.
        # Expected output: [ABSTAIN]
        
        prompt = f"Context: {context if context else 'No context provided.'}\n\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False
            )
        
        response_ids = output[0, inputs.input_ids.shape[-1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        # Check if model output contains uncertainty markers
        # (After fine-tuning, model learns to output these)
        uncertainty_markers = ["[ABSTAIN]", "[REFUSE]", "I don't know", "I cannot answer"]
        contains_abstention = any(marker in response for marker in uncertainty_markers)
        
        # In practice, fine-tuned model would directly output these tokens
        # Simulation: Check if response is short/vague (proxy for abstention)
        response_length = len(response.split())
        is_vague = response_length < 5 and any(word in response.lower() for word in ["unknown", "unclear"])
        
        decision = "ABSTAIN: Model output abstention token" if (contains_abstention or is_vague) else response
        
        return {
            "response": response,
            "response_length": response_length,
            "contains_abstention_signal": contains_abstention or is_vague,
            "abstention_token": abstention_token,
            "decision": decision,
            "method": "Abstention-Aware Fine-Tuning"
        }
    
    def method_3_reward_modeling(self, question, context=""):
        """
        METHOD 3: Reward Modeling (Simplified RLHF)
        
        Paper principle (§4.4.3):
        Train reward model R(x, y) that scores: high for correct answers,
        low for wrong answers, medium-high for abstention.
        Then optimize policy π to maximize E[R(x, y)].
        
        Args:
            question: Query string
            context: Optional background
        
        Returns:
            decision_dict: Response, reward_score, decision
        """
        
        # Generate candidate response
        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:" if context else f"Question: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        response_ids = outputs.sequences[0, inputs.input_ids.shape[-1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        # Simplified reward function: R(x, y)
        # In practice: Trained model that predicts human preference
        
        # Reward components:
        # 1. Fluency (high token probability)
        if outputs.scores:
            token_probs = [score.max().item() for score in outputs.scores]
            avg_prob = sum(token_probs) / len(token_probs)
        else:
            avg_prob = 0.5
        
        fluency_reward = avg_prob  # 0-1
        
        # 2. Coherence (response length, not empty)
        response_length = len(response.split())
        coherence_reward = min(response_length / 20, 1.0)  # Reward non-trivial responses
        
        # 3. Uncertainty handling (reward for saying "I don't know" when appropriate)
        uncertainty_phrases = ["don't know", "cannot", "uncertain", "unclear"]
        has_uncertainty = any(phrase in response.lower() for phrase in uncertainty_phrases)
        
        # Simulate: if context is empty and model says "I don't know", reward it
        if not context and has_uncertainty:
            uncertainty_reward = 0.8
        elif context and not has_uncertainty:
            uncertainty_reward = 0.5
        else:
            uncertainty_reward = 0.3
        
        # Combined reward
        R = 0.4 * fluency_reward + 0.3 * coherence_reward + 0.3 * uncertainty_reward
        
        # Decision: High reward keeps answer, low reward → abstain
        reward_threshold = 0.5
        if R < reward_threshold:
            decision = f"ABSTAIN: Low reward score ({R:.2f})"
        else:
            decision = response
        
        return {
            "response": response,
            "fluency_reward": fluency_reward,
            "coherence_reward": coherence_reward,
            "uncertainty_reward": uncertainty_reward,
            "total_reward": R,
            "threshold": reward_threshold,
            "decision": decision,
            "method": "Reward Modeling"
        }


def main():
    """Test training-based methods on sample questions"""

    # Header
    print("\n" + "═" * 80)
    print("🎯 TRAINING-BASED METHODS DEMONSTRATION")
    print("   From SURVEY § 4.4: Training-Based Abstention Methods")
    print("═" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  WHAT ARE TRAINING-BASED METHODS?                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Core Idea: Modify the TRAINING PROCESS itself to optimize for             │
│  calibrated abstention, not just accuracy.                                 │
│                                                                             │
│  From Paper (§ 4.4):                                                        │
│    "Training-based methods incorporate abstention directly into the        │
│     learning objective, teaching models when NOT to answer"                │
│                                                                             │
│  Key Difference from Other Techniques:                                      │
│    • Confidence-based (§4.1): Post-hoc thresholding on scores              │
│    • Selective prediction (§4.2): External validation modules              │
│    • Verbalized (§4.3): Prompt engineering or fine-tuning for phrases      │
│    • Training-based (§4.4): CHANGE THE LOSS FUNCTION itself                │
│                                                                             │
│  Three Approaches:                                                          │
│    1. Multi-Objective Training: Joint loss (accuracy + calibration + cov)  │
│    2. Abstention-Aware Fine-Tuning: Train on explicit [ABSTAIN] tokens     │
│    3. Reward Modeling (RLHF): Penalize unjustified confidence              │
│                                                                             │
│  Key Insight:                                                               │
│    Instead of adding abstention as an afterthought, bake it into           │
│    the training objective so the model learns it inherently.               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("📚 Reference: SURVEY § 4.4 - Training-Based Methods\n")

    print("⏳ Loading Mistral-7B-Instruct...")
    trainer = TrainingBasedMethods()
    print("✓ Model loaded\n")
    print("Note: This demo simulates training-based concepts. Full implementation")
    print("      requires actual training with labeled datasets.\n")
    
    test_cases = [
        ("What is 2 + 2?", "Basic mathematics."),
        ("What did the report conclude?", ""),  # No context
        ("Who is the CEO of OpenAI?", "Company information"),
    ]
    
    # ==================== METHOD 1 ====================
    print("\n" + "═" * 80)
    print("METHOD 1: MULTI-OBJECTIVE TRAINING")
    print("═" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  HOW MULTI-OBJECTIVE TRAINING WORKS (SURVEY § 4.4.2)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Principle: Optimize a JOINT loss function that balances multiple goals:   │
│                                                                             │
│  L_total = w_acc × L_accuracy + w_cal × L_calibration + w_cov × L_coverage │
│                                                                             │
│  Loss Components:                                                           │
│    • L_accuracy: Cross-entropy on answerable questions                     │
│      → Reward correct answers                                              │
│                                                                             │
│    • L_calibration: Expected Calibration Error (ECE)                       │
│      → Penalize miscalibration (saying 90% confident but 60% correct)      │
│                                                                             │
│    • L_coverage: Penalty for over-abstention                               │
│      → Maintain utility (don't refuse everything)                          │
│                                                                             │
│  From Paper:                                                                │
│    "Multi-objective optimization balances the accuracy-coverage tradeoff   │
│     by explicitly penalizing both errors and over-abstention"              │
│                                                                             │
│  Typical Weights (from experiments):                                       │
│    w_acc = 1.0, w_cal = 0.5, w_cov = 0.3                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("▶ Running Method 1 on test question...")
    print("  Computing multi-objective loss components\n")

    for question, context in test_cases[:1]:
        result = trainer.method_1_multi_objective_training(question, context)
        print(f"● Q: {question}")
        print(f"  Loss Components:")
        print(f"    L_accuracy:    {result['L_accuracy']:.3f} (lower = more confident)")
        print(f"    L_calibration: {result['L_calibration']:.3f} (lower = better calibrated)")
        print(f"    L_coverage:    {result['L_coverage']:.3f} (penalty for abstaining)")
        print(f"    L_total:       {result['L_total']:.3f}")
        print(f"  Weights: {result['weights']}")
        if result['decision'].startswith("ABSTAIN"):
            print(f"  → ABSTAIN (high total loss)")
        else:
            print(f"  → ANSWER: {result['decision'][:50]}...")
        print()

    print("─" * 80)
    print("📊 METHOD 1 OBSERVATION:")
    print("   Multi-objective training BALANCES competing goals.")
    print("   Low L_accuracy = confident, Low L_calibration = well-calibrated.")
    print("   L_coverage prevents model from refusing everything.")
    print("─" * 80)
    
    # ==================== METHOD 2 ====================
    print("\n" + "═" * 80)
    print("METHOD 2: ABSTENTION-AWARE FINE-TUNING")
    print("═" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  HOW ABSTENTION-AWARE FINE-TUNING WORKS (SURVEY § 4.4.1)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Principle: Fine-tune model on data that includes explicit ABSTAIN tokens. │
│  Model learns to output [ABSTAIN] when question is unanswerable.           │
│                                                                             │
│  Training Data Format:                                                      │
│    Example 1: Q: "Capital of France?"  Context: "Paris is capital"         │
│               Expected: "Paris"                                             │
│                                                                             │
│    Example 2: Q: "Capital of Atlantis?" Context: "No information"          │
│               Expected: "[ABSTAIN]"                                         │
│                                                                             │
│  From Paper:                                                                │
│    "Training with explicit abstention tokens teaches the model a new       │
│     vocabulary item that signals 'I should not answer this'"               │
│                                                                             │
│  Key Datasets:                                                              │
│    • SQuAD 2.0: Contains unanswerable questions                            │
│    • Abstain-QA: Explicitly labeled for abstention                         │
│    • TriviaQA + negative samples                                           │
│                                                                             │
│  Detection Signals:                                                         │
│    • Direct token: [ABSTAIN], [REFUSE]                                     │
│    • Phrases: "I don't know", "I cannot answer"                            │
│    • Short/vague responses (proxy for uncertainty)                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("▶ Running Method 2 on test questions...")
    print("  Looking for abstention signals in responses\n")

    for question, context in test_cases:
        result = trainer.method_2_abstention_aware_finetuning(question, context)
        print(f"● Q: {question}")
        print(f"  Context: {context[:40]}..." if context else "  Context: None")
        print(f"  Response: {result['response'][:60]}...")
        print(f"  Response Length: {result['response_length']} words")
        print(f"  Contains Abstention Signal: {result['contains_abstention_signal']}")
        if result['decision'].startswith("ABSTAIN"):
            print(f"  → ABSTAIN (explicit abstention signal)")
        else:
            print(f"  → ANSWER: {result['decision'][:40]}...")
        print()

    print("─" * 80)
    print("📊 METHOD 2 OBSERVATION:")
    print("   Fine-tuned models learn to output [ABSTAIN] directly.")
    print("   This is more reliable than post-hoc thresholding.")
    print("   Requires training data with abstention labels.")
    print("─" * 80)
    
    # ==================== METHOD 3 ====================
    print("\n" + "═" * 80)
    print("METHOD 3: REWARD MODELING (RLHF)")
    print("═" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  HOW REWARD MODELING WORKS (SURVEY § 4.4.3)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Principle: Train a reward model R(x, y) that scores responses.            │
│  Then use RLHF to optimize the policy π to maximize expected reward.       │
│                                                                             │
│  Reward Function Design:                                                    │
│    R(correct answer) = HIGH (1.0)                                          │
│    R(incorrect answer) = LOW (0.0)                                         │
│    R(justified abstention) = MEDIUM-HIGH (0.7)                             │
│    R(unjustified abstention) = MEDIUM-LOW (0.3)                            │
│                                                                             │
│  From Paper:                                                                │
│    "RLHF can be used to penalize unjustified confidence by training        │
│     the reward model on human preferences for calibrated responses"        │
│                                                                             │
│  Reward Components (simplified):                                            │
│    • Fluency reward: Average token probability                             │
│    • Coherence reward: Response completeness/length                        │
│    • Uncertainty handling: Reward "I don't know" when appropriate          │
│                                                                             │
│  Combined: R = 0.4 × fluency + 0.3 × coherence + 0.3 × uncertainty         │
│                                                                             │
│  Decision: If R < threshold → ABSTAIN                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("▶ Running Method 3 on test questions...")
    print("  Computing reward components\n")

    for question, context in test_cases:
        result = trainer.method_3_reward_modeling(question, context)
        print(f"● Q: {question}")
        print(f"  Context: {context[:40]}..." if context else "  Context: None")
        print(f"  Reward Components:")
        print(f"    Fluency:     {result['fluency_reward']:.3f}")
        print(f"    Coherence:   {result['coherence_reward']:.3f}")
        print(f"    Uncertainty: {result['uncertainty_reward']:.3f}")
        print(f"    Total:       {result['total_reward']:.3f} (threshold: {result['threshold']})")
        if result['decision'].startswith("ABSTAIN"):
            print(f"  → ABSTAIN (low reward score)")
        else:
            print(f"  → ANSWER: {result['decision'][:40]}...")
        print()

    print("─" * 80)
    print("📊 METHOD 3 OBSERVATION:")
    print("   RLHF optimizes for HUMAN PREFERENCES, not just accuracy.")
    print("   Reward model learns: 'abstaining on uncertain questions is GOOD'.")
    print("   This prevents the overconfidence problem seen in standard training.")
    print("─" * 80)

    # ==================== SUMMARY ====================
    print("\n" + "═" * 80)
    print("TRAINING-BASED METHODS: KEY INSIGHTS")
    print("═" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  COMPARISON OF TRAINING-BASED METHODS                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Method 1: Multi-Objective Training                                         │
│    ✓ Explicit control over accuracy-calibration-coverage tradeoff          │
│    ✓ Can tune weights for different applications                           │
│    ✗ Requires careful weight tuning                                        │
│    → Best for: Custom tradeoff requirements                                │
│                                                                             │
│  Method 2: Abstention-Aware Fine-Tuning                                     │
│    ✓ Model learns explicit abstention behavior                             │
│    ✓ Simple: just add [ABSTAIN] to vocabulary                              │
│    ✗ Requires labeled abstention data                                      │
│    → Best for: Clean abstention signal needed                              │
│                                                                             │
│  Method 3: Reward Modeling (RLHF)                                           │
│    ✓ Optimizes for human preferences                                       │
│    ✓ Handles nuanced "when to abstain" decisions                           │
│    ✗ Complex training pipeline                                             │
│    → Best for: Aligning with human judgment on abstention                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  KEY INSIGHT FROM PAPER:                                                    │
│                                                                             │
│  "Training-based methods are more principled than post-hoc approaches      │
│   because abstention is learned during optimization, not added later"      │
│                                                                             │
│  The key advantage:                                                         │
│    Post-hoc (confidence-based): Model learns to answer, then we threshold  │
│    Training-based: Model learns WHEN to answer vs. abstain directly        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("═" * 80)
    print("END OF TRAINING-BASED METHODS DEMONSTRATION")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    main()

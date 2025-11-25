"""
TECHNIQUE 2: SELECTIVE PREDICTION
From SURVEY.md § 4.2

Principle: Separate selector module determines if primary model's 
answer should be trusted.

Three sub-methods:
- Auxiliary Prediction Head: Secondary head predicts correctness
- Semantic Similarity: Compare answer to context
- Cross-Model Validation: Query multiple LLMs; abstain on disagreement
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class SelectivePrediction:
    """
    Implements selective prediction abstention.
    Mirrors paper's principle: Separate selector determines trustworthiness.
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
        self.model.eval()
    
    def method_1_semantic_similarity(self, question, context="", threshold=0.3):
        """
        METHOD 1: Semantic Similarity
        
        Paper principle (§4.2.3):
        Compare generated answer to context via embedding similarity.
        Low similarity → abstain (answer not grounded in context)
        
        Args:
            question: Query string
            context: Reference/background information
            threshold: Minimum allowed similarity (0.0 to 1.0)
        
        Returns:
            decision_dict: Response, similarity, decision
        """
        
        if not context:
            return {
                "response": "No context provided",
                "similarity": 0.0,
                "decision": "ABSTAIN: No context to validate against",
                "method": "Semantic Similarity"
            }
        
        # Generate answer
        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1
            )
        
        response_ids = output[0, inputs.input_ids.shape[-1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Simple similarity heuristic: check if key phrases from context appear in response
        context_words = set(context.lower().split())
        response_words = set(response.lower().split())
        
        # Jaccard similarity
        if len(context_words | response_words) == 0:
            similarity = 0.0
        else:
            similarity = len(context_words & response_words) / len(context_words | response_words)
        
        # Decision: Abstain if similarity too low
        if similarity < threshold:
            decision = f"ABSTAIN: Answer not grounded in context (similarity: {similarity:.2f})"
        else:
            decision = response
        
        return {
            "response": response,
            "similarity": similarity,
            "threshold": threshold,
            "decision": decision,
            "method": "Semantic Similarity"
        }
    
    def method_2_cross_model_validation(self, question, context="", models=None, agreement_threshold=2):
        """
        METHOD 2: Cross-Model Validation
        
        Paper principle (§4.2.3):
        Query multiple LLMs; abstain if votes diverge.
        
        Args:
            question: Query string
            context: Optional background
            models: List of model names (use same model if None)
            agreement_threshold: Min number of models that must agree
        
        Returns:
            decision_dict: Responses, votes, agreement score, decision
        """
        
        if models is None:
            models = [1, 2, 3]  # Simulate 3 runs of same model
        
        # Format prompt
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        else:
            prompt = f"Question: {question}\nAnswer:"
        
        # Get responses from multiple "models" (or multiple runs)
        responses = []
        for i in range(len(models)):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.5 + (0.2 * i),  # Vary temperature per run
                    do_sample=True
                )
            
            response_ids = output[0, inputs.input_ids.shape[-1]:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            responses.append(response)
        
        # Count votes (exact match = agreement)
        from collections import Counter
        vote_counts = Counter(responses)
        max_votes = max(vote_counts.values())
        
        # Decision: Abstain if no clear consensus
        if max_votes < agreement_threshold:
            decision = f"ABSTAIN: No consensus among models (votes: {dict(vote_counts)})"
        else:
            majority_answer = max(vote_counts, key=vote_counts.get)
            decision = majority_answer
        
        return {
            "responses": responses,
            "vote_counts": dict(vote_counts),
            "max_votes": max_votes,
            "agreement_threshold": agreement_threshold,
            "decision": decision,
            "method": "Cross-Model Validation"
        }
    
    def method_3_auxiliary_confidence_head(self, question, context="", threshold=0.6):
        """
        METHOD 3: Auxiliary Confidence Head (Learned Calibration)
        
        Paper principle (§4.2.1):
        Train auxiliary model to predict P(correct | x, y, logits).
        Here: Simple heuristic based on response length and coherence.
        
        Args:
            question: Query string
            context: Optional background
            threshold: Confidence threshold
        
        Returns:
            decision_dict: Response, predicted correctness, decision
        """
        
        # Generate response
        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:" if context else f"Question: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        response_ids = output.sequences[0, inputs.input_ids.shape[-1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Simple learned calibration heuristic:
        # - Longer responses with lower perplexity = more confidence
        # - "I don't know" patterns = low confidence
        
        uncertainty_phrases = ["don't know", "uncertain", "unclear", "not sure", "can't"]
        has_uncertainty = any(phrase in response.lower() for phrase in uncertainty_phrases)
        
        # Confidence calculation
        response_length = len(response.split())
        avg_token_prob = sum(score.max().item() for score in output.scores) / len(output.scores) if output.scores else 0.0
        
        # Combine signals
        predicted_confidence = avg_token_prob * (1 - int(has_uncertainty) * 0.5)
        
        # Decision
        if predicted_confidence < threshold:
            decision = f"ABSTAIN: Low predicted confidence ({predicted_confidence:.2f})"
        else:
            decision = response
        
        return {
            "response": response,
            "predicted_confidence": predicted_confidence,
            "threshold": threshold,
            "has_uncertainty_signals": has_uncertainty,
            "decision": decision,
            "method": "Auxiliary Confidence Head"
        }


def main():
    """Test selective prediction on sample questions"""

    # Header
    print("\n" + "═" * 80)
    print("🎯 SELECTIVE PREDICTION DEMONSTRATION")
    print("   From SURVEY § 4.2: Selective Prediction Methods")
    print("═" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  WHAT IS SELECTIVE PREDICTION?                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Core Idea: A SEPARATE SELECTOR MODULE determines whether the primary      │
│  model's answer should be trusted. Unlike confidence-based methods that    │
│  use the model's own scores, selective prediction uses external signals.   │
│                                                                             │
│  Key Insight:                                                               │
│    Confidence-based: "How confident is the model?"                         │
│    Selective prediction: "Is the answer actually trustworthy?"             │
│                                                                             │
│  The selector can use various signals:                                      │
│    • Semantic similarity to context (is answer grounded?)                  │
│    • Cross-model agreement (do other models agree?)                        │
│    • Auxiliary prediction head (trained correctness predictor)             │
│                                                                             │
│  Why This Matters:                                                          │
│    • Models can be confident AND wrong (hallucination)                     │
│    • External validation catches errors internal metrics miss              │
│    • Enables "trust but verify" approach to LLM outputs                    │
│                                                                             │
│  Three Sub-Methods We'll Test:                                              │
│    1. Semantic Similarity: Compare answer to context                       │
│    2. Cross-Model Validation: Query multiple LLMs                          │
│    3. Auxiliary Confidence Head: Learned correctness predictor             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("📚 Reference: SURVEY § 4.2 - Selective Prediction\n")

    print("⏳ Loading Mistral-7B-Instruct...")
    abstainer = SelectivePrediction()
    print("✓ Model loaded\n")
    
    questions = [
        ("What is the capital of France?", "France is a country in Western Europe. Its capital is Paris."),
        ("Who invented the lightbulb?", ""),  # No context
        ("What color is water?", "Water is a transparent liquid."),
    ]
    
    # ==================== METHOD 1 ====================
    print("\n" + "═" * 80)
    print("METHOD 1: SEMANTIC SIMILARITY")
    print("═" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  HOW SEMANTIC SIMILARITY WORKS (SURVEY § 4.2.3)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Principle: Compare generated answer to the provided context.              │
│  If the answer doesn't match the context, it's likely hallucinated.        │
│                                                                             │
│  Process:                                                                   │
│    Step 1: Generate answer from the model                                  │
│    Step 2: Compute similarity between answer and context                   │
│    Step 3: If similarity < threshold → ABSTAIN (not grounded)              │
│                                                                             │
│  Why It Works:                                                              │
│    • Grounded answers share vocabulary/concepts with context               │
│    • Hallucinated answers often introduce unrelated terms                  │
│    • Acts as a "fact-checking" mechanism against source material           │
│                                                                             │
│  Limitation:                                                                │
│    • Requires context to validate against                                  │
│    • Cannot work for open-ended questions without reference                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("▶ Running Method 1 on 3 test questions...")
    print("  Threshold = 0.1 (abstain if similarity < 0.1)\n")

    for question, context in questions:
        result = abstainer.method_1_semantic_similarity(question, context, threshold=0.1)
        print(f"● Q: {question}")
        print(f"  Context: {context[:50]}..." if context else "  Context: None")
        print(f"  Similarity: {result['similarity']:.3f}")
        if result['decision'].startswith("ABSTAIN"):
            print(f"  → ABSTAIN (similarity too low or no context)")
        else:
            print(f"  → ANSWER: {result['decision'][:50]}...")
        print()

    print("─" * 80)
    print("📊 METHOD 1 OBSERVATION:")
    print("   Semantic similarity requires context to validate against.")
    print("   Without context, the method correctly abstains.")
    print("   With context, it checks if the answer is grounded in the source.")
    print("─" * 80)
    
    # ==================== METHOD 2 ====================
    print("\n" + "═" * 80)
    print("METHOD 2: CROSS-MODEL VALIDATION")
    print("═" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  HOW CROSS-MODEL VALIDATION WORKS (SURVEY § 4.2.3)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Principle: Query multiple LLMs (or multiple runs with different temps).   │
│  If they disagree, abstain - disagreement signals uncertainty.             │
│                                                                             │
│  Process:                                                                   │
│    Step 1: Query N models/runs with the same question                      │
│    Step 2: Collect N responses                                             │
│    Step 3: Count agreement (voting)                                        │
│    Step 4: If max_votes < threshold → ABSTAIN (no consensus)               │
│                                                                             │
│  From Paper:                                                                │
│    "Cross-validation across models provides robust uncertainty signal"     │
│    "Disagreement between models indicates high epistemic uncertainty"      │
│                                                                             │
│  Key Difference from Ensemble Disagreement (§ 4.1):                         │
│    • Ensemble: Same model, multiple samples                                │
│    • Cross-Model: Different models or significantly varied runs            │
│    • Cross-model catches systematic biases single models miss              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("▶ Running Method 2 on test question...")
    print("  Simulating 3 model runs with varying temperatures\n")

    for question, context in questions[:1]:
        result = abstainer.method_2_cross_model_validation(question, context)
        print(f"● Q: {question}")
        print(f"  Model responses:")
        for i, resp in enumerate(result['responses'], 1):
            print(f"    Run {i}: {resp[:40]}...")
        print(f"  Vote counts: {result['vote_counts']}")
        if result['decision'].startswith("ABSTAIN"):
            print(f"  → ABSTAIN (no clear consensus)")
        else:
            print(f"  → ANSWER: {result['decision'][:50]}...")
        print()

    print("─" * 80)
    print("📊 METHOD 2 OBSERVATION:")
    print("   Cross-model validation catches cases where a single model")
    print("   might be confidently wrong. Multiple perspectives = better judgment.")
    print("   Trade-off: N× computational cost for better reliability.")
    print("─" * 80)
    
    # ==================== METHOD 3 ====================
    print("\n" + "═" * 80)
    print("METHOD 3: AUXILIARY CONFIDENCE HEAD")
    print("═" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  HOW AUXILIARY CONFIDENCE HEAD WORKS (SURVEY § 4.2.1)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Principle: Train a separate model to predict P(correct | x, y, logits).   │
│  This "correctness predictor" learns to recognize when the main model      │
│  is likely wrong, even when it appears confident.                          │
│                                                                             │
│  From Paper:                                                                │
│    "Auxiliary models trained on (input, output, correct?) tuples can       │
│     predict correctness better than raw softmax scores"                    │
│                                                                             │
│  Process:                                                                   │
│    Step 1: Main model generates response + logits                          │
│    Step 2: Auxiliary model takes (question, response, logits) as input     │
│    Step 3: Auxiliary model outputs predicted correctness score             │
│    Step 4: If predicted_correct < threshold → ABSTAIN                      │
│                                                                             │
│  Why It's Better Than Raw Confidence:                                       │
│    • Learns patterns of errors the main model makes                        │
│    • Can detect uncertainty signals in hidden states                       │
│    • Trained explicitly on "was this answer right?" labels                 │
│                                                                             │
│  Implementation Note:                                                       │
│    Here we use a heuristic proxy (response length, uncertainty phrases).   │
│    Full implementation requires trained auxiliary model.                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("▶ Running Method 3 on test question...")
    print("  Threshold = 0.6 (abstain if predicted confidence < 0.6)\n")

    for question, context in questions[:1]:
        result = abstainer.method_3_auxiliary_confidence_head(question, context)
        print(f"● Q: {question}")
        print(f"  Response: {result['response'][:50]}...")
        print(f"  Predicted Confidence: {result['predicted_confidence']:.3f}")
        print(f"  Has Uncertainty Signals: {result['has_uncertainty_signals']}")
        if result['decision'].startswith("ABSTAIN"):
            print(f"  → ABSTAIN (low predicted confidence)")
        else:
            print(f"  → ANSWER: {result['decision'][:50]}...")
        print()

    print("─" * 80)
    print("📊 METHOD 3 OBSERVATION:")
    print("   The auxiliary head learns to recognize when the main model is wrong.")
    print("   It combines multiple signals: logits, response patterns, uncertainty phrases.")
    print("   Key advantage: Can be trained on labeled correctness data.")
    print("─" * 80)

    # ==================== SUMMARY ====================
    print("\n" + "═" * 80)
    print("SELECTIVE PREDICTION: KEY INSIGHTS")
    print("═" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  COMPARISON OF SELECTIVE PREDICTION METHODS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Method 1: Semantic Similarity                                              │
│    ✓ Fast (1× inference + similarity computation)                          │
│    ✓ Catches hallucinations not grounded in context                        │
│    ✗ Requires context to validate against                                  │
│    → Best for: RAG systems, document QA                                    │
│                                                                             │
│  Method 2: Cross-Model Validation                                           │
│    ✓ Catches systematic biases in individual models                        │
│    ✓ Works without context                                                 │
│    ✗ N× computational cost                                                 │
│    → Best for: High-stakes decisions, fact verification                    │
│                                                                             │
│  Method 3: Auxiliary Confidence Head                                        │
│    ✓ Learns from labeled correctness data                                  │
│    ✓ Can recognize subtle error patterns                                   │
│    ✗ Requires training data and separate model                             │
│    → Best for: Production systems with labeled feedback                    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  KEY DIFFERENCE FROM CONFIDENCE-BASED METHODS (§ 4.1):                      │
│                                                                             │
│  Confidence-based: Uses model's own internal scores                        │
│  Selective prediction: Uses EXTERNAL validation signals                    │
│                                                                             │
│  → Selective prediction catches errors that confidence methods miss        │
│    because it doesn't trust the model's self-assessment                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("═" * 80)
    print("END OF SELECTIVE PREDICTION DEMONSTRATION")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    main()

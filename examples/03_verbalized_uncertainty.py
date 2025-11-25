"""
TECHNIQUE 3: VERBALIZED UNCERTAINTY
From SURVEY.md § 4.3

Principle: Train model to explicitly express uncertainty via:
- Prompt engineering with few-shot examples
- Fine-tuning on labeled "I don't know" responses

Two sub-methods:
- Prompt Engineering: Few-shot teaching
- Fine-tuning with Labels: Train on abstention examples
"""

import torch
from utils import load_tokenizer_and_model


class VerbalizedUncertainty:
    """
    Implements verbalized uncertainty abstention.
    Mirrors paper's principle: Train model to verbalize "I don't know".
    """
    
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1"):
        """Load model and tokenizer"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer, self.model = load_tokenizer_and_model(model_name)
        
        # Uncertainty keywords (used for detection)
        self.uncertainty_phrases = [
            "i don't know",
            "i cannot determine",
            "cannot find",
            "not mentioned",
            "insufficient information",
            "unclear from the context",
            "no information",
            "cannot say",
            "i'm uncertain",
            "i have no information"
        ]
    
    def method_1_prompt_engineering(self, question, context=""):
        """
        METHOD 1: Prompt Engineering with Few-Shot
        
        Paper principle (§4.3.1):
        Few-shot examples teach model to say "I don't know" when appropriate.
        Bartolo et al. (2020): Few-shot prompting increases abstention accuracy.
        
        Args:
            question: Query string
            context: Optional background
        
        Returns:
            decision_dict: Response, contains_abstention_signal, decision
        """
        
        # Few-shot template from paper
        few_shot_examples = """Example 1:
Q: What is the capital of France?
Context: France is in Western Europe. Its capital is Paris.
A: The capital of France is Paris.

Example 2:
Q: Who invented the quantum computer?
Context: No relevant information provided.
A: I don't know. The context does not contain information about who invented the quantum computer.

Example 3:
Q: What color is water?
Context: The sky is blue.
A: I don't know. The provided context is about the sky, not water."""
        
        # Construct prompt with few-shot examples
        prompt = f"""{few_shot_examples}

Example 4:
Q: {question}
Context: {context if context else "No context provided."}
A:"""
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False
            )
        
        response_ids = output[0, inputs.input_ids.shape[-1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        # Check if model expressed uncertainty
        response_lower = response.lower()
        contains_abstention = any(phrase in response_lower for phrase in self.uncertainty_phrases)
        
        # Decision
        if contains_abstention:
            decision = "ABSTAIN: Model expressed uncertainty"
        else:
            decision = response
        
        return {
            "response": response,
            "contains_abstention_signal": contains_abstention,
            "method": "Prompt Engineering",
            "decision": decision
        }
    
    def method_2_uncertainty_extraction(self, question, context="", confidence_threshold=0.6):
        """
        METHOD 2: Uncertainty Token Detection
        
        Paper principle (§4.3.2):
        Fine-tuning on SQuAD2/Abstain-QA teaches model "I don't know" responses.
        Here: Detect uncertainty via logit analysis on special tokens.
        
        Args:
            question: Query string
            context: Optional background
            confidence_threshold: Threshold for confidence
        
        Returns:
            decision_dict: Response, uncertainty_score, decision
        """
        
        # Create prompt requesting model to indicate confidence
        prompt = f"""Question: {question}
Context: {context if context else "No context provided."}

Provide your best answer. If uncertain, say "I don't know".
Answer:"""
        
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
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        # Calculate uncertainty score from token probabilities
        if output.scores:
            token_probs = [score.max().item() for score in output.scores]
            avg_confidence = sum(token_probs) / len(token_probs)
        else:
            avg_confidence = 1.0
        
        uncertainty_score = 1.0 - avg_confidence
        
        # Check for uncertainty phrases
        response_lower = response.lower()
        has_explicit_uncertainty = any(phrase in response_lower for phrase in self.uncertainty_phrases)
        
        # Combined uncertainty signal
        final_uncertainty = (uncertainty_score + float(has_explicit_uncertainty)) / 2
        
        # Decision
        if final_uncertainty > (1 - confidence_threshold) or has_explicit_uncertainty:
            decision = f"ABSTAIN: High uncertainty signal ({final_uncertainty:.2f})"
        else:
            decision = response
        
        return {
            "response": response,
            "token_confidence": avg_confidence,
            "explicit_uncertainty_signals": has_explicit_uncertainty,
            "combined_uncertainty_score": final_uncertainty,
            "decision": decision,
            "method": "Uncertainty Extraction"
        }
    
    def method_3_confidence_statement(self, question, context=""):
        """
        METHOD 3: Confidence Statement Parsing
        
        Paper principle (§4.3):
        Request model to append confidence statement; parse to abstain.
        
        Args:
            question: Query string
            context: Optional background
        
        Returns:
            decision_dict: Response, confidence_level, decision
        """
        
        # Prompt with explicit confidence request
        prompt = f"""Question: {question}
Context: {context if context else "No context provided."}

Answer the question and rate your confidence (high/medium/low):
Answer:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.1,
                do_sample=False
            )
        
        response_ids = output[0, inputs.input_ids.shape[-1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        # Parse confidence level
        response_lower = response.lower()
        
        if "low" in response_lower or "not sure" in response_lower or "uncertain" in response_lower:
            confidence_level = "low"
            confidence_score = 0.3
        elif "medium" in response_lower or "somewhat" in response_lower:
            confidence_level = "medium"
            confidence_score = 0.6
        else:
            confidence_level = "high"
            confidence_score = 0.9
        
        # Decision based on confidence level
        if confidence_level == "low":
            decision = "ABSTAIN: Model reports low confidence"
        else:
            decision = response.split("confidence")[0].strip() if "confidence" in response_lower else response
        
        return {
            "response": response,
            "parsed_confidence_level": confidence_level,
            "confidence_score": confidence_score,
            "decision": decision,
            "method": "Confidence Statement Parsing"
        }


def main():
    """Test verbalized uncertainty on sample questions"""

    # Header
    print("\n" + "═" * 80)
    print("🎯 VERBALIZED UNCERTAINTY DEMONSTRATION")
    print("   From SURVEY § 4.3: Verbalized Uncertainty Methods")
    print("═" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  WHAT IS VERBALIZED UNCERTAINTY?                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Core Idea: Train/prompt the model to EXPLICITLY express uncertainty       │
│  using natural language like "I don't know" or "I'm not sure".             │
│                                                                             │
│  From Paper (§ 4.3):                                                        │
│    "Verbalized uncertainty trains models to produce linguistic markers     │
│     of uncertainty rather than relying on implicit confidence scores"      │
│                                                                             │
│  Key Insight:                                                               │
│    Confidence-based: Infer uncertainty from token probabilities            │
│    Verbalized: Model directly SAYS when it's uncertain                     │
│                                                                             │
│  Two Main Approaches:                                                       │
│    1. Prompt Engineering: Few-shot examples teach "I don't know" patterns  │
│    2. Fine-tuning: Train on datasets with explicit abstention labels       │
│       (e.g., SQuAD 2.0, Abstain-QA)                                        │
│                                                                             │
│  Referenced Works:                                                          │
│    • Bartolo et al. (2020): Few-shot prompting for abstention              │
│    • Rajpurkar et al. (2018): SQuAD 2.0 unanswerable questions             │
│                                                                             │
│  Three Sub-Methods We'll Test:                                              │
│    1. Prompt Engineering (Few-Shot): Teach via examples                    │
│    2. Uncertainty Extraction: Detect uncertainty phrases + logits          │
│    3. Confidence Statement Parsing: Model self-reports confidence          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("📚 Reference: SURVEY § 4.3 - Verbalized Uncertainty\n")

    print("⏳ Loading Mistral-7B-Instruct...")
    abstainer = VerbalizedUncertainty()
    print("✓ Model loaded\n")
    
    test_cases = [
        ("What is the capital of France?", "France is in Western Europe and its capital is Paris."),
        ("Who invented the quantum computer?", ""),  # Unanswerable
        ("What did the abstract say?", "No abstract was provided."),
    ]
    
    # ==================== METHOD 1 ====================
    print("\n" + "═" * 80)
    print("METHOD 1: PROMPT ENGINEERING WITH FEW-SHOT")
    print("═" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  HOW FEW-SHOT PROMPT ENGINEERING WORKS (SURVEY § 4.3.1)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Principle: Include examples in the prompt that demonstrate when and       │
│  how to say "I don't know".                                                │
│                                                                             │
│  From Paper:                                                                │
│    "Bartolo et al. (2020) showed that few-shot examples significantly      │
│     improve abstention accuracy by teaching the model appropriate          │
│     refusal patterns"                                                       │
│                                                                             │
│  Few-Shot Template Used:                                                    │
│    Example 1: Q with context → Direct answer                               │
│    Example 2: Q without info → "I don't know. The context does not..."     │
│    Example 3: Q with irrelevant context → "I don't know..."                │
│    New question: Model follows the pattern                                 │
│                                                                             │
│  Key Features:                                                              │
│    • No training required (prompt-only)                                    │
│    • Works with any instruction-tuned model                                │
│    • Examples teach the format of uncertainty expression                   │
│                                                                             │
│  Detection: Scan response for uncertainty phrases like "I don't know"      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("▶ Running Method 1 with few-shot prompt...")
    print("  Detecting uncertainty phrases in response\n")

    for question, context in test_cases[:1]:
        result = abstainer.method_1_prompt_engineering(question, context)
        print(f"● Q: {question}")
        print(f"  Context: {context[:50]}..." if context else "  Context: None")
        print(f"  Response: {result['response'][:80]}...")
        print(f"  Contains abstention signal: {result['contains_abstention_signal']}")
        if result['decision'].startswith("ABSTAIN"):
            print(f"  → ABSTAIN (model expressed uncertainty)")
        else:
            print(f"  → ANSWER: {result['decision'][:50]}...")
        print()

    print("─" * 80)
    print("📊 METHOD 1 OBSERVATION:")
    print("   Few-shot examples teach the model WHEN to say 'I don't know'.")
    print("   The model learns the pattern from examples, no fine-tuning needed.")
    print("   Detection relies on scanning for uncertainty phrases.")
    print("─" * 80)
    
    # ==================== METHOD 2 ====================
    print("\n" + "═" * 80)
    print("METHOD 2: UNCERTAINTY EXTRACTION")
    print("═" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  HOW UNCERTAINTY EXTRACTION WORKS (SURVEY § 4.3.2)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Principle: Combine EXPLICIT signals (uncertainty phrases) with            │
│  IMPLICIT signals (token probabilities) for robust detection.              │
│                                                                             │
│  From Paper:                                                                │
│    "Fine-tuning on SQuAD 2.0 and Abstain-QA teaches models to produce      │
│     'I don't know' responses for unanswerable questions"                   │
│                                                                             │
│  Combined Signals:                                                          │
│    1. Explicit: Does response contain "I don't know", "unclear", etc.?     │
│    2. Implicit: Are token probabilities low (model hesitating)?            │
│    3. Combined: (explicit_uncertainty + implicit_uncertainty) / 2          │
│                                                                             │
│  Why Combine Both:                                                          │
│    • Explicit alone: Model might say "I don't know" incorrectly            │
│    • Implicit alone: Low probs might be writing style, not uncertainty     │
│    • Combined: More robust signal from multiple sources                    │
│                                                                             │
│  Uncertainty Score Formula:                                                 │
│    uncertainty = 1 - avg_token_probability                                 │
│    final = (uncertainty + has_explicit_phrases) / 2                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("▶ Running Method 2 on test questions...")
    print("  Combining explicit and implicit uncertainty signals\n")

    for question, context in test_cases:
        result = abstainer.method_2_uncertainty_extraction(question, context)
        print(f"● Q: {question}")
        print(f"  Response: {result['response'][:60]}...")
        print(f"  Token Confidence: {result['token_confidence']:.3f}")
        print(f"  Has Explicit Uncertainty: {result['explicit_uncertainty_signals']}")
        print(f"  Combined Uncertainty Score: {result['combined_uncertainty_score']:.3f}")
        if result['decision'].startswith("ABSTAIN"):
            print(f"  → ABSTAIN (high uncertainty)")
        else:
            print(f"  → ANSWER: {result['decision'][:40]}...")
        print()

    print("─" * 80)
    print("📊 METHOD 2 OBSERVATION:")
    print("   Combining explicit + implicit signals is more robust than either alone.")
    print("   A model saying 'I don't know' with low confidence → strong abstention signal.")
    print("   A confident answer without uncertainty phrases → proceed with answer.")
    print("─" * 80)
    
    # ==================== METHOD 3 ====================
    print("\n" + "═" * 80)
    print("METHOD 3: CONFIDENCE STATEMENT PARSING")
    print("═" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  HOW CONFIDENCE STATEMENT PARSING WORKS (SURVEY § 4.3)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Principle: Ask the model to explicitly rate its own confidence,           │
│  then parse that rating to decide whether to abstain.                      │
│                                                                             │
│  Prompt Template:                                                           │
│    "Answer the question and rate your confidence (high/medium/low)"        │
│                                                                             │
│  Parsing Logic:                                                             │
│    • Response contains "high" or "certain" → confidence = 0.9              │
│    • Response contains "medium" or "somewhat" → confidence = 0.6           │
│    • Response contains "low" or "uncertain" → confidence = 0.3             │
│                                                                             │
│  Decision:                                                                  │
│    • If parsed confidence = "low" → ABSTAIN                                │
│    • Otherwise → return the answer                                         │
│                                                                             │
│  Key Advantage:                                                             │
│    • Model self-reports in interpretable terms                             │
│    • Easy for humans to understand the abstention reason                   │
│    • Can be calibrated by training on confidence-labeled data              │
│                                                                             │
│  Limitation:                                                                │
│    • Models may not be well-calibrated in self-assessment                  │
│    • Confidence statements may be overconfident (like token probs)         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("▶ Running Method 3 on test question...")
    print("  Parsing self-reported confidence level\n")

    for question, context in test_cases[:1]:
        result = abstainer.method_3_confidence_statement(question, context)
        print(f"● Q: {question}")
        print(f"  Response: {result['response'][:80]}...")
        print(f"  Parsed Confidence Level: {result['parsed_confidence_level']}")
        print(f"  Confidence Score: {result['confidence_score']:.2f}")
        if result['decision'].startswith("ABSTAIN"):
            print(f"  → ABSTAIN (low self-reported confidence)")
        else:
            print(f"  → ANSWER: {result['decision'][:50]}...")
        print()

    print("─" * 80)
    print("📊 METHOD 3 OBSERVATION:")
    print("   Self-reported confidence is interpretable but may not be calibrated.")
    print("   Models often over-report confidence (similar to softmax issue).")
    print("   Works best when combined with other uncertainty signals.")
    print("─" * 80)

    # ==================== SUMMARY ====================
    print("\n" + "═" * 80)
    print("VERBALIZED UNCERTAINTY: KEY INSIGHTS")
    print("═" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  COMPARISON OF VERBALIZED UNCERTAINTY METHODS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Method 1: Few-Shot Prompt Engineering                                      │
│    ✓ No training required                                                  │
│    ✓ Works with any instruction-tuned model                                │
│    ✗ Relies on model following examples consistently                       │
│    → Best for: Quick deployment without fine-tuning                        │
│                                                                             │
│  Method 2: Uncertainty Extraction (Explicit + Implicit)                     │
│    ✓ Robust: combines multiple signals                                     │
│    ✓ Catches uncertainty even when model doesn't verbalize                 │
│    ✗ Requires access to logits/probabilities                               │
│    → Best for: Maximum detection accuracy                                  │
│                                                                             │
│  Method 3: Confidence Statement Parsing                                     │
│    ✓ Interpretable: human-readable confidence                              │
│    ✓ Simple to implement                                                   │
│    ✗ Models may be overconfident in self-assessment                        │
│    → Best for: Systems requiring explainable abstention                    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  KEY INSIGHT FROM PAPER:                                                    │
│                                                                             │
│  "Verbalized uncertainty is more interpretable than confidence scores,     │
│   but requires either careful prompting or fine-tuning on abstention       │
│   datasets like SQuAD 2.0 to achieve reliable performance"                 │
│                                                                             │
│  The key is teaching the model WHEN to say "I don't know" -               │
│  either through examples (few-shot) or training data (fine-tuning).        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("═" * 80)
    print("END OF VERBALIZED UNCERTAINTY DEMONSTRATION")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    main()

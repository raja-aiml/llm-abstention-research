"""
TECHNIQUE 5: MULTI-AGENT SYSTEMS
From SURVEY.md § 4.5

Principle: Multiple agents collaborate to verify and cross-validate
responses before committing.

Three sub-methods:
- Voting/Consensus: Multiple agents, abstain on disagreement
- Hierarchical Refinement: Agent 1 generates, Agent 2 reviews
- Reasoning Verification: Agent 1 reasons, Agent 2 validates logic
"""

import torch
from collections import Counter
from utils import load_tokenizer_and_model


class MultiAgentSystem:
    """
    Implements multi-agent abstention.
    Mirrors paper's principle: Multiple agents + validation = better abstention.
    """
    
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1", num_agents=3):
        """Load model and tokenizer for all agents (same model, different prompts)"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer, self.model = load_tokenizer_and_model(model_name)
        self.num_agents = num_agents
    
    def _generate_response(self, prompt, temperature=0.7, max_tokens=50):
        """Helper: Generate single response"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True
            )
        
        response_ids = output[0, inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    
    def method_1_voting_consensus(self, question, context="", agreement_threshold=0.7):
        """
        METHOD 1: Voting/Consensus
        
        Paper principle (§4.5.1):
        Query N agents. Abstain if consensus < threshold.
        Formula: abstain if |{i : ŷ_i = ȳ}| / N < threshold
        
        Args:
            question: Query string
            context: Optional background
            agreement_threshold: Min agreement ratio (0-1)
        
        Returns:
            decision_dict: Votes, agreement, decision
        """
        
        prompt_template = "Context: {}\n\nQuestion: {}\nAnswer:"
        prompt = prompt_template.format(context if context else "No context.", question)
        
        # Get responses from multiple agents
        responses = []
        for agent_id in range(self.num_agents):
            # Vary temperature per agent for diversity
            temp = 0.5 + (0.2 * agent_id)
            response = self._generate_response(prompt, temperature=temp)
            responses.append(response)
        
        # Count votes
        vote_counts = Counter(responses)
        total_votes = len(responses)
        
        # Find majority
        if vote_counts:
            majority_response = max(vote_counts, key=vote_counts.get)
            majority_votes = vote_counts[majority_response]
            agreement_ratio = majority_votes / total_votes
        else:
            agreement_ratio = 0.0
            majority_response = None
        
        # Decision: Abstain if no clear consensus
        if agreement_ratio < agreement_threshold:
            decision = f"ABSTAIN: No consensus. Votes: {dict(vote_counts)}"
        else:
            decision = majority_response
        
        return {
            "responses": responses,
            "vote_counts": dict(vote_counts),
            "agreement_ratio": agreement_ratio,
            "agreement_threshold": agreement_threshold,
            "majority_answer": majority_response,
            "decision": decision,
            "method": "Voting/Consensus"
        }
    
    def method_2_hierarchical_refinement(self, question, context="", confidence_threshold=0.6):
        """
        METHOD 2: Hierarchical Refinement
        
        Paper principle (§4.5.2):
        Agent 1 generates answer. Agent 2 reviews and rates confidence.
        Abstain if reviewer is not confident.
        
        Args:
            question: Query string
            context: Optional background
            confidence_threshold: Min confidence from reviewer
        
        Returns:
            decision_dict: Agent 1 response, Agent 2 review, decision
        """
        
        # Agent 1: Generate answer
        prompt1 = f"Context: {context if context else 'No context.'}\n\nQuestion: {question}\nAnswer:"
        agent1_response = self._generate_response(prompt1, temperature=0.1)
        
        # Agent 2: Review agent 1's response
        review_prompt = f"""Question: {question}
Context: {context if context else 'No context.'}
Proposed answer: {agent1_response}

Is this answer correct and well-supported? Rate your confidence (0-1) and briefly explain:
Confidence:"""
        
        agent2_review = self._generate_response(review_prompt, temperature=0.1, max_tokens=30)
        
        # Parse confidence from agent 2
        # Try to extract a number
        confidence_score = self._extract_confidence(agent2_review)
        
        # Decision
        if confidence_score < confidence_threshold:
            decision = f"ABSTAIN: Reviewer confidence {confidence_score:.2f} below threshold"
        else:
            decision = agent1_response
        
        return {
            "agent1_response": agent1_response,
            "agent2_review": agent2_review,
            "agent2_confidence": confidence_score,
            "confidence_threshold": confidence_threshold,
            "decision": decision,
            "method": "Hierarchical Refinement"
        }
    
    def method_3_reasoning_verification(self, question, context=""):
        """
        METHOD 3: Reasoning Verification
        
        Paper principle (§4.5.3):
        Agent 1 generates reasoning steps. Agent 2 verifies logic.
        Abstain if logical errors detected or reasoning is weak.
        
        Args:
            question: Query string
            context: Optional background
        
        Returns:
            decision_dict: Reasoning, verification, decision
        """
        
        # Agent 1: Generate reasoning chain
        reasoning_prompt = f"""Context: {context if context else 'No context.'}
Question: {question}

Let me think step by step:"""
        
        reasoning = self._generate_response(reasoning_prompt, temperature=0.3, max_tokens=100)
        
        # Agent 2: Verify reasoning
        verification_prompt = f"""Question: {question}
Reasoning provided:
{reasoning}

Check this reasoning for:
1. Logical validity (are conclusions supported?)
2. Evidence usage (does it use given context?)
3. Completeness (does it address the question?)

Is the reasoning sound? (yes/no):"""
        
        verification = self._generate_response(verification_prompt, temperature=0.1, max_tokens=50)
        
        # Parse verification result
        verification_lower = verification.lower()
        is_sound = any(pos in verification_lower for pos in ["yes", "correct", "valid", "sound"])
        has_issues = any(neg in verification_lower for neg in ["no", "incorrect", "invalid", "unsound", "error"])
        
        reasoning_quality = 1.0 if is_sound else (0.0 if has_issues else 0.5)
        
        # Decision
        if reasoning_quality < 0.6:
            decision = "ABSTAIN: Reasoning failed verification"
        else:
            # Generate final answer based on verified reasoning
            answer_prompt = f"{reasoning}\n\nFinal answer to '{question}':"
            final_answer = self._generate_response(answer_prompt, temperature=0.1, max_tokens=50)
            decision = final_answer
        
        return {
            "reasoning": reasoning,
            "verification": verification,
            "reasoning_quality": reasoning_quality,
            "is_sound": is_sound,
            "decision": decision,
            "method": "Reasoning Verification"
        }
    
    def _extract_confidence(self, text):
        """Extract confidence score (0-1) from text"""
        text_lower = text.lower()
        
        # Try to find numeric pattern
        import re
        numbers = re.findall(r'0?\.\d+|[0-1]', text_lower)
        if numbers:
            try:
                score = float(numbers[0])
                return min(max(score, 0.0), 1.0)
            except:
                pass
        
        # Heuristic: keyword-based
        if any(word in text_lower for word in ["very confident", "definitely", "certain", "high"]):
            return 0.8
        elif any(word in text_lower for word in ["somewhat", "likely", "probably"]):
            return 0.6
        elif any(word in text_lower for word in ["uncertain", "unsure", "low", "doubt"]):
            return 0.3
        else:
            return 0.5


def main():
    """Test multi-agent systems on sample questions"""

    # Header
    print("\n" + "═" * 80)
    print("🎯 MULTI-AGENT SYSTEMS DEMONSTRATION")
    print("   From SURVEY § 4.5: Multi-Agent Abstention Methods")
    print("═" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  WHAT ARE MULTI-AGENT SYSTEMS FOR ABSTENTION?                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Core Idea: Multiple agents collaborate to verify and cross-validate       │
│  responses before committing to an answer.                                 │
│                                                                             │
│  From Paper (§ 4.5):                                                        │
│    "Multi-agent approaches leverage the wisdom of crowds principle -       │
│     disagreement between agents signals uncertainty"                       │
│                                                                             │
│  Key Insight:                                                               │
│    Single model: Might be confidently wrong (hallucination)                │
│    Multiple agents: Disagreement reveals uncertainty                       │
│                                                                             │
│  Three Approaches:                                                          │
│    1. Voting/Consensus: Query N agents, abstain if no agreement            │
│    2. Hierarchical Refinement: Agent 1 generates, Agent 2 reviews          │
│    3. Reasoning Verification: Agent 1 reasons, Agent 2 validates logic     │
│                                                                             │
│  Connection to Other Techniques:                                            │
│    • Similar to Ensemble Disagreement (§ 4.1) but with distinct agents     │
│    • Similar to Cross-Model Validation (§ 4.2) but with structured roles   │
│    • Adds explicit verification/review steps                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("📚 Reference: SURVEY § 4.5 - Multi-Agent Systems\n")

    print("⏳ Loading Mistral-7B-Instruct with 3 simulated agents...")
    agents = MultiAgentSystem(num_agents=3)
    print("✓ Model loaded (same model, different roles/temperatures)\n")
    
    test_cases = [
        ("What is the capital of France?", "France is in Western Europe."),
        ("Who invented the transistor?", ""),  # No context
        ("Explain quantum entanglement", "Physics topic"),
    ]
    
    # ==================== METHOD 1 ====================
    print("\n" + "═" * 80)
    print("METHOD 1: VOTING/CONSENSUS")
    print("═" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  HOW VOTING/CONSENSUS WORKS (SURVEY § 4.5.1)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Principle: Query N agents with the same question.                         │
│  If they don't agree, abstain - disagreement signals uncertainty.          │
│                                                                             │
│  Formula:                                                                   │
│    agreement_ratio = |{i : ŷ_i = majority}| / N                            │
│    ABSTAIN if agreement_ratio < threshold                                  │
│                                                                             │
│  Process:                                                                   │
│    Step 1: Query N agents (different temperatures for diversity)           │
│    Step 2: Collect N responses                                             │
│    Step 3: Find majority answer and count votes                            │
│    Step 4: If agreement < 70% → ABSTAIN                                    │
│                                                                             │
│  From Paper:                                                                │
│    "Voting-based consensus leverages the intuition that if multiple        │
│     independent agents disagree, the question is likely ambiguous"         │
│                                                                             │
│  Key Parameters:                                                            │
│    • N = number of agents (typically 3-5)                                  │
│    • agreement_threshold = minimum consensus (typically 0.6-0.8)           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("▶ Running Method 1 with 3 agents...")
    print("  Agreement threshold = 0.7 (abstain if < 70% agreement)\n")

    for question, context in test_cases[:1]:
        result = agents.method_1_voting_consensus(question, context)
        print(f"● Q: {question}")
        print(f"  Agent Responses:")
        for i, resp in enumerate(result['responses'], 1):
            print(f"    Agent {i}: {resp[:50]}...")
        print(f"  Vote Counts: {result['vote_counts']}")
        print(f"  Agreement Ratio: {result['agreement_ratio']:.2f}")
        if result['decision'].startswith("ABSTAIN"):
            print(f"  → ABSTAIN (no consensus)")
        else:
            print(f"  → ANSWER: {result['decision'][:50]}...")
        print()

    print("─" * 80)
    print("📊 METHOD 1 OBSERVATION:")
    print("   When agents agree → high confidence in the answer.")
    print("   When agents disagree → uncertainty signal → abstain.")
    print("   Trade-off: N× inference cost for better reliability.")
    print("─" * 80)
    
    # ==================== METHOD 2 ====================
    print("\n" + "═" * 80)
    print("METHOD 2: HIERARCHICAL REFINEMENT")
    print("═" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  HOW HIERARCHICAL REFINEMENT WORKS (SURVEY § 4.5.2)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Principle: Two agents with distinct roles:                                │
│    Agent 1 (Generator): Generates the answer                               │
│    Agent 2 (Reviewer): Reviews and rates confidence in Agent 1's answer    │
│                                                                             │
│  Process:                                                                   │
│    Step 1: Agent 1 generates answer to the question                        │
│    Step 2: Agent 2 receives (question, context, Agent 1's answer)          │
│    Step 3: Agent 2 reviews and rates confidence (0-1)                      │
│    Step 4: If Agent 2's confidence < threshold → ABSTAIN                   │
│                                                                             │
│  From Paper:                                                                │
│    "Hierarchical refinement separates generation from verification,        │
│     allowing specialized evaluation of answer quality"                     │
│                                                                             │
│  Agent 2's Review Prompt:                                                   │
│    "Is this answer correct and well-supported?                             │
│     Rate your confidence (0-1) and briefly explain"                        │
│                                                                             │
│  Key Advantage:                                                             │
│    • Separation of concerns: generate vs. validate                         │
│    • Reviewer can catch errors generator misses                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("▶ Running Method 2 with Generator + Reviewer agents...")
    print("  Confidence threshold = 0.6 (abstain if reviewer < 0.6)\n")

    for question, context in test_cases:
        result = agents.method_2_hierarchical_refinement(question, context)
        print(f"● Q: {question}")
        print(f"  Context: {context[:40]}..." if context else "  Context: None")
        print(f"  Agent 1 (Generator): {result['agent1_response'][:50]}...")
        print(f"  Agent 2 (Reviewer): {result['agent2_review'][:50]}...")
        print(f"  Reviewer Confidence: {result['agent2_confidence']:.2f}")
        if result['decision'].startswith("ABSTAIN"):
            print(f"  → ABSTAIN (reviewer not confident)")
        else:
            print(f"  → ANSWER: {result['decision'][:40]}...")
        print()

    print("─" * 80)
    print("📊 METHOD 2 OBSERVATION:")
    print("   Reviewer acts as a 'second opinion' on the answer.")
    print("   Low reviewer confidence → abstain even if generator was confident.")
    print("   Catches errors that single-agent systems miss.")
    print("─" * 80)
    
    # ==================== METHOD 3 ====================
    print("\n" + "═" * 80)
    print("METHOD 3: REASONING VERIFICATION")
    print("═" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  HOW REASONING VERIFICATION WORKS (SURVEY § 4.5.3)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Principle: Two agents with reasoning-focused roles:                       │
│    Agent 1 (Reasoner): Generates step-by-step reasoning                    │
│    Agent 2 (Verifier): Validates the logical soundness of reasoning        │
│                                                                             │
│  Process:                                                                   │
│    Step 1: Agent 1 generates reasoning chain ("Let me think step by step") │
│    Step 2: Agent 2 receives reasoning and validates:                       │
│            - Logical validity (are conclusions supported?)                 │
│            - Evidence usage (does it use given context?)                   │
│            - Completeness (does it address the question?)                  │
│    Step 3: If reasoning fails verification → ABSTAIN                       │
│                                                                             │
│  From Paper:                                                                │
│    "Reasoning verification catches logical errors and unsupported          │
│     conclusions that confidence-based methods miss"                        │
│                                                                             │
│  Verification Categories:                                                   │
│    • Sound/Valid/Correct → reasoning_quality = 1.0                         │
│    • Issues detected → reasoning_quality = 0.0                             │
│    • Unclear → reasoning_quality = 0.5                                     │
│                                                                             │
│  Key Advantage:                                                             │
│    • Catches logical errors in chain-of-thought reasoning                  │
│    • More rigorous than simple agreement checking                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("▶ Running Method 3 with Reasoner + Verifier agents...")
    print("  Reasoning quality threshold = 0.6 (abstain if quality < 0.6)\n")

    for question, context in test_cases[:1]:
        result = agents.method_3_reasoning_verification(question, context)
        print(f"● Q: {question}")
        print(f"  Agent 1 (Reasoning):")
        print(f"    {result['reasoning'][:70]}...")
        print(f"  Agent 2 (Verification):")
        print(f"    {result['verification'][:70]}...")
        print(f"  Reasoning Quality: {result['reasoning_quality']:.2f}")
        print(f"  Is Sound: {result['is_sound']}")
        if result['decision'].startswith("ABSTAIN"):
            print(f"  → ABSTAIN (reasoning failed verification)")
        else:
            print(f"  → ANSWER: {result['decision'][:40]}...")
        print()

    print("─" * 80)
    print("📊 METHOD 3 OBSERVATION:")
    print("   Verifier checks LOGICAL SOUNDNESS, not just agreement.")
    print("   Even if generator seems confident, bad reasoning → abstain.")
    print("   Particularly useful for complex multi-step questions.")
    print("─" * 80)

    # ==================== SUMMARY ====================
    print("\n" + "═" * 80)
    print("MULTI-AGENT SYSTEMS: KEY INSIGHTS")
    print("═" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  COMPARISON OF MULTI-AGENT METHODS                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Method 1: Voting/Consensus                                                 │
│    ✓ Simple: just count agreements                                         │
│    ✓ Works for factual questions                                           │
│    ✗ N× inference cost                                                     │
│    → Best for: Simple factual verification                                 │
│                                                                             │
│  Method 2: Hierarchical Refinement                                          │
│    ✓ Separation of generate vs. validate                                   │
│    ✓ Reviewer can be specialized for QA                                    │
│    ✗ 2× inference cost                                                     │
│    → Best for: Answer quality assessment                                   │
│                                                                             │
│  Method 3: Reasoning Verification                                           │
│    ✓ Catches logical errors in reasoning chains                            │
│    ✓ More rigorous than agreement checking                                 │
│    ✗ 2× cost + more complex prompts                                        │
│    → Best for: Complex multi-step reasoning                                │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  KEY INSIGHT FROM PAPER:                                                    │
│                                                                             │
│  "Multi-agent systems trade computational cost for reliability.            │
│   The cost is justified when errors are expensive - medical, legal,        │
│   financial decisions benefit from multi-agent verification"               │
│                                                                             │
│  When to use multi-agent:                                                   │
│    • High-stakes decisions (medical, legal, financial)                     │
│    • Complex reasoning (multi-step, mathematical)                          │
│    • When single-model confidence is unreliable                            │
│                                                                             │
│  When NOT to use:                                                           │
│    • High-throughput systems (chatbots, autocomplete)                      │
│    • Simple factual queries with clear answers                             │
│    • Latency-sensitive applications                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("═" * 80)
    print("END OF MULTI-AGENT SYSTEMS DEMONSTRATION")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    main()

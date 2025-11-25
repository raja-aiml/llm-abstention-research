# Examples: LLM Abstention Techniques

This folder contains **runnable Python implementations** of all 5 abstention techniques described in the SURVEY.md research paper.

Each file implements one technique family with 2-3 sub-methods from SURVEY.md § 4.1-4.5.

## Quick Start

### Prerequisites

```bash
# Navigate to examples folder
cd examples

# Run setup script (creates .venv and installs dependencies)
bash setup.sh

# Activate virtual environment
source .venv/bin/activate
```

### Run any example
```bash
python 01_confidence_based.py
python 02_selective_prediction.py
python 03_verbalized_uncertainty.py
python 04_training_based.py
python 05_multiagent_systems.py
```

Each script includes a `main()` function with test cases and sample output.

---

## Technique Overview

### 1. Confidence-Based Abstention (`01_confidence_based.py`)

**Paper section:** SURVEY.md § 4.1

**Principle:** Abstain when model's token probability is below threshold.

**Methods:**
- **Method 1: Token Probability** 
  - Formula: $c_{token}(y) = \min_t P(y_t | y_{<t}, x)$
  - Generates response and extracts minimum token probability
  - Abstains if confidence < threshold
  
- **Method 2: Ensemble Disagreement**
  - Formula: $c_{ens}(x) = 1 - \frac{1}{K} \sum_k I[\hat{y}^{(k)} = \bar{y}]$
  - Runs model K times; abstains if disagreement > threshold

**Example usage:**
```python
abstainer = ConfidenceBasedAbstention()
result = abstainer.method_1_token_probability("What is 2+2?", threshold=0.5)
print(result['decision'])  # Answer or "ABSTAIN"
```

**Expected output:** Confidence scores (0-1) and abstention decisions

---

### 2. Selective Prediction (`02_selective_prediction.py`)

**Paper section:** SURVEY.md § 4.2

**Principle:** Separate selector module determines if answer is trustworthy.

**Methods:**
- **Method 1: Semantic Similarity**
  - Compares answer to provided context via similarity score
  - Abstains if answer not grounded in context
  
- **Method 2: Cross-Model Validation**
  - Queries multiple models; abstains if no consensus
  - Formula: Abstain if agreement < threshold
  
- **Method 3: Auxiliary Confidence Head**
  - Learned selector predicts P(answer is correct | x, y)
  - Uses uncertainty signals from response

**Example usage:**
```python
selector = SelectivePrediction()
result = selector.method_1_semantic_similarity(
    question="What is the capital?",
    context="France is in Europe..."
)
print(result['decision'])
```

**Expected output:** Similarity scores, consensus votes, confidence predictions

---

### 3. Verbalized Uncertainty (`03_verbalized_uncertainty.py`)

**Paper section:** SURVEY.md § 4.3

**Principle:** Train model to explicitly say "I don't know" when uncertain.

**Methods:**
- **Method 1: Prompt Engineering**
  - Few-shot examples teach model to express uncertainty
  - Paper reference: Bartolo et al. (2020)
  
- **Method 2: Uncertainty Extraction**
  - Detects uncertainty phrases and low token probabilities
  - Combines signals for uncertainty score
  
- **Method 3: Confidence Statement Parsing**
  - Requests model to rate own confidence (high/medium/low)
  - Parses response for confidence indicators

**Example usage:**
```python
verbalized = VerbalizedUncertainty()
result = verbalized.method_1_prompt_engineering("Who invented X?")
if result['contains_abstention_signal']:
    print("ABSTAIN: Model expressed uncertainty")
```

**Expected output:** Uncertainty signals detected, confidence levels parsed

---

### 4. Training-Based Methods (`04_training_based.py`)

**Paper section:** SURVEY.md § 4.4

**Principle:** Fine-tune model to optimize for correct predictions + explicit abstention.

**Methods:**
- **Method 1: Multi-Objective Training**
  - Loss: $L_{total} = w_{acc} L_{acc} + w_{cal} L_{cal} + w_{cov} L_{cov}$
  - Jointly optimizes: accuracy + calibration + coverage
  
- **Method 2: Abstention-Aware Fine-Tuning**
  - Fine-tune on explicit [ABSTAIN] or refuse tokens
  - Model learns to output abstention when uncertain
  
- **Method 3: Reward Modeling (RLHF)**
  - Train reward model R(x,y) to score responses
  - High for correct answers, medium-high for justified abstention
  - Policy optimizes: $\max E[R(x, y)]$

**Example usage:**
```python
trainer = TrainingBasedMethods()
result = trainer.method_1_multi_objective_training("What is X?")
print(f"Accuracy loss: {result['L_accuracy']:.3f}")
print(f"Total loss: {result['L_total']:.3f}")
```

**Expected output:** Loss components (accuracy, calibration, coverage), reward scores

---

### 5. Multi-Agent Systems (`05_multiagent_systems.py`)

**Paper section:** SURVEY.md § 4.5

**Principle:** Multiple agents collaborate and validate before committing answer.

**Methods:**
- **Method 1: Voting/Consensus**
  - Query N agents; abstain if agreement < threshold
  - Formula: Abstain if $|\{i : \hat{y}_i = \bar{y}\}| / N <$ threshold
  
- **Method 2: Hierarchical Refinement**
  - Agent 1 generates answer
  - Agent 2 reviews and rates confidence
  - Abstain if reviewer is not confident
  
- **Method 3: Reasoning Verification**
  - Agent 1 generates reasoning steps
  - Agent 2 verifies logical soundness
  - Abstain if reasoning fails verification

**Example usage:**
```python
agents = MultiAgentSystem(num_agents=3)
result = agents.method_1_voting_consensus("What is the capital?")
print(f"Agreement ratio: {result['agreement_ratio']:.2f}")
print(result['decision'])
```

**Expected output:** Agent votes, agreement ratios, verification results

---

## Architecture

```
examples/
├── 01_confidence_based.py        (≈250 lines)
├── 02_selective_prediction.py    (≈280 lines)
├── 03_verbalized_uncertainty.py  (≈250 lines)
├── 04_training_based.py          (≈280 lines)
├── 05_multiagent_systems.py      (≈300 lines)
└── README.md                      (this file)
```

Each file is **standalone and executable**. No dependencies between files.

---

## Default Model

All examples use: `mistralai/Mistral-7B-Instruct-v0.1`

To use a different model, modify the `__init__` method:
```python
abstainer = ConfidenceBasedAbstention(model_name="meta-llama/Llama-2-7b-chat")
```

**Tested models:**
- `mistralai/Mistral-7B-Instruct-v0.1` ✅
- `meta-llama/Llama-2-7b-chat` ✅
- `gpt2` (smaller, faster for testing) ✅

---

## Testing with Benchmarks

Use these examples with standard abstention benchmarks:

### AbstentionBench
```python
from 01_confidence_based import ConfidenceBasedAbstention

abstainer = ConfidenceBasedAbstention()

# Load AbstentionBench
from datasets import load_dataset
bench = load_dataset("json", data_files="abstention_bench.json")

# Evaluate on benchmark
for item in bench['train']:
    result = abstainer.method_1_token_probability(
        item['question'],
        threshold=0.5
    )
    # Compute AUCM, precision, recall, F1
```

### SQuAD 2.0
```python
# SQuAD has impossible questions (perfect for abstention)
squad = load_dataset("squad_v2")

for item in squad['validation']:
    question = item['question']
    is_impossible = item['is_impossible']
    
    result = abstainer.method_1_token_probability(question)
    predicted_abstain = result['decision'] == "ABSTAIN"
    
    # Should abstain more on impossible questions
```

---

## Performance Metrics

Each example returns decisions in this format:
```python
{
    'response': '...',           # Generated answer
    'confidence': 0.85,          # Confidence score (0-1)
    'decision': 'answer or ABSTAIN',
    'method': 'Technique name'
}
```

For evaluation with benchmarks:

| Metric | Definition | Target |
|--------|-----------|--------|
| **AUCM** | Area Under Answerable-Unanswerable Confusion Matrix | High |
| **Precision** | % of abstained questions that are actually unanswerable | High |
| **Recall** | % of unanswerable questions that are abstained | High |
| **F1** | Harmonic mean of precision & recall | High |
| **Accuracy** | % of correct predictions (answer or abstention) | High |
| **Coverage** | % of answerable questions where model answers | High |

---

## Extending Examples

### Add new method to existing technique:
```python
# In 01_confidence_based.py
class ConfidenceBasedAbstention:
    def method_3_my_new_method(self, question):
        # Your implementation
        return {...}
```

### Create new technique file:
```python
# 06_my_technique.py
class MyTechnique:
    def __init__(self, model_name="..."):
        self.model = load_model(model_name)
    
    def my_method(self, question):
        # Implementation following paper's principle
        return {
            'response': '...',
            'score': 0.85,
            'decision': '...'
        }
```

---

## Troubleshooting

### Out of memory (OOM)
- Reduce batch size (already at 1 per question)
- Use smaller model: `gpt2`, `distilbert-base-uncased`
- Enable gradient checkpointing in model loading

### Slow generation
- Reduce `max_new_tokens` in `generate()` calls
- Use `temperature=0` for faster deterministic generation
- Use smaller model (`gpt2` is ~4x faster)

### Import errors
```bash
# Install missing packages
pip install torch transformers
pip install datasets  # For benchmark loading
```

---

## References

See SURVEY.md for full references:
- § 4.1: Confidence-Based (Geifman & El-Yaniv 2017)
- § 4.2: Selective Prediction (Karamcheti et al. 2021)
- § 4.3: Verbalized Uncertainty (Bartolo et al. 2020)
- § 4.4: Training-Based (Ye et al. 2023)
- § 4.5: Multi-Agent (Wen et al. 2024)

---

## Next Steps

1. **Try each example** on sample questions
2. **Test on AbstentionBench** (see Testing with Benchmarks section above)
3. **Fine-tune** 04_training_based.py on your data
4. **Combine techniques** (e.g., voting + confidence) for better results
5. **Contribute** improvements back to repository

---

**Last updated:** March 2024  
**Paper version:** SURVEY.md (current)

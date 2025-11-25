# Beginner's Guide: Understanding LLM Abstention

> 👋 **New here?** Start with this guide for simple explanations. For the full technical survey, see [SURVEY.md](SURVEY.md). For implementation details, see [examples/README.md](examples/README.md).

Welcome! This guide explains the core concepts of this survey in simple terms.

---

## 🎯 What's the Big Problem?

ChatGPT and similar AI models are **trained to always give an answer**, even when they don't actually know it. This causes them to:

- **Hallucinate**: Make up false information confidently
- **Confabulate**: Create coherent-sounding but completely wrong stories
- **Give dangerous advice**: Suggest wrong medical treatments, legal advice, etc.

**Example of a Hallucination**:
```
User: "What is the capital of Atlantis?"
ChatGPT: "The capital of Atlantis is Poseidia, located on the western coast..."
(Atlantis is fictional! The model should say "I don't know")
```

---

## 💡 What's the Solution?

**Abstention** = Teaching AI models to say **"I don't know"** when appropriate.

Instead of always guessing, the model should recognize uncertainty and refuse to answer.

**Example of Abstention**:
```
User: "What's the treatment for XYZ rare disease?"
ChatGPT (with abstention): "I'm not confident enough to give medical advice on this rare condition. Please consult a doctor."
```

This is much safer than confidently giving wrong medical information.

---

## 🔍 How Do We Know When a Model is Uncertain?

There are five main ways to detect uncertainty:

### 1. **Confidence Score** (Easiest)
- The model naturally produces a confidence score for each answer
- If confidence < threshold → abstain
- Think of it like the model saying "I'm only 30% sure"

### 2. **Multiple Attempts** (Ensemble)
- Ask the model the same question 5 times
- If it gives different answers each time → it's uncertain
- If it gives the same answer every time → more reliable

### 3. **Check Internal Signals**
- Look at the "probability" the model assigns to each word
- Lower probabilities = less confident

### 4. **Train a Confidence Predictor**
- Build a separate small model that learns: "When is the main model correct?"
- This learned model becomes your confidence checker

### 5. **Ask for Reasoning**
- Prompt: "Explain your answer step-by-step"
- If the explanation has gaps or contradictions → uncertain
- Example: "I think the capital is... wait, actually I'm not sure"

---

## 📊 How Do We Measure If Abstention Works?

Imagine you're testing a model on 100 questions:

**Traditional Accuracy**:
```
Correct answers: 85/100 = 85% accuracy
```

**But with Abstention**, we care about different things:

```
Example Results:
- 60 questions: Model answers (58 correct, 2 wrong) ✅ Good accuracy
- 30 questions: Model abstains (correctly knows it's uncertain) ✅ Good
- 10 questions: Model answers wrong (hallucinations) ❌ Bad

Metrics We Track:
- Accuracy on questions it answers: 58/60 = 96.7% (high!)
- Abstention precision: 30/30 = 100% (only abstains when right to)
- Not over-abstaining: Model answers enough questions
```

**The Trade-off**:
- If you make abstention threshold high → model refuses to answer a lot (safe but not helpful)
- If you make threshold low → model answers more (helpful but less safe)
- Goal: Find the sweet spot

---

## 🏆 What Are the Main Approaches?

### Approach 1: **Confidence-Based**
"Check how sure the model is"
- Simplest method
- Often not accurate enough on its own
- Example: ChatGPT's internal probability scores

### Approach 2: **Selective Prediction**
"Only answer if similar questions are in training data"
- Compares new question to training examples
- If very different → abstain
- Example: "This question style wasn't in my training data"

### Approach 3: **Verbalized Uncertainty**
"Ask the model to tell us if it's uncertain"
- Prompt: "Rate your confidence: High/Medium/Low"
- Model's own assessment matters
- Example: "Medium confidence - I've seen similar topics but not exactly this"

### Approach 4: **Training-Based**
"Teach the model abstention during training"
- Modify the training process to reward saying "I don't know"
- Most fundamental approach
- Example: Fine-tune on questions with explicit "unknown" labels

### Approach 5: **Multi-Agent Systems**
"Have multiple models vote"
- Run the same question through 3 different models
- If they disagree significantly → abstain
- If they agree → more confident
- Example: Ask GPT-4, Claude, and Llama the same question

---

## 🧪 Real-World Benchmarks

Researchers test abstention on datasets:

### **SQuAD2** (2018)
- 📖 [Dataset Link](https://rajpurkar.github.io/SQuAD-explorer/)
- Reading comprehension dataset
- 20% of questions are **unanswerable**
- Tests if model can recognize when no answer exists
- Example question: "What color was Napoleon's hat?" (answerable)
- Example unanswerable: "What was Napoleon's favorite pizza topping?" (never mentioned)

### **AbstentionBench** (2025, Meta AI)
- 🔗 [GitHub Repository](https://github.com/facebookresearch/AbstentionBench)
- 35,000+ questions across 20 datasets
- Specifically designed to test abstention
- Tests different LLMs (GPT-4, Claude, Llama, etc.)
- Key finding: **Reasoning models (o1, DeepSeek) paradoxically do WORSE at abstention**
  - Why? Because they're trained to answer everything thoroughly
  - They "reason through" even uncertain topics

### **Do-Not-Answer** (Safety-focused)
- 📄 [Paper & Dataset](https://arxiv.org/abs/2308.13387)
- Tests if model refuses harmful requests
- Example: "How do I make a weapon?"
- Model should refuse (not just be uncertain)

---

## 🚨 Key Findings from Research

1. **Scaling alone doesn't help**
   - Bigger models ≠ better at knowing when to abstain
   - GPT-4 is not necessarily more honest than GPT-3.5

2. **Fine-tuning backfires**
   - Training models to reason better actually makes abstention WORSE
   - Counterintuitive! But makes sense: reasoning models are trained to "figure it out"

3. **Simple methods work okay, but not great**
   - Just checking confidence scores catches maybe 60-70% of uncertain cases
   - Need combining multiple signals

4. **Prompting is temporary**
   - Asking nicely helps: "Be honest about uncertainty"
   - But models revert without reinforcement training

5. **Real solution requires architectural changes**
   - Need to change how models are trained from scratch
   - Current training processes reward confident answers

---

## 💼 Real-World Applications

### Healthcare
```
Doctor: "What's wrong with this patient?"
AI (without abstention): "Clearly lupus" (wrong!)
AI (with abstention): "This could be several things. Recommend blood tests."
```

### Legal
```
Lawyer: "Is this precedent relevant?"
AI (without): "Yes, Section 5 applies here" (hallucinated section!)
AI (with): "I'm not confident in this domain. Consult a law database."
```

### Finance
```
Investor: "What's Apple's Q3 2025 earnings?"
AI (without): "$125 billion" (made up!)
AI (with): "I don't have real-time data. Check official SEC filings."
```

---

## 🎓 Key Concepts to Remember

| Concept | Definition | Example |
|---------|-----------|---------|
| **Hallucination** | AI confidently generates false information | ChatGPT inventing a fake study |
| **Abstention** | Model refuses to answer when uncertain | "I don't have enough information" |
| **Confidence Score** | Model's own assessment of correctness (0-1) | 0.92 = "92% sure" |
| **Calibration** | Does confidence match actual accuracy? | If model says "90% confident," is it right 90%? |
| **Threshold** | Cutoff for when to abstain | Abstain if confidence < 0.7 |
| **Precision** | Of answers given, how many are correct? | 95/100 correct = 95% precision |
| **Recall** | Of all questions, how many does it answer? | Answers 100/150 questions = 67% recall |

---

## 🔬 How to Get Involved

If you're interested in this research:

1. **Explore the benchmarks**
   - Try [AbstentionBench](https://github.com/facebookresearch/AbstentionBench)
   - Test your favorite model

2. **Read key papers**
   - Start with: "Know Your Limits" (Wen et al., 2024)
   - Then: "Do LLMs Know When to NOT Answer?" (Madhusudhan et al., 2024)

3. **Experiment yourself**
   - Try prompting techniques: "Rate your confidence"
   - Compare multiple models on same questions
   - Build simple confidence baselines

4. **Ask interesting questions**
   - What happens with specialized domains?
   - Can we detect hallucinations before generation?
   - How do fine-tuned models compare?

---

## 📚 Further Reading

- **Main Survey**: Read [SURVEY.md](SURVEY.md) for comprehensive technical overview
- **Implementation Guide**: See [examples/README.md](examples/README.md) for code examples and testing

---

**Questions?** Feel free to explore the other documentation files or reach out to the research team!

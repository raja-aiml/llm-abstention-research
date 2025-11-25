# LLM Abstention Research

A comprehensive survey on abstention techniques in Large Language Models—methods enabling AI models to recognize uncertainty and appropriately refuse to answer.

## 📚 Documentation

Choose your starting point based on your background:

### 🎓 **For Beginners & Undergraduates**
→ **[BEGINNER_GUIDE.md](BEGINNER_GUIDE.md)**
- Simple explanations of core concepts
- Real-world examples (healthcare, legal, finance)
- Five main approaches explained clearly
- Key findings summarized for students
- Links to benchmarks and papers to explore

### 📖 **For Researchers & Practitioners**
→ **[SURVEY.md](SURVEY.md)**
- Comprehensive technical survey (50+ papers reviewed)
- Unified taxonomy of abstention techniques
- Evaluation frameworks and metrics (AUCM)
- Benchmark analysis (AbstentionBench, SQuAD2, Do-Not-Answer)
- Empirical findings across 20+ LLMs
- Discussion of limitations and open challenges

### 🛠️ **For Implementation & Testing**
→ **[examples/README.md](examples/README.md)**
- Runnable Python implementations of all 5 techniques
- How to use AbstentionBench for testing
- Step-by-step guide to evaluate your LLM
- AUCM metrics calculation
- Comparing multiple models



---

## 🎯 Quick Start

### Setup Environment (Required for Running Examples)

```bash
# Clone or navigate to repository
cd llm-abstention-research

# Go to examples folder
cd examples

# Run setup script (creates .venv and installs dependencies)
bash setup.sh

# Activate virtual environment
source .venv/bin/activate
```

**Just want to understand the basics?**
```
Start here: BEGINNER_GUIDE.md
Time: 10-15 minutes
```

**Need comprehensive technical details?**
```
Start here: SURVEY.md
Time: 30-60 minutes
```

**Want to run the code examples?**
```
1. Run: bash setup.sh
2. Then: python examples/01_confidence_based.py
   or explore: examples/README.md
Time: 20-40 minutes
```

---

## 🔍 What Is Abstention?

**The Problem**: ChatGPT and similar models are trained to always give an answer, even when they don't know it. This leads to hallucinations (making up false information confidently).

**The Solution**: **Abstention** = Teaching models to say "I don't know" when appropriate.

**Example**:
```
Question: "What's treatment for XYZ rare disease?"

❌ Without abstention: "Take this medication..." (probably wrong!)
✅ With abstention: "I'm not confident enough to advise on this rare condition. Consult a doctor."
```

---

## 🏆 Key Findings

1. **Abstention remains unsolved** — Current methods provide only modest improvements
2. **Scaling doesn't help** — Bigger models aren't automatically better at knowing when to abstain
3. **Reasoning backfires** — Fine-tuning for reasoning actually makes abstention worse
4. **Prompting is temporary** — Asking nicely helps briefly, but needs reinforcement training
5. **Architecture matters** — Fundamental changes to training are needed

---

## 📊 Repository Structure

```
llm-abstention-research/
├── README.md                    ← You are here (start here!)
├── BEGINNER_GUIDE.md            ← Easy explanations for beginners
├── SURVEY.md                    ← Full technical survey
├── .gitignore                   ← Git ignore rules
├── examples/                    ← Runnable implementations + testing guide
│   ├── README.md                ← Examples & AbstentionBench guide
│   ├── setup.sh                 ← Setup script (creates .venv)
│   ├── requirements.txt         ← Python dependencies
│   ├── 01_confidence_based.py
│   ├── 02_selective_prediction.py
│   ├── 03_verbalized_uncertainty.py
│   ├── 04_training_based.py
│   ├── 05_multiagent_systems.py
│   └── .venv/                   ← Virtual environment (created by setup.sh)
└── .git/                        ← Version control
```

---

## 📊 Benchmarks Discussed

| Benchmark | Year | Questions | Purpose | Link |
|-----------|------|-----------|---------|------|
| **SQuAD2** | 2018 | 100,000+ | Reading comprehension with unanswerable questions | [Dataset](https://rajpurkar.github.io/SQuAD-explorer/) |
| **AbstentionBench** | 2025 | 35,000+ | Comprehensive abstention evaluation (20+ LLMs) | [GitHub](https://github.com/facebookresearch/AbstentionBench) |
| **Do-Not-Answer** | 2023 | 939 | Safety-focused: model refusal of harmful requests | [arXiv](https://arxiv.org/abs/2308.13387) |
| **XSafety** | 2024 | 10,000+ | Multilingual safety across 14 languages | [arXiv](https://arxiv.org/abs/2405.18132) |

---

## 🔬 Five Approaches to Abstention

1. **Confidence-Based** — Check how sure the model is
2. **Selective Prediction** — Only answer if similar questions are in training data
3. **Verbalized Uncertainty** — Ask model to express its confidence
4. **Training-Based** — Teach abstention during model training
5. **Multi-Agent Systems** — Have multiple models vote

→ See **BEGINNER_GUIDE.md** for detailed explanations

---

## 📖 Recommended Reading Path

**Undergraduate / New to topic:**
1. Read: **BEGINNER_GUIDE.md** (15 min)
2. Explore: Benchmark links in guide

**Researcher / Quick overview:**
1. Read: **SURVEY.md** (60 min)

**Developer / Want to test:**
1. Read: **[examples/README.md](examples/README.md)** (30 min)
2. Follow: Step-by-step code examples
3. Test: On AbstentionBench with your LLM

---

## 🎓 Key Concepts Explained

| Term | Meaning | Example |
|------|---------|---------|
| **Hallucination** | AI confidently generates false info | Inventing a fake medical study |
| **Abstention** | Model refuses to answer when uncertain | "I don't have enough information" |
| **Calibration** | Does confidence match actual accuracy? | If 90% confident, is the answer correct 90% of the time? |
| **Threshold** | Cutoff for when to abstain | Abstain if confidence < 0.7 |
| **AUCM** | Answerable-Unanswerable Confusion Matrix | Metric framework for evaluating abstention |

---

## 💡 Real-World Applications

### Healthcare
AI assistant should refuse to diagnose rare diseases rather than guess

### Legal
Lawyer's AI tool should acknowledge unfamiliar case law rather than invent precedents

### Finance
Financial advisor AI should admit lack of real-time data rather than predict stock prices

---

## 🚀 Getting Started

1. **Choose your path** (beginner or researcher above)
2. **Read the relevant document** (takes 15-60 minutes)
3. **Explore the benchmarks** (links provided in guides)

---

## 🔗 External Resources

### Key Papers
- **Wen et al. (2024)**: "Know Your Limits" — [Direct MIT Press Link](https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl_a_00754)
- **Brahman et al. (2025)**: "AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions" — [arXiv](https://arxiv.org/abs/2506.09038)
- **Madhusudhan et al. (2024)**: "Do LLMs Know When to NOT Answer?" — [arXiv](https://arxiv.org/abs/2407.16221)

### Benchmarks
- **SQuAD2**: https://rajpurkar.github.io/SQuAD-explorer/
- **AbstentionBench**: https://github.com/facebookresearch/AbstentionBench
- **Do-Not-Answer**: https://arxiv.org/abs/2308.13387

### Tools & Communities
- **LaTeX Help**: [Stack Exchange TeX](https://tex.stackexchange.com/)
- **arXiv Submission**: [arXiv Help](https://arxiv.org/help/)
- **Research Discussion**: [NeurIPS Open Problems](https://openreview.net/), [ACL Anthology](https://aclanthology.org/)

---

## 📝 Citation

If you use this survey, cite it as:

```bibtex
@article{YourName2025,
  title={Abstention in Large Language Models: A Comprehensive Survey on Uncertainty Recognition and Refusal Mechanisms},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## 📬 Questions?

- **Beginner questions?** → Start with **BEGINNER_GUIDE.md**
- **Technical questions?** → Check **SURVEY.md** (§8 Discussion)
- **Implementation help?** → See **[examples/README.md](examples/README.md)** (with code)
- **General info?** → Check the README (you are here)

---

**Status**: Active Research Survey | **Last Updated**: November 2025 | **Papers Reviewed**: 50+ | **Coverage**: 2023-2025

**Happy learning!** 🚀

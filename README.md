# FinanceQA Lab  
**Model Failure Dashboard: Comprehensive Analysis of gpt-4o-mini on SEC-style FinanceQA test questions.**

[🔗 View the live app on Streamlit](https://ibraheemshaikh5-financeqa-lab-app-waktp3.streamlit.app/) | 
[🔗 View the project demo on Loom](https://www.loom.com/share/a855367165804854845eb09de1dfe93f)

---

## 1. Project Motivation  
In modern finance and equity research workflows, accuracy and reliability of language models are mission-critical. Mistakes in numerical reasoning, accounting conventions, or interpreting financial disclosures can lead to poor decisions.  

This project was built to:  
- **Evaluate a “weak” model** (`gpt‑4o‑mini`) across a set of real FinanceQA questions derived from SEC-style disclosures.  
- **Leverage a “strong” model** (`gpt‑4o`) to label and categorize failures into a refined taxonomy of error types (arithmetic, accounting convention, wrong metric, context misuse/hallucination, etc.).  
- **Surface actionable insights** via a polished dashboard for analysts, researchers, and developers.  

Most importantly, this workflow is a **step toward breaking the black-box of AI models**. By systematically cataloging **why models fail** and **how they reason incorrectly**, we generate structured datasets that reveal their shortcomings. This transparency allows us to **train, fine-tune, or guide models** to overcome these obstacles, turning model evaluation into **model improvement** — in line with the AfterQuery methodology of weak-to-strong reasoning pipelines.

---

## 2. What’s Included  
- `app.py` — Streamlit dashboard that loads the labeled dataset, computes key metrics, and lets users explore failures in depth.  
- `data/financeqa_labeled.csv` — Output of the evaluation pipeline: each question, the correct answer, `gpt‑4o‑mini` answer, assigned error label, and rationale from `gpt‑4o`.  
- Labeling script — Orchestrates sampling from `AfterQuery/FinanceQA`, calls `gpt‑4o‑mini` for answers, then `gpt‑4o` to assign JSON‑schema‑validated error labels.  
- **Error Taxonomy**: `CORRECT`, `ARITHMETIC_ERROR`, `ACCOUNTING_CONVENTION_ERROR`, `MISSING_OR_WRONG_ASSUMPTION`, `WRONG_METRIC_OR_CONCEPT`, `CONTEXT_MISUSE_OR_HALLUCINATION`, `NON_ANSWER_OR_GENERIC`.  
- **Polished UI** — KPI cards, error chips, step-through viewer, internal scroll panels, color-coded badges, and clean layout that's modeled after AfterQuery's UI.

---

## 3. How to Run Locally  
1. Clone repository:  
   ```bash
   git clone https://github.com/ibraheemshaikh5/financeqa-lab.git
   cd financeqa-lab

2. Install dependencies:
   ```bash
   pip install streamlit pandas altair python-dotenv

3. Ensure the labeled CSV exists at **data/financeqa_labeled.csv**.

4. Launch the dashboard 
   ```bash
   streamlit run app.py

5. Open browser to **http://localhost:8501** to view KPIs, error breakdowns, and deep-dive examples.

---

## 4. Dashboard Features

- **KPI Cards** – Display total samples, model accuracy, most common failure mode, and total failures.  
- **Error Distribution Chart** – Horizontal bar chart visualizing counts by error type.  
- **Interactive Table** – Searchable and filterable table showing each question with truncated text, ground truth answer, model answer, error label, and rationale.  
- **Step-Through Viewer** – Carousel-style panel to navigate individual examples:  
  - **Left panel:** Metadata and context (company, question type, source file).  
  - **Right panel:** Question, model answer, and rationale for the assigned error label.  
- **Design Features** – Fixed-height panels prevent infinite scrolling; color-coded badges, tight spacing, and consistent visual styling align with AfterQuery dashboards.

---

## 5. Extensions & Next Steps

- **HallBayes Integration** – Add hallucination-risk scoring per question to highlight outputs likely to be unreliable, enabling proactive monitoring and intervention.  
- **Failure Prediction Models** – Train classifiers (logistic regression, transformers) to predict weak-model failures before evaluation, based on question complexity, entity count, or context length.  
- **Multi-Model Comparison** – Compare multiple models side-by-side (gpt‑4o‑mini, open-source LLMs) to visualize relative strengths, weaknesses, and trends in improvement.  
- **Report Export / PDF** – Enable exporting dashboard state (metrics, filtered tables) for stakeholders and documentation.  
- **Live Evaluation / Upload Mode** – Allow uploading new CSVs or querying new questions for real-time evaluation pipelines.

---

## 6. Why This Approach Matters

- **From Evaluation to Understanding** – Rather than treating models as black boxes, this pipeline reveals the reasoning and failure patterns of gpt‑4o‑mini.  
- **Data-Driven Improvement** – The structured output (question, model answer, error type, rationale) creates a dataset that can be used to teach models how to avoid common mistakes, closing the gap between weak and strong performance.  
- **AfterQuery Methodology** – Weak-to-strong evaluation ensures systematic failure analysis and reproducibility. This is not just auditing a model — it is **enabling models to learn from their own failures**.

---

## 7. Attribution

- **Dataset:** FinanceQA (Hugging Face Hub)
- **Models:** gpt‑4o‑mini & gpt‑4o (OpenAI API)  
- **UI Design:** Streamlit dashboards inspired by AfterQuery; CSS styling generated with assistance from ChatGPT


This project is a first step toward transparent, accountable, and improvable AI in finance. By pairing weak-model evaluation with strong-model labeling and visualization, we move past the black box toward real understanding and targeted model improvement.

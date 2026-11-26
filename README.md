# Self-RAG–style Q&A over Indian Railways manuals

A question-answering system over two Indian Railways commercial manuals that checks its own work before answering. It is inspired by the Self-RAG paper (Asai et al., 2023), applied to a new domain and built at small scale. It is **not** a full reproduction of the paper.

The central question: *can a RAG system say "this is not in the documents" instead of inventing an answer?*

## How it works

```
Question
  -> 1. Retrieve?   Does this need the manuals at all?
  -> hybrid search (vector + keyword), top 5 chunks
  -> 2. Relevant?   Is each chunk useful for this exact question?
  -> 3. Draft       Write an answer from the relevant chunks
  -> 4. Supported?  Is every claim in the draft stated in those chunks?
  -> Final answer, or "I cannot find this in the documents."
```

A Streamlit app (`app.py`) shows each of these steps live. It runs locally; it is not deployed.

## Data

- Two Indian Railways commercial manuals ([fill in exact titles and the source URL]), downloaded on [date].
- Text extracted with PyMuPDF, cleaned, and split into **962 chunks** of about 300 words with 50 words of overlap.
- Search: Azure AI Search hybrid query (vector + keyword). Form codes such as `Com./R-14` are normalised to a single token, because plain keyword search splits them into useless pieces.

## Evaluation set (159 questions)

| Type | Count | How it was made |
|------|-------|-----------------|
| Answerable | 123 | Drafted by an LLM from random chunks, then reviewed. The first ~23 were checked by hand; the rest were reviewed by a second LLM, and I hand-checked a random sample of 30 (0 problems found). |
| Not answerable | 36 | "Decoy" questions that sound like they belong to the manuals but whose answer is not given (for example, the manual says "as prescribed" and gives no number). Each one was checked by hand against the top search results. |

Chunks and neighbours behind the test questions were excluded from all training data.

## Systems compared

1. **Normal RAG.** Search, then answer. No checks.
2. **Prompt Self-RAG.** The four steps above, with the checks done by prompting an Azure OpenAI model.
3. **Fine-tuned Self-RAG.** The same four steps, all done by a LoRA-tuned Qwen2.5-3B-Instruct (4-bit, Unsloth, free Colab T4). Trained on 1,975 examples labelled by the Azure model (distillation).
4. **Hybrid.** Fine-tuned 3B model does the checks, Azure model writes the answers.
5. **Azure Self-RAG.** System 2 again, running on Azure AI Search retrieval (the code path of the demo app).

Answers were graded by an Azure OpenAI model against reference answers.

## Results

| System | Answerable correct (of 123) | Answerable wrongly refused | Unanswerable correctly refused (of 36) | Balanced score |
|---|---|---|---|---|
| Normal RAG | 120 | 0 | 27 | 86% |
| Prompt Self-RAG | 118 | 1 | 33 | **94%** |
| Fine-tuned Self-RAG (3B) | 89 | 16 | 35 | 85% |
| Hybrid (3B checks, Azure writes) | 104 | 10 | 31 | 85% |
| Azure Self-RAG (Azure retrieval) | 118 | 1 | 33 | **94%** |

*Balanced score = average of (correct / answerable) and (correctly refused / unanswerable).*

Retrieval: the correct chunk was in the top 5 for 111/123 questions with a local Chroma + BM25 hybrid, and 115/123 with Azure AI Search.

Validation accuracy of the fine-tuned model on its three checks (held-out examples): Retrieve 27/27, Relevance 90/119 (76%), Support 39/45 (87%).

## What I learned

- **Self-checks work.** Prompt-based Self-RAG cut invented answers on unanswerable questions from 9/36 to 3/36 while losing only two answerable questions.
- **The fine-tuned 3B model did not beat its teacher.** It was the most cautious (1 invented answer in 36) but refused or got wrong 34 answerable questions. Failure analysis: about a third came from the small model's weaker answers, the rest from over-strict support and relevance checks.
- **Training data rebalancing can hurt calibration.** I downsampled "supported" examples to balance the support check, and the model then said "unsupported" too readily.
- **A debugging lesson.** The first training run silently used only 669 of 1,975 examples because a helper dropped examples; I found this by reading the trainer's log, and fixed it with my own label masking.
- **Known failure.** "For how many days is a 1,000 km single journey ticket valid?" can be answered wrongly: the rule for *circular* tickets is retrieved and both checks accept it.

## Limitations

- Small evaluation set (36 unanswerable questions): differences of 2-3 questions are not meaningful.
- Each system was run once, and repeat runs vary by several questions.
- An LLM graded the answers, and the same model family acted as teacher, so the grading may favour similar answers.
- The answerable questions are mostly single-fact lookups.

## Future work

More and cleaner training data for the relevance check; calibrate the support check's threshold instead of rebalancing the data; a larger base model; a control that fine-tunes the same model on the same data without reflection; repeated runs at temperature 0; measuring judge agreement against human labels.

## Repository guide

| Step | Files |
|------|-------|
| Data prep | `prep.py` |
| Local search | `build_index.py`, `hybrid.py` |
| Azure search | `build_azure_index.py`, `azure_search.py`, `eval_retrieval.py` |
| Evaluation set | `gen_questions.py`, `review.py`, `spot_check.py`, `check_unanswerable.py`, `make_eval_set.py` |
| Systems | `rag_baseline.py`, `selfrag_prompt.py`, `core.py`, `sft_prompts.py` |
| Running and grading | `run_baseline.py`, `run_selfrag_prompt.py`, `run_eval_core.py`, `judge.py`, `compare.py` |
| Fine-tuning data | `gen_train_questions.py`, `label_train.py`, `build_sft.py` |
| Fine-tuning | `notebooks/finetune_lora_colab.ipynb` |
| Analysis | `diagnose_ft.py`, `compare_runs.py`, `find_demo_examples.py` |
| Demo app | `app.py` |

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (never commit it) with:

```
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_DEPLOYMENT=
AZURE_EMBEDDING_DEPLOYMENT=
AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_ADMIN_KEY=
AZURE_SEARCH_INDEX=
```

The manuals and derived chunks are not included in this repository. Download the PDFs from the original source, put them in `data/raw/`, then run the scripts in the order of the table above. To launch the demo: `streamlit run app.py`.

## Reference

Asai, A., Wu, Z., Wang, Y., Sil, A., Hajishirzi, H. *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* arXiv:2310.11511 (2023); ICLR 2024.

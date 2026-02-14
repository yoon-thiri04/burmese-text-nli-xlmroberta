# mmNLI: Burmese Text Natural Language Inference with XLM-RoBERTa

A **Burmese Natural Language Inference (NLI)** model fine-tuned from **`xlm-roberta-base`**, trained on a curated Burmese NLI dataset combining cleaned native data, manual annotations, and translated English NLI samples.

This model predicts the relationship between a **premise** and a **hypothesis** as one of:

* **Entailment**
* **Neutral**
* **Contradiction**
## Model & Demo
- Model >> https://huggingface.co/emilyyy04/xlm-roberta-base-burmese-nli-v3
- Space Demo Link >> https://huggingface.co/spaces/emilyyy04/burmese-text-nli-xlmroberta

---
## Model Details

* **Base model:** `xlm-roberta-base`
* **Language:** Burmese (Myanmar)
* **Task:** Natural Language Inference (NLI)
* **Labels:** `entailment`, `neutral`, `contradiction`
* **Framework:** Transformers / PyTorch
---
## Dataset and its Structure

The dataset consists of **~10k [10,443] samples** across three classes:

| Label | Class         | Count |
| ----: | ------------- | ----: |
|     0 | Entailment    | 3,608 |
|     1 | Neutral       | 3,466 |
|     2 | Contradiction | 3,369 |

and the dataset is prepared from:

* Cleaned Burmese NLI data (source: *[(https://huggingface.co/datasets/akhtet/myanmar-xnli)]*)
* Additional **manually created** samples
* **Translated English NLI** (SNLI, multiNLI) data for diversity
* Most samples follow a **1 premise → 3 hypotheses** structure
* Each hypothesis has a **different NLI label**
* An additional **`genre`** field is included

  * Intended for **future zero-shot / cross-genre experiments**
  * Not used during training yet

## Preprocessing

Since a pretrained multilingual LLM is used, **no manual tokenization** (word-level or syllable-level) is applied.

Steps:

1. **Unicode normalization** (NFC)
2. **Zawgyi detection**
3. **Automatic conversion to Unicode** if Zawgyi text is detected
4. Rely on **XLM-R subword tokenizer** for tokenization

## Data Splitting Strategy

To prevent data leakage caused by shared premises:

* **Train:** 70%
* **Validation:** 15%
* **Test:** 15%

Instead of random shuffling:

* **GroupShuffleSplit** is used
* Samples with the **same premise always stay in the same split**
* Prevents:

  * Premise overlap across splits
  * Hypothesis leakage between train / validation / test sets
---


## Training Setup

The model was trained using **Hugging Face Transformers** with the following key configurations:

* **Model:** facebook/xlm-roberta-base
* **Epochs:** Up to **10 epochs**
* **Early Stopping:** Enabled (patience = 1)
* **Learning Rate:** `1.5e-5`
* **Batch Size:** `16` (train & evaluation)
* **Weight Decay:** `0.02`
* **Warmup Ratio:** `0.1`
* **FP16 Training:** Enabled
* **Best Model Selection:** Based on **F1-score**
* **Seed:** `42`

The best checkpoint was automatically loaded at the end of training using **early stopping** and **F1-based model selection**.

---

## Evaluation Metrics

The model performance is evaluated using:

* **Accuracy**
* **Macro F1-score**

## Training Results

| Epoch | Train Loss | Val Loss | Accuracy |     F1 |
| ----: | ---------: | -------: | -------: | -----: |
|     1 |     0.7579 |   0.7919 |   0.6415 | 0.6134 |
|     2 |     0.6764 |   0.6207 |   0.7291 | 0.7265 |
|     3 |     0.5801 |   0.6861 |   0.7443 | 0.7462 |
|     4 |     0.4554 |   0.6415 |   0.7481 | 0.7488 |
|     5 |     0.3848 |   0.6434 |   0.7646 | 0.7646 |
|     6 |     0.3564 |   0.7296 |   0.7608 | 0.7607 |

Training stopped early at **epoch 6** due to validation performance plateau and the best model saved is at the f1 score of 0.7646.


## Test Set Performance

```json
{
  "eval_loss": 0.5780,
  "eval_accuracy": 0.7877,
  "eval_f1": 0.7876
}
```

## Confusion Matrix on Test Set

Label order: **entailment, neutral, contradiction**

```
[[481  50  18]
 [ 78 418  20]
 [ 14  20 476]]
```

Rows represent **true labels**, columns represent **predicted labels**.

## Classification Report (Test Set)

| Class            | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| Entailment       | 0.84      | 0.88   | 0.86     | 549     |
| Neutral          | 0.86      | 0.81   | 0.83     | 516     |
| Contradiction    | 0.93      | 0.93   | 0.93     | 510     |
| **Accuracy**     |           |        | **0.87** | 1575    |
| **Macro Avg**    | 0.87      | 0.87   | 0.87     | 1575    |
| **Weighted Avg** | 0.87      | 0.87   | 0.87     | 1575    |


----
## Inference Example

You can use the model as follows:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "emilyyy04/xlm-roberta-base-burmese-nli"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

premise = "သူမသည် ဆေးရုံတွင် ဆရာဝန်အဖြစ် အလုပ်လုပ်နေသည်။"
hypothesis = "သူမသည် ကျန်းမာရေးလုပ်ငန်းတွင် အလုပ်လုပ်နေသည်။"

inputs = tokenizer(
    premise,
    hypothesis,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128
)

outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits, dim=1).item()

label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
print("Predicted label:", label_map[predicted_class])
# conf
probs = torch.softmax(outputs.logits, dim=-1)[0]
print("Confidence:", {k: round(float(probs[i]), 3) for i, k in label_map.items()})


```
---
## Limitations & Future Work

* Genre-aware and **zero-shot classification** is planned but not yet implemented
* Performance may vary for:

  * Very long inputs
  * Out-of-domain or highly informal Burmese
* Future improvements:

  * Larger native Burmese NLI dataset
  * Explicit genre-based evaluation
  * Domain adaptation

---

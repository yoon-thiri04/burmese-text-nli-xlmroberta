# MyanmarNLI: Burmese Text Natural Language Inference with XLM-RoBERTa

A **Burmese Natural Language Inference (NLI)** model fine-tuned from **`xlm-roberta-base`**, trained on a curated Burmese NLI dataset combining cleaned native data, manual annotations, and translated English NLI samples.

This model predicts the relationship between a **premise** and a **hypothesis** as one of:

* **Entailment**
* **Neutral**
* **Contradiction**

## Model Details

* **Base model:** `xlm-roberta-base`
* **Language:** Burmese (Myanmar)
* **Task:** Natural Language Inference (NLI)
* **Labels:** `entailment`, `neutral`, `contradiction`
* **Framework:** Transformers / PyTorch
---
## Dataset

The model is trained on an **~8K Burmese NLI dataset**, prepared from:

* Cleaned Burmese NLI data (source: *[(https://huggingface.co/datasets/akhtet/myanmar-xnli)]*)
* Additional **manually created** samples
* **Translated English NLI** data for diversity

### Dataset Structure

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

* **Epochs:** 4
* **Learning rate:** `2e-5`
* **Batch size:** 8
* **Weight decay:** `0.01`
* **Warmup ratio:** `0.1`
* **FP16 training**
* **Best model selected by:** Validation **F1 score**
* **Seed:** 42

## Evaluation Metrics

The model is evaluated using:

* **Accuracy**
* **Macro F1-score**

## Training Results 

| Epoch | Train Loss | Val Loss | Accuracy |     F1 |
| ----: | ---------: | -------: | -------: | -----: |
|     1 |     0.9509 |   0.8948 |   0.5602 | 0.5143 |
|     2 |     0.7850 |   0.6888 |   0.7233 | 0.7153 |
|     3 |     0.6067 |   0.6367 |   0.7660 | 0.7603 |
|     4 |     0.4301 |   0.7060 |   0.7455 | 0.7411 |

Best checkpoint selected based on **F1 score**.

## Test Set Performance

```json
{
  "eval_loss": 0.7713,
  "eval_accuracy": 0.7868,
  "eval_f1": 0.7852
}
```

## Confusion Matrix on Test Set

```
[[352  57  38]
 [ 53 270  34]
 [ 27  46 319]]
```

Rows represent **true labels**, columns represent **predicted labels**
(Label order: entailment, neutral, contradiction)

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
    padding=True
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

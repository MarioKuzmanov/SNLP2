## NOTE 
### for the purpose of our tutorial, we have only used a very small subset of the original data
### you can download it from [here](https://huggingface.co/datasets/syedkhalid076/Sentiment-Analysis).

---
datasets:
  - sentiment-analysis-dataset
language:
  - en
task_categories:
  - text-classification
task_ids:
  - sentiment-classification
tags:
  - sentiment-analysis
  - text-classification
  - balanced-dataset
  - oversampling
  - csv
pretty_name: Sentiment Analysis Dataset (Imbalanced)
dataset_info:
  features:
    - name: text
      dtype: string
    - name: label
      dtype: int64
  splits:
    - name: train
      num_examples: 83989
    - name: validation
      num_examples: 10499
    - name: test
      num_examples: 10499
  format: csv
---


# Sentiment Analysis Dataset

## Overview

This dataset is designed for sentiment analysis tasks, providing labeled examples across three sentiment categories:
- **0**: Negative
- **1**: Neutral
- **2**: Positive

It is suitable for training, validating, and testing text classification models in tasks such as social media sentiment analysis, customer feedback evaluation, and opinion mining.

---

## Dataset Details

### Key Features

- **Type**: CSV
- **Language**: English
- **Labels**: 
  - `0`: Negative 
  - `1`: Neutral 
  - `2`: Positive
- **Pre-processing**:
  - Duplicates removed
  - Null values removed
  - Cleaned for consistency

### Dataset Split

| Split        | Rows   |
|--------------|--------|
| **Train**    | 83,989 |
| **Validation** | 10,499 |
| **Test**     | 10,499 |

### Format

Each row in the dataset consists of the following columns:
- `text`: The input text data (e.g., sentences, comments, or tweets).
- `label`: The corresponding sentiment label (`0`, `1`, or `2`).

---

## Usage

### Installation

Download the dataset from the [Hugging Face Hub](https://huggingface.co/datasets/your-dataset-path) or your preferred storage location.

### Loading the Dataset

#### Using Pandas

```python
import pandas as pd

# Load the train dataset
train_df = pd.read_csv("path_to_train.csv")
print(train_df.head())

# Columns: text, label
```

#### Using Hugging Face's `datasets` Library

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("your-dataset-path")

# Access splits
train_data = dataset["train"]
validation_data = dataset["validation"]
test_data = dataset["test"]

# Example: Printing a sample
print(train_data[0])
```

---

## Example Usage

Here’s an example of using the dataset to fine-tune a sentiment analysis model with the [Hugging Face Transformers](https://huggingface.co/transformers) library:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("your-dataset-path")

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Train model
trainer.train()
```

---

## Applications

This dataset can be used for:
1. **Social Media Sentiment Analysis**: Understand the sentiment of posts or tweets.
2. **Customer Feedback Analysis**: Evaluate reviews or feedback.
3. **Product Sentiment Trends**: Monitor public sentiment about products or services.

---

## License

This dataset is released under the **[Insert Your Chosen License Here]**. Ensure proper attribution if used in academic or commercial projects.

---

## Citation

If you use this dataset, please cite it as follows:

```
@dataset{your_name_2024,
  title        = {Sentiment Analysis Dataset},
  author       = {Syed Khalid Hussain},
  year         = {2024},
  url          = {https://huggingface.co/datasets/syedkhalid076/Sentiment-Analysis}
}
```

---

## Acknowledgments

This dataset was curated and processed by **Syed Khalid Hussain**. The author takes care to ensure high-quality data, enabling better model performance and reproducibility.

---

**Author**: Syed Khalid Hussain
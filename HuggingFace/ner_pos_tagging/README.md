# Named Entity Recognition (NER) and Part-of-Speech (POS) Tagging

This project implements token-level classification for Named Entity Recognition (NER) and Part-of-Speech (POS) tagging using BERT on the CoNLL-03 dataset, with comprehensive handling of subword tokenization challenges.

## 📋 Project Overview

**Objective**: Build a token classification system that identifies named entities (persons, organizations, locations, miscellaneous) and grammatical categories in text using BERT.

**Key Components**:
- Token-level classification with BERT
- BIO tagging scheme implementation
- Subword tokenization alignment
- Label alignment for transformer models

## 🛠️ Implementation Details

### 1. Dataset Loading
```python
from datasets import load_dataset

# Load the CoNLL-03 NER dataset
dataset = load_dataset("conll2003")
```

**CoNLL-03 Structure**:
```python
{
  'tokens': ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'],
  'pos_tags': [22, 42, 16, 21, 35, 37, 16, 21, 7],
  'chunk_tags': [11, 21, 11, 12, 21, 22, 11, 12, 0],
  'ner_tags': [3, 0, 7, 0, 0, 0, 7, 0, 0]
}
```

### 2. Understanding NER and POS Tasks

#### Named Entity Recognition (NER)
**Purpose**: Identify and classify specific entities in text into predefined categories.

**Categories**:
- **B-PER/I-PER**: Person names (Barack Obama)
- **B-ORG/I-ORG**: Organizations (UNICEF) 
- **B-LOC/I-LOC**: Locations (New York)
- **B-MISC/I-MISC**: Miscellaneous entities
- **O**: Outside any named entity

#### Part-of-Speech (POS) Tagging
**Purpose**: Assign grammatical categories to each word.

**Categories**: Noun (NN), Verb (VB), Adjective (JJ), Preposition (IN), etc.

### 3. Tokenization Challenge
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Problem: Subword tokenization creates token/label mismatch
tokens = dataset["train"][0]["tokens"]
ner_tags = dataset["train"][0]["ner_tags"]

tokenized = tokenizer.tokenize(tokens, is_split_into_words=True)
# tokens: ['EU', 'rejects', 'German', 'call', ...]
# tokenized: ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'la', '##mb', '.']
```

**Core Problem**: `len(tokenized) != len(ner_tags)` due to subword splitting.

### 4. Label Alignment Solution
```python
def align_ner_target(word_ids, ner_tags):
    aligned_labels = []
    prev_word_id = None
    
    for word_id in word_ids:
        if word_id is None:
            # Special tokens ([CLS], [SEP])
            aligned_labels.append(-100)  # Ignore in loss computation
            
        elif word_id != prev_word_id:
            # First token of a word
            aligned_labels.append(ner_tags[word_id])
            prev_word_id = word_id
            
        else:
            # Subword token (repeated word_id)
            label = ner_tags[word_id]
            if label != 0 and label % 2 == 1:  # B-tags (odd indices)
                aligned_labels.append(label + 1)  # Convert B-XXX → I-XXX
            else:
                aligned_labels.append(label)  # O or already I-XXX
                
    return aligned_labels
```

**Key Concept**: BIO (Beginning-Inside-Outside) format consistency.

### 5. Understanding BIO Format
```python
# Example: "John Smith" gets split into ["John", "Sm", "##ith"]
# Original labels: [B-PER, I-PER]  
# Aligned labels:  [B-PER, I-PER, I-PER]

# Label mapping:
# B-PER (1) → I-PER (2)  [add 1]
# B-ORG (3) → I-ORG (4)  [add 1] 
# B-LOC (5) → I-LOC (6)  [add 1]
# B-MISC (7) → I-MISC (8) [add 1]
```

### 6. Complete Tokenization Pipeline
```python
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_label = align_ner_target(word_ids, label)
        labels.append(aligned_label)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
```

### 7. Model Configuration
```python
from transformers import AutoModelForTokenClassification

# Get label names from dataset
label_names = dataset["train"].features["ner_tags"].feature.names
# ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(label_names),
    id2label={i: label for i, label in enumerate(label_names)},
    label2id={label: i for i, label in enumerate(label_names)}
)
```

### 8. Data Collation
```python
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True
)
```

**Function**: Handles padding and ensures proper label alignment in batches.

### 9. Evaluation Metrics
```python
from evaluate import load

metric = load("seqeval")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens) and convert to labels
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
```

**seqeval**: Proper entity-level evaluation (not token-level).

### 10. Training Setup
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="ner_tags",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

## 🔑 Key Technical Concepts

### Word IDs and Offset Mapping
```python
sample = tokenizer(dataset["train"][0]["tokens"], is_split_into_words=True)
word_ids = sample.word_ids()
# [None, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, None]
#  [CLS] EU rejects German call to boycott British la ##mb . [SEP]
```

**Critical Insight**: Word ID `7` appears twice for "lamb" → ["la", "##mb"]

### Special Token Handling
- **[CLS], [SEP]**: Assigned `word_id = None`, get label `-100`
- **-100**: PyTorch ignores these positions in loss computation
- **Padding tokens**: Also get `-100` labels

### BIO Format Consistency
**Rule**: Within a multi-token entity, only the first token gets `B-` tag, subsequent tokens get `I-` tag.

## 📊 Applications

### Real-World Use Cases:
- **Information Extraction**: Extract entities from documents
- **Search Enhancement**: Better query understanding
- **Content Analysis**: Categorize and analyze text content
- **Privacy Protection**: Identify and mask personal information
- **Knowledge Graph Construction**: Build entity relationship networks

### Advanced Applications:
- **Nested NER**: Handle overlapping entities
- **Few-shot Learning**: New entity types with minimal data
- **Cross-lingual NER**: Multiple languages
- **Domain-Specific**: Medical, legal, financial entities

## 🚀 Extension Possibilities

### Model Improvements:
- **RoBERTa/DeBERTa**: More powerful base models
- **Span-based NER**: Alternative to token classification
- **Conditional Random Fields (CRF)**: Structured prediction
- **Multi-task Learning**: Joint NER and POS tagging

### Data Enhancement:
- **Active Learning**: Smart annotation strategies
- **Weak Supervision**: Leverage existing knowledge bases
- **Data Augmentation**: Synthetic entity generation

## 💡 Key Learnings

1. **Tokenization Alignment**: Critical challenge in transformer-based NLP
2. **BIO Format**: Proper entity boundary representation
3. **Label Propagation**: Handling subword tokens correctly
4. **Entity-level Evaluation**: Using seqeval for proper metrics
5. **Token Classification**: Different from sequence classification

This project demonstrates the complete pipeline for building production-ready token classification systems, essential for information extraction and text understanding applications.

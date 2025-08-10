# Fine-Tuning DistilBERT for Sentiment Classification

This project demonstrates how to fine-tune a pre-trained transformer model (DistilBERT) for sentiment analysis using the Hugging Face ecosystem.

## 📋 Project Overview

**Objective**: Fine-tune DistilBERT on sentiment analysis datasets (Amazon Polarity and GLUE SST-2) to classify text as positive or negative.

**Key Components**:
- Dataset loading and exploration
- Text tokenization for BERT models
- Model configuration and training
- Performance evaluation
- Inference pipeline creation

## 🛠️ Implementation Details

### 1. Dataset Loading
```python
from datasets import load_dataset

# Load two different sentiment datasets for comparison
raw_datasets_1 = load_dataset("amazon_polarity")
raw_datasets_2 = load_dataset("glue", "sst2")
```

**Datasets Used**:
- **Amazon Polarity**: Amazon product reviews with binary sentiment
- **GLUE SST-2**: Stanford Sentiment Treebank with binary classification

### 2. Tokenization Process
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenizer_fun(data):
    return tokenizer(data["sentence"], truncation=True)

tokenized_dataset = raw_datasets_2.map(tokenizer_fun, batched=True)
```

**Key Features**:
- Uses DistilBERT tokenizer (compatible with BERT but more efficient)
- Automatic truncation for long sequences
- Batch processing for efficiency

### 3. Model Setup
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=2
)
```

**Configuration**:
- Base model: DistilBERT (faster, smaller version of BERT)
- Task: Binary sequence classification
- Automatic classification head addition

### 4. Training Configuration
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1,
)
```

### 5. Evaluation Metrics
```python
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    pred = predictions.argmax(axis=1)
    acc = accuracy_score(labels, pred)
    return {"accuracy": acc}
```

### 6. Training Process
```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
```

### 7. Inference Pipeline
```python
from transformers import pipeline

classifier = pipeline("text-classification", model="mymodel", tokenizer=tokenizer)
result = classifier("This is a bad movie")
```

## 📊 Results

The model learns to classify sentiment with the following output format:
- **LABEL_0**: Negative sentiment (class 0)
- **LABEL_1**: Positive sentiment (class 1)
- **Score**: Confidence probability for the prediction

## 🔑 Key Learnings

1. **Pre-trained Models**: Provide excellent starting points for fine-tuning
2. **Tokenization**: Critical preprocessing step for transformer models
3. **Transfer Learning**: Adapts general language understanding to specific tasks
4. **Evaluation**: Continuous monitoring prevents overfitting
5. **Pipeline Interface**: Simplifies deployment and inference

## 📈 Next Steps

- Experiment with different model architectures (RoBERTa, ALBERT)
- Try multi-class sentiment analysis
- Implement more sophisticated evaluation metrics
- Add cross-validation for robust evaluation
- Deploy the model using FastAPI or Gradio

## 💡 Use Cases

- **Product Review Analysis**: Automatically classify customer feedback
- **Social Media Monitoring**: Track sentiment on social platforms
- **Content Moderation**: Filter negative comments
- **Market Research**: Analyze public opinion on products/services

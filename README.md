# Medical Data NER with Fine-tuned BERT

This project leverages BERT for Named Entity Recognition (NER) on a medical dataset. The notebook provides a step-by-step guide, from dataset preparation to fine-tuning and saving the trained model for medical NER tasks.

## Project Structure

1. **Dataset Loading**  
   Upload and load your medical dataset in JSON format:

   ```python
   from google.colab import files
   import json

   # Upload and load JSON data
   uploaded = files.upload()
   file_name = list(uploaded.keys())[0]
   with open(file_name, 'r') as f:
       dataset = json.load(f)
   ```

2. **Tokenization and BIO Tagging**  
   Use BERT's tokenizer to tokenize sentences and assign BIO tags to each token:

   ```python
   from transformers import BertTokenizer

   tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

   def tokenize_and_create_labels(sentence, entities, tokenizer):
       # Tokenize and label tokens
       tokenized_sentence = tokenizer.tokenize(sentence)
       labels = ['O'] * len(tokenized_sentence)
       # Assign 'B-' and 'I-' labels as per BIO tagging format
       # (Sample logic for locating and labeling entities)
       return tokenized_sentence, labels

   # Process each sentence in the dataset
   tokenized_data = []
   for item in dataset:
       tokens, labels = tokenize_and_create_labels(item["sentence"], item["entities"], tokenizer)
       tokenized_data.append({"tokens": tokens, "labels": labels})
   ```

3. **Tensor Conversion**  
   Convert tokenized data to tensors with padding for uniform input size:

   ```python
   import torch
   from sklearn.model_selection import train_test_split

   # Convert tokens to IDs, create attention masks, and pad inputs
   def convert_to_tensors(tokenized_data, tokenizer, max_length=128):
       # Example function for tensor conversion and padding
       return input_ids, attention_masks, label_ids

   input_ids, attention_masks, label_ids = convert_to_tensors(tokenized_data, tokenizer)
   train_inputs, test_inputs, train_labels, test_labels, train_masks, test_masks = train_test_split(
       input_ids, label_ids, attention_masks, test_size=0.2
   )
   ```

4. **Model Training**  
   Fine-tune the BERT model on the medical dataset using Hugging Face’s `Trainer`:

   ```python
   from transformers import BertForTokenClassification, Trainer, TrainingArguments

   model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(unique_labels))

   training_args = TrainingArguments(
         output_dir='./results',
         evaluation_strategy="epoch",  # Evaluate at the end of every epoch
         per_device_train_batch_size=16,
         per_device_eval_batch_size=16,
         num_train_epochs=15,
         weight_decay=0.01,
         logging_dir='./logs',
         logging_steps=100,
   )

   trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
   trainer.train()
   ```

5. **Saving the Model**  
   Save the trained model weights for future use:
   ```python
   model_save_path = "model.pt"
   torch.save(model.state_dict(), model_save_path)
   print(f"Model saved as {model_save_path}")
   ```

6. **Evaluation Metrics**
   
      | Metric                  | BERT Model |
      |-------------------------|------------|
      | **Accuracy**            | 96%        |
      | **Precision (Average)** | 96%        |
      | **Recall (Average)**    | 96%        |
      | **F1 Score (Average)**  | 96%        |
   
8. **NER Inferences**
   Enter a sentence for entity recognition: Mrs. Williams attends physical therapy to improve her mobility after hip replacement surgery.

   ```
      Token-Level Predictions:
      [
      {
         "token": "Mrs",
         "label": "B-Person"
      },
      {
         "token": ".",
         "label": "I-Person"
      },
      {
         "token": "Williams",
         "label": "I-Person"
      },
      {
         "token": "attends",
         "label": "O"
      },
      {
         "token": "physical",
         "label": "B-Service"
      },
      {
         "token": "therapy",
         "label": "I-Service"
      },
      {
         "token": "to",
         "label": "O"
      },
      {
         "token": "improve",
         "label": "B-Outcome"
      },
      {
         "token": "her",
         "label": "O"
      },
      {
         "token": "mobility",
         "label": "B-Outcome"
      },
      {
         "token": "after",
         "label": "O"
      },
      {
         "token": "hip",
         "label": "B-MedicalProcedure"
      },
      {
         "token": "replacement",
         "label": "I-MedicalProcedure"
      },
      {
         "token": "surgery",
         "label": "I-MedicalProcedure"
      },
      {
         "token": ".",
         "label": "O"
      }
      ]

      Entity-Level JSON Output:
      {
      "sentence": "Mrs. Williams attends physical therapy to improve her mobility after hip replacement surgery.",
      "entities": [
         {
            "text": "Mrs . Williams",
            "label": "Person"
         },
         {
            "text": "physical therapy",
            "label": "Service"
         },
         {
            "text": "improve",
            "label": "Outcome"
         },
         {
            "text": "mobility",
            "label": "Outcome"
         },
         {
            "text": "hip replacement surgery",
            "label": "MedicalProcedure"
         }
      ]
      }
   ```

## Requirements

- Python 3.x
- Libraries: `transformers`, `torch`, `scikit-learn`

## How to Run

1. **Install Requirements**
   ```bash
   pip install transformers torch scikit-learn
   ```
2. **Upload the Dataset**  
   Place your JSON dataset in the same directory or upload it in the notebook.
3. **Run the Notebook**  
   Execute each cell sequentially to preprocess data, train, and save the model.

## Acknowledgments

- Hugging Face `transformers`
- PyTorch for tensor manipulation

---

Let me know if you'd like further customization!

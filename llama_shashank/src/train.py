"""
PLASMA: Perspective-aware Language Model Adaptation for Medical Summarization
Complete Implementation Pipeline with Llama 3.2 and Two-Stage Adaptation
"""

import json
import argparse
import os
import sys
import math
import pandas as pd
import numpy as np
import torch
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    PeftModel, 
    TaskType
)
from scipy.spatial.distance import cosine
from transformers import (
    RobertaForSequenceClassification, 
    RobertaTokenizer,
    BertTokenizer,
    BertModel
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from bert_score import score as bert_score

# Suppress warnings
warnings.filterwarnings("ignore")

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#######################
# EVALUATION COMPONENTS
#######################

# Load BERT and RoBERTa for perspective detection
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5).to(device)

# Load classifier checkpoint if available
ckpt_path = "./classifier/checkpoints/best_ckpt_epoch=5_valid_loss=0.2561.ckpt"
if os.path.exists(ckpt_path):
    print("Loading the trained classifier checkpoint...")
    ckpt = torch.load(ckpt_path, weights_only=False)
    roberta_model.load_state_dict(ckpt['model_state_dict'])
    print("Classifier loaded successfully.")
else:
    print("Warning: Classifier checkpoint not found. Using base model.")

def get_bert_embedding(text):
    """Get BERT embeddings for a text"""
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def calculate_rouge_score_for_each_phrase(predictions, references):
    """Calculate ROUGE scores between predictions and references"""
    rouge = Rouge()
    rouge_l_f1_scores = []

    for prediction, reference in zip(predictions, references):
        scores = rouge.get_scores(prediction.lower(), reference.lower())[0]
        rouge_l_f1 = scores["rouge-1"]["f"]
        rouge_l_f1_scores.append(rouge_l_f1)

    return rouge_l_f1_scores

def score_all_phrases(summary, phrases):
    """Score a summary against a list of phrases"""
    start_of_summary = ' '.join(summary.split()[:4])
    predictions = [start_of_summary] * len(phrases)
    references = phrases
    rouge_l_f1_results = calculate_rouge_score_for_each_phrase(predictions, references)
    phrase_scores = dict(zip(phrases, rouge_l_f1_results))
    return phrase_scores

def Ep(generated_summary):
    """Perspective-specific energy function"""
    inputs = roberta_tokenizer(generated_summary, padding=True, truncation=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = roberta_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = outputs.logits.argmax(dim=-1)

    class_labels = {0: "EXPERIENCE", 1: "SUGGESTION", 2: "INFORMATION", 3: "CAUSE", 4: "QUESTION"}
    predicted_label = class_labels[predictions[0].item()]

    final = {}
    for i in range(0, 5):
        final[class_labels[i]] = probabilities[0][i].cpu().numpy().item()

    return final

def Es(generated_summary):
    """Style-specific energy function"""
    phrases = [
        "In user's experience…",
        "It is suggested",
        "For information purposes",
        "Some of the causes",
        "It is inquired"
    ]
    
    phrase_scores = score_all_phrases(generated_summary, phrases)
    return phrase_scores

def Et(generated_summary):
    """Tone-specific energy function"""
    l_sugg = ["Advisory", "Recommending", "Cautioning", "Prescriptive", "Guiding", "Prescriptive"]
    l_exp = ["Personal", "Narrative", "Introspective", "Exemplary", "Insightful", "Emotional"]
    l_info = ["Clinical", "Scientific", "Informative", "Educational", "Factual", "Informing", "Academic", "Analytical"]
    l_cause = ["Diagnostic", "Explanatory", "Causal", "Due to", "Resulting from", "Attributable to"]
    l_qs = ["Inquiry", "Rhetorical", "Exploratory Questioning", "Clarifying Inquiry", "Problem-Solving Deliberation"]

    summary_embedding = get_bert_embedding(generated_summary)
    cosine_similarities = {}

    for label, word_list in zip(['sugg', 'exp', 'info', 'cause', 'qs'], 
                              [l_sugg, l_exp, l_info, l_cause, l_qs]):
        combined_text = ' '.join(word_list)
        word_embedding = get_bert_embedding(combined_text)
        similarity = 1 - cosine(summary_embedding.cpu().detach().numpy(), 
                               word_embedding.cpu().detach().numpy())
        cosine_similarities[label] = similarity

    return cosine_similarities

def compute_perspective_loss(model, tokenizer, input_ids, attention_mask, perspective):
    """Compute perspective-aware energy-based loss"""
    model.eval()
    
    # Generate summary
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=50,
        temperature=0.9,
        do_sample=True,
        top_p=0.92,
        top_k=50
    )
    
    generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if len(generated_summary) <= 0:
        generated_summary = 'None'
    
    # Calculate energies
    Ep_dict = Ep(generated_summary)
    Es_dict = Es(generated_summary)
    Et_dict = Et(generated_summary)

    # Weights for different energy components
    alpha = 0.7  
    beta = 0.3   
    gamma = 0.5  

    perspective_types = ["EXPERIENCE", "SUGGESTION", "INFORMATION", "CAUSE", "QUESTION"]
    
    # Combined energy function
    E_X = {
        "EXPERIENCE": alpha * Ep_dict["EXPERIENCE"] + beta * Es_dict["In user's experience…"] + gamma * Et_dict['exp'],
        "SUGGESTION": alpha * Ep_dict["SUGGESTION"] + beta * Es_dict["It is suggested"] + gamma * Et_dict['sugg'],
        "INFORMATION": alpha * Ep_dict["INFORMATION"] + beta * Es_dict["For information purposes"] + gamma * Et_dict['info'],
        "CAUSE": alpha * Ep_dict["CAUSE"] + beta * Es_dict["Some of the causes"] + gamma * Et_dict['cause'],
        "QUESTION": alpha * Ep_dict["QUESTION"] + beta * Es_dict["It is inquired"] + gamma * Et_dict['qs']
    }

    # Compute probabilities using exponential of negative energy
    exp_E_X = {k: math.exp(-1/(v + 1e-10)) for k, v in E_X.items()}
    Z = sum(exp_E_X.values())
    P_X = {k: v / Z for k, v in exp_E_X.items()}
    
    # Set target labels
    Y = {k: 0.0 for k in perspective_types}
    if perspective[0] in Y:
        Y[perspective[0]] = 1.0
    
    # Convert to tensors
    P_X_tensor = torch.tensor([P_X[k] for k in perspective_types]).to(device)
    Y_tensor = torch.tensor([Y[k] for k in perspective_types]).to(device)
    
    # Avoid numerical issues
    P_X_tensor = torch.clamp(P_X_tensor, min=1e-10)
    
    # Cross-entropy loss between target perspective and predicted distribution
    loss = -torch.sum(Y_tensor * torch.log(P_X_tensor))
    return loss

#######################
# DATASETS
#######################

class MedicalDataset(Dataset):
    """Dataset for medical domain adaptation"""
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format for medical domain adaptation
        prompt = f"""<|system|>
You are a healthcare AI assistant specialized in providing accurate medical information to patients and healthcare professionals.
<|user|>
{item['Question']}
<|assistant|>
"""
        
        # Full text includes prompt and response
        full_text = f"{prompt}{item['Answer']}"
        
        # Tokenize with padding
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoded.input_ids.squeeze()
        attention_mask = encoded.attention_mask.squeeze()
        
        # Determine prompt length to mask loss
        prompt_encoded = self.tokenizer(prompt, return_tensors="pt")
        prompt_length = prompt_encoded.input_ids.shape[1]
        
        # Create labels: -100 for prompt tokens, actual ids for target tokens
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class PerspectiveDataset(Dataset):
    """Dataset for perspective-aware summarization with structured prompts"""
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        perspective = item['Perspective'].strip().upper()
        summary = item['Summary']

        definitions = {
            "SUGGESTION": "Defined as advice or recommendations to assist users in making informed medical decisions, solving problems, or improving health issues.",
            "INFORMATION": "Defined as knowledge about diseases, disorders, and health-related facts, providing insights into symptoms and diagnosis.",
            "EXPERIENCE": "Defined as individual experiences, anecdotes, or firsthand insights related to health, medical treatments, medication usage, and coping strategies",
            "CAUSE": "Defined as reasons responsible for the occurrence of a particular medical condition, symptom, or disease",
            "QUESTION": "Defined as inquiry made for deeper understanding."
        }

        beginnings = {
            "SUGGESTION": "It is suggested",
            "INFORMATION": "For information purposes",
            "EXPERIENCE": "In user's experience",
            "CAUSE": "Some of the causes",
            "QUESTION": "It is inquired"
        }

        tones = {
            "SUGGESTION": "Advisory, Recommending",
            "INFORMATION": "Informative, Educational",
            "EXPERIENCE": "Personal, Narrative",
            "CAUSE": "Explanatory, Causal",
            "QUESTION": "Seeking Understanding"
        }

        definition = definitions.get(perspective, "")
        begin_summary_with = beginnings.get(perspective, "")
        tone = tones.get(perspective, "")

        # Ensure summary starts correctly
        if not any(begin_summary_with in summary[:len(begin_summary_with) + 10] for _ in [0]):
            summary = f"{begin_summary_with} {summary}"

        combined_answers = ' '.join([ans.replace('\n', '') for ans in item['answers']])
        
        # Prompt format
        prompt_template = f"""<|system|>
You are an AI assistant specialized in healthcare summarization. Your task is to summarize medical information from a specific perspective while maintaining accuracy.
<|user|>
I need you to summarize the following medical content from the {perspective} perspective.
{perspective} Definition: {definition}
Begin Summary with: {begin_summary_with}
Tone of summary: {tone}
Content to summarize:
{combined_answers}
Associated medical question: {item.get('question', '')}
Please provide a concise, accurate summary from the {perspective} perspective.
<|assistant|>
"""

        full_text = f"{prompt_template}{summary}"

        # Tokenization
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded.input_ids.squeeze()
        attention_mask = encoded.attention_mask.squeeze()

        # Mask loss for the prompt
        prompt_encoded = self.tokenizer(prompt_template, truncation=True, return_tensors="pt")
        prompt_length = prompt_encoded.input_ids.shape[1]

        labels = input_ids.clone()
        labels[:prompt_length] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'Perspective': perspective,
            'Summary': summary
        }

def create_dataloader(dataset, batch_size, shuffle=True):
    """Create DataLoader from Dataset"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

#######################
# TRAINING FUNCTIONS
#######################

def train_medical_adapter(args, tokenizer, csv_path="../data/adapter_train.csv"):
    """Train medical domain adapter"""
    print("Starting medical domain adaptation...")
    
    # Load medical data from CSV
    print(f"Loading medical data from {csv_path}")
    medical_df = pd.read_csv(csv_path)
    medical_data = medical_df.to_dict('records')
    print(f"Loaded {len(medical_data)} medical QA pairs")
    
    # Initialize model with quantization
    print("Loading base Llama model with quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        quantization_config=bnb_config
    )
    
    # Configure LoRA for medical adaptation
    print("Configuring LoRA for medical domain adaptation...")
    medical_lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    model = get_peft_model(base_model, medical_lora_config)
    model.print_trainable_parameters()
    
    # Create dataset and dataloader
    medical_dataset = MedicalDataset(medical_data, tokenizer)
    medical_dataloader = create_dataloader(
        medical_dataset,
        batch_size=args.batch_size_medical
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.medical_lr)
    steps_per_epoch = len(medical_dataloader)
    total_steps = steps_per_epoch * args.medical_epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.medical_warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"Starting medical domain training for {args.medical_epochs} epochs...")
    model.train()
    
    for epoch in range(args.medical_epochs):
        epoch_losses = []
        progress_bar = tqdm(medical_dataloader, desc=f"Medical Epoch {epoch+1}/{args.medical_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update progress
            epoch_losses.append(loss.item())
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Log every 50 steps
            if step % 50 == 0:
                print(f"Medical Epoch {epoch+1}, Step {step}/{steps_per_epoch}, Loss: {loss.item():.4f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Medical Epoch {epoch+1} complete, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        os.makedirs(args.medical_ckpt_dir, exist_ok=True)
        checkpoint_path = f"{args.medical_ckpt_dir}/medical_adapter_epoch_{epoch+1}"
        model.save_pretrained(checkpoint_path)
        print(f"Saved medical adapter checkpoint to {checkpoint_path}")
    
    # Save final medical adapter
    final_adapter_path = f"{args.medical_ckpt_dir}/final_medical_adapter"
    model.save_pretrained(final_adapter_path)
    print(f"Medical domain adaptation complete. Saved to {final_adapter_path}")
    
    return final_adapter_path

def validate(model, tokenizer, eval_dataloader, epoch):
    """Validate the perspective-aware model"""
    print(f"Running validation for epoch {epoch}...")
    model.eval()
    
    valid_losses = []
    perspective_losses = []
    generated_summaries = []
    reference_summaries = []
    perspectives = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Validation")):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Calculate perspective loss
            perspective_loss = compute_perspective_loss(
                model, 
                tokenizer, 
                input_ids, 
                attention_mask, 
                batch["Perspective"]
            )
            
            # Combined loss
            loss = outputs.loss + perspective_loss
            
            valid_losses.append(outputs.loss.item())
            perspective_losses.append(perspective_loss.item())
            
            # Generate text for evaluation
            if batch_idx % 10 == 0:  # Generate every 10 batches to save time
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=100,
                    temperature=0.9,
                    do_sample=True,
                    top_p=0.92,
                    top_k=50
                )
                
                generated_full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                assistant_tag = "<|assistant|>"
                if assistant_tag in generated_full_text:
                    generated_text = generated_full_text.split(assistant_tag)[1].strip()
                else:
                    generated_text = generated_full_text

                generated_summaries.append(generated_text)
                reference_summaries.append(batch['Summary'][0])
                perspectives.append(batch['Perspective'][0])
                
                print(f"\nValidation Sample {batch_idx}:")
                print(f"Perspective: {batch['Perspective'][0]}")
                print(f"Generated: {generated_text[:100]}...")
                print(f"Reference: {batch['Summary'][0][:100]}...")
    
    # Calculate average losses
    avg_loss = sum(valid_losses) / len(valid_losses)
    avg_perspective_loss = sum(perspective_losses) / len(perspective_losses)
    
    print(f"Validation CE Loss: {avg_loss:.4f}")
    print(f"Validation Perspective Loss: {avg_perspective_loss:.4f}")
    print(f"Total Validation Loss: {avg_loss + avg_perspective_loss:.4f}")
    
    # Calculate metrics if we have generated samples
    if generated_summaries:
        # Calculate ROUGE
        rouge = Rouge()
        rouge_scores = rouge.get_scores(generated_summaries, reference_summaries, avg=True)
        
        # Calculate perspective adherence
        perspective_adherence = []
        
        for gen, persp in zip(generated_summaries, perspectives):
            Ep_dict = Ep(gen)
            target_score = Ep_dict[persp]
            perspective_adherence.append(target_score)
        
        avg_adherence = sum(perspective_adherence) / len(perspective_adherence)
        
        print(f"ROUGE-1: {rouge_scores['rouge-1']['f']:.4f}")
        print(f"ROUGE-2: {rouge_scores['rouge-2']['f']:.4f}")
        print(f"ROUGE-L: {rouge_scores['rouge-l']['f']:.4f}")
        print(f"Perspective Adherence: {avg_adherence:.4f}")
    
    return avg_loss + avg_perspective_loss

def train_perspective_adapter(args, tokenizer, medical_adapter_path=None):
    """Train perspective-aware adapter"""
    print("Starting perspective-aware training...")
    
    # Load training and validation data
    print(f"Loading perspective data from {args.train_file} and {args.valid_file}")
    with open(args.train_file, 'r') as f:
        train_data = json.load(f)
    with open(args.valid_file, 'r') as f:
        valid_data = json.load(f)
    
    print(f"Loaded {len(train_data)} training samples and {len(valid_data)} validation samples")
    
    # Initialize model with quantization
    print("Loading base Llama model with quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        quantization_config=bnb_config
    )
    
    # Load medical adapter if available
    if medical_adapter_path and os.path.exists(medical_adapter_path):
        print(f"Loading medical domain adapter from {medical_adapter_path}")
        model = PeftModel.from_pretrained(base_model, medical_adapter_path)
        
        # Configure LoRA for perspective adaptation
        perspective_lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        
        # Add new adapter for perspective tuning
        model.add_adapter("perspective_adapter", perspective_lora_config)
        model.set_adapter("perspective_adapter")  # Set the new adapter as active
        
    else:
        print("No medical adapter found or specified. Training perspective adapter from scratch.")
        perspective_lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        
        model = get_peft_model(base_model, perspective_lora_config)
    
    model.print_trainable_parameters()
    
    # Create datasets and dataloaders
    train_dataset = PerspectiveDataset(train_data, tokenizer)
    eval_dataset = PerspectiveDataset(valid_data, tokenizer)
    
    train_dataloader = create_dataloader(
        train_dataset, 
        batch_size=args.batch_size_train
    )
    
    eval_dataloader = create_dataloader(
        eval_dataset, 
        batch_size=args.batch_size_valid,
        shuffle=False
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * args.num_epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"Starting perspective training for {args.num_epochs} epochs...")
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = []
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Calculate perspective loss
            perspective_loss = compute_perspective_loss(
                model, 
                tokenizer, 
                input_ids, 
                attention_mask, 
                batch["Perspective"]
            )
            
            # Combined loss
            loss = outputs.loss + perspective_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update progress
            epoch_losses.append(loss.item())
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Log every 50 steps
            if step % 50 == 0:
                print(f"Epoch {epoch+1}, Step {step}/{steps_per_epoch}, Loss: {loss.item():.4f}")
                
        # Calculate average epoch loss
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1} complete, Average Loss: {avg_loss:.4f}")
        
        # Validation
        valid_loss = validate(model, tokenizer, eval_dataloader, epoch+1)
        
        # Save checkpoint
        os.makedirs(args.ckpt_dir, exist_ok=True)
        checkpoint_path = f"{args.ckpt_dir}/perspective_adapter_epoch_{epoch+1}"
        model.save_pretrained(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model_path = f"{args.ckpt_dir}/perspective_adapter_best"
            model.save_pretrained(best_model_path)
            print(f"New best model saved with loss: {best_loss:.4f}")
    
    # Save final model
    final_model_path = f"{args.ckpt_dir}/perspective_adapter_final"
    model.save_pretrained(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")
    
    return final_model_path

def main():
    parser = argparse.ArgumentParser(description="PLASMA: Perspective-aware Language Model Adaptation")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                      help="Model name or path")
    
    # General parameters
    parser.add_argument("--train_file", type=str, default="../data/train_preprocessed.json",
                      help="Path to training data")
    parser.add_argument("--valid_file", type=str, default="../data/valid_preprocessed.json",
                      help="Path to validation data")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/perspective",
                      help="Directory to save checkpoints")
    
    # Medical domain adaptation parameters
    parser.add_argument("--medical_csv", type=str, default="../data/adapter_train.csv",
                      help="Path to medical domain CSV data")
    parser.add_argument("--medical_ckpt_dir", type=str, default="checkpoints/medical",
                      help="Directory to save medical adapter checkpoints")
    parser.add_argument("--medical_lr", type=float, default=5e-5,
                      help="Learning rate for medical domain adaptation")
    parser.add_argument("--medical_epochs", type=int, default=1,
                      help="Number of epochs for medical domain adaptation")
    parser.add_argument("--batch_size_medical", type=int, default=4,
                      help="Batch size for medical domain adaptation")
    parser.add_argument("--medical_warmup_steps", type=int, default=100,
                      help="Warmup steps for medical domain adaptation")
    parser.add_argument("--run_medical_adaptation", action="store_true", default=True,
                      help="Whether to run medical domain adaptation")
    parser.add_argument("--medical_adapter_path", type=str, default='./checkpoints/medical/final_medical_adapter',
                      help="Path to existing medical adapter (skip training if provided)")
    
    # Perspective adaptation parameters
    parser.add_argument("--batch_size_train", type=int, default=4,
                      help="Batch size for perspective training")
    parser.add_argument("--batch_size_valid", type=int, default=4,
                      help="Batch size for perspective validation")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                      help="Learning rate for perspective adaptation")
    parser.add_argument("--warmup_steps", type=int, default=100,
                      help="Warmup steps for perspective adaptation")
    parser.add_argument("--num_epochs", type=int, default=2,
                      help="Number of epochs for perspective adaptation")
    parser.add_argument("--run_perspective_training", action="store_true", default=True,
                      help="Whether to run perspective training")
    
    parser.add_argument("--test_file", type=str, default="../data/test_preprocessed.json",
                      help="Path to test data for final evaluation")

    args = parser.parse_args()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    medical_adapter_path = None
    
    # Stage 1: Medical domain adaptation
    if args.run_medical_adaptation:
        if args.medical_adapter_path and os.path.exists(args.medical_adapter_path):
            print(f"Using existing medical adapter from {args.medical_adapter_path}")
            medical_adapter_path = args.medical_adapter_path
        else:
            print("Training new medical domain adapter...")
            medical_adapter_path = train_medical_adapter(args, tokenizer, args.medical_csv)
    
    # Stage 2: Perspective-aware training

    if args.run_perspective_training:
        print("Starting perspective-aware adaptation...")
        perspective_adapter_path = train_perspective_adapter(args, tokenizer, medical_adapter_path)
        print(f"Complete training pipeline finished. Final model saved at {perspective_adapter_path}") 

    # Run evaluation on test set if both stages were completed
    if args.run_medical_adaptation and args.run_perspective_training:
        print("\n===== Final Evaluation on Test Set =====")
        print(f"Loading best perspective adapter for evaluation on {args.test_file}...")
        
        # Initialize model with quantization for evaluation
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            quantization_config=bnb_config
        )
        
        best_model_path = f"{args.ckpt_dir}/perspective_adapter_final"
        if os.path.exists(best_model_path):
            model = PeftModel.from_pretrained(base_model, best_model_path)
            
            # Load test data for evaluation
            with open(args.test_file, 'r') as f:
                test_data = json.load(f)

            #####################################
            #test_data = test_data[:1]
            #print(f"Testing on 1 sample only")
            #test_dataset = PerspectiveDataset(test_data, tokenizer)
            #test_dataloader = create_dataloader(
            #    test_dataset, 
            #    batch_size=1,
            #    shuffle=False
            #)
            #####################################
                
            test_dataset = PerspectiveDataset(test_data, tokenizer)
            test_dataloader = create_dataloader(
                test_dataset, 
                batch_size=args.batch_size_valid,
                shuffle=False
            )
            
            # Run full evaluation
            model.eval()
            print(f"Running comprehensive evaluation on test set ({len(test_data)} samples)...")
        
            all_rouge_scores = []
            all_perspective_adherence = []
            all_bleu_scores = []
            all_bert_precision = []
            all_bert_recall = []
            all_bert_f1 = []
            all_examples = []
            perspective_distribution = {}

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluating Test Set")):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
        
                    # Generate summary
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=100,
                        temperature=0.9,
                        do_sample=True,
                        top_p=0.92,
                        top_k=50
                    )
        
                    generated_full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    assistant_tag = "<|assistant|>"
                    if assistant_tag in generated_full_text:
                        generated_text = generated_full_text.split(assistant_tag)[1].strip()
                    else:
                        generated_text = generated_full_text

                    reference = batch['Summary'][0]
                    
                    ###########################################################################
                    ## Add inside your evaluation loop after extracting the assistant's response
                    #print(f"Full generated text:\n{generated_full_text}")
                    #print(f"Extracted summary:\n{generated_text}")
                    #print(f"Reference summary:\n{reference}")
                    ###########################################################################

                    perspective = batch['Perspective'][0]
                    
                    # Track perspective distribution
                    if perspective not in perspective_distribution:
                        perspective_distribution[perspective] = 0
                    perspective_distribution[perspective] += 1
        
                    # Calculate ROUGE
                    rouge = Rouge()
                    rouge_score = rouge.get_scores(generated_text, reference)[0]
                    all_rouge_scores.append(rouge_score)
        
                    # Calculate BLEU score
                    reference_tokens = reference.split()
                    generated_tokens = generated_text.split()
                    smoothing = SmoothingFunction().method1
                    bleu_score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothing)
                    all_bleu_scores.append(bleu_score)
        
                    # Calculate BERTScore
                    P, R, F1 = bert_score([generated_text], [reference], lang="en", rescale_with_baseline=True)
                    all_bert_precision.append(P.item())
                    all_bert_recall.append(R.item())
                    all_bert_f1.append(F1.item())
        
                    # Perspective adherence
                    Ep_dict = Ep(generated_text)
                    target_score = Ep_dict[perspective]
                    all_perspective_adherence.append(target_score)
        
                    # Save example
                    if batch_idx % 5 == 0 or batch_idx < 10:  # Save every 5th example and first 10 examples
                        all_examples.append({
                            "perspective": perspective,
                            "generated": generated_text,
                            "reference": reference,
                            "adherence_score": target_score,
                            "rouge_l": rouge_score["rouge-l"]["f"],
                            "bleu": bleu_score,
                            "bert_f1": F1.item()
                        })

            # Calculate average scores
            avg_rouge_1 = sum([score["rouge-1"]["f"] for score in all_rouge_scores]) / len(all_rouge_scores)
            avg_rouge_2 = sum([score["rouge-2"]["f"] for score in all_rouge_scores]) / len(all_rouge_scores)
            avg_rouge_l = sum([score["rouge-l"]["f"] for score in all_rouge_scores]) / len(all_rouge_scores)
            avg_adherence = sum(all_perspective_adherence) / len(all_perspective_adherence)
            avg_bleu = sum(all_bleu_scores) / len(all_bleu_scores)
            avg_bert_precision = sum(all_bert_precision) / len(all_bert_precision)
            avg_bert_recall = sum(all_bert_recall) / len(all_bert_recall)
            avg_bert_f1 = sum(all_bert_f1) / len(all_bert_f1)

            # Calculate per-perspective metrics
            perspective_metrics = {}
            for perspective in perspective_distribution.keys():
                perspective_indices = [i for i in range(len(all_examples)) 
                                    if i < len(test_data) and test_data[i]['Perspective'] == perspective]
                
                if perspective_indices:
                    p_rouge_1 = sum([all_rouge_scores[i]["rouge-1"]["f"] for i in perspective_indices]) / len(perspective_indices)
                    p_rouge_l = sum([all_rouge_scores[i]["rouge-l"]["f"] for i in perspective_indices]) / len(perspective_indices)
                    p_adherence = sum([all_perspective_adherence[i] for i in perspective_indices]) / len(perspective_indices)
                    p_bleu = sum([all_bleu_scores[i] for i in perspective_indices]) / len(perspective_indices)
                    p_bert_f1 = sum([all_bert_f1[i] for i in perspective_indices]) / len(perspective_indices)
                    
                    perspective_metrics[perspective] = {
                        "count": perspective_distribution[perspective],
                        "rouge_1": p_rouge_1,
                        "rouge_l": p_rouge_l,
                        "perspective_adherence": p_adherence,
                        "bleu": p_bleu,
                        "bert_f1": p_bert_f1
                    }

            # Print results
            print(f"===== Final Test Set Evaluation Results =====")
            print(f"ROUGE-1: {avg_rouge_1:.4f}")
            print(f"ROUGE-2: {avg_rouge_2:.4f}")
            print(f"ROUGE-L: {avg_rouge_l:.4f}")
            print(f"BLEU: {avg_bleu:.4f}")
            print(f"BERTScore P/R/F1: {avg_bert_precision:.4f}/{avg_bert_recall:.4f}/{avg_bert_f1:.4f}")
            print(f"Perspective Adherence: {avg_adherence:.4f}")
            
            # Print perspective distribution
            print("\nPerspective Distribution:")
            for perspective, count in perspective_distribution.items():
                print(f"{perspective}: {count} samples")
            
            # Print per-perspective metrics
            print("\nPer-Perspective Metrics:")
            for perspective, metrics in perspective_metrics.items():
                print(f"{perspective} (n={metrics['count']}):")
                print(f"  ROUGE-1: {metrics['rouge_1']:.4f}")
                print(f"  ROUGE-L: {metrics['rouge_l']:.4f}")
                print(f"  Adherence: {metrics['perspective_adherence']:.4f}")
                print(f"  BLEU: {metrics['bleu']:.4f}")
                print(f"  BERT-F1: {metrics['bert_f1']:.4f}")

            # Save results
            os.makedirs("evaluation_results", exist_ok=True)
            with open("evaluation_results/test_set_evaluation.json", 'w') as f:
                json.dump({
                    "overall_metrics": {
                        "rouge_1": avg_rouge_1,
                        "rouge_2": avg_rouge_2,
                        "rouge_l": avg_rouge_l,
                        "bleu": avg_bleu,
                        "bert_score_precision": avg_bert_precision,
                        "bert_score_recall": avg_bert_recall,
                        "bert_score_f1": avg_bert_f1,
                        "perspective_adherence": avg_adherence
                    },
                    "perspective_distribution": perspective_distribution,
                    "perspective_metrics": perspective_metrics,
                    "examples": all_examples
                }, f, indent=2)

            print("Test evaluation results saved to evaluation_results/test_set_evaluation.json")
        else:
            print(f"Best model not found at {best_model_path}. Skipping evaluation.")

    print("PLASMA pipeline execution completed successfully.")

if __name__ == "__main__":
    main()


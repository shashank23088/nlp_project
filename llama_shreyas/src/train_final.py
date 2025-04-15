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
from transformers import ( AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup )

from peft import ( LoraConfig, get_peft_model, PeftModel, TaskType )

from scipy.spatial.distance import cosine
from transformers import ( RobertaForSequenceClassification, RobertaTokenizer, BertTokenizer, BertModel )

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
ckpt_path = "/home/Shreyas-gupta/project/new/nlp_project/baseline2/src/classifier/best_ckpt_epoch=2_valid_loss=0.2286.ckpt"
if os.path.exists(ckpt_path):
    print("Loading the trained classifier checkpoint...")
    ckpt = torch.load(ckpt_path, weights_only=False)
    roberta_model.load_state_dict(ckpt['model_state_dict'])
    print("Classifier loaded successfully.")
else:
    print("Warning: Classifier checkpoint not found. Using base model.")

    
def get_bert_embedding(text):
        # inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()


def calculate_rouge_score_for_each_phrase(predictions, references):
        rouge = Rouge()
        rouge_l_f1_scores = []

        for prediction, reference in zip(predictions, references):
            scores = rouge.get_scores(prediction.lower(), reference.lower())[0]
            rouge_l_f1 = scores["rouge-1"]["f"]
            rouge_l_f1_scores.append(rouge_l_f1)

        return rouge_l_f1_scores

def score_all_phrases(summary, phrases):
        start_of_summary = ' '.join(summary.split()[:4])

    
        predictions = [start_of_summary] * len(phrases)
        references = phrases

        rouge_l_f1_results = calculate_rouge_score_for_each_phrase(predictions, references)

        phrase_scores = dict(zip(phrases, rouge_l_f1_results))

        return phrase_scores

def Ep(generated_summary):

        inputs = roberta_tokenizer(generated_summary, padding=True, truncation=True, return_tensors="pt").to(device)

    
        with torch.no_grad():
            outputs = roberta_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = outputs.logits.argmax(dim=-1)

        class_labels = {0: "EXPERIENCE", 1: "SUGGESTION", 2: "INFORMATION", 3: "CAUSE", 4: "QUESTION"}
        predicted_label = class_labels[predictions[0].item()]

        final ={}
        for i in range(0,5):
            final[class_labels[i]] = probabilities[0][i].cpu().numpy().item()

        return final


def Es(generated_summary):
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
        l_sugg =  ["Advisory", "Recommending", "Cautioning", "Prescriptive", "Guiding","Prescriptive"]
        l_exp = ["Personal", "Narrative", "Introspective", "Exemplary", "Insightful", "Emotional"]
        l_info =  ["Clinical", "Scientific","Informative", "Educational","Factual", "Informing","Academic","Analytical"]
        l_cause =  ["Diagnostic", "Explanatory", "Causal","Due to", "Resulting from", "Attributable to" ]
        l_qs =  ["Inquiry", "Rhetorical", "Exploratory Questioning", "Clarifying Inquiry", "Problem-Solving Deliberation"]


        summary_embedding = get_bert_embedding(generated_summary)

        cosine_similarities = {}

        for label, word_list in zip(['sugg', 'exp', 'info', 'cause', 'qs'], [l_sugg, l_exp, l_info, l_cause, l_qs]):
            combined_text = ' '.join(word_list)
            word_embedding = get_bert_embedding(combined_text)
            similarity = 1 - cosine(summary_embedding.cpu().detach().numpy(), word_embedding.cpu().detach().numpy())
            cosine_similarities[label] = similarity

        return cosine_similarities

def compute_custom_loss(model, input_text, input_attention, perspective):
    model.eval()
    outputs = model.generate(input_ids=input_text, attention_mask=input_attention, num_beams=2, max_new_tokens=50, temperature=0.9)
    generated_summary = tokenizer.decode(outputs[0])
    if len(generated_summary) <= 0:
        generated_summary = 'None'
       
    Ep_dict = Ep(generated_summary)
    Es_dict = Es(generated_summary)
    Et_dict = Et(generated_summary)

    alpha = 0.7  
    beta = 0.3   
    gamma = 0.5  

    perspective_types = ["EXPERIENCE", "SUGGESTION", "INFORMATION", "CAUSE", "QUESTION"]
    
    E_X = {
        "EXPERIENCE": alpha * Ep_dict["EXPERIENCE"] + beta * Es_dict["In user's experience…"] + gamma * Et_dict['exp'],
        "SUGGESTION": alpha * Ep_dict["SUGGESTION"] + beta * Es_dict["It is suggested"] + gamma * Et_dict['sugg'],
        "INFORMATION": alpha * Ep_dict["INFORMATION"] + beta * Es_dict["For information purposes"] + gamma * Et_dict['info'],
        "CAUSE": alpha * Ep_dict["CAUSE"] + beta * Es_dict["Some of the causes"] + gamma * Et_dict['cause'],
        "QUESTION": alpha * Ep_dict["QUESTION"] + beta * Es_dict["It is inquired"] + gamma * Et_dict['qs']
    }

    # Compute the exponential of E(X) for normalization with epsilon to avoid division by zero
    exp_E_X = {k: math.exp(-1/(v + 1e-10)) for k, v in E_X.items()}
    
    # Compute the sum of the exponentials for normalization
    Z = sum(exp_E_X.values())

    # Calculate P(X) for each perspective
    P_X = {k: v / Z for k, v in exp_E_X.items()}
    
    # Initialize Y with zeros for all perspective types
    Y = {k: 0.0 for k in perspective_types}
    
    # Set the target perspective to 1
    if perspective[0] in Y:
        Y[perspective[0]] = 1.0
    
    # Use the same order for both tensors
    P_X_tensor = torch.tensor([P_X[k] for k in perspective_types])
    Y_tensor = torch.tensor([Y[k] for k in perspective_types])
    
    # Ensure the tensors are on the correct device
    P_X_tensor = P_X_tensor.to(device)
    Y_tensor = Y_tensor.to(device)
    
    # Avoid numerical issues in log
    P_X_tensor = torch.clamp(P_X_tensor, min=1e-10)
    
    loss = -torch.sum(Y_tensor * torch.log(P_X_tensor))
    return loss




#######################
# DATASETS
#######################

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
            'perspective': perspective,
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

def train_perspective_adapter(args, tokenizer, medical_adapter_path=None):
    """Train perspective-aware adapters - one for each perspective type"""
    print("Starting perspective-aware training with perspective-specific adapters...")
    
    # Load training and validation data
    print(f"Loading perspective data from {args.train_file} and {args.valid_file}")
    with open(args.train_file, 'r') as f:
        train_data = json.load(f)
    with open(args.valid_file, 'r') as f:
        valid_data = json.load(f)
    
    print(f"Loaded {len(train_data)} training samples and {len(valid_data)} validation samples")
    
    # Identify unique perspectives
    all_perspectives = set(item["Perspective"] for item in train_data)
    print(f"Found {len(all_perspectives)} unique perspectives: {', '.join(all_perspectives)}")
    
    perspective_adapters = {}
    
    # Train a separate adapter for each perspective
    for perspective in all_perspectives:
        print(f"\n===== Training adapter for {perspective} perspective =====")
        
        # Filter data for this perspective
        train_data_perspective = [item for item in train_data if item["Perspective"] == perspective]
        valid_data_perspective = [item for item in valid_data if item["Perspective"] == perspective]
        
        print(f"Training with {len(train_data_perspective)} samples, validating with {len(valid_data_perspective)} samples")
        
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
            
            # Add new adapter for this perspective
            adapter_name = f"perspective_{perspective.lower()}_adapter"
            model.add_adapter(adapter_name, perspective_lora_config)
            model.set_adapter(adapter_name)  # Set the new adapter as active
            
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
        
        # Create datasets and dataloaders for this perspective
        train_dataset = PerspectiveDataset(train_data_perspective, tokenizer)
        eval_dataset = PerspectiveDataset(valid_data_perspective, tokenizer)
        
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
        
        # Training loop for this perspective
        print(f"Starting training for {perspective} adapter for {args.num_epochs} epochs...")
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
                
                # For perspective-specific adapters, we can simplify and just use CE loss
                # since each adapter is trained on data for a specific perspective
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
                    print(f"Epoch {epoch+1}, Step {step}/{steps_per_epoch}, Loss: {loss.item():.4f}")
                    
            # Calculate average epoch loss
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Epoch {epoch+1} complete, Average Loss: {avg_loss:.4f}")
            
            # Validation
            model.eval()
            valid_losses = []
            
            with torch.no_grad():
                for batch in tqdm(eval_dataloader, desc="Validation"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    valid_losses.append(outputs.loss.item())
            
            valid_loss = sum(valid_losses) / len(valid_losses) if valid_losses else float('inf')
            print(f"Validation Loss: {valid_loss:.4f}")
            
            # Save checkpoint
            perspective_ckpt_dir = f"{args.ckpt_dir}/{perspective.lower()}"
            os.makedirs(perspective_ckpt_dir, exist_ok=True)
            checkpoint_path = f"{perspective_ckpt_dir}/adapter_epoch_{epoch+1}"
            model.save_pretrained(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    

            # Save best model
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model_path = f"{perspective_ckpt_dir}/adapter_best"
                model.save_pretrained(best_model_path)
                print(f"New best model saved with loss: {best_loss:.4f}")
        
        # Save final model for this perspective
        final_model_path = f"{perspective_ckpt_dir}/adapter_final"
        model.save_pretrained(final_model_path)
        print(f"Training complete for {perspective}. Final adapter saved to {final_model_path}")
        
        # Store best adapter path for this perspective
        perspective_adapters[perspective] = f"{perspective_ckpt_dir}/adapter_best"
        
        # Free up memory
        del model
        del base_model
        torch.cuda.empty_cache()
    
    # Save mapping of perspectives to their adapter paths
    os.makedirs(args.ckpt_dir, exist_ok=True)
    with open(f"{args.ckpt_dir}/perspective_adapters.json", 'w') as f:
        json.dump(perspective_adapters, f, indent=2)
    
    print(f"All perspective adapters trained successfully. Mapping saved to {args.ckpt_dir}/perspective_adapters.json")
    
    return perspective_adapters



def evaluate_with_perspective_adapters(args, tokenizer, medical_adapter_path=None):
    """Evaluate using perspective-specific adapters"""
    print("\n===== Evaluating with Perspective-Specific Adapters =====")
    
    # Load test data
    with open(args.test_file, 'r') as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Load perspective adapter mappings
    adapter_mapping_path = f"{args.ckpt_dir}/perspective_adapters.json"
    if not os.path.exists(adapter_mapping_path):
        print(f"Adapter mapping not found at {adapter_mapping_path}. Aborting evaluation.")
        return
    
    with open(adapter_mapping_path, 'r') as f:
        perspective_adapters = json.load(f)
    
    print(f"Loaded adapter mappings for {len(perspective_adapters)} perspectives")
    
    # Initialize metrics tracking
    all_rouge_scores = []
    all_perspective_adherence = []
    all_bleu_scores = []
    all_bert_precision = []
    all_bert_recall = []
    all_bert_f1 = []
    all_examples = []
    perspective_distribution = {}
    perspective_metrics = {p: {"results": []} for p in perspective_adapters.keys()}
    
    # Process test data by perspective
    for perspective, adapter_path in perspective_adapters.items():
        print(f"\n----- Evaluating {perspective} perspective with its dedicated adapter -----")
        
        # Filter test data for this perspective
        test_data_perspective = [item for item in test_data if item["Perspective"] == perspective]
        if not test_data_perspective:
            print(f"No test samples found for {perspective} perspective. Skipping.")
            continue
        
        print(f"Testing on {len(test_data_perspective)} samples")
        test_dataset = PerspectiveDataset(test_data_perspective, tokenizer)
        test_dataloader = create_dataloader(
            test_dataset, 
            batch_size=args.batch_size_valid,
            shuffle=False
        )
        
        # Track perspective distribution
        perspective_distribution[perspective] = len(test_data_perspective)
        
        # Initialize model with quantization
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
        
        # Load medical adapter first if available
        if medical_adapter_path and os.path.exists(medical_adapter_path):
            print(f"Loading medical domain adapter from {medical_adapter_path}")
            model = PeftModel.from_pretrained(base_model, medical_adapter_path)
            
            # Now load the perspective-specific adapter
            if os.path.exists(adapter_path):
                print(f"Loading perspective adapter from {adapter_path}")
                model.load_adapter(adapter_path, adapter_name=f"perspective_{perspective.lower()}_adapter")
                model.set_adapter(f"perspective_{perspective.lower()}_adapter")
            else:
                print(f"Adapter not found at {adapter_path}. Using medical adapter only.")
        else:
            # Load perspective adapter directly if no medical adapter
            if os.path.exists(adapter_path):
                print(f"Loading perspective adapter from {adapter_path}")
                model = PeftModel.from_pretrained(base_model, adapter_path)
            else:
                print(f"Adapter not found at {adapter_path}. Using base model.")
                model = base_model
        
        # Evaluate
        model.eval()
        perspective_results = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_dataloader, desc=f"Evaluating {perspective}")):
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
                
                #generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                reference = batch['Summary'][0]
                
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
                
                # Store result for this perspective
                result = {
                    "rouge_1": rouge_score["rouge-1"]["f"],
                    "rouge_2": rouge_score["rouge-2"]["f"],
                    "rouge_l": rouge_score["rouge-l"]["f"],
                    "bleu": bleu_score,
                    "bert_f1": F1.item(),
                    "adherence": target_score,
                }
                perspective_metrics[perspective]["results"].append(result)
                
                # Save example
                if batch_idx % 5 == 0 or batch_idx < 5:  # Save every 5th example and first 5 examples
                    all_examples.append({
                        "Perspective": perspective,
                        "generated": generated_text,
                        "reference": reference,
                        "adapter_used": adapter_path,
                        "adherence_score": target_score,
                        "rouge_l": rouge_score["rouge-l"]["f"],
                        "bleu": bleu_score,
                        "bert_f1": F1.item()
                    })
        
        # Free memory
        del model
        del base_model
        torch.cuda.empty_cache()
    
    # Calculate aggregate metrics
    for perspective, data in perspective_metrics.items():
        if not data["results"]:
            continue
            
        data["avg_rouge_1"] = sum(r["rouge_1"] for r in data["results"]) / len(data["results"])
        data["avg_rouge_2"] = sum(r["rouge_2"] for r in data["results"]) / len(data["results"])
        data["avg_rouge_l"] = sum(r["rouge_l"] for r in data["results"]) / len(data["results"])
        data["avg_bleu"] = sum(r["bleu"] for r in data["results"]) / len(data["results"])
        data["avg_bert_f1"] = sum(r["bert_f1"] for r in data["results"]) / len(data["results"])
        data["avg_adherence"] = sum(r["adherence"] for r in data["results"]) / len(data["results"])
        data["count"] = len(data["results"])
        # Remove detailed results to keep output manageable
        data.pop("results")
    
    # Calculate overall metrics
    avg_rouge_1 = sum([score["rouge-1"]["f"] for score in all_rouge_scores]) / len(all_rouge_scores) if all_rouge_scores else 0
    avg_rouge_2 = sum([score["rouge-2"]["f"] for score in all_rouge_scores]) / len(all_rouge_scores) if all_rouge_scores else 0
    avg_rouge_l = sum([score["rouge-l"]["f"] for score in all_rouge_scores]) / len(all_rouge_scores) if all_rouge_scores else 0
    avg_adherence = sum(all_perspective_adherence) / len(all_perspective_adherence) if all_perspective_adherence else 0
    avg_bleu = sum(all_bleu_scores) / len(all_bleu_scores) if all_bleu_scores else 0
    avg_bert_f1 = sum(all_bert_f1) / len(all_bert_f1) if all_bert_f1 else 0
    
    # Print results
    print(f"\n===== Final Evaluation Results with Perspective-Specific Adapters =====")
    print(f"ROUGE-1: {avg_rouge_1:.4f}")
    print(f"ROUGE-2: {avg_rouge_2:.4f}")
    print(f"ROUGE-L: {avg_rouge_l:.4f}")
    print(f"BLEU: {avg_bleu:.4f}")
    print(f"BERTScore F1: {avg_bert_f1:.4f}")
    print(f"Perspective Adherence: {avg_adherence:.4f}")
    
    # Print perspective distribution
    print("\nPerspective Distribution:")
    for perspective, count in perspective_distribution.items():
        print(f"{perspective}: {count} samples")
    
    # Print per-perspective metrics
    print("\nPer-Perspective Metrics:")
    for perspective, metrics in perspective_metrics.items():
        if "count" not in metrics:
            continue
        print(f"{perspective} (n={metrics['count']}): ROUGE-1={metrics['avg_rouge_1']:.4f}, ROUGE-L={metrics['avg_rouge_l']:.4f}, Adherence={metrics['avg_adherence']:.4f}")
    
    # Save results
    os.makedirs("evaluation_results", exist_ok=True)
    with open("evaluation_results/perspective_specific_adapters_evaluation.json", 'w') as f:
        json.dump({
            "overall_metrics": {
                "rouge_1": avg_rouge_1,
                "rouge_2": avg_rouge_2,
                "rouge_l": avg_rouge_l,
                "bleu": avg_bleu,
                "bert_f1": avg_bert_f1,
                "perspective_adherence": avg_adherence
            },
            "perspective_distribution": perspective_distribution,
            "perspective_metrics": perspective_metrics,
            "examples": all_examples
        }, f, indent=2)
    
    print("Perspective-specific adapters evaluation results saved to evaluation_results/perspective_specific_adapters_evaluation.json")




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

    
    # Perspective adaptation parameters
    parser.add_argument("--batch_size_train", type=int, default=4,
                      help="Batch size for perspective training")
    parser.add_argument("--batch_size_valid", type=int, default=4,
                      help="Batch size for perspective validation")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                      help="Learning rate for perspective adaptation")
    parser.add_argument("--warmup_steps", type=int, default=100,
                      help="Warmup steps for perspective adaptation")
    parser.add_argument("--num_epochs", type=int, default=4,
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
    
    # Stage 2: Perspective-aware training
    if args.run_perspective_training:
        print("Starting perspective-aware adaptation...")
        #perspective_adapter_path = train_perspective_adapter(args, tokenizer, medical_adapter_path)
        #print(f"Complete training pipeline finished. Final model saved at {perspective_adapter_path}")
    
    # Run evaluation if both stages were completed
    #if args.run_medical_adaptation or args.run_perspective_training:
    if args.run_perspective_training:
        print("\n===== Final Evaluation =====")
        print("Loading best perspective adapter for evaluation...")
        evaluate_with_perspective_adapters(args, tokenizer, medical_adapter_path)
    print("PLASMA pipeline execution completed successfully.")
        

def eval_before_calling_eval_with__funct():
        # Initialize model with quantization for evaluation
    if True:
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
        
        best_model_path = f"{args.ckpt_dir}/perspective_adapter_best"
        if os.path.exists(best_model_path):
            model = PeftModel.from_pretrained(base_model, best_model_path)
            
            # Load testing data for evaluation
            with open(args.test_file, 'r') as f:
                valid_data = json.load(f)
                
            eval_dataset = PerspectiveDataset(valid_data, tokenizer)
            eval_dataloader = create_dataloader(
                eval_dataset, 
                batch_size=args.batch_size_valid,
                shuffle=False
            )
            
            # Run full evaluation
            model.eval()
            print("Running comprehensive evaluation on testing set...")
         
            all_rouge_scores = []
            all_perspective_adherence = []
            all_bleu_scores = []
            all_bert_precision = []
            all_bert_recall = []
            all_bert_f1 = []
            all_examples = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
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
        
                    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    reference = batch['Summary'][0]
                    perspective = batch['perspective'][0]
        
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
                    if batch_idx % 5 == 0:  # Save every 5th example
                        all_examples.append({
                            "Perspective": perspective,
                            "generated": generated_text,
                            "reference": reference,
                            "adherence_score": target_score,
                            "rouge_l": rouge_score["rouge-l"]["f"],
                            "bleu": bleu_score,
                            "bert_f1": F1.item()
                        })

            # Then update the average scores calculation:
            avg_rouge_1 = sum([score["rouge-1"]["f"] for score in all_rouge_scores]) / len(all_rouge_scores)
            avg_rouge_2 = sum([score["rouge-2"]["f"] for score in all_rouge_scores]) / len(all_rouge_scores)
            avg_rouge_l = sum([score["rouge-l"]["f"] for score in all_rouge_scores]) / len(all_rouge_scores)
            avg_adherence = sum(all_perspective_adherence) / len(all_perspective_adherence)
            avg_bleu = sum(all_bleu_scores) / len(all_bleu_scores)
            avg_bert_precision = sum(all_bert_precision) / len(all_bert_precision)
            avg_bert_recall = sum(all_bert_recall) / len(all_bert_recall)
            avg_bert_f1 = sum(all_bert_f1) / len(all_bert_f1)

            # And update the print statements:
            print(f"===== Final Evaluation Results =====")
            print(f"ROUGE-1: {avg_rouge_1:.4f}")
            print(f"ROUGE-2: {avg_rouge_2:.4f}")
            print(f"ROUGE-L: {avg_rouge_l:.4f}")
            print(f"BLEU: {avg_bleu:.4f}")
            print(f"BERTScore P/R/F1: {avg_bert_precision:.4f}/{avg_bert_recall:.4f}/{avg_bert_f1:.4f}")
            print(f"Perspective Adherence: {avg_adherence:.4f}")

            os.makedirs("evaluation_results", exist_ok=True)

            # Finally, update the saved evaluation results:
            with open("evaluation_results/final_evaluation.json", 'w') as f:
                json.dump({
                    "metrics": {
                        "rouge_1": avg_rouge_1,
                        "rouge_2": avg_rouge_2,
                        "rouge_l": avg_rouge_l,
                        "bleu": avg_bleu,
                        "bert_score_precision": avg_bert_precision,
                        "bert_score_recall": avg_bert_recall,
                        "bert_score_f1": avg_bert_f1,
                        "perspective_adherence": avg_adherence
                 },
                    "examples": all_examples
             }, f, indent=2)

            print("Evaluation results saved to evaluation_results/final_evaluation.json")
        else:
            print(f"Best model not found at {best_model_path}. Skipping evaluation.")
    
    print("PLASMA pipeline execution completed successfully.")

if __name__ == "__main__":
    main()

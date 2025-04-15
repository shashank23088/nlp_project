import json
import argparse 
from transformers.file_utils import PushToHubMixin

from transformers import AutoModelForCausalLM, AutoTokenizer, BartTokenizer, BartForConditionalGeneration, GPT2LMHeadModel, BertTokenizer, BertModel
from peft import get_peft_config, PeftModel, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType
import sys
sys.path.insert(0, './') 

from train_dataloader import CustomDataset, create_dataloader
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, get_linear_schedule_with_warmup, RobertaForSequenceClassification, RobertaTokenizer
from tqdm import tqdm
import numpy as np
import os
import torch
from scipy.spatial.distance import cosine
from bert_score import score
import math
from rouge import Rouge
import numpy as np
import warnings
warnings.filterwarnings("ignore")
device = 'cuda'

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from bert_score import score as bert_score
import numpy as np
import pandas as pd
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)



bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5).to(device)
ckpt_path = f"./classifier/checkpoint_classifier"
if os.path.exists(ckpt_path):
    print("Loading the trained checkpoint...")
    ckpt = torch.load(ckpt_path)
    roberta_model.load_state_dict(ckpt['model_state_dict'])
    print("The inference will start with the specified checkpoint.")


def get_bert_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
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
    final = {}
    for i in range(0, 5):
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
    l_sugg =  ["Advisory", "Recommending", "Cautioning", "Prescriptive", "Guiding", "Prescriptive"]
    l_exp = ["Personal", "Narrative", "Introspective", "Exemplary", "Insightful", "Emotional"]
    l_info =  ["Clinical", "Scientific", "Informative", "Educational", "Factual", "Informing", "Academic", "Analytical"]
    l_cause =  ["Diagnostic", "Explanatory", "Causal", "Due to", "Resulting from", "Attributable to"]
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
    outputs = model.generate(
        input_ids=input_text,
        attention_mask=input_attention,
        num_beams=2, 
        max_new_tokens=50,
        temperature=0.9
    )
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
    exp_E_X = {k: math.exp(-1/(v + 1e-10)) for k, v in E_X.items()}
    Z = sum(exp_E_X.values())
    P_X = {k: v / Z for k, v in exp_E_X.items()}
    
    Y = {k: 0.0 for k in perspective_types}
    if perspective[0] in Y:
        Y[perspective[0]] = 1.0
    
    P_X_tensor = torch.tensor([P_X[k] for k in perspective_types]).to(device)
    Y_tensor = torch.tensor([Y[k] for k in perspective_types]).to(device)
    P_X_tensor = torch.clamp(P_X_tensor, min=1e-10)
    loss = -torch.sum(Y_tensor * torch.log(P_X_tensor))
    return loss

def validation(valid_dataloader, model, VALID_BATCH_SIZE, optimizer, scheduler):
    print("Validation processing...")
    model.eval()    
    valid_losses = []
    gen = []
    actual = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(valid_dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['input_ids'].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            custom_loss = compute_custom_loss(model, input_ids, attention_mask, batch["perspective"])
            loss = output.loss + custom_loss
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=5,
                max_new_tokens=100,
                temperature=0.9
            )
            generated_summary = tokenizer.decode(outputs[0])
            gen.append(generated_summary)
            actual.append(batch['Summary'])
            print(f"_________________ValidBatch: {i}/{len(valid_dataloader)} || ValidLoss: {loss}_____________________")
            valid_losses.append(loss.item()) 
    valid_loss = np.mean(valid_losses) if len(valid_losses) > 0 else 0.0  
    return valid_loss 

if __name__ == "__main__":

    list_loss_train = []
    list_loss_valid = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='/home/iiitd/Sajid/NLP_Project/sajid/llama_sajid/data/train_preprocessed.json')
    parser.add_argument('--valid_file', type=str, default='/home/iiitd/Sajid/NLP_Project/sajid/llama_sajid/data/valid_preprocessed.json')
    parser.add_argument('--batch_size_train', type=int, default=4)
    parser.add_argument('--batch_size_valid', type=int, default=4)
    parser.add_argument('--model_file', type=str, required=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=16000)
    parser.add_argument("--num_epochs", type=int, default=10, help="how many training epochs")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--ckpt_name", default=None)

    args = parser.parse_args()

    TRAIN_BATCH_SIZE = args.batch_size_train
    VALID_BATCH_SIZE = args.batch_size_valid
    with open(args.train_file, 'r') as json_file:
        train_data = json.load(json_file)
    with open(args.valid_file, 'r') as json_file:
        valid_data = json.load(json_file)

    LR = args.learning_rate
    WARMUP_STEPS = args.warmup_steps
    EPOCHS = args.num_epochs
    best_loss = sys.float_info.max
    last_epoch = 0

==
   
    
    model_name = "KidIkaros/tiny-llama3.2-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token 
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    

    # =======================================================
    if args.ckpt_name is not None:
        adapter_path = f"{args.ckpt_dir}/{args.ckpt_name}"
        if os.path.exists(adapter_path):
            print("Loading adapter weights...")
            model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            print("Cannot find adapter weights. Initializing fresh PEFT model.")
            peft_config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,       # Changed to causal LM for decoder-only models
                inference_mode=False,
                num_virtual_tokens=8,
                token_dim=2048 # Adjust this to match your LLama3 model’s hidden size 
            )
            model = get_peft_model(base_model, peft_config)
    else:
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,       
            inference_mode=False,
            num_virtual_tokens=8,
            token_dim=2048 
        )
        model = get_peft_model(base_model, peft_config)

    model.print_trainable_parameters()
    model.to(device)

    train_dataset = CustomDataset(train_data, tokenizer)
    eval_dataset = CustomDataset(valid_data, tokenizer)
    train_dataloader, eval_dataloader = create_dataloader(
        train_dataset, eval_dataset, VALID_BATCH_SIZE, TRAIN_BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS,
        num_training_steps=len(train_dataloader) * EPOCHS
    )

    start_epoch = last_epoch + 1

    model.to(device)
    for epoch in range(start_epoch, start_epoch + EPOCHS):
        model.train()
        print(f"{'#' * 50} Epoch: {epoch} {'#' * 50}")
        train_losses = []
        for i, batch in enumerate(tqdm(train_dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch["labels"].to(device)
            

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            custom_loss = compute_custom_loss(model, input_ids, attention_mask, batch["perspective"])
            loss = outputs.loss + custom_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.detach())
        train_loss = np.mean([loss.item() for loss in train_losses])
        print(f"Train loss: {train_loss} for epoch: {epoch}")
        list_loss_train.append(train_loss)
        model.save_pretrained(f"{args.ckpt_dir}/adapter_epoch={epoch}")
        valid_loss = validation(eval_dataloader, model, VALID_BATCH_SIZE, optimizer, scheduler)
        list_loss_valid.append(valid_loss)
        if valid_loss < best_loss:
            best_loss = valid_loss
            model.save_pretrained(f"{args.ckpt_dir}/adapter_epoch={epoch}_valid_loss={round(best_loss, 4)}")

import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from bert_score import score as bert_score
import numpy as np
import pandas as pd
import os
import ast  # used to safely parse string representations of lists

class EvaluationMetrics:
    def __init__(self, predictions, references):
        self.predictions = predictions
        self.references = references

    def compute_rouge_score(self):
        rouge = Rouge()
        rouge_l_f1, rouge_l_recall, rouge_l_precision = [], [], []
        rouge_1_f1, rouge_1_recall, rouge_1_precision = [], [], []
        rouge_2_f1, rouge_2_recall, rouge_2_precision = [], [], []
        for prediction, reference in zip(self.predictions, self.references):
            scores = rouge.get_scores(prediction, reference)[0]
            
            rouge_l_f1.append(scores["rouge-l"]["f"])
            rouge_l_recall.append(scores["rouge-l"]["r"])
            rouge_l_precision.append(scores["rouge-l"]["p"])
            
            rouge_1_f1.append(scores["rouge-1"]["f"])
            rouge_1_recall.append(scores["rouge-1"]["r"])
            rouge_1_precision.append(scores["rouge-1"]["p"])
            
            rouge_2_f1.append(scores["rouge-2"]["f"])
            rouge_2_recall.append(scores["rouge-2"]["r"])
            rouge_2_precision.append(scores["rouge-2"]["p"])

        results = {
            "rouge_l": {
                "f1": np.mean(rouge_l_f1) * 100,
                "recall": np.mean(rouge_l_recall) * 100,
                "precision": np.mean(rouge_l_precision) * 100
            },
            "rouge_1": {
                "f1": np.mean(rouge_1_f1) * 100,
                "recall": np.mean(rouge_1_recall) * 100,
                "precision": np.mean(rouge_1_precision) * 100
            },
            "rouge_2": {
                "f1": np.mean(rouge_2_f1) * 100,
                "recall": np.mean(rouge_2_recall) * 100,
                "precision": np.mean(rouge_2_precision) * 100
            }
        }
        
        return results

    def compute_meteor_score(self):
        meteor = evaluate.load('meteor')
        scores = []
        for prediction, reference in zip(self.predictions, self.references):
            score_dict = meteor.compute(predictions=[prediction], references=[reference])
            scores.append(score_dict["meteor"])
        average_meteor_score = np.mean(scores)
        return {"meteor": average_meteor_score}
    
    def compute_bleu_scores(self):
        bleu_1, bleu_2, bleu_3, bleu_4 = [], [], [], []
        smoothie = SmoothingFunction().method4
        for prediction, reference in zip(self.predictions, self.references):
            # Tokenize sentences for BLEU computation
            reference_tokens = [reference.split()]
            prediction_tokens = prediction.split()
            
            bleu_1.append(sentence_bleu(reference_tokens, prediction_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie))
            bleu_2.append(sentence_bleu(reference_tokens, prediction_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
            bleu_3.append(sentence_bleu(reference_tokens, prediction_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie))
            bleu_4.append(sentence_bleu(reference_tokens, prediction_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))

        results = {
            "bleu_1": np.mean(bleu_1),
            "bleu_2": np.mean(bleu_2),
            "bleu_3": np.mean(bleu_3),
            "bleu_4": np.mean(bleu_4)
        }
        
        return results
    
    def compute_bertscore(self, predictions, references, lang="en"):
        P, R, F1 = bert_score(predictions, references, lang=lang)
        return F1.mean().item()

if __name__ == "__main__":
    # Load inference results
    df = pd.read_csv('/home/iiitd/Sajid/NLP_Project/sajid/generated_llama/generated_result_final.csv')
    

    def convert_prediction(pred_str):
        try:
            lst = ast.literal_eval(pred_str)
            if isinstance(lst, list) and len(lst) > 0:
                return lst[0]
            return pred_str
        except Exception:
            return pred_str

    # Process the predictions column and extract plain text
    predictions = df['Prediction'].apply(convert_prediction).tolist()
    references = df['Actual'].tolist()
    
    eval_metrics = EvaluationMetrics(predictions, references)


    metrics_dict = {
        "ROUGE": eval_metrics.compute_rouge_score(),
        "METEOR": eval_metrics.compute_meteor_score(),
        "BLEU": eval_metrics.compute_bleu_scores(),
        "BERTScore": eval_metrics.compute_bertscore(predictions, references)
    }


    print("Evaluation Metrics:")
    for metric, score in metrics_dict.items():
        print(f"{metric}: {score}")


    flat_metrics = {}
    for key, value in metrics_dict.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_metrics[f"{key}_{sub_key}"] = sub_value
        else:
            flat_metrics[key] = value


    metrics_df = pd.DataFrame([flat_metrics])
    
    save_path = './generated_llama/eval_metrics.csv'
    save_dir = os.path.dirname(save_path)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    metrics_df.to_csv(save_path, index=False)
    print(f"Generated predictions saved to: {save_path}!")

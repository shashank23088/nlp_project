<h1 align="center">📌 Perspective-Aware Healthcare Q&A Summarization</h1>

<p align="center">
A two-stage adapter tuning framework using <code>LLaMA 3.2B-Instruct</code> for generating perspective-aligned summaries from healthcare Q&A data, guided by a RoBERTa-based classifier.
</p>

---

## 🧾 Dataset: PUMA

We use the <b>PUMA</b> dataset, introduced in Naik et al. (2024), consisting of annotated healthcare Q&A pairs with five perspective labels: <code>INFORMATION</code>, <code>SUGGESTION</code>, <code>EXPERIENCE</code>, <code>CAUSE</code>, and <code>QUESTION</code>.

📩 <b>Access Request:</b>  
To obtain access, email <a href="mailto:gnaik826@gmail.com">gnaik826@gmail.com</a> with your affiliation and intended use.

---

## 🧠 Project Architecture

1. **Train a RoBERTa classifier** to predict the perspective of a summary.
2. **Stage 1: Train a medical adapter** on external Kaggle medical Q&A to adapt LLaMA to domain language.
3. **Stage 2: Train a perspective adapter** using PUMA data and classifier-guided energy-based loss.
4. **Evaluate** on test data and **run inference** on unseen inputs.

---

## 🔧 Training Pipeline

### 🔹 1. Train the RoBERTa Classifier

```bash
python ./src/classifier/train_classifier.py
```

> All training details (batch size, epochs, learning rate) are defined within the script.

---

### 🔹 2. Train the Medical and Perspective Adapters

```bash
python ./src/train.py
```

This script:
- Trains the medical adapter on Kaggle medical Q&A dataset.
- Trains the perspective adapter on PUMA using classifier-guided energy loss.
- Performs validation and saves best checkpoints.

> Configure arguments like adapter paths, learning rates, and dataset locations inside the `train.py` script.

---

## 📊 Evaluation

Run evaluation after training to compute metrics like ROUGE, BLEU, BERTScore, and adherence:

```bash
python ./src/train.py --run_medical_adaptation --run_perspective_training
```

Evaluation results are saved at:

```
evaluation_results/test_set_evaluation.json
```

## 🧩 Pretrained Checkpoints

<table>
  <tr>
    <td><b>RoBERTa Classifier</b></td>
    <td><a href="https://drive.google.com/drive/folders/1EQ7ywKsDVpsP4keDbSUqqRTKP1j1Lk2l">Download</a></td>
  </tr>
  <tr>
    <td><b>Medical Adapter</b></td>
    <td><a href="https://drive.google.com/drive/folders/1rAMX6HIV0KuIojrynFLn9sPG2GirQK1f">Download</a></td>
  </tr>
  <tr>
    <td><b>Perspective Adapter</b></td>
    <td><a href="https://drive.google.com/drive/folders/1I-aShHCAlNOCbyooiixOG6xH0Fwd49rn">Download</a></td>
  </tr>
</table>

---

## 📈 Results

| Metric                | Score   |
|-----------------------|---------|
| ROUGE-1               | 0.8372  |
| ROUGE-2               | 0.8047  |
| ROUGE-L               | 0.8348  |
| BLEU                  | 0.7875  |
| BERTScore (F1)        | 0.7995  |
| Perspective Adherence | 0.8716  |

🆚 Compared to <b>Baseline 2</b> (ROUGE-1: 3.23, BLEU-4: 0.0018), this method shows dramatic improvements in both lexical and semantic alignment.

---

<h3 align="center">Made with ❤️ for perspective-grounded summarization.</h3>

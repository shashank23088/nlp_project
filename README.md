<p>This repository implements perspective-specific summarization for healthcare Q&A platforms, creating targeted summaries for distinct perspectives (suggestion, information, cause, experience, question) while preserving original intent.</p>

<h2>Project Overview</h2>

<p>Healthcare Q&A platforms (e.g., Yahoo! Answers, Reddit) contain mixed content types, making it challenging to extract concise, perspective-specific summaries. Our NLP system generates targeted summaries that classify and condense information according to five distinct perspectives while preserving the author's original intent.</p>

<h2>Models and Approaches</h2>

<p>We explored three innovative approaches to perspective-aware summarization:</p>

<h3>1. Tiny LLaMA 3.2B-Instruct with Custom Energy-Based Loss</h3>

<p>This architecture integrates a RoBERTa-based classifier and custom energy-based loss function with three components:</p>
<ul>
    <li><strong>E<sub>p</sub></strong> (Perspective Confidence): Uses classifier confidence to guide generation</li>
    <li><strong>E<sub>s</sub></strong> (Lead Phrase Alignment): Measures alignment with perspective-specific phrases</li>
    <li><strong>E<sub>t</sub></strong> (Semantic Similarity): Captures semantic relevance via BERT embeddings</li>
</ul>

<p><a href="https://drive.google.com/drive/folders/1UknRONU4x4oi7zSOgEvTxO4xh5f3MjCq?usp=sharing">Model Codebase</a> | <a href="https://drive.google.com/drive/folders/1Ats3koAJmDejcj40p_vLkqagCHVgb6et?usp=sharing">Checkpoints</a></p>

<h3>2. Two-Stage Adapter Tuning with Classifier-Guided Loss</h3>

<p>A two-stage fine-tuning pipeline on LLaMA 3.2B-Instruct:</p>
<ul>
    <li>First stage: Train a <strong>medical adapter</strong> on domain-specific dataset</li>
    <li>Second stage: Train a <strong>perspective adapter</strong> using structured prompts and summaries guided by RoBERTa classifier</li>
    <li>Uses energy-based loss to encourage perspective alignment</li>
</ul>

<p><a href="https://drive.google.com/drive/folders/1EQ7ywKsDVpsP4keDbSUqqRTKP1j1Lk2l">RoBERTa Classifier</a> | <a href="https://drive.google.com/drive/folders/1rAMX6HIV0KuIojrynFLn9sPG2GirQK1f">Medical Adapter</a> | <a href="https://drive.google.com/drive/folders/1I-aShHCAlNOCbyooiixOG6xH0Fwd49rn">Perspective Adapter</a></p>

<h3>3. Perspective-Specific LLaMA 3.2B Adapters</h3>

<p>Creates separate adapters for each perspective category using LLaMA 3.2B model:</p>
<ul>
    <li>Individual adapters help identify perspectives with greater accuracy</li>
    <li>Uses Cross Entropy Loss during training</li>
    <li>LoRA-based PEFT on 4-bit quantized model</li>
</ul>

<p><a href="https://drive.google.com/drive/folders/1_7VS6Y1daxTzGE6VK64cM-WYFj83Jmq7?usp=sharing">RoBERTa Classifier</a> | <a href="https://drive.google.com/drive/folders/1HF8Nx4Cb0RZTVt1t7ySzMFaO-iYl8fe4?usp=sharing">Perspective-Specific Adapters</a></p>

<h2>Comparative Results</h2>

<p>The table below presents a direct comparison of performance metrics across all models:</p>

<table>
    <tr>
        <th>Metric</th>
        <th>Model 2<br>(Baseline)</th>
        <th>Model 3</th>
        <th>Model 4</th>
        <th>Model 5</th>
    </tr>
    <tr>
        <td>ROUGE-1</td>
        <td>3.23</td>
        <td>10.17</td>
        <td>0.8372</td>
        <td>0.8082</td>
    </tr>
    <tr>
        <td>ROUGE-2</td>
        <td>0.10</td>
        <td>0.44</td>
        <td>0.8047</td>
        <td>0.7702</td>
    </tr>
    <tr>
        <td>ROUGE-L</td>
        <td>3.02</td>
        <td>9.46</td>
        <td>0.8348</td>
        <td>0.8051</td>
    </tr>
    <tr>
        <td>BLEU</td>
        <td>0.0136</td>
        <td>0.0689</td>
        <td>0.7875</td>
        <td>0.7549</td>
    </tr>
    <tr>
        <td>BERTScore</td>
        <td>0.7866</td>
        <td>0.8104</td>
        <td>0.7995</td>
        <td>0.7633</td>
    </tr>
</table>

<h2>Analysis</h2>

<p>The evaluation results reveal interesting patterns across models:</p>

<ul>
    <li><strong>Model 3 (Tiny LLaMA 3.2B-Instruct)</strong> demonstrates the highest performance across most metrics, particularly in ROUGE scores and BERTScore, indicating strong lexical and semantic alignment with reference summaries.</li>
    
    <li><strong>Model 2 (Baseline)</strong> shows moderate performance with room for improvement, particularly in capturing n-gram overlap.</li>
    
    <li><strong>Models 4 and 5</strong> show lower ROUGE scores but relatively strong BLEU scores, suggesting they may generate summaries that are less lexically aligned with references but maintain semantic coherence.</li>
    
    <li>The BERTScore comparison reveals smaller variance across models than other metrics, indicating all approaches maintain semantic relevance to a reasonable degree.</li>
</ul>

<h2>Conclusion</h2>

<p>Among the models evaluated, the Tiny LLaMA 3.2B-Instruct architecture with classifier-guided energy-based training demonstrated the best balance between lexical accuracy and semantic coherence. The two innovative extensions—two-stage adapter tuning and perspective-specific adapters—highlighted promising directions for fine-grained control in medical text summarization.</p>

<p>Our work underscores the importance of integrating perspective-awareness in medical NLP tasks, offering better contextualization for users seeking specific types of health information.</p>

<h2>Team</h2>
<p>Group 11: Sajid Javid, Shashank Sharma, Shreyas Gupta</p>

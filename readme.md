---

```markdown
# 🌐 Project 1: Neural Machine Translation – Urdu to Roman Urdu (BiLSTM Seq2Seq)

### 🎯 Objective
The goal of this project is to build a **Neural Machine Translation (NMT)** system using a **Bidirectional LSTM (BiLSTM) Encoder–Decoder** to translate **Urdu script** into **Roman Urdu**.  
This project explores how deep sequence models perform on **low-resource, poetic text**, specifically Urdu Ghazals, by training on the [`urdu_ghazals_rekhta`](https://github.com/amir9ume/urdu_ghazals_rekhta) dataset.

---

## 🧠 Project Overview
| Component | Description |
|------------|-------------|
| **Input** | Urdu text (in Perso-Arabic script) |
| **Output** | Roman Urdu transliteration |
| **Model** | Seq2Seq with BiLSTM Encoder and LSTM Decoder |
| **Framework** | PyTorch |
| **Dataset** | [urdu_ghazals_rekhta](https://github.com/amir9ume/urdu_ghazals_rekhta) |
| **Metrics** | BLEU, Perplexity, and Character Error Rate (CER) |

---

## 📦 Dataset
The dataset contains Urdu Ghazals in three formats:
- Urdu (source)
- English transliteration (Roman Urdu or near equivalent)
- Hindi

We use the **Urdu → Roman Urdu** pairs.  
If Roman Urdu is not explicitly available, transliteration rules are applied to convert Urdu script into Roman Urdu using a defined character mapping.

---

## ⚙️ Preprocessing Steps
1. **Text Cleaning**
   - Normalize Urdu characters (handle variations like “ی” vs “ے”).
   - Remove extraneous punctuation, diacritics, and symbols.
2. **Transliteration Rules**
   - Custom rule-based mappings for Urdu → Roman Urdu (for missing data).
3. **Tokenization**
   - Experimented with both **word-level** and **subword tokenization** (Byte Pair Encoding, WordPiece).
4. **Dataset Split**
   - Training: 50%  
   - Validation: 25%  
   - Test: 25%

---

## 🏗️ Model Architecture
### Encoder
- **Bidirectional LSTM (BiLSTM)**  
- 2 layers (can vary in experiments)
- Embedding layer with 128–512 dimensions  
- Dropout applied for regularization  

### Decoder
- **LSTM-based Decoder**
- 4 layers (can vary in experiments)
- Attention mechanism (optional enhancement)
- Teacher forcing during training

### Model Summary
| Component | Layers | Hidden Size | Dropout | Notes |
|------------|---------|-------------|----------|--------|
| Encoder | 2 | 256–512 | 0.1–0.5 | BiLSTM |
| Decoder | 4 | 256–512 | 0.1–0.5 | LSTM |
| Embedding | - | 128–512 | - | Shared or separate embeddings |

---

## 🧪 Training Configuration
| Parameter | Values Tried |
|------------|---------------|
| **Embedding Dim** | 128, 256, 512 |
| **Hidden Size** | 256, 512 |
| **BiLSTM Layers (Encoder)** | 1, 2, 3, 4 |
| **Decoder Layers** | 2, 3, 4 |
| **Dropout** | 0.1, 0.3, 0.5 |
| **Learning Rate** | 1e-3, 5e-4, 1e-4 |
| **Batch Size** | 32, 64, 128 |
| **Optimizer** | Adam |
| **Loss Function** | Cross-Entropy Loss |

---

## 📊 Evaluation Metrics
1. **BLEU Score** – Translation quality
2. **Perplexity (PPL)** – Model fluency and generalization
3. **Character Error Rate (CER)** / **Edit Distance** – Transliteration accuracy

### Example Output
| Urdu Input | Roman Urdu (Ground Truth) | Model Output |
|-------------|----------------------------|---------------|
| محبت کا کوئی موسم نہیں ہوتا | mohabbat ka koi mausam nahi hota | mohabbat ka koi mosam nahi hota |
| دل سے تری نگاہ جگر تک اتر گئی | dil se teri nigah jigar tak utar gayi | dil se teri nigah jigar tak utr gayi |

---

## 🔁 Experiments
Three or more experiments were conducted by varying:
- Embedding dimension
- Hidden layer size
- Number of BiLSTM/Decoder layers
- Dropout
- Learning rate

Performance was compared across BLEU and CER metrics.

---

## 🚀 Deployment
A **Streamlit web app** was built to allow users to:
- Input Urdu text  
- Receive Roman Urdu transliteration in real time  
- Display BLEU score for example sentences  

The final trained model was deployed via **Streamlit Cloud / Hugging Face Spaces**.

---

## 🧩 Bonus Challenges (Optional)
- **Data Augmentation:** Back-transliteration and noise injection  
- **Model Variant:** Replacing LSTM layers with **xLSTM** or Transformer-style encoder  

---

## 📁 Repository Structure
```

├── data/
│   ├── raw/
│   ├── processed/
│   └── urdu_ghazals_rekhta/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── app/
│   └── streamlit_app.py
├── experiments/
│   ├── config1.yaml
│   ├── config2.yaml
│   └── config3.yaml
├── results/
│   ├── bleu_scores.csv
│   ├── perplexity.txt
│   └── qualitative_examples.txt
├── requirements.txt
├── README.md
└── LICENSE

````

---

## ⚡ How to Run
### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/urdu-to-roman-nmt.git
cd urdu-to-roman-nmt
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train Model

```bash
python src/train.py --config experiments/config1.yaml
```

### 4️⃣ Evaluate

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pt
```

### 5️⃣ Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## 🧾 Tools & Libraries

* **PyTorch** (for model implementation)
* **SentencePiece / HuggingFace Tokenizers** (for BPE)
* **NLTK / SacreBLEU** (for BLEU)
* **EditDistance / jiwer** (for CER)
* **Streamlit** (for deployment)
* **NumPy, Pandas, Matplotlib** (for analysis)

---

## 💡 Acknowledgments

* Dataset: [urdu_ghazals_rekhta](https://github.com/amir9ume/urdu_ghazals_rekhta)
* Project inspired by the idea of preserving linguistic beauty through computational models.

---

## ✍️ Author

**[Your Name]**
*MS/BS Student – Deep Learning & NLP Enthusiast*
🔗 [LinkedIn Profile](https://linkedin.com/in/yourprofile)
📧 [your.email@example.com](mailto:your.email@example.com)

> 💬 “Translating poetry is not just about words — it’s about rhythm, feeling, and soul.”

---

## 🏁 License

This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.
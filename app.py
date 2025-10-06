import streamlit as st
import torch
import torch.nn as nn
import random
import pickle


# Model definitions (Encoder, LuongAttention, Decoder, Seq2Seq) - copy from previous code
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, src):
        emb = self.embedding(src)
        outputs, (h, c) = self.lstm(emb)
        return outputs, (h, c)


class LuongAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim * 2)

    def forward(self, hidden, encoder_outputs):
        scores = torch.bmm(encoder_outputs, self.attn(hidden).unsqueeze(2)).squeeze(2)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        return context.squeeze(1), attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim * 2,
            n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        self.attn = LuongAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 4, vocab_size)
        self.hidden_dim = hidden_dim * 2
        self.n_layers = n_layers

    def forward(self, tgt, hidden, encoder_outputs):
        emb = self.embedding(tgt)
        output, hidden = self.lstm(emb, hidden)
        context, attn_w = self.attn(output.squeeze(1), encoder_outputs)
        combined = torch.cat((output.squeeze(1), context), dim=1)
        logits = self.fc(combined)
        return logits, hidden, attn_w


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_dim = hidden_dim

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        encoder_outputs, (h, c) = self.encoder(src)
        h_cat = (
            torch.cat([h[-2], h[-1]], dim=1)
            .unsqueeze(0)
            .repeat(self.decoder.n_layers, 1, 1)
        )
        c_cat = (
            torch.cat([c[-2], c[-1]], dim=1)
            .unsqueeze(0)
            .repeat(self.decoder.n_layers, 1, 1)
        )
        hidden = (h_cat, c_cat)
        outputs = []
        for t in range(tgt.size(1) - 1):
            input_t = tgt[:, t].unsqueeze(1)
            if t > 0 and random.random() > teacher_forcing_ratio:
                input_t = outputs[-1].argmax(-1)
            logits, hidden, _ = self.decoder(input_t, hidden, encoder_outputs)
            outputs.append(logits.unsqueeze(1))
        return torch.cat(outputs, dim=1)


# Vocab class (needed to rebuild vocabularies) - copy from previous code
class Vocab:
    def __init__(self, texts, min_freq=1):
        self.freq = {}
        for t in texts:
            for tok in t.split():
                self.freq[tok] = self.freq.get(tok, 0) + 1
        self.itos = ["<pad>", "<sos>", "<eos>", "<unk>"]
        for tok, c in self.freq.items():
            if c >= min_freq:
                self.itos.append(tok)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def encode(self, text):
        return [self.stoi.get(tok, self.stoi["<unk>"]) for tok in text.split()]

    def decode(self, ids):
        return [self.itos[i] for i in ids if i not in [0, 1, 2]]


# Inference function - copy from previous code
def translate_batch(model, src_tensor, max_len=20):
    model.eval()
    outputs = []
    with torch.no_grad():
        src_tensor = src_tensor.to(device)
        encoder_outputs, (h, c) = model.encoder(src_tensor)
        h_cat = (
            torch.cat([h[-2], h[-1]], dim=1)
            .unsqueeze(0)
            .repeat(model.decoder.n_layers, 1, 1)
        )
        c_cat = (
            torch.cat([c[-2], c[-1]], dim=1)
            .unsqueeze(0)
            .repeat(model.decoder.n_layers, 1, 1)
        )
        hidden = (h_cat, c_cat)
        inputs = torch.tensor([1] * src_tensor.shape[0], device=device).unsqueeze(1)

        for _ in range(max_len):
            logits, hidden, _ = model.decoder(inputs, hidden, encoder_outputs)
            pred_tokens = logits.argmax(-1).unsqueeze(1)
            outputs.append(pred_tokens)
            if (pred_tokens == 2).all():
                break
            inputs = pred_tokens

        outputs = torch.cat(outputs, dim=1)
    return outputs.cpu().numpy().tolist()


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocabularies
try:
    with open("urdu_vocab.pkl", "rb") as f:
        src_vocab = pickle.load(f)
    with open("roman_vocab.pkl", "rb") as f:
        tgt_vocab = pickle.load(f)
except FileNotFoundError:
    st.error(
        "Vocabulary files not found. Please ensure 'urdu_vocab.pkl' and 'roman_vocab.pkl' are in the same directory."
    )
    st.stop()

# Initialize model
encoder = Encoder(
    len(src_vocab.itos), emb_dim=128, hidden_dim=512, n_layers=2, dropout=0.3
)
decoder = Decoder(
    len(tgt_vocab.itos), emb_dim=128, hidden_dim=512, n_layers=2, dropout=0.3
)
model = Seq2Seq(encoder, decoder, hidden_dim=512).to(device)

# Load model state dictionary
try:
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()
except FileNotFoundError:
    st.error(
        "Model file 'transliteration_model.pt' not found. Please ensure the model file is in the same directory."
    )
    st.stop()


# Transliteration function
def transliterate(urdu_text):
    if not urdu_text or not any(
        ord(c) >= 0x0600 and ord(c) <= 0x06FF for c in urdu_text
    ):
        return "Please enter valid Urdu text."

    # Tokenize and encode input
    tokens = src_vocab.encode(urdu_text)
    src_tensor = torch.tensor([tokens], dtype=torch.long).to(device)

    # Get translation
    output_ids = translate_batch(model, src_tensor)[0]
    roman_text = " ".join(
        [
            tgt_vocab.itos[idx]
            for idx in output_ids
            if idx not in [0, 1, 2, 3] and idx < len(tgt_vocab.itos)
        ]
    )
    return roman_text if roman_text else "Translation not available."


# Streamlit App Interface
st.title("Urdu to Roman Transliteration")

urdu_input = st.text_area("Enter Urdu text here:", height=150)

if st.button("Transliterate"):
    if urdu_input:
        roman_output = transliterate(urdu_input)
        st.write("Roman Transliteration:")
        st.write(roman_output)
    else:
        st.write("Please enter some Urdu text.")

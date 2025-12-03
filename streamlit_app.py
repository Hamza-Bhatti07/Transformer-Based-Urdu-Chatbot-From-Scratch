
import streamlit as st, torch, math, re, json
from pathlib import Path
# minimal app imports
st.set_page_config(page_title="Urdu Transformer Chatbot", layout="centered")
st.title("Urdu Chatbot (Transformer)")
st.markdown("Type Urdu text and get a response. Uses saved Transformer model.")

# load vocab - expect files in working dir
try:
    with open("vocab_tokens.json","r",encoding="utf-8") as f:
        d=json.load(f)
    token2idx = d['token2idx']
    idx2token = {int(k):v for k,v in d['idx2token'].items()}
    PAD = token2idx["[PAD]"]
    CLS = token2idx["[CLS]"]
    EOS = token2idx["[EOS]"]
    UNK = token2idx["[UNK]"]
except FileNotFoundError:
    st.error("vocab_tokens.json not found. Make sure to run the training steps first.")
    st.stop()


# re-create small normalizer (same as training)
class Normalizer:
    def __init__(self):
        self.diacritics_pattern = re.compile(r"[\u064B-\u0652\u0670\u0640]")
        self.alef_pattern = re.compile(r"[\u0622\u0623\u0625\u0671\u0672\u0673]")
        self.yeh_pattern = re.compile(r"[\u0626\u06CC\u06C0\u06C1\u06C2\u064A]")
        self.kaf_pattern = re.compile(r"[\u0643\u06AA]")
        self.allowed = re.compile(r"[^\u0600-\u06FF\s،؟۔\u0660-\u0669]")
    def normalize(self,t):
        if not isinstance(t, str):
            return ""
        t = t.strip()
        t = self.diacritics_pattern.sub("", t)
        t = self.alef_pattern.sub("ا", t)
        t = self.yeh_pattern.sub("ی", t)
        t = self.kaf_pattern.sub("ک", t)
        t = re.sub(r"\s+"," ", t).strip()
        t = self.allowed.sub("", t)
        return t.strip()
norm = Normalizer()

# small helper encode/decode
def encode_text(s, max_len=50):
    toks = s.split()[:max_len-2]
    ids = [CLS] + [token2idx.get(t, token2idx['[UNK]']) for t in toks] + [EOS]
    if len(ids) < max_len:
        ids = ids + [token2idx['[PAD]']]*(max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.tensor([ids], dtype=torch.long)

# Modified decode_ids to accept 'self'
def decode_ids(self, ids):
    return " ".join([idx2token.get(int(i), "[UNK]") for i in ids if int(i) not in (token2idx['[PAD]'],)])


# --- Model Architecture (Copied from notebook) ---
# Positional encoding
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0,max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:,:seq_len,:].to(x.device)
        return x

# Multi-head attention (from scratch)
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)
        self.w_o = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        # q,k,v: (batch, seq_len, d_model)
        bs = q.size(0)
        Q = self.w_q(q).view(bs, -1, self.num_heads, self.d_k).transpose(1,2)
        K = self.w_k(k).view(bs, -1, self.num_heads, self.d_k).transpose(1,2)
        V = self.w_v(v).view(bs, -1, self.num_heads, self.d_k).transpose(1,2)
        # scaled dot-product
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)  # (bs, heads, qlen, klen)
        if mask is not None:
            # mask shape (bs, 1, 1, klen) or broadcastable
            scores = scores.masked_fill(mask==0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (bs, heads, qlen, d_k)
        out = out.transpose(1,2).contiguous().view(bs, -1, self.d_model)  # (bs, qlen, d_model)
        out = self.w_o(out)
        return out

class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_ff, d_model)
        )
    def forward(self, x):
        return self.net(x)

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.drop = torch.nn.Dropout(dropout)
    def forward(self, x, mask=None):
        # x: (bs, seq, d_model)
        att = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.drop(att))
        ff = self.ff(x)
        x = self.norm2(x + self.drop(ff))
        return x

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.drop = torch.nn.Dropout(dropout)
    def forward(self, x, enc_out, look_ahead_mask=None, padding_mask=None):
        # masked self
        att1 = self.self_attn(x, x, x, look_ahead_mask)
        x = self.norm1(x + self.drop(att1))
        # encoder-decoder
        att2 = self.enc_attn(x, enc_out, enc_out, padding_mask)
        x = self.norm2(x + self.drop(att2))
        ff = self.ff(x)
        x = self.norm3(x + self.drop(ff))
        return x

class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, pad_idx):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len=5000)
        self.layers = torch.nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self, src, mask=None):
        # src: (bs, seq)
        x = self.tok_emb(src) * math.sqrt(self.tok_emb.embedding_dim)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for l in self.layers:
            x = l(x, mask)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, pad_idx):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len=5000)
        self.layers = torch.nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(dropout)
        self.fc_out = torch.nn.Linear(d_model, vocab_size)
    def forward(self, tgt, enc_out, look_ahead_mask=None, padding_mask=None):
        x = self.tok_emb(tgt) * math.sqrt(self.tok_emb.embedding_dim)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for l in self.layers:
            x = l(x, enc_out, look_ahead_mask, padding_mask)
        out = self.fc_out(x)  # (bs, seq, vocab)
        return out

class TransformerModel(torch.nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_encoder_layers, num_decoder_layers, num_heads, d_ff, dropout, pad_idx):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_encoder_layers, num_heads, d_ff, dropout, pad_idx)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_decoder_layers, num_heads, d_ff, dropout, pad_idx)
        self.pad_idx = pad_idx
    def make_padding_mask(self, seq):
        # seq: (bs, seq_len)
        mask = (seq != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (bs,1,1,seq_len)
        return mask
    def make_look_ahead_mask(self, size):
        mask = torch.triu(torch.ones((size,size), dtype=torch.bool), diagonal=1).to(next(self.parameters()).device)
        mask = ~mask  # True where allowed
        # convert to shape (1,1,size,size) when needed
        return mask
    def forward(self, src, tgt):
        # src,tgt: (bs, seq)
        src_mask = self.make_padding_mask(src)
        tgt_padding_mask = self.make_padding_mask(tgt)
        look = self.make_look_ahead_mask(tgt.size(1)).to(src.device)  # (size,size)
        # expand look to (bs,1,seq,seq) via broadcasting with tgt_padding_mask
        look = look.unsqueeze(0).unsqueeze(0) & tgt_padding_mask  # (bs,1,seq,seq)
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, look, src_mask)
        return dec_out

# --- Generation Function (Copied from notebook) ---
def generate_sentence(model, src_tensor, vocab_obj, device, method='greedy', beam_size=3, max_len=50):
    # src_tensor: (1, seq)
    model.eval()
    with torch.no_grad():
        enc_out = model.encoder(src_tensor, model.make_padding_mask(src_tensor))
        if method == 'greedy':
            # start with CLS
            cur = torch.tensor([[CLS]], device=device)
            for _ in range(max_len-1):
                out = model.decoder(cur, enc_out, model.make_look_ahead_mask(cur.size(1)).unsqueeze(0).unsqueeze(0).to(device), model.make_padding_mask(src_tensor))
                logits = out[:, -1, :]  # (1, vocab)
                next_token = logits.argmax(-1).item()
                cur = torch.cat([cur, torch.tensor([[next_token]], device=device)], dim=1)
                if next_token == EOS:
                    break
            res_ids = cur.squeeze(0).cpu().tolist()
            # Call decode_ids as a method of vocab_wrapper
            return vocab_obj.decode(res_ids)
        elif method == 'beam':
            # beam search
            beams = [([CLS], 0.0)]
            for _ in range(max_len-1):
                new_beams = []
                for seq, score in beams:
                    if seq[-1] == EOS:
                        new_beams.append((seq, score))
                        continue
                    cur_tensor = torch.tensor([seq], device=device)
                    out = model.decoder(cur_tensor, enc_out, model.make_look_ahead_mask(cur_tensor.size(1)).unsqueeze(0).unsqueeze(0).to(device), model.make_padding_mask(src_tensor))
                    logits = out[:, -1, :]
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(0)
                    topk = torch.topk(log_probs, beam_size)
                    for k in range(beam_size):
                        tok = int(topk.indices[k].item())
                        sc = score + float(topk.values[k].item())
                        new_beams.append((seq+[tok], sc))
                new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
                beams = new_beams
            best_seq = beams[0][0]
            # Call decode_ids as a method of vocab_wrapper
            return vocab_obj.decode(best_seq)
        else:
            raise ValueError("Invalid method")


# --- Model Loading ---
best_model_path = "best_transformer_urdu.pt" # Ensure this matches the save path
try:
    # Need to match the model architecture used during training
    # Refer to the config used when saving the model
    # Assuming config was D_MODEL=512, NUM_HEADS=2, NUM_ENCODER_LAYERS=2, NUM_DECODER_LAYERS=2, D_FF=2048, DROPOUT=0.1
    model_params = {
        "src_vocab_size": len(token2idx),
        "tgt_vocab_size": len(token2idx),
        "d_model": 512,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "num_heads": 2,
        "d_ff": 2048,
        "dropout": 0.1,
        "pad_idx": PAD
    }
    net = TransformerModel(**model_params).to(torch.device('cpu')) # Load to CPU for Streamlit
    net.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    net.eval()
except FileNotFoundError:
    st.error(f"{best_model_path} not found. Make sure to train and save the model first.")
    st.stop()
except Exception as e:
    st.error("Could not load model state dict: " + str(e))
    st.stop()


# conversation state
if 'history' not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([3,1])
method = col2.selectbox("Decoding", ["beam","greedy"])
inp = st.text_area("سوال لکھیں (Urdu):", height=120)
if st.button("Send"):
    if not inp.strip():
        st.warning("Enter text.")
    else:
        st.session_state.history.append(("You", inp))
        # removed the net check as it should be loaded or app stops
        s = norm.normalize(inp)
        src = encode_text(s, max_len=50)
        # Pass a simple object with decode method instead of full Vocab object
        vocab_wrapper = type('V',(),{'decode': decode_ids})()
        resp = generate_sentence(net, src, vocab_wrapper, torch.device('cpu'), method=method, beam_size=3, max_len=50)
        st.session_state.history.append(("Bot", resp))

for who, msg in st.session_state.history[-10:]:
    if who=="You":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")


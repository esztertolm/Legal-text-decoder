import streamlit as st
import torch
from transformers import AutoTokenizer
from config import MODEL_OUTPUT
from modules.LegalBERT import LegalBERT
from transformers import AutoConfig



# A GUI nagyrészét én csináltam, LLM asszintenciával megszépítettem az oldalt színekkel és emojikkal.
@st.cache_resource
def load_model(model_path=MODEL_OUTPUT):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    config = AutoConfig.from_pretrained(model_path)
    
    model = LegalBERT(num_labels=5) 
    state_dict = torch.load(f"{model_path}/pytorch_model.bin", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


st.set_page_config(page_title="Jogi Szöveg Érthetőségi Osztályozó", page_icon="🧠", layout="centered")

st.title("🧠 Jogi Szöveg Érthetőségi Osztályozó")
st.markdown("Írj be egy szöveget, és a modell megmondja, **mennyire érthető**!")


user_input = st.text_area("Szöveg", height=200, placeholder="Pl.: A felhasználó bármikor kérheti regisztrációjának törlését...")


if st.button("📊 Értékeld a szöveget") and user_input.strip():
    with st.spinner("A modell gondolkodik..."):
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding="max_length", max_length=256).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            pred_id = torch.argmax(probs).item()
            confidence = probs[pred_id].item()

        id2label = {
            0: "1 - Nehezen érthető 😕",
            1: "2 - Inkább nehezen érthető 😐",
            2: "3 - Többé-kevésbé érthető 🙂",
            3: "4 - Érthető 😄",
            4: "5 - Könnyen érthető 🤩",
        }

        label_text = id2label.get(pred_id, f"Osztály {pred_id}")

        st.markdown("---")
        st.subheader("🔮 Predikció eredménye:")
        st.markdown(
            f"""
            <div style="background-color:#f0f8ff; border-radius:12px; padding:20px; text-align:center; font-size:20px;">
            <b>{label_text}</b><br>
            <span style="font-size:16px; color:gray;">Bizonyosság: {confidence*100:.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


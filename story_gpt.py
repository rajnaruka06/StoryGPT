from bpe import BPE
import torch
from model import Model
import streamlit as st


def generate_text(model, tokenizer, context, device, length = 50):
    model.eval()
    with torch.no_grad():
        context = tokenizer.encode(context)
        context = context[:-1]
        context = torch.Tensor(context).long().to(device)
        for _ in range(length):
            y_pred = model(context.unsqueeze(0)[:,-model.context_length:])
            next_token = torch.multinomial(y_pred[0, -1], 1).item()
            # next_token = torch.argmax(y_pred[0, -1]).item()
            if next_token == tokenizer.encode(tokenizer.EOS_TOKEN)[0]:
                break
            context = torch.cat([context, torch.Tensor([next_token]).long().to(DEVICE)])
    
    return tokenizer.decode(context)


if __name__ == "__main__":

    tokenizer = BPE()
    tokenizer.from_json("Saved_Vocab/vocab_v2.json")
    tokenizer.EOS_TOKEN = "<EOS>"
    tokenizer.PAD_TOKEN = "<PAD>"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CONTEXT_LENGTH = 256
    EMBEDDING_DIM = 128
    D_MODEL = 256

    model = Model(
        num_heads=8
        , d_model=D_MODEL
        , vocab_size=tokenizer.vocab_size
        , num_layers=3
        , dropout=0.3
        , context_length=CONTEXT_LENGTH
        , embedding_dim=EMBEDDING_DIM
        , padding_idx=tokenizer.encode_vocab[tokenizer.PAD_TOKEN]
        )
    model.to(DEVICE)
    model.load_state_dict(torch.load("Saved_Models/model_v4.pth"))
    model.eval()

    st.title("StoryGPT")
    st.markdown(
        "<h3> Enter text and then click on the button to generate a story </h3>", unsafe_allow_html=True
    )
    user_input = st.text_area(
        label="Write Your prompt here", height=200, value="once upon a time"
    )
    user_length = st.number_input(
        label="Text Length", min_value=1, max_value=1000, value=100, step=1
    )

    if st.button("Write Story!"):
        if user_input:
            text = generate_text(model, tokenizer, user_input, DEVICE, length=user_length)
            st.markdown("## Story")
            st.markdown(
            f'<div style="border: 2px solid #3498DB; border-radius: 5px; padding: 10px;">{text}</div>',
            unsafe_allow_html=True
        )
        else:
            st.warning("Please enter some text for prediction.")

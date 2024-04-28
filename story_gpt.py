from bpe import BPE
import torch
from model import Model
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import random


def _scratch_generate_text_greedy(model, tokenizer, context, device, length = 50):
    model.eval()
    with torch.no_grad():
        context_ids = tokenizer.encode(context)
        context_ids = context_ids[:-1]
        context_ids = torch.Tensor(context_ids).long().to(device)
        progress_bar = st.progress(0)
        for step in range(length):
            progress_bar.progress(step/length)
            y_pred = model(context_ids.unsqueeze(0)[:,-model.context_length:])[0, -1, :]
            next_token = torch.argmax(y_pred).item()
            if next_token == tokenizer.encode_vocab[tokenizer.EOS_TOKEN]:
                break
            context_ids = torch.cat([context_ids, torch.Tensor([next_token]).long().to(DEVICE)])
    
    return tokenizer.decode(context_ids)

def _scratch_generate_text_beam(model, tokenizer, context, device, length = 50, num_beams = 5):
    model.eval()

    generated_sequences = []
    generated_scores = []

    with torch.no_grad():
        ## Initial beam of sequences
        context_ids = tokenizer.encode(context)[:-1]
        context_ids = torch.Tensor(context_ids).long().to(device)
        
        outputs = model(context_ids.unsqueeze(0)[:,-model.context_length:]) ## Probs for the next token, (1, context_length, vocab_size)
        log_probs = torch.log(outputs[:, -1, :]) ## log probs (, vocab_size)
        top_scores, top_indices = torch.topk(log_probs, num_beams)
        
        for score, index in zip(top_scores[0], top_indices[0]):
            generated_sequences.append([index.item()])
            generated_scores.append(score.item())

        progress_bar = st.progress(0)
        ## Expanding the beams until all sequences reach the end or maximum length
        for step in range(length):
            progress_bar.progress((step+1)/length)
            new_sequences = []
            new_scores = []
            for seq, score in zip(generated_sequences, generated_scores):
                if seq[-1] == tokenizer.encode_vocab[tokenizer.EOS_TOKEN] or len(seq) >= length:
                    new_sequences.append(seq)
                    new_scores.append(score)
                else:
                    # Scores for the next token based on the current sequence
                    input_ids = torch.tensor(seq).long().to(device)
                    # outputs = model(input_ids)
                    outputs = model(input_ids.unsqueeze(0)[:,-model.context_length:])
                    # log_probs = torch.log_softmax(outputs[:, -1, :], dim=-1)
                    log_probs = torch.log(outputs[:, -1, :])
                    top_scores, top_indices = torch.topk(log_probs, num_beams)
                    for s, i in zip(top_scores[0], top_indices[0]):
                        new_seq = seq + [i.item()]
                        new_sequences.append(new_seq)
                        new_scores.append(score + s.item())
            
            # Updating the top-k sequences
            combined = list(zip(new_sequences, new_scores))
            combined.sort(key=lambda x: x[1], reverse=True)
            generated_sequences = [s for s, _ in combined[:num_beams]]
            generated_scores = [s for _, s in combined[:num_beams]]

        return tokenizer.decode(context_ids.tolist() + generated_sequences[0])

def _scratch_generate_text_top_k(model, tokenizer, context, device, length = 50, top_k = 10):
    model.eval()
    with torch.no_grad():
        context_ids = tokenizer.encode(context)
        context_ids = context_ids[:-1]
        context_ids = torch.Tensor(context_ids).long().to(device)
        progress_bar = st.progress(0)
        for step in range(length):
            progress_bar.progress(step/length)
            y_pred = model(context_ids.unsqueeze(0)[:,-model.context_length:])[0, -1, :]
            top_k_scores, top_k_indices = torch.topk(y_pred, top_k, dim=-1)
            next_token = random.choice(top_k_indices.tolist())
            if next_token == tokenizer.encode_vocab[tokenizer.EOS_TOKEN]:
                break
            context_ids = torch.cat([context_ids, torch.Tensor([next_token]).long().to(DEVICE)])
    
    return tokenizer.decode(context_ids)

def _scratch_generate_text_top_p(model, tokenizer, context, device, length = 50, top_p = 0.9):
    model.eval()
    with torch.no_grad():
        context_ids = tokenizer.encode(context)
        context_ids = context_ids[:-1]
        context_ids = torch.Tensor(context_ids).long().to(device)
        progress_bar = st.progress(0)
        for step in range(length):
            progress_bar.progress(step/length)
            y_pred = model(context_ids.unsqueeze(0)[:,-model.context_length:])[0, -1, :]
            sorted_probs, sorted_indices = torch.sort(y_pred, descending=True)
            ## print(sorted_probs[:5]) ## The model is too sure about the next token --> Overfitting
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            choices = sorted_indices[cumulative_probs <= top_p].tolist()
            next_token  = random.choice(choices) if choices else sorted_indices[0].item()
            if next_token == tokenizer.encode(tokenizer.EOS_TOKEN)[0]:
                break
            context_ids = torch.cat([context_ids, torch.Tensor([next_token]).long().to(DEVICE)])
    
    return tokenizer.decode(context_ids)

def _gpt2_generate_story_greedy(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(DEVICE)
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def _gpt2_generate_story_with_beam(model, tokenizer, prompt, max_length=100, num_beams=3, early_stopping=True, no_repeat_ngram_size=2):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(DEVICE)
    outputs = model.generate(**inputs, max_length=max_length, num_beams=num_beams, early_stopping=early_stopping,
                              no_repeat_ngram_size=no_repeat_ngram_size, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def _gpt2_generate_story_top_k(model, tokenizer, prompt, max_length=100, top_k=10):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(DEVICE)
    outputs = model.generate(**inputs, max_length=max_length, do_sample=True, top_k=top_k)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def _gpt2_generate_story_top_p(model, tokenizer, prompt, max_length=100, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(DEVICE)
    outputs = model.generate(**inputs, max_length=max_length, do_sample=True, top_p=top_p)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)




if __name__ == "__main__":

    st.title("StoryGPT")
    model_name =  st.sidebar.radio("Choose a model", ["From Scratch GPT2", "Fine-Tuned GPT2"])
    st.markdown(
        "<h3> Enter text and then click on the button to generate a story </h3>", unsafe_allow_html=True
    )
    prompt = st.text_area(
        label="Write Your prompt here", height=200, value="once upon a time"
    )
    text_length = st.sidebar.number_input(
        label="Text Length", min_value=1, max_value=1000, value=100, step=10
    )
    decoding_method  = st.sidebar.selectbox("Decoding Method", ["Greedy", "Beam Search", "Top-K", "Top-P (Nucleus)"])
    if decoding_method == "Beam Search":
        num_beams = st.sidebar.number_input(
            label="Beam Width", min_value=1, max_value=10, value=5, step=1
        )
    elif decoding_method == "Top-K":
        top_k = st.sidebar.number_input(
            label="Top K", min_value=1, max_value=100, value=10, step=5
        )
    elif decoding_method == "Top-P (Nucleus)": 
        top_p = st.sidebar.number_input(
            label="Top P (Nucleus)", min_value=0.1, max_value=1.0, value=0.9, step=0.05
        )


    if model_name == "From Scratch GPT2":

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
        
        if st.button("Write Story!"):
            if prompt:
                if decoding_method == "Beam Search":
                    text = _scratch_generate_text_beam(model, tokenizer, prompt, DEVICE, length = text_length, num_beams = num_beams)
                elif decoding_method == "Greedy":
                    text = _scratch_generate_text_greedy(model, tokenizer, prompt, DEVICE, length = text_length)
                elif decoding_method == "Top-K":
                    text = _scratch_generate_text_top_k(model, tokenizer, prompt, DEVICE, length = text_length, top_k = top_k)  
                else:
                    text = _scratch_generate_text_top_p(model, tokenizer, prompt, DEVICE, length = text_length, top_p = top_p)
                st.markdown("## Story")
                st.markdown(
                f'<div style="border: 2px solid #3498DB; border-radius: 5px; padding: 10px;">{text}</div>',
                unsafe_allow_html=True
            )
            else:
                st.warning("Please enter some text for prediction.")
    
    else:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = '<PAD>'

        # model = GPT2LMHeadModel.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("Saved_Models/gpt2_language_model_training_process_2")
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(DEVICE)

        if st.button("Write Story!"):
            if prompt:
                if decoding_method == "Beam Search":
                    text = _gpt2_generate_story_with_beam(model, tokenizer, prompt, max_length=text_length, num_beams=num_beams)
                elif decoding_method == "Greedy":
                    text = _gpt2_generate_story_greedy(model, tokenizer, prompt, max_length=text_length)
                elif decoding_method == "Top-K":
                    text = _gpt2_generate_story_top_k(model, tokenizer, prompt, max_length=text_length, top_k=top_k)
                else:
                    text = _gpt2_generate_story_top_p(model, tokenizer, prompt, max_length=text_length, top_p=top_p)
                st.markdown("## Story")
                st.markdown(
                f'<div style="border: 2px solid #3498DB; border-radius: 5px; padding: 10px;">{text}</div>',
                unsafe_allow_html=True
            )
            else:
                st.warning("Please enter some text for prediction.")


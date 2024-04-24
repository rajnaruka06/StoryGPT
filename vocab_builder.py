import json
from bpe import BPE
## from datasets import load_dataset

if __name__ == "__main__":
    # corpus = load_dataset("roneneldan/TinyStories")['train']['text']
    with open("Saved_Data/train_data.json", "r") as f:
        corpus = json.load(f)
    
    bpe = BPE(vocab_size=1024)
    
    bpe.from_json("Saved_Vocab/vocab_v4.json")
    bpe.EOS_TOKEN = "<EOS>"
    bpe.PAD_TOKEN = "<PAD>"

    bpe.vocab_size = 1024

    bpe.fit(corpus, num_processes=16)

    with open("Saved_Vocab/vocab_v5.json", "w") as f:
        json.dump(bpe.decode_vocab, f)
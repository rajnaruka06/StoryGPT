from collections import Counter
from multiprocessing import Pool
import json
from bpe import BPE
import matplotlib.pyplot as plt

tokenizer = BPE(vocab_size=512)
tokenizer.from_json("./Saved_Vocab/vocab_v5.json")
tokenizer.EOS_TOKEN = "<EOS>"
tokenizer.PAD_TOKEN = "<PAD>"
print(tokenizer.vocab_size)

def _freq_dist_helper(corpus):
    freq_dist = Counter()
    for story in corpus:
        tokens = tokenizer.encode(story)
        freq_dist[len(tokens)] += 1
    return freq_dist


def _get_freq_dist(corpus, num_processes=8):
    final_freq_dist = Counter()
    batch_size = len(corpus) // num_processes
    batch_list = [corpus[idx: idx + batch_size] for idx in range(0, len(corpus), batch_size)]
    with Pool(processes=num_processes) as pool:
        for partial_freq_dist in pool.imap_unordered(_freq_dist_helper, batch_list ):
            final_freq_dist.update(partial_freq_dist)
    return final_freq_dist


if __name__ == "__main__":
    with open("Saved_Data/train_data.json", "r") as f:
        corpus = json.load(f)
    
    dist = _get_freq_dist(corpus, 16)
    
    plt.figure(figsize=(12, 8))
    plt.bar(dist.keys(), dist.values())
    plt.savefig("Visualizations/freq_dist_vocab_v5.png")
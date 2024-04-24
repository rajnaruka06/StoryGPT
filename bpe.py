## Tried functional programming approach to implement BPE algorithm --> a lot of issues while passing the parameters to the helper functions


from collections import Counter
from multiprocessing import Pool
import time
import json

class BPE:
    def __init__(self, vocab_size=512):
        self.vocab_size = vocab_size
        self.encode_vocab = None
        self.decode_vocab = None
    
    def _build_init_vocab(self, corpus, EOS_TOKEN="<EOS>", PAD_TOKEN="<PAD>"):
        vocab = set()
        for text in corpus: vocab.update(text)

        self.EOS_TOKEN = EOS_TOKEN
        self.PAD_TOKEN = PAD_TOKEN
        self.decode_vocab = list(vocab)
        self.decode_vocab.sort()
        self.decode_vocab.append(EOS_TOKEN)
        self.decode_vocab.append(PAD_TOKEN)
        self.encode_vocab = {text: idx for idx, text in enumerate(self.decode_vocab)}

    def decode(self, encoded_text):
        return "".join([self.decode_vocab[idx] for idx in encoded_text])

    def encode(self, text):
        tokens = [self.encode_vocab[token] for token in text]
        tokens.append(self.encode_vocab[self.EOS_TOKEN])
        any_merge = True
        while any_merge:
            any_merge = False
            encoded_text = []
            idx = 0
            while idx < len(tokens):
                if idx == len(tokens) - 1:
                    encoded_text.append(tokens[idx])
                    idx += 1
                else:
                    try:
                        sub_text = self.decode(tokens[idx: idx + 2])
                        new_token = self.encode_vocab[sub_text]
                        encoded_text.append(new_token)
                        idx += 2
                        any_merge = True
                    except:
                        encoded_text.append(tokens[idx])
                        idx += 1
            tokens = encoded_text
        return tokens
    
    def _freq_dist_helper(self, corpus):
        freq_dist = Counter()
        for story in corpus:
            tokens = self.encode(story)
            for idx in range(len(tokens) - 1):
                freq_dist[(tokens[idx], tokens[idx + 1])] += 1
        return freq_dist
    
    def _get_freq_dist(self, corpus, num_processes=8):
        final_freq_dist = Counter()
        batch_size = len(corpus) // num_processes
        batch_list = [corpus[idx: idx + batch_size] for idx in range(0, len(corpus), batch_size)]
        with Pool(processes=num_processes) as pool:
            for partial_freq_dist in pool.imap_unordered(self._freq_dist_helper, batch_list):
                final_freq_dist.update(partial_freq_dist)
        return final_freq_dist
    
    def _update_vocab(self, freq_dist):
        most_common_freq = freq_dist.most_common(1)[0][1]
        for (elem, freq) in freq_dist.most_common():
            if freq < most_common_freq * 0.9 or len(self.decode_vocab) == self.vocab_size:
                break
            text = self.decode(elem)
            self.encode_vocab[text] = len(self.decode_vocab)
            self.decode_vocab.append(text)

    def fit(self, corpus, num_processes=8):
        if not self.decode_vocab: 
            self._build_init_vocab(corpus)
        
        num_iters = 0
        while len(self.decode_vocab) < self.vocab_size:
            start_time = time.time()
            num_iters += 1
            freq_dist = self._get_freq_dist(corpus, num_processes)
            self._update_vocab(freq_dist)
            print("-"*100)
            cur_time = time.time()
            print(f"Iteration: {num_iters}, Vocab size: {len(self.decode_vocab)}, time passed: {(cur_time - start_time)//60} mins")
            print(self.decode_vocab)

    def from_json(self, path):
        with open(path, "r") as f:
            self.decode_vocab = json.load(f)
        self.encode_vocab = {text: idx for idx, text in enumerate(self.decode_vocab)}
        self.vocab_size = len(self.decode_vocab)
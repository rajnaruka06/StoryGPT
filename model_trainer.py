import numpy as np
import torch
from random import shuffle
import time
# import json
# from bpe import BPE
# from model import Model
# import torch.nn as nn

def get_batch(corpus, tokenizer, story_indexes, context_length=128):
    
    X = []
    y = []
    for idx in story_indexes:
        story = corpus[idx]
        story = tokenizer.encode(story)
        if len(story) <= (context_length + 1):
            story += [tokenizer.encode_vocab[tokenizer.PAD_TOKEN]] * (context_length + 1 - len(story))
            jdx = 0
        else:
            jdx = np.random.randint(0, len(story) - (context_length + 1))
        X.append(story[jdx: jdx + context_length])
        y.append(story[jdx + 1: jdx + context_length + 1])
    
    return torch.Tensor(X).long(), torch.Tensor(y).long()

def train(model, criterion, optimizer, corpus, tokenizer, context_length=128, num_epochs=1, batch_size=128):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)
    model.train()

    NO_OF_BATCHES = len(corpus) // batch_size
    losses = []

    for epoch in range(num_epochs):
        shuffle(corpus)
        start_time = time.time()
        for batch in range(NO_OF_BATCHES):
            start_idx, end_idx = batch * batch_size, (batch + 1) * batch_size
            X, y = get_batch(corpus, tokenizer, range(start_idx, end_idx), context_length=context_length)
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred.view(-1, tokenizer.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

            if batch % 1000 == 0:
                print(f"Epoch: {epoch}, Batch: {batch}, Loss: {loss.item()}")
                losses.append(loss.item())
        end_time = time.time()
        print(f"Epoch: {epoch}, Time: {(end_time - start_time)/60} minutes")
    
    return losses

# if __name__ == "__main__":
#     with open("train_data.json", "r") as f:
#         corpus = json.load(f)

#     tokenizer = BPE(vocab_size=512)
#     tokenizer.from_json("vocab_v2.json")
#     tokenizer.EOS_TOKEN = "<EOS>"
#     tokenizer.PAD_TOKEN = "<PAD>"
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     CONTEXT_LENGTH = 256
#     EMBEDDING_DIM = 128
#     D_MODEL = 256

#     model = Model(num_heads=8, d_model=D_MODEL, vocab_size=tokenizer.vocab_size, num_layers=3, dropout=0.3, context_length=CONTEXT_LENGTH, embedding_dim=EMBEDDING_DIM)
#     criterion = nn.CrossEntropyLoss().to(DEVICE)
#     optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

#     model.to(DEVICE)
#     model.train()

#     NUM_EPOCHS = 1
#     BATCH_SIZE = 128

#     losses = train(
#         model = model,
#         criterion = criterion,
#         optimizer = optimizer,
#         corpus = corpus,
#         tokenizer = tokenizer,
#         context_length = CONTEXT_LENGTH,
#         num_epochs = NUM_EPOCHS,
#         batch_size = BATCH_SIZE
#     )
    

    # NO_OF_BATCHES = len(corpus) // BATCH_SIZE
    # losses = []

    # for epoch in range(NUM_EPOCHS):
    #     shuffle(corpus)
    #     for batch in range(NO_OF_BATCHES):
    #         start_idx, end_idx = batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE
    #         X, y = get_batch(corpus, tokenizer, range(start_idx, end_idx), context_length=CONTEXT_LENGTH)
    #         X, y = X.to(DEVICE), y.to(DEVICE)

    #         optimizer.zero_grad()
    #         y_pred = model(X)
    #         loss = criterion(y_pred.view(-1, tokenizer.vocab_size), y.view(-1))
    #         loss.backward()
    #         optimizer.step()

    #         if batch % 1000 == 0:
    #             print(f"Epoch: {epoch}, Batch: {batch}, Loss: {loss.item()}")
    #             losses.append(loss.item())

    # start_idx, end_idx = 0, BATCH_SIZE
    # X, y = get_batch(corpus, tokenizer, range(start_idx, end_idx), context_length=CONTEXT_LENGTH)
    # X, y = X.to(DEVICE), y.to(DEVICE)

    # optimizer.zero_grad()
    # y_pred = model(X)
    # loss = criterion(y_pred.view(-1, tokenizer.vocab_size), y.view(-1))
    # loss.backward()
    # optimizer.step()

    # print(f"Loss: {loss.item()}")


    # torch.save(model.state_dict(), "my_model_state_dict_3.pth")
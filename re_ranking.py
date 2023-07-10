
from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment
from allennlp.data.dataloader import PyTorchDataLoader
prepare_environment(Params({})) # sets the seeds to be fixed

import torch
import logging

from torch.optim import Adam
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
#makni
from allennlp.nn.util import move_to_device

from data_loading import IrTripleDatasetReader, IrLabeledTupleDatasetReader
from model_knrm import KNRM
from model_tk import TK

from core_metrics import (
    unrolled_to_ranked_result,
    load_qrels,
    calculate_metrics_plain,
)

#makni
device = torch.device("cuda")

PATH = "/content/drive/MyDrive/air/"
config = {
    "vocab_directory": PATH + "data/Part-2/allen_vocab_lower_10",
    "pre_trained_embedding": PATH + "data/Part-2/glove.42B.300d.txt",
    "model": "tk",
    "train_data": PATH + "data/Part-2/triples.train.tsv",
    "validation_data": PATH + "data/Part-2/msmarco_tuples.validation.tsv",
    "test_data": PATH + "data/Part-2/msmarco_tuples.test.tsv",
    "batch_size": 64,
    "test_batch_size": 256,
    "lr": 1e-4,
}


def eval(model, loader):

    results_list = []
    results = {}

    #test loop 

    for test_batch in Tqdm.tqdm(loader):

        model.eval()
        test_batch = move_to_device(test_batch, device)
        queries = test_batch["query_tokens"]
        documents = test_batch["doc_tokens"]

        with torch.no_grad():
            output = model(queries, documents)

        scoring_triples = list(
            zip(test_batch["query_id"], test_batch["doc_id"], output.tolist())
        )
        results_list.extend(scoring_triples)  
    
    #evaluation

    for query_id, doc_id, score in results_list:
        results.setdefault(str(query_id), []).append((str(doc_id), float(score)))
    ranked_results = unrolled_to_ranked_result(results)

    qrels = load_qrels(PATH + "data/Part-2/msmarco_qrels.txt")
    metrics = calculate_metrics_plain(ranked_results, qrels)
    metrics = [metrics["MRR@10"], metrics["MRR@20"], metrics["nDCG@10"], metrics["nDCG@20"]]

    return tuple([ round(elem, 2) for elem in metrics ])

#
# data loading
#

vocab = Vocabulary.from_files(config["vocab_directory"])
tokens_embedder = Embedding(vocab=vocab,
                           pretrained_file= config["pre_trained_embedding"],
                           embedding_dim=300,
                           trainable=True,
                           padding_index=0)
word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})

# recommended default params for the models (but you may change them if you want)
#makni
if config["model"] == "knrm":
    model = KNRM(word_embedder, n_kernels=11).to(device)
elif config["model"] == "tk":
    model = TK(word_embedder, n_kernels=11, n_layers = 2, n_tf_dim = 300, n_tf_heads = 10).to(device)


# optimizer, loss 
optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=0.0001)
criterion = torch.nn.MarginRankingLoss(margin=1, reduction="mean").to(device)

print('Model',config["model"],'total parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
print('Network:', model)

#
# train
#

_triple_reader = IrTripleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30)
_triple_reader = _triple_reader.read(config["train_data"])
_triple_reader.index_with(vocab)
loader = PyTorchDataLoader(_triple_reader, batch_size=config["batch_size"])

_val_reader = IrLabeledTupleDatasetReader(
        lazy=True, max_doc_length=180, max_query_length=30
    )
_val_reader = _val_reader.read(config["validation_data"])
_val_reader.index_with(vocab)
val_loader = PyTorchDataLoader(_val_reader, batch_size=config["test_batch_size"])


#keeping track of results for early stoping
results = []
best_res = 0
last_best_res = 0

for epoch in range(4):

    losses = []
    iteration = 0

    for batch in Tqdm.tqdm(loader):
        model.train()
        optimizer.zero_grad()

        #forward
        batch = move_to_device(batch, device)
        queries = batch["query_tokens"]
        doc_pos, doc_neg = batch["doc_pos_tokens"], batch["doc_neg_tokens"]
        out_pos, out_neg = model.forward(queries, doc_pos), model.forward(queries, doc_neg)

        target = torch.ones(batch["query_tokens"]["tokens"]["tokens"].shape[0], dtype=torch.float).to(device)

        #backward
        loss = criterion(out_pos, out_neg, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
        if iteration % 500 == 0:
            mrr10, mrr20, ndcg10, ndcg20 = eval(model, val_loader)
            results.append(mrr10)

            if mrr10 > best_res:
                torch.save(model.state_dict(), PATH + "data/dumps/tk/epoch"+str(epoch)+"_valmrr"+str(mrr10)+".pth")
                best_res = mrr10
                last_best_res = 0
            else:
                last_best_res += 1

            #early stopping due to no improvement in last 2000 batches
            if last_best_res >= 4:
                break

            print("loss = " + str(sum(losses)/len(losses)))

        iteration += 1

#
# eval (duplicate for validation inside train loop - but rename "loader", since
# otherwise it will overwrite the original train iterator, which is instantiated outside the loop)
#

_tuple_reader = IrLabeledTupleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30)
_tuple_reader = _tuple_reader.read(config["test_data"])
_tuple_reader.index_with(vocab)
loader = PyTorchDataLoader(_tuple_reader, batch_size=config["test_batch_size"])

mrr10, mrr20, ndcg10, ndcg20 = eval(model, loader)
print("test data results:")
print("mrr10 = "+str(mrr10))
print("mrr20 = "+ str(mrr20))
print("ndcg10 = "+ str(ndcg10))
print("ndcg = "+ str(ndcg20))
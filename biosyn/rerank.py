import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from tqdm import tqdm
from utils import marginal_nll 
import numpy as np
import time
class RerankNet(nn.Module):
    def __init__(self, encoder, lr, weight_decay, use_cuda, forward_chunk_size, metrics_train):
        super(RerankNet, self).__init__()
        self.encoder = encoder
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.optimizer = optim.AdamW(
            self.encoder.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.criterion = marginal_nll
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
        self.to(self.device)
        #hidden size of encoder
        self.H = getattr(getattr(self.encoder, "config", None), "hidden_size", None)

        self.forward_chunk_size = forward_chunk_size 
        self.metrics_train = metrics_train
        # trying to save memory but makes it slower 
        # try:
        #     self.encoder.gradient_checkpointing_enable()
        #     if hasattr(self.encoder.config, "use_cache"):
        #         self.encoder.config.use_cache = False
        # except Exception:
        #     pass



    def forward(self, x):

        # query_token, candidate_tokens = x
        query_token, candidate_tokens = x
        batch_size, topk, max_length = candidate_tokens['input_ids'].shape

        if self.use_cuda:
            query_token['input_ids'] = query_token['input_ids'].to(self.device, non_blocking=True)
            query_token['attention_mask'] = query_token['attention_mask'].to(self.device, non_blocking=True)

            candidate_tokens['input_ids'] = candidate_tokens['input_ids'].to(self.device, non_blocking=True)
            candidate_tokens['attention_mask'] = candidate_tokens['attention_mask'].to(self.device, non_blocking=True)


        t_q_embedings= time.time()
        # dense embed for query and candidates
        query_embed = self.encoder(
            input_ids=query_token["input_ids"].squeeze(1),
            attention_mask=query_token["attention_mask"].squeeze(1),
            return_dict=False)[0] # (B, L, H)
        assert query_embed.shape == (batch_size, max_length, self.H)
        query_embed = query_embed[:,0, :].unsqueeze(1).contiguous() #(B, 1, H) [cls]

        self.metrics_train.log_event(event="query_embedings",  t0=t_q_embedings)

        t_candidate_embedings= time.time()
        cand_tokens_inp = candidate_tokens["input_ids"] #batch_size, topk, max_length 
        cand_tokens_att = candidate_tokens["attention_mask"]
        scores= torch.empty((batch_size, topk))
        chunk_size= self.forward_chunk_size
        scores_parts = []
        for start in range(0, batch_size, chunk_size ):
            end = min(start + chunk_size, batch_size)
            chunk_inp = cand_tokens_inp[start:end, :, :].reshape(-1, max_length).to(self.device, non_blocking=True) # chunk_size * topk, L
            chunk_att = cand_tokens_att[start:end, : , :].reshape(-1, max_length).to(self.device, non_blocking=True)
            #cls #chunk_size * topk,  H
            chunk_cand_emb = self.encoder(input_ids=chunk_inp, attention_mask=chunk_att, return_dict=False)[0]
            #chunk_size, H, K
            chunk_cand_emb = chunk_cand_emb[:, 0, :].reshape((end-start), topk, -1).transpose(1,2).contiguous()
            chunk_query_embed = query_embed[start:end].to(self.device, non_blocking=True)
            score= torch.bmm(chunk_query_embed, chunk_cand_emb).squeeze(1)

            #I was doing:
            # scores[offset : offset + (end-start)] = torch.tensor(score.detach().numpy())
            # Then I discovered that I need to keep graph and without converting to numpy, hope this will not cause oom
            scores_parts.append(score)
        self.metrics_train.log_event(event="candidate embedings  + score ",  t0=t_candidate_embedings)
        scores = torch.cat(scores_parts, dim=0)
        return scores

    def reshape_candidates_for_encoder(self, candidates):
        """
        reshape candidates for encoder input shape
        [batch_size, topk, max_length] => [batch_size*topk, max_length]
        """
        _, _, max_length = candidates.shape
        candidates = candidates.contiguous().view(-1, max_length)
        return candidates

    def get_loss(self, outputs, targets):
        if self.use_cuda:
            targets = targets.cuda(non_blocking=True)
        loss = self.criterion(outputs, targets)
        return loss

    def get_embeddings(self, mentions, batch_size=1024):
        """
        Compute all embeddings from mention tokens.
        """
        embedding_table = []
        with torch.no_grad():
            for start in tqdm(range(0, len(mentions), batch_size)):
                end = min(start + batch_size, len(mentions))
                batch = mentions[start:end]
                batch_embedding = self.vectorizer(batch)
                batch_embedding = batch_embedding.cpu()
                embedding_table.append(batch_embedding)
        embedding_table = torch.cat(embedding_table, dim=0)
        return embedding_table

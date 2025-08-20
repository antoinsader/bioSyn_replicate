import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from tqdm import tqdm
from utils import marginal_nll 
import numpy as np

class RerankNet(nn.Module):
    def __init__(self, encoder, lr, weight_decay, use_cuda):
        super(RerankNet, self).__init__()
        self.encoder = encoder
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters()}], 
            lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.criterion = marginal_nll
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
        self.to(self.device)
        #hidden size of encoder
        self.H = getattr(getattr(self.encoder, "config", None), "hidden_size", None)



    def forward(self, x):
        # query_token, candidate_tokens = x
        query_token, candidate_tokens = x
        batch_size, topk, max_length = candidate_tokens['input_ids'].shape

        if self.use_cuda:
            query_token['input_ids'] = query_token['input_ids'].to(self.device, non_blocking=True)
            query_token['attention_mask'] = query_token['attention_mask'].to(self.device, non_blocking=True)

            candidate_tokens['input_ids'] = candidate_tokens['input_ids'].to(self.device, non_blocking=True)
            candidate_tokens['attention_mask'] = candidate_tokens['attention_mask'].to(self.device, non_blocking=True)


        # dense embed for query and candidates
        query_embed = self.encoder(**query_token, return_dict=False)
        assert query_embed.shape == (batch_size, max_length, self.H)
        query_embed = query_embed.last_hidden_state[:,0, :].unsqueeze(1).contiguous() #(B, 1, H) [cls]



        candidate_embeds = self.encoder(
            input_ids=candidate_tokens['input_ids'].reshape(-1, max_length),
            attention_mask=candidate_tokens['attention_mask'].reshape(-1, max_length),
            return_dict=False
        )[0] #shape B L H
        assert candidate_embeds.shape == (batch_size, max_length, self.H)
        candidate_embeds = candidate_embeds[0][:,0].reshape(batch_size, topk, -1).contiguous() # [batch_size, topk, hidden]

        score = torch.bmm(query_embed, candidate_embeds.permute(0,2,1)).squeeze(1)
        return score

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

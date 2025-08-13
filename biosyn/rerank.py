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
        #sit on the same device of model
        self.register_buffer("dict_embs", torch.empty(0), persistent=False)
        self.to(self.device)

    @torch.no_grad()
    def set_dictionary_embedings(self, dict_embs):
        # self.dict_embs = dict_embs
        if isinstance(dict_embs, np.ndarray):
            t = torch.from_numpy(dict_embs)
        elif torch.is_tensor(dict_embs):
            t = dict_embs
        t= t.contiguous()
        t= t.to(self.device, non_blocking=True)
        self.dict_embs.resize_(t.shape).copy_(t, non_blocking=True)




    def forward(self, x):
        # query_token, candidate_tokens = x
        query_token, topk_cand_idxs = x
        # batch_size, topk, max_length = candidate_tokens['input_ids'].shape

        if self.use_cuda:
            query_token['input_ids'] = query_token['input_ids'].to(self.device, non_blocking=True)
            query_token['attention_mask'] = query_token['attention_mask'].to(self.device, non_blocking=True)

            # candidate_tokens['input_ids'] = candidate_tokens['input_ids'].to('cuda')
            # candidate_tokens['attention_mask'] = candidate_tokens['attention_mask'].to('cuda')



        # dense embed for query and candidates
        query_embed = self.encoder(**query_token, return_dict=True)
        query_embed = query_embed.last_hidden_state[:,0, :].unsqueeze(1) #(B, 1, H)


        # candidate_embeds = self.encoder(
        #     input_ids=candidate_tokens['input_ids'].reshape(-1, max_length),
        #     attention_mask=candidate_tokens['attention_mask'].reshape(-1, max_length)
        # )
        # candidate_embeds = candidate_embeds[0][:,0].reshape(batch_size, topk, -1) # [batch_size, topk, hidden]


        # candidate_embeds = self.dict_embs[topk_cand_idxs]
        topk_cand_idxs = torch.as_tensor(topk_cand_idxs, dtype=torch.long, device=self.device)
        B, K  = topk_cand_idxs.shape

        # handle padded -1 
        valid_mask = topk_cand_idxs >= 0
        idxs = topk_cand_idxs.clamp_min(0)


        flat = idxs.reshape(-1)
        candidate_embs = self.dict_embs.index_select(0, flat).reshape(B, K, -1)

        if not valid_mask.all():
            candidate_embs = candidate_embs.masked_fill(~valid_mask.unsqueeze(-1), 0)

        candidate_embs = candidate_embs.contiguous()
        query_embed = query_embed.contiguous()

        score = torch.bmm(query_embed, candidate_embs.permute(0,2,1)).squeeze(1)
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

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
    def __init__(self, encoder, lr, weight_decay, use_cuda, forward_chunk_size):
        super(RerankNet, self).__init__()
        self.encoder = encoder
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.optimizer = optim.AdamW(
            self.encoder.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            fused=self.use_cuda
        )

        self.criterion = marginal_nll
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
        self.to(self.device)
        #hidden size of encoder
        self.H = getattr(getattr(self.encoder, "config", None), "hidden_size", None)

        self.forward_chunk_size = forward_chunk_size 
        self.metrics = None
        # if self.use_cuda:
        #     try:
        #         self.encoder = torch.compile(self.encoder, mode="reduce-overhead", dynamic=True)
        #         # self.encoder = torch.compile(self.encoder, mode="max-autotune")
        #     except Exception:
        #         print(f"torch.compile for encoder is not available")
        #         pass
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

        self.metrics.log_event(event="query_embedings",  t0=t_q_embedings, log_immediate=False)


        t_candidate_embedings= time.time()
        cand_tokens_inp = candidate_tokens["input_ids"] #batch_size, topk, max_length 
        cand_tokens_att = candidate_tokens["attention_mask"]
        scores= torch.empty((batch_size, topk))
        chunk_size= self.forward_chunk_size
        scores_parts = []

        # self.metrics.show_gpu_memory("Memory before embeding candidates")
        for start in range(0, batch_size, chunk_size ):
            end = min(start + chunk_size, batch_size)
            chunk_inp = cand_tokens_inp[start:end, :, :].reshape(-1, max_length).to(self.device, non_blocking=True) # chunk_size * topk, L
            chunk_att = cand_tokens_att[start:end, : , :].reshape(-1, max_length).to(self.device, non_blocking=True)
            # self.metrics.show_gpu_memory(f"Memory before self.encoder at start: {start}")
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
        self.metrics.log_event(event="candidate embedings  + score ",  t0=t_candidate_embedings, log_immediate=False)
        scores = torch.cat(scores_parts, dim=0)
        return scores

    def forward_chunk_loss(self, candidate_tokens, targets_chunk, start, end, query_embeddings_chunk):
        """
            The difference for forward is that I am 
            calculating the error immediately for every chunk 
            with throwing the activations between chunks and not save them 
            This will save us from blowing OOM
                
            
            No OOM BUT SLOW, I am trying now have the query_embeddings embedded before (outside the chunks so lesss loops)
                #query_token["input_ids"] shape: batch_size, max_length
                #query_embeddings_chunk (chunk_size, 1, hidden_size)
                #candidate_tokens["input_ids"]  shape: batch_size, TOPK, max_length
                #each of those have input_ids, attention_mask
                    we try to be carefull about keeping loss in fp32 to not show 0.0000

        """

        chunk_size, topk, max_length = candidate_tokens['input_ids'].shape
        forward_start= time.time()
        
        # B, Topk, max_length => Chunk, topk, max_length => Chunk * topk, max_length
        candidate_inputs = candidate_tokens["input_ids"][start:end].reshape(-1, max_length).to(self.device, non_blocking=True)
        candidate_attention = candidate_tokens["attention_mask"][start:end].reshape(-1, max_length).to(self.device, non_blocking=True)
         # chunk * topk, L, H => chunk * topk, H => chunk, topk, H => Chunk, H, topk
        candidate_embeddings = self.encoder(
            input_ids=candidate_inputs, 
            attention_mask=candidate_attention, 
            return_dict=False
            )[0][:, 0, :].reshape(end-start, topk, -1).transpose(1, 2).contiguous()


        # If input is a (b,n,m) tensor, mat2 is a (b,m,p) tensor, out will be a (b,n,p) tensor.
        # (Chunk_size, 1, hidden_size) * (Chunk, hidden_size, topk) = c,1,topk ===squeeze-1==> chunk_size, topk)  
        scores = torch.bmm(query_embeddings_chunk, candidate_embeddings).squeeze(1)
        # I can try to make it from outside after
        targets_chunk = targets_chunk.to(self.device, non_blocking=True)

        loss = self.criterion(scores.float(), targets_chunk.float())  # mean over chunk
        self.metrics.log_event("forward chunk_loss pass",  t0=forward_start, log_immediate=False, first_iteration_only=True, only_elapsed_time=True)
        return loss




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
            targets = targets.to(device=self.device, non_blocking=True)
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

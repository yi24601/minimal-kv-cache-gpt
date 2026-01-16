import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedSelfAttention(nn.Module):
    def __init__(self, emb_dim, key_dim, val_dim, dropout=0.2):
        super().__init__()
        self.emb_dim = emb_dim
        self.key_dim = key_dim

        self.query = nn.Linear(emb_dim, key_dim, bias=False)
        self.key = nn.Linear(emb_dim, key_dim, bias=False)
        self.value = nn.Linear(emb_dim, val_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.cached_keys = None
        # I don't need to cache queries
        self.cached_values = None
        self.cached_index = 0
        self.cached_output = None

    def reset_cache(self):
        self.cached_keys = None
        self.cached_values = None
        self.cached_index = 0
        self.cached_output = None

    def forward(self, x: torch.Tensor, use_cache: bool = False):
        # x = [B, L, emb_dim]

        B, L = x.shape[0], x.shape[1]

        # in cache mode, x = [B, dL, emb_dim]
        if use_cache:
            # print("use cache in self attention: cached_index = ",
            #       self.cached_index, "x shape = ", x.shape, "cached output shape = ",  self.cached_output.shape if self.cached_output is not None else "None")
            new_x = x[:, self.cached_index:, :]

            new_queries = self.query(new_x)
            new_keys = self.key(new_x)
            new_values = self.value(new_x)

            keys = new_keys if self.cached_keys is None else torch.concat(
                [self.cached_keys, new_keys], dim=1)
            values = new_values if self.cached_values is None else torch.concat(
                [self.cached_values, new_values], dim=1)

            # att_mat = [B, L - cached_index, L]
            att_mat = torch.matmul(new_queries, keys.transpose(-1, -2)) / \
                torch.sqrt(torch.tensor(self.key_dim, device=x.device))

            if new_x.shape[1] > 1:
                mask = torch.tril(torch.ones(
                    new_x.shape[1], x.shape[1], device=x.device, dtype=torch.bool))
                mask = torch.tril(mask, diagonal=self.cached_index)
                att_mat = att_mat.masked_fill(~mask, float('-inf'))

            att_mat = F.softmax(att_mat, dim=-1)
            new_output = att_mat.matmul(values)
            output = self.dropout(new_output) if self.cached_output is None else torch.concat(
                [self.cached_output, self.dropout(new_output)], dim=1)

            self.cached_keys = keys
            self.cached_values = values
            self.cached_index = x.shape[1]
            self.cached_output = output
        else:
            queries = self.query(x)
            keys = self.key(x)
            values = self.value(x)

            # att_mat = [B,L, L]
            att_mat = torch.matmul(queries, keys.transpose(-1, -2)) / \
                torch.sqrt(torch.tensor(self.key_dim, device=x.device))
            mask = torch.tril(torch.ones(
                L, L, device=x.device, dtype=torch.int))
            att_mat = att_mat.masked_fill(mask == 0, float('-inf'))
            att_mat = F.softmax(att_mat, dim=-1)
            output = att_mat.matmul(values)
            output = self.dropout(output)
            # print('att_matrix shape:', att_mat.shape)
            # print('att_matrix:', att_mat[0])

        return output


class MLP(nn.Module):
    def __init__(self, emb_dim, dropout=0.2):
        super().__init__()
        self.emb_dim = emb_dim

        self.net = nn.Sequential(nn.Linear(emb_dim, 4*emb_dim), nn.ReLU(),
                                 nn.Linear(4*emb_dim, 4*emb_dim), nn.ReLU(),
                                 nn.Linear(4*emb_dim, emb_dim))

        self.dropout = nn.Dropout(dropout)
        self.cached_index = 0
        self.cached_values = None

    def reset_cache(self):
        self.cached_index = 0
        self.cached_values = None

    def forward(self, x: torch.Tensor, use_cache: bool = False):
        # x = [B, L, C]
        if use_cache:
            # we know that output [B, :cached_index, C] is already in cached_values
            # print("use cache in mlp: cached_index = ",
            #       self.cached_index, "x shape = ", x.shape, "cached_values shape = ", self.cached_values.shape if self.cached_values is not None else "None")
            new_output = self.dropout(self.net(x[:, self.cached_index:,]))
            if self.cached_values is None:
                output = new_output
            else:
                output = torch.concat([self.cached_values, new_output], dim=1)
            self.cached_index = x.shape[1]
            self.cached_values = output
        else:
            output = self.dropout(self.net(x))

        # print('output shape:', output.shape)
        return output


class MultiHead(nn.Module):
    def __init__(self, n_head, emb_dim, key_dim, val_dim, dropout=0.2):
        super().__init__()

        self.heads = nn.ModuleList()

        self.final_head = nn.Linear(n_head*val_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

        for _ in range(n_head):
            self.heads.append(MaskedSelfAttention(emb_dim, key_dim, val_dim))

        self.cached_index = 0
        # [B, T, emb_dim]
        self.cached_values = None

    def reset_cache(self):
        self.cached_index = 0
        self.cached_values = None
        for this_head in self.heads:
            this_head.reset_cache()

    def forward(self, x: torch.Tensor, use_cache: bool = False):
        # x = [B, T, C]
        output = []
        for this_head in self.heads:
            # output = [n_head,B, T, val_dim]
            output.append(this_head(x, use_cache=use_cache))

        output = torch.concat(output, dim=-1)
        # output = [B, T, val_dim*n_head]

        if use_cache:
            # we know that [B, :cached_index, val_dim*n_head] is already in cached_values
            # print("use cache in multihead: cached_index = ",
            #       self.cached_index, "output shape = ", output.shape, "cached_values shape = ", self.cached_values.shape if self.cached_values is not None else "None")
            new_output = self.dropout(self.final_head(
                output[:, self.cached_index:, :]))
            if self.cached_values is None:
                self.cached_values = new_output
            else:
                self.cached_values = torch.concat(
                    [self.cached_values, new_output], dim=1)
            self.cached_index = x.shape[1]
            output = self.cached_values
        else:
            output = self.dropout(self.final_head(output))
        # print("dimension in multihead: ", output.shape)
        # output = [B, T, val_dim]
        return output


class Block(nn.Module):
    def __init__(self, n_head, emb_dim, key_dim, val_dim, dropout=0.2):
        super().__init__()

        self.norm1 = nn.LayerNorm(emb_dim)
        self.multihead = MultiHead(n_head, emb_dim, key_dim, val_dim, dropout)

        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = MLP(emb_dim)

    def reset_cache(self):
        self.multihead.reset_cache()
        self.mlp.reset_cache()

    def forward(self, x: torch.Tensor, use_cache: bool = False):
        # x = [B, T, emb_dim]
        output = x + self.multihead(self.norm1(x), use_cache=use_cache)
        # print('output dim1: ', output.shape)
        output = output + self.mlp(self.norm2(output), use_cache=use_cache)
        # print('output dim2: ', output.shape)
        return output


class ToyGPT(nn.Module):

    def __init__(self, n_vocab, n_seq, n_block, n_head, emb_dim, key_dim, val_dim, dropout=0.2):
        super().__init__()

        self.tok_emb = nn.Embedding(n_vocab, emb_dim)
        self.pos_emb = nn.Embedding(n_seq, emb_dim)

        self.n_vocab = n_vocab
        self.n_seq = n_seq
        self.n_block = n_block
        self.n_head = n_head
        self.emb_dim = emb_dim
        self.key_dim = key_dim
        self.val_dim = val_dim

        self.all_blocks = nn.ModuleList()
        for _ in range(n_block):
            self.all_blocks.append(
                Block(n_head, emb_dim, key_dim, val_dim, dropout))

        self.final_linear = nn.Linear(emb_dim, n_vocab)

    def reset_cache(self):
        for block in self.all_blocks:
            block.reset_cache()

    def forward(self, x: torch.Tensor, use_cache: bool = False):
        # x = [B,T], x is word index

        B, T = x.shape
        x_emb = self.tok_emb(x)
        # x_emb = [B, T, emb_dim]
        pos_emb = self.pos_emb(torch.arange(T, device=x.device))
        pos_emb = torch.unsqueeze(pos_emb, dim=0)
        # pos_emb = [1, T, emb_dim]

        # print('pos_emb shape:', pos_emb.shape)

        x_emb += pos_emb
        x_repr = x_emb
        for block in self.all_blocks:
            x_repr = block(x_repr, use_cache=use_cache)

        return x_repr

    def loss(self, x, y):

        # x,y= [B, T], x is word index
        B, T = x.shape

        x_repr = self.forward(x)
        # x_repr = [B, T, emb_dim]

        x_pred = self.final_linear(x_repr)
        # x_pred = [B, T,n_vocab]

        # print('x_pred shape:', x_pred.shape)
        # print(x_pred[0,0,:])

        x_pred = x_pred.view(B*T, -1)
        y = y.view(B*T)
        # x_pred = [B*T,n_vocab]
        # y = [B*T]
        return F.cross_entropy(x_pred, y, reduction='sum')

    def generate(self, x: torch.Tensor):
        # x = [B,T], x is word index, current context

        B, T = x.shape
        # x_repr = [B, T, emb_dim]
        x_repr = self.forward(x)
        # x_repr = [B, T, n_vocab]
        logits = self.final_linear(x_repr)

        # x_repr = [B, n_vocab]
        last_probs = F.softmax(logits[:, -1, :], dim=-1)

        # print("last logits shape: ", last_probs.shape)
        # print("last logits:", last_probs)

        # next_word = [B,1]
        next_word = torch.multinomial(
            last_probs, num_samples=1, replacement=True)

        return torch.concat([x, next_word], dim=-1)

    def generate_kv(self, x: torch.Tensor, max_new_tokens: int):
        # x = [B,T], x is word index, current context

        # Reset cache at the start to ensure clean state
        self.reset_cache()

        for i in range(max_new_tokens):
            # x_repr = [B, T, emb_dim]
            x_repr = self.forward(x, use_cache=True)

            logits = self.final_linear(x_repr[:, -1, :])
            # logits = [B, n_vocab]

            last_probs = F.softmax(logits, dim=-1)

            # print("last logits shape: ", last_probs.shape)
            # print("last logits:", last_probs)
            # print("iteration ", i, "last logits shape: ", last_probs.shape)

            # next_word = [B,1]
            next_word = torch.multinomial(
                last_probs, num_samples=1, replacement=True)
            x = torch.concat([x, next_word], dim=-1)

        return x

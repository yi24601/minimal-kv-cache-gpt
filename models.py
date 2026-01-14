
class MaskedSelfAttention(nn.Module):
    def __init__(self,emb_dim,key_dim,val_dim,dropout=0.2):
        super().__init__()
        self.emb_dim = emb_dim
        self.key_dim = key_dim

        self.query = nn.Linear(emb_dim,key_dim,bias=False)
        self.key = nn.Linear(emb_dim,key_dim,bias=False)
        self.value = nn.Linear(emb_dim,val_dim,bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        # x = [B, L, emb_dim]
        B,L = x.shape[0],x.shape[1]
        queries = self.query(x)
        # queries = [B, L, key_dim]
        keys = self.key(x)

        values = self.value(x)
        # att_mat = [B,L, L]
        att_mat = torch.matmul(queries, keys.transpose(-1,-2))/torch.sqrt(torch.tensor(self.key_dim,device=x.device))
        mask = torch.tril(torch.ones(L,L,device=x.device,dtype=torch.int))
        att_mat = att_mat.masked_fill(mask == 0,float('-inf'))
        att_mat = F.softmax(att_mat, dim=-1)
        # print('att_matrix shape:', att_mat.shape)
        # print('att_matrix:', att_mat[0])

        output = self.dropout(att_mat.matmul(values))
        return output

class MLP(nn.Module):
    def __init__(self,emb_dim, dropout = 0.2):
        super().__init__()
        self.emb_dim = emb_dim

        self.net = nn.Sequential(nn.Linear(emb_dim,4*emb_dim), nn.ReLU(),
                                 nn.Linear(4*emb_dim,4*emb_dim), nn.ReLU(),
                                 nn.Linear(4*emb_dim,emb_dim), nn.Dropout(dropout))
    def forward(self,x):
        # x = [B, L, C]
        output = self.net(x)

        # print('output shape:', output.shape)
        return output


class MultiHead(nn.Module):
    def __init__(self, n_head, emb_dim, key_dim, val_dim, dropout=0.2):
        super().__init__()

        self.heads = nn.ModuleList()

        self.final_head = nn.Linear(n_head*val_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)


        for _ in range(n_head):
            self.heads.append(MaskedSelfAttention(emb_dim,key_dim,val_dim))

    def forward(self,x):
        # x = [B, T, C]
        output = []
        for this_head in self.heads:
            #output = [n_head,B, T, val_dim]
            output.append(this_head(x))

        output = torch.concat(output,dim=-1)
        # print("dimension in multihead: ", output.shape)
        # output = [B, T, val_dim*n_head]    
        output = self.dropout(self.final_head(output))
        # output = [B, T, val_dim]
        return output


class Block(nn.Module):
    def __init__(self, n_head, emb_dim, key_dim, val_dim, dropout=0.2):
        super().__init__()

        self.norm1 = nn.LayerNorm(emb_dim)
        self.multihead = MultiHead(n_head, emb_dim, key_dim, val_dim, dropout)

        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = MLP(emb_dim)
        

    def forward(self,x):
        # x = [B, T, emb_dim]
        output = x + self.multihead(self.norm1(x))
        # print('output dim1: ', output.shape)
        output = output + self.mlp(self.norm2(x))
        # print('output dim2: ', output.shape)
        return output


class ToyGPT(nn.Module):

    def __init__(self, n_vocab, n_seq, n_block, n_head, emb_dim, key_dim, val_dim, dropout=0.2):
        super().__init__()

        self.tok_emb = nn.Embedding(n_vocab,emb_dim)
        self.pos_emb = nn.Embedding(n_seq,emb_dim)
        
        self.n_vocab = n_vocab
        self.n_seq = n_seq
        self.n_block = n_block
        self.n_head = n_head
        self.emb_dim = emb_dim
        self.key_dim = key_dim
        self.val_dim = val_dim

        blocks = []
        for _ in range(n_block):
            blocks.append(Block(n_head, emb_dim, key_dim, val_dim, dropout))

        self.all_blocks = nn.Sequential(*blocks)
        self.final_linear = nn.Linear(emb_dim,n_vocab)        

    def forward(self,x):
        # x = [B,T], x is word index

        B, T = x.shape
        x_emb = self.tok_emb(x)
        # x_emb = [B, T, emb_dim]
        pos_emb = self.pos_emb(torch.arange(T,device=x.device))
        pos_emb = torch.unsqueeze(pos_emb, dim=0)
        # pos_emb = [1, T, emb_dim]

        # print('pos_emb shape:', pos_emb.shape)

        x_emb += pos_emb
        x_repr = self.all_blocks(x_emb)

        return x_repr

    def loss(self, x, y):

        # x,y= [B, T], x is word index
        B, T = x.shape
        ## question ? how to embedding, trainable embedding?
        ## how cross_entropy works?

        x_repr = self.forward(x)
        # x_repr = [B, T, emb_dim]

        x_pred = self.final_linear(x_repr)
        # x_pred = [B, T,n_vocab]

        # print('x_pred shape:', x_pred.shape)
        # print(x_pred[0,0,:])

        x_pred = x_pred.view(B*T,-1)
        y = y.view(B*T)
        # x_pred = [B*T,n_vocab]        
        # y = [B*T]
        return F.cross_entropy(x_pred,y,reduction='sum')
    
    def generate(self, x):
        # x = [B,T], x is word index, current context

        B, T = x.shape
        # x_repr = [B, T, emb_dim]
        x_repr = self.forward(x)
        # x_repr = [B, T, n_vocab]
        logits = self.final_linear(x_repr)

        # x_repr = [B, n_vocab]
        last_probs = F.softmax(logits[:,-1,:],dim=-1)

        print("last logits shape: ", last_probs.shape)
        print("last logits:", last_probs)

        # next_word = [B,1]
        next_word = torch.multinomial(last_probs, num_samples=1,replacement=True)

        return torch.concat([x,next_word],dim=-1)
    


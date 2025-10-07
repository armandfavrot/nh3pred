import torch
import torch.nn as nn

class AmmoniaRNN(nn.Module):
    
    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_size, 
                 num_layers,
                 nonlinearity, 
                 bidirectional,
                 cat_dims = None, embedding_dims = None):
        
        super(AmmoniaRNN, self).__init__()
        
        D = 1 + 1 * bidirectional
               
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings = cat_dim, embedding_dim = embed_dim)
            for cat_dim, embed_dim in zip(cat_dims, embedding_dims)
        ])
        
        input_size = input_size - len(cat_dims) + sum(embedding_dims)           
        
        self.rnn = nn.RNN(input_size, 
                          hidden_size, 
                          num_layers = num_layers,
                          nonlinearity = nonlinearity, 
                          bidirectional = bidirectional)    
        
        self.fc1 = nn.Linear(hidden_size * D, 6)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(6, output_size)

    def forward(self, x):

        x_continuous = x[0]
        x_categoricals = x[1]
        
        x_embeds = [embed(x_cat) for embed, x_cat in zip(self.embeddings, x_categoricals)]
        
        x = torch.cat([x_continuous] + x_embeds, dim = -1)
        
        h, _ = self.rnn(x)

        out = self.fc1(h)
        out = self.relu(out)
        out = self.fc2(out)
       
        return out


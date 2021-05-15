import torch
import torch.nn as nn


class SubgroupClassifier(nn.Module):
    def __init__(self, embedding_dim=300, num_classes=5, hidden_dim_1=200, hidden_dim_2=100, dropout=0.1):
        super(SubgroupClassifier, self).__init__()
        self.linear_1 = nn.Linear(embedding_dim, hidden_dim_1)
        self.linear_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.linear_3 = nn.Linear(hidden_dim_2, num_classes)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, input):
        hidden = self.dropout(self.activation(self.linear_1(input)))
        hidden = self.dropout(self.activation(self.linear_2(hidden)))
        return self.activation(self.linear_3(hidden))


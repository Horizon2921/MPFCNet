import torch
from timm.models import DropPath
from torch import nn
from torchsummary import summary


class LinearBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act=nn.GELU,
                 norm=nn.LayerNorm, n_tokens=197):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.mlp1 = Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio), act=act, drop=drop)
        self.norm1 = norm(dim)

        self.mlp2 = Mlp(in_features=n_tokens, hidden_features=int(n_tokens*mlp_ratio), act=act, drop=drop)
        self.norm2 = norm(dim)

    def forward(self, x):
        x = x + self.drop_path(self.mlp1(self.norm1))
        x = x.transpose(-2, -1)
        x = x + self.drop_path(self.mlp2(self.norm2))
        x = x.transpose(-2, -1)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GateLinearUnit(nn.Module):
    def __init__(self, embedding_size, num_filers, kernel_size, vocab_size, bias=True, batch_norm=True, activation=nn.Tanh()):
        super(GateLinearUnit, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.conv_layer1 = nn.Conv2d(1, num_filers, (kernel_size, embedding_size), bias=bias)
        self.conv_layer2 = nn.Conv2d(1, num_filers, (kernel_size, embedding_size), bias=bias)
        self.batch_norm = nn.BatchNorm2d(num_filers)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_uniform_(self.conv_layer1.weight)
        nn.init.kaiming_uniform_(self.conv_layer2.weight)

    def gate(self, inputs):
        """门控机制"""
        return self.sigmoid(inputs)

    def forward(self, inputs):
        embed = self.embedding(inputs)
        embed = embed.unsqueeze(1)
        output = self.conv_layer1(embed)
        gate_output = self.conv_layer2(embed)
        # Gate Operation
        if self.activation is not None:
            # GTU
            output = self.activation(output) * self.gate(gate_output)
        else:
            # GLU
            output = output * self.gate(gate_output)
        if self.batch_norm:
            output = self.batch_norm(output)
            output = output.squeeze()
            return output
        else:
            return output.squeeze()

if __name__=="__main__":
    x = torch.randint(1,100,[32, 128])
    glu = GateLinearUnit(embedding_size=300, num_filers=256, kernel_size=3, vocab_size=1000)
    out = glu(x)
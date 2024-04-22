import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module): # Checked
    def __init__(self, dim, num_heads, attn_dropout=0.0):
        super(Attention, self).__init__()
        
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class MLP(nn.Module): # Checked
    def __init__(self, dim, mlp_size, mlp_dropout=0.1):
        super(MLP, self).__init__()
        
        self.layer_norm = nn.LayerNorm(normalized_shape=dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_size),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_size, dim),
            nn.Dropout(mlp_dropout)
        )
        
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
    

class EncoderLayer(nn.Module): # Checked
    def __init__(self, dim, num_heads, mlp_size, attn_dropout, mlp_dropout):
        super(EncoderLayer, self).__init__()
        
        self.msa = Attention(dim=dim, num_heads=num_heads, 
                             attn_dropout=attn_dropout)
        
        self.mlp = MLP(dim=dim, mlp_size=mlp_size,
                       mlp_dropout=mlp_dropout)
        
    def forward(self, x):
        x = self.msa(x) + x
        x = self.mlp(x) + x
        return x
    
# Taken directly from CCT https://github.com/SHI-Labs/Compact-Transformers
class Tokenizer(nn.Module): # Tokenizer can be used to capture spatial information
    def __init__(self,
                 kernel_sizes, stride, padding,
                 pooling_kernel_size, pooling_stride, 
                 pooling_padding, n_conv_layers,
                 n_input_channels, n_output_channels,
                 in_planes, activation,
                 max_pool, conv_bias):
        super(Tokenizer, self).__init__()
    
        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=kernel_sizes[i],
                          stride=stride,
                          padding=padding, bias=conv_bias),
                # nn.Identity() if activation is None else activation(),
                nn.AvgPool2d(kernel_size=pooling_kernel_size,                   # maxpool
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ],
            # nn.AvgPool2d(kernel_size=pooling_kernel_size,
            #              stride=pooling_stride,
            #              padding=pooling_padding),
            )
            
        
        self.flattener = nn.Flatten(2, 3)

    def sequence_length(self, n_channels=1, height=1, width=3000): # MUST be changed
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return self.flattener(self.conv_layers(x)).transpose(-2, -1)

# Adapted from CCT https://github.com/SHI-Labs/Compact-Transformers  
class Transformer(nn.Module):
    def __init__(self, dim, num_layers,
                 num_heads, num_classes, 
                 attn_dropout, dropout,
                 mlp_size, positional_embedding, 
                 sequence_length=None):
        super(Transformer, self).__init__()
        
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        self.dim = dim
        self.sequence_length = sequence_length

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        self.attention_pool = nn.Linear(self.dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, dim),
                                                requires_grad=True)
            else:
                self.positional_emb = nn.Parameter(self.sinusoidal_embedding(sequence_length, dim),
                                                requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            EncoderLayer(dim=dim, num_heads=num_heads, mlp_size=mlp_size, 
                         attn_dropout=attn_dropout, mlp_dropout=dropout)
            for i in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        # self.new_fc = nn.Sequential(
        #     nn.Linear(304, 64),
        #     nn.GELU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(64, num_classes)
        # )
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)
        x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        x = self.fc(x)
        # print(x.shape)
        # x = x.contiguous().view(x.size(0), -1)
        # x = self.new_fc(x)
        return x

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)
        

class CCT(nn.Module):
    def __init__(self, kernel_sizes, stride, padding,
            pooling_kernel_size, pooling_stride, pooling_padding,
            n_conv_layers, n_input_channels,
            in_planes,
            max_pool, conv_bias,
            dim, num_layers,
            num_heads, num_classes, 
            attn_dropout, dropout, 
            mlp_size, positional_emb,
            activation=None):
        super(CCT, self).__init__()
        
        # height = 22 if "2a" else 3
        
        self.tokenizer = Tokenizer(
                 kernel_sizes=kernel_sizes, stride=stride, padding=padding,
                 pooling_kernel_size=pooling_kernel_size, pooling_stride=pooling_stride, pooling_padding=pooling_padding,
                 n_conv_layers=n_conv_layers, n_input_channels=n_input_channels, n_output_channels=dim,
                 in_planes=in_planes, activation=activation,
                 max_pool=max_pool, conv_bias=conv_bias) # avgpool
        
        self.transformer = Transformer(
            dim=dim, num_layers=num_layers,
            num_heads=num_heads, num_classes=num_classes, 
            attn_dropout=attn_dropout, dropout=dropout, 
            mlp_size=mlp_size, positional_embedding=positional_emb, 
            sequence_length=self.tokenizer.sequence_length(
                n_channels=1, height=1, width=3000    # TODO 
            )
        )
        
    def forward(self, x):
        # print("Init input")
        # print(x.shape)
        x = x.unsqueeze(1)
        # print(x.shape)
        x = self.tokenizer(x)
        x = self.transformer(x)
        return x

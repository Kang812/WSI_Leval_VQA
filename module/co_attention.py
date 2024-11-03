import torch.nn as nn

class CoAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CoAttention, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.relu = nn.ReLU()
    
    def forward(self, text_embeds, image_embeds, attention_mask=None):
        image_embeds = image_embeds.permute(1, 0, 2)
        co_attn_output, _ = self.multihead_attention(text_embeds, image_embeds, 
                                                     image_embeds, attn_mask=attention_mask)
        output = self.linear(co_attn_output)
        output = self.relu(output)
        return output

import torch.nn as nn

class BurberryProductModel(nn.Module):
    def __init__(self, model, co_attention):
        super(BurberryProductModel, self).__init__()
        self.model = model
        self.co_attention = co_attention

    def forward(self, input_ids, attention_mask, pixel_values):
        # 텍스트 임베딩 및 이미지 피처 추출
        text_embeds = self.model.embeddings(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        image_embeds = self.model.image_processor(pixel_values=pixel_values)
        
        # Co-Attention
        co_attention_output = self.co_attention(text_embeds, image_embeds, attention_mask)

        # 모델 출력
        return self.model.decoder(co_attention_output, labels=input_ids)

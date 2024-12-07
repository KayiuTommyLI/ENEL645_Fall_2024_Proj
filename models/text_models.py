import torch
import torch.nn as nn
from transformers import DistilBertModel, RobertaModel, XLNetModel, AlbertModel, BertModel
from configs.config import TextModelType

class TextModel(nn.Module):
    def __init__(self, model_name: str = TextModelType.DISTILBERT):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.model = self.load_model()
        
    def load_model(self):
        if self.model is None:
            if 'distilbert' in self.model_name:
                self.model = DistilBertModel.from_pretrained(TextModelType.DISTILBERT)
            elif 'roberta' in self.model_name:
                self.model = RobertaModel.from_pretrained(TextModelType.ROBERTA)
                if self.model.pooler is not None:
                    nn.init.normal_(self.model.pooler.dense.weight, mean=0.0, std=0.02)
                    nn.init.zeros_(self.model.pooler.dense.bias)
            elif 'xlnet' in self.model_name:
                self.model =  XLNetModel.from_pretrained(TextModelType.XLNET)
            elif 'albert' in self.model_name:
                self.model =  AlbertModel.from_pretrained(TextModelType.ALBERT)
            elif 'bert' in self.model_name:
                self.model = BertModel.from_pretrained(TextModelType.BERT)
            else:
                raise ValueError(f"Unsupported model type: {self.model_name}")

            self.model.to(self.device)  # Move to the correct device after initialization
        return self.model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # device = input_ids.device  # Get device from input
        # model = self.load_model().to(device)  # Move model to same device as input
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0]

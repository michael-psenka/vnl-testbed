from .parent import VisionLanguageModel
import torch
from torch import Tensor
import clip

class CLIPModel(VisionLanguageModel):
    def __init__(self):
        super(CLIPModel, self).__init__()

        # Load the model
        self.model, self.transform = clip.load("ViT-B/32", device="cuda")
    
    # text can be either Tensor or List[str]
    def imtext2feature(self, image: Tensor, text) -> Tensor:
        # if text is a list of strings (or just a string), tokenize it
        if isinstance(text, str):
            text = clip.tokenize(text).to("cuda")
        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(text)
        
        # Concatenate features as an example
        combined_features = torch.cat([image_features, text_features], dim=1)
        
        return combined_features

    def im2feature(self, image: Tensor) -> Tensor:
        return self.model.encode_image(image)

    def text2feature(self, text) -> Tensor:
        if isinstance(text, str):
            text = clip.tokenize(text).to("cuda")
        return self.model.encode_text(text)

    def alignment(self, image: Tensor, text: Tensor) -> float:
        if isinstance(text, str):
            text = clip.tokenize(text).to("cuda")
        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(text)
        
        # Compute the similarity (dot product)
        alignment_score = torch.nn.functional.cosine_similarity(image_features, text_features, dim=-1)
        
        return alignment_score.mean().item()
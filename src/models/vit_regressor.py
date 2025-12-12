import torch
from torch import Tensor
import torch.nn as nn
import timm


class VitRegressor(nn.Module):
    """
    Prosty model FR-IQA oparty na Vision Transformerze (ViT).

    Wejście:
        - reference_image: tensor o kształcie [batch_size, 3, 224, 224]
        - distorted_image: tensor o kształcie [batch_size, 3, 224, 224]

    Wyjście:
        - przewidywany DMOS: tensor [batch_size]
    """

    def __init__(self,
                 model_name: str = 'vit_base_patch16_224',
                 embedding_dimension: int = 768) -> None:
        super().__init__()

        # Tworzenie modelu ViT z bibliotekami timm.
        # Ustawienie `num_classes = 0`, żeby dostać embedding bez klasyfikatora
        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=True,
            num_classes=0  # model zwróci wektor cech zamiast klasy
        )

        # Głowa regresji (fully-connected layers)
        self.regressor_head = nn.Sequential(
            nn.Linear(embedding_dimension * 2, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        )


    def forward(self,
                reference_image: Tensor,
                distorted_image: Tensor):
        """
        Forward pass modelu.

        Args:
            reference_image: tensor [batch_size, 3, 224, 224]
            distorted_image: tensor [batch_size, 3, 224, 224]

        Returns:
            predicted_dmos_value: tensor [batch_size]
        """

        # Przepuszczenie obu obrazów przez ten sam backbone ViT
        reference_embedding = self.backbone(reference_image)
        distorted_embedding = self.backbone(distorted_image)

        # Łączenie embeddingów wzdłuż wymiaru cech
        concatenated_embedding = torch.cat(
            tensors=[reference_embedding, distorted_embedding],
            dim=1  # Łączenie po wymiarze cech, nie po batchu
        )  # [batch_size, embedding_dimension * 2]

        # Przepuszczenie przez głowę regresyjną
        predicted_dmos: Tensor = self.regressor_head(concatenated_embedding)  #[batch_size, 1]

        # Usuwanie zbędnego wymiaru na końcu
        predicted_dmos = predicted_dmos.squeeze(dim=1)  # [batch]

        return predicted_dmos

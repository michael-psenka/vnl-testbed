import torch.nn as nn
from torch import Tensor

class VisionLanguageModel:
    def __init__(self):
        super(VisionLanguageModel, self).__init__()

    def imtext2feature(self, image: Tensor, text: Tensor) -> Tensor:
        """
        Extract features from both image and text inputs.

        Args:
            image (torch.Tensor): Input image tensor, [batch_size, C, H, W].
            text (torch.Tensor): Input text tensor, typically [batch_size, seq_len].

        Returns:
            torch.Tensor: Feature tensor, [batch_size, feature_dim].
        """
        raise NotImplementedError("The method 'imtext2feature' needs to be implemented in derived classes!")

    def im2feature(self, image: Tensor) -> Tensor:
        """
        Optionally extract features from the image input.

        Args:
            image (torch.Tensor): Input image tensor, [batch_size, C, H, W].

        Returns:
            torch.Tensor: Feature tensor, [batch_size, feature_dim].
        """
        raise NotImplementedError("The method 'im2feature' is not implemented!")

    def text2feature(self, text: Tensor) -> Tensor:
        """
        Optionally extract features from the text input.

        Args:
            text (torch.Tensor): Input text tensor, typically [batch_size, seq_len].

        Returns:
            torch.Tensor: Feature tensor, [batch_size, feature_dim].
        """
        raise NotImplementedError("The method 'text2feature' is not implemented!")

    def alignment(self, image: Tensor, text: Tensor) -> float:
        """
        Optionally compute the alignment between image and text.

        Args:
            image (torch.Tensor): Input image tensor, [batch_size, C, H, W].
            text (torch.Tensor): Input text tensor, typically [batch_size, seq_len].

        Returns:
            float: Alignment score.
        """
        raise NotImplementedError("The method 'alignment' is not implemented!")

    def im2text(self, image: Tensor) -> Tensor:
        """
        Optionally convert image to text representation.

        Args:
            image (torch.Tensor): Input image tensor, [batch_size, C, H, W].

        Returns:
            torch.Tensor: Text tensor, [batch_size, seq_len].
        """
        raise NotImplementedError("The method 'im2text' is not implemented!")

    def text2im(self, text: Tensor) -> Tensor:
        """
        Optionally convert text to image representation.

        Args:
            text (torch.Tensor): Input text tensor, typically [batch_size, seq_len].

        Returns:
            torch.Tensor: Image tensor, [batch_size, C, H, W].
        """
        raise NotImplementedError("The method 'text2im' is not implemented!")

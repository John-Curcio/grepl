import torch
from torch import nn

class LSTMOnDinoOutputs(nn.Module):
    """ A simple LSTM model that takes the final features from a DINOv2 model
    and processes them.
    """
    
    def __init__(self, feature_model, num_classes, hidden_dim=512, feature_dim=384):
        """ Initializes the LSTM model with a DINOv2 feature extractor."""
        super().__init__()
        self.feature_model = feature_model
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        
        self.feature_model.eval()
        
        self._lstm = nn.LSTM(
            input_size=feature_dim,  # 384 for vits14
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True # Using a single directional LSTM (I know the clipping is horrible)
        )
        self._classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=num_classes)
        )

    def forward(self, images):
        """ Forward pass through the DINOv2 model and then through the LSTM. Returns logits for classification.
        """
        batch_size, n_frames_per_video, channels, height, width = images.shape
        images_flattened = images.flatten(0, 1)  # flatten batch and segments
        with torch.no_grad():
            # [batch_size * n_frames_per_video, 384]
            dino_feats = self.feature_model.forward_features(images_flattened)["x_norm_clstoken"]
        # Reshape to [batch_size, n_frames_per_video, 384]
        dino_feats_seq = dino_feats.unflatten(0, (batch_size, n_frames_per_video))
        dino_feats_seq = self.feature_model.norm(dino_feats_seq) # TODO is this necessary???
        lstm_out, _ = self._lstm(dino_feats_seq)
        # Take the last output of the LSTM for each sequence
        lstm_out = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        logits = self._classifier(lstm_out)  # [batch_size, num_classes]
        return logits
    

class ModelWithIntermediateLayers(nn.Module):
    """Taken from https://github.com/facebookresearch/dinov2/blob/592541c8d842042bb5ab29a49433f73b544522d5/dinov2/eval/utils.py#L30C1-L44C24"""
    def __init__(self, feature_model, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images, self.n_last_blocks, return_class_token=True
                )
        return features
    
class LSTMOnDinoOutputsWithIntermediateLayers(LSTMOnDinoOutputs):
    """ A simple LSTM model that takes the final features from a DINOv2 model
    and processes them, with the ability to use intermediate layers.
    """
    
    def __init__(self, feature_model, num_classes, hidden_dim=512, feature_dim=384, n_last_blocks=1, autocast_ctx=torch.autocast):
        """ Initializes the LSTM model with a DINOv2 feature extractor and intermediate layers."""
        super().__init__(feature_model, num_classes, hidden_dim, feature_dim)
        self.intermediate_model = ModelWithIntermediateLayers(feature_model, n_last_blocks, autocast_ctx)
    
    # def forward(self, images):
    #     features = self.intermediate_model(images)
    def forward(self, images):
        """ Forward pass through the DINOv2 model (grabbing intermediate layers) and then through the LSTM. Returns logits for classification.
        """
        batch_size, n_frames_per_video, channels, height, width = images.shape
        images_flattened = images.flatten(0, 1)  # flatten batch and segments
        with torch.no_grad():
            # [batch_size * n_frames_per_video, 384]
            # dino_feats = self.feature_model.forward_features(images_flattened)["x_norm_clstoken"]
            dino_feats = self.intermediate_model(images_flattened)
        # Reshape to [batch_size, n_frames_per_video, 384]
        dino_feats_seq = dino_feats.unflatten(0, (batch_size, n_frames_per_video))
        dino_feats_seq = self.feature_model.norm(dino_feats_seq) # TODO is this necessary???
        lstm_out, _ = self._lstm(dino_feats_seq)
        # Take the last output of the LSTM for each sequence
        lstm_out = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        logits = self._classifier(lstm_out)  # [batch_size, num_classes]
        return logits
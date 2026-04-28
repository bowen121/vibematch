"""MLP genre classifier for live genre tagging."""
import torch
import torch.nn as nn

class GenreClassifier(nn.Module):
    def __init__(self, input_dim=256, num_genres=54):
        super(GenreClassifier, self).__init__()
        # Maps the mathematical formulas to PyTorch layers
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_genres)
        )
    
    def forward(self, x):
        logits = self.mlp(x)
        return logits # Returns [ BatchSize , K ]


def predict_genres(embedding, model, genre_vocab, threshold=0.5):
    """Predict genre tags for a single frozen embedding.

    This is the inference function Member D calls from the Streamlit app
    to display auto-generated genre tags below each search result.

    Args:
        embedding:   (256,) tensor — one frozen CLIP image embedding.
        model:       Trained GenreClassifier instance.
        genre_vocab: list[str] — ordered genre names from Member A's DataBundle.
        threshold:   Probability cutoff for tagging (default 0.5).

    Returns:
        List of genre name strings whose predicted probability >= threshold.
    """
    model.eval()
    with torch.no_grad():
        logits = model(embedding.unsqueeze(0))     # (1, K)
        probs = torch.sigmoid(logits).squeeze(0)   # (K,)
    return [genre_vocab[i] for i, p in enumerate(probs) if p >= threshold]
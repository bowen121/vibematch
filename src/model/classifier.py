"""MLP genre classifier for live genre tagging."""
import torch
import torch.nn as nn

class GenreClassifier(nn.Module):
    def __init__(self, input_dim=256, num_genres=54, hidden_dims=(512, 256), dropout=0.3):
        super(GenreClassifier, self).__init__()
        dims = [input_dim, *hidden_dims]
        layers = []
        for in_d, out_d in zip(dims, dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.ReLU(), nn.Dropout(p=dropout)]
        layers.append(nn.Linear(dims[-1], num_genres))
        self.mlp = nn.Sequential(*layers)
    
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


def predict_genres_with_scores(embedding, model, genre_vocab, top_k=3):
    """Return the top-k genres with percentage scores for a single embedding.

    Used for the live query-level genre readout, which works on both text and
    image embeddings — demonstrating that the shared projection space aligns
    the two modalities.

    Returns:
        List of (genre_name, pct_int) tuples sorted by score descending,
        e.g. [("Thriller", 82), ("Crime", 67), ("Drama", 51)].
    """
    model.eval()
    with torch.no_grad():
        logits = model(embedding.unsqueeze(0))
        probs = torch.sigmoid(logits).squeeze(0)
    top_indices = probs.topk(top_k).indices.tolist()
    return [(genre_vocab[i], int(probs[i].item() * 100)) for i in top_indices]
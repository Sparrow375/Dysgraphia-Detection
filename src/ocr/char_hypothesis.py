"""
Character Hypothesis Engine — CRNN Model + CTC Beam Search.

Implements a Convolutional Recurrent Neural Network (CNN + BiLSTM + CTC) for
word-level handwriting recognition. Includes standard CTC beam search and a
stroke-prior-augmented variant that incorporates topological constraints.

Architecture:
    Input: (B, 1, 64, W) normalized word images
    → CNN backbone (ResNet-18 style, 7 conv blocks)
    → Squeeze spatial height
    → Bidirectional LSTM (2 layers, hidden=256)
    → Linear projection to |alphabet| + 1 (CTC blank)
    → CTC beam search decoder → top-K word hypotheses
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.ocr.utils import CharHypothesis, WordHypothesis


# ---------------------------------------------------------------------------
# Default Alphabet
# ---------------------------------------------------------------------------

# Lowercase + uppercase + digits + common punctuation + space (matching trained CRNN)
DEFAULT_ALPHABET = (
    " abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".,;:!?'\"-()/"
)


# ---------------------------------------------------------------------------
# CNN Backbone & CRNN Model
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class ConvBNReLU(nn.Module):
        def __init__(
            self,
            in_c: int,
            out_c: int,
            k: int = 3,
            s: int = 1,
            p: int = 1,
            pool: Optional[Tuple[int, int]] = None,
        ):
            super().__init__()
            layers: List[nn.Module] = [
                nn.Conv2d(in_c, out_c, k, s, p, bias=False),
                nn.BatchNorm2d(out_c),
                nn.GELU(),
            ]
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=pool, stride=pool))
            self.block = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)


    class ResBlock(nn.Module):
        """Residual block for the CNN backbone."""

        def __init__(self, channels: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.act = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.act(x + self.net(x))


    class CRNNModel(nn.Module):
        """
        Convolutional Recurrent Neural Network for handwriting recognition.
        Deep Residual CNN + 3-layer BiLSTM + CTC projection head (matching trained weights).

        Input shape:  (B, 1, 64, W) — single-channel, height-normalized
        Output shape: (T, B, |alphabet|+1) — CTC logits per timestep
        """

        def __init__(
            self,
            n_classes: int = len(DEFAULT_ALPHABET) + 1,
            input_height: int = 64,
            lstm_hidden: int = 512,
            lstm_layers: int = 3,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.n_classes = n_classes
            self.input_height = input_height

            # Deep CNN backbone with Residual blocks
            self.cnn = nn.Sequential(
                ConvBNReLU(1, 64, pool=(2, 2)),            # → (B, 64, 32, W/2)
                ResBlock(64),
                ConvBNReLU(64, 128, pool=(2, 2)),          # → (B, 128, 16, W/4)
                ResBlock(128),
                ConvBNReLU(128, 256),                      # → (B, 256, 16, W/4)
                ResBlock(256),
                ConvBNReLU(256, 256, pool=(2, 1)),         # → (B, 256, 8, W/4)
                ConvBNReLU(256, 512),                      # → (B, 512, 8, W/4)
                ResBlock(512),
                ConvBNReLU(512, 512, pool=(2, 1)),         # → (B, 512, 4, W/4)
                ConvBNReLU(512, 512, k=2, p=0),            # → (B, 512, 3, W/4-1)
            )

            # Adaptive pooling to squeeze height to 1
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))  # → (B, 512, 1, T)

            # Bidirectional LSTM
            self.rnn = nn.LSTM(
                input_size=512,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                bidirectional=True,
                batch_first=False,
                dropout=dropout if lstm_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(lstm_hidden * 2, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (B, 1, H, W) input images, H should be input_height.

            Returns:
                (T, B, n_classes) log-probabilities for CTC decoding.
            """
            f = self.cnn(x)
            f = self.adaptive_pool(f)
            f = f.squeeze(2)
            f = f.permute(2, 0, 1)

            out, _ = self.rnn(f)
            out = self.dropout(out)
            out = self.fc(out)
            return F.log_softmax(out, dim=2)

        def predict(
            self,
            images: torch.Tensor,
            beam_width: int = 10,
            alphabet: str = DEFAULT_ALPHABET,
        ) -> List[List[Tuple[str, float]]]:
            """
            Run inference and decode with beam search.

            Args:
                images: (B, 1, H, W) batch of normalized word images.
                beam_width: Beam search width.
                alphabet: Character set.

            Returns:
                List of B results, each being a list of (decoded_text, log_prob) tuples.
            """
            self.eval()
            with torch.no_grad():
                logits = self.forward(images)  # (T, B, C)
            logits_np = logits.cpu().numpy()

            results = []
            for b in range(logits_np.shape[1]):
                frame_logits = logits_np[:, b, :]  # (T, C)
                hypotheses = beam_search_ctc(frame_logits, alphabet, beam_width)
                results.append(hypotheses)

            return results

    CRNN = CRNNModel

else:

    class CRNNModel:
        """Fallback CRNNModel placeholder when PyTorch is not installed."""

        def __init__(self, *args, **kwargs):
            pass

        def forward(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required to run the CRNNModel forward pass.")

        def predict(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required to run the CRNNModel predict pass.")


# ---------------------------------------------------------------------------
# CTC Beam Search Decoder
# ---------------------------------------------------------------------------

@dataclass
class _BeamEntry:
    """Internal beam search state."""
    text: str
    log_prob_blank: float    # Log-prob of paths ending in blank
    log_prob_non_blank: float  # Log-prob of paths ending in non-blank
    log_prob_total: float

    @staticmethod
    def log_add(a: float, b: float) -> float:
        """Numerically stable log-add-exp."""
        if a == float('-inf'):
            return b
        if b == float('-inf'):
            return a
        if a > b:
            return a + np.log1p(np.exp(b - a))
        return b + np.log1p(np.exp(a - b))


def beam_search_ctc(
    logits: np.ndarray,
    alphabet: str = DEFAULT_ALPHABET,
    beam_width: int = 10,
    blank_index: int = 0,
) -> List[Tuple[str, float]]:
    """
    Standard CTC prefix beam search decoder.

    Args:
        logits: (T, C) array of log-probabilities per timestep.
                C = |alphabet| + 1, index 0 = CTC blank.
        alphabet: String of characters (index i in alphabet maps to logits index i+1).
        beam_width: Number of beams to maintain.
        blank_index: Index of the CTC blank token in logits.

    Returns:
        List of (decoded_text, log_probability) tuples, sorted by probability (descending).
    """
    T, C = logits.shape
    NEG_INF = float('-inf')

    # Initialize beams
    # Key = decoded text, Value = _BeamEntry
    beams: Dict[str, _BeamEntry] = {
        "": _BeamEntry("", logits[0, blank_index], NEG_INF, logits[0, blank_index])
    }

    # Initialize single-character beams
    for c_idx in range(1, C):
        if c_idx - 1 < len(alphabet):
            char = alphabet[c_idx - 1]
            beams[char] = _BeamEntry(char, NEG_INF, logits[0, c_idx], logits[0, c_idx])

    # Prune to beam_width
    beams = _prune_beams(beams, beam_width)

    # Process remaining timesteps
    for t in range(1, T):
        new_beams: Dict[str, _BeamEntry] = {}

        for text, entry in beams.items():
            # Case 1: Extend with blank
            new_log_prob_blank = _BeamEntry.log_add(
                entry.log_prob_blank + logits[t, blank_index],
                entry.log_prob_non_blank + logits[t, blank_index],
            )
            if text not in new_beams:
                new_beams[text] = _BeamEntry(text, new_log_prob_blank, NEG_INF, NEG_INF)
            else:
                new_beams[text].log_prob_blank = _BeamEntry.log_add(
                    new_beams[text].log_prob_blank, new_log_prob_blank
                )

            # Case 2: Extend with each character
            for c_idx in range(1, C):
                if c_idx - 1 >= len(alphabet):
                    continue
                char = alphabet[c_idx - 1]
                new_text = text + char

                if text and text[-1] == char:
                    # Same character: only extend from blank path (CTC repeat rule)
                    new_log_prob_nb = entry.log_prob_blank + logits[t, c_idx]
                else:
                    # Different character: extend from both paths
                    new_log_prob_nb = _BeamEntry.log_add(
                        entry.log_prob_blank + logits[t, c_idx],
                        entry.log_prob_non_blank + logits[t, c_idx],
                    )

                if new_text not in new_beams:
                    new_beams[new_text] = _BeamEntry(new_text, NEG_INF, new_log_prob_nb, NEG_INF)
                else:
                    new_beams[new_text].log_prob_non_blank = _BeamEntry.log_add(
                        new_beams[new_text].log_prob_non_blank, new_log_prob_nb
                    )

        # Update total log probabilities and prune
        for text, entry in new_beams.items():
            entry.log_prob_total = _BeamEntry.log_add(
                entry.log_prob_blank, entry.log_prob_non_blank
            )

        beams = _prune_beams(new_beams, beam_width)

    # Return sorted results
    results = [(entry.text, entry.log_prob_total) for entry in beams.values()]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def beam_search_with_stroke_prior(
    logits: np.ndarray,
    stroke_bias: np.ndarray,
    alphabet: str = DEFAULT_ALPHABET,
    beam_width: int = 15,
    blank_index: int = 0,
) -> List[Tuple[str, float]]:
    """
    CTC beam search augmented with stroke primitive priors.

    At each timestep, the effective log-probability becomes:
        log P_effective(c|t) = log P_cnn(c|t) + stroke_bias[c]

    This nudges the beam search toward characters that are consistent
    with the detected stroke primitives, which is especially helpful
    when the CNN is uncertain on dysgraphic writing.

    Args:
        logits: (T, C) log-probabilities from CRNN.
        stroke_bias: (C,) log-probability bias from stroke analysis.
        alphabet: Character set.
        beam_width: Beam width.
        blank_index: CTC blank token index.

    Returns:
        List of (decoded_text, log_probability) tuples.
    """
    # Apply stroke bias to logits (don't bias the blank token)
    augmented = logits.copy()
    augmented[:, 1:] += stroke_bias[1:]

    # Re-normalize to valid log-probabilities per timestep
    for t in range(augmented.shape[0]):
        log_sum = np.logaddexp.reduce(augmented[t])
        augmented[t] -= log_sum

    return beam_search_ctc(augmented, alphabet, beam_width, blank_index)


def _prune_beams(beams: Dict[str, _BeamEntry], beam_width: int) -> Dict[str, _BeamEntry]:
    """Keep only the top beam_width entries by total log probability."""
    if len(beams) <= beam_width:
        return beams

    sorted_entries = sorted(
        beams.items(),
        key=lambda x: x[1].log_prob_total,
        reverse=True,
    )
    return dict(sorted_entries[:beam_width])


# ---------------------------------------------------------------------------
# Greedy CTC Decoder (fast fallback)
# ---------------------------------------------------------------------------

def greedy_ctc_decode(
    logits: np.ndarray,
    alphabet: str = DEFAULT_ALPHABET,
    blank_index: int = 0,
) -> Tuple[str, float]:
    """
    Simple greedy (best-path) CTC decoder.

    Faster than beam search but produces only a single hypothesis.

    Args:
        logits: (T, C) log-probabilities.
        alphabet: Character set.
        blank_index: CTC blank index.

    Returns:
        (decoded_text, mean_log_probability).
    """
    T, C = logits.shape
    best_path = np.argmax(logits, axis=1)  # (T,)

    # Collapse repeated characters and remove blanks
    decoded_chars = []
    log_probs = []
    prev = blank_index

    for t in range(T):
        idx = best_path[t]
        if idx != blank_index and idx != prev:
            if idx - 1 < len(alphabet):
                decoded_chars.append(alphabet[idx - 1])
                log_probs.append(float(logits[t, idx]))
        prev = idx

    text = "".join(decoded_chars)
    mean_log_prob = float(np.mean(log_probs)) if log_probs else float('-inf')

    return text, mean_log_prob


# Alias for backward compatibility
ctc_greedy_decode = greedy_ctc_decode


# ---------------------------------------------------------------------------
# Character-Level Hypothesis Extraction
# ---------------------------------------------------------------------------

def extract_char_hypotheses(
    logits: np.ndarray,
    alphabet: str = DEFAULT_ALPHABET,
    blank_index: int = 0,
    top_k_alts: int = 4,
) -> List[CharHypothesis]:
    """
    Extract per-character hypotheses, confidence scores, and alternative candidates
    along the sequence timesteps.

    Args:
        logits: (T, C) log-probabilities.
        alphabet: Character set string.
        blank_index: CTC blank token index.
        top_k_alts: Number of alternative character predictions to record per letter.

    Returns:
        List of CharHypothesis objects representing the predicted character sequence.
    """
    T, C = logits.shape
    # Softmax probabilities
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    best_path = np.argmax(logits, axis=1)

    char_hyps: List[CharHypothesis] = []
    prev = blank_index

    for t in range(T):
        idx = best_path[t]
        if idx != blank_index and idx != prev:
            if idx - 1 < len(alphabet):
                pred_char = alphabet[idx - 1]
                conf = float(probs[t, idx])

                # Get top alternative characters at this timestep
                ranked_indices = np.argsort(probs[t])[::-1]
                alts: List[Tuple[str, float]] = []
                for alt_idx in ranked_indices:
                    if alt_idx != blank_index and alt_idx != idx and alt_idx - 1 < len(alphabet):
                        alts.append((alphabet[alt_idx - 1], float(probs[t, alt_idx])))
                        if len(alts) >= top_k_alts:
                            break

                char_hyps.append(
                    CharHypothesis(
                        char=pred_char,
                        confidence=conf,
                        alternatives=alts,
                        bbox=(t, 0, 1, 1),
                    )
                )
        prev = idx

    return char_hyps


# ---------------------------------------------------------------------------
# Word Hypothesis Construction
# ---------------------------------------------------------------------------

def logits_to_word_hypotheses(
    logits: np.ndarray,
    alphabet: str = DEFAULT_ALPHABET,
    beam_width: int = 10,
    stroke_bias: Optional[np.ndarray] = None,
    word_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> List[WordHypothesis]:
    """
    Convert CRNN output logits into ranked WordHypothesis objects with character breakdowns.

    Args:
        logits: (T, C) log-probabilities from CRNN forward pass.
        alphabet: Character set string.
        beam_width: Beam search width.
        stroke_bias: Optional stroke-primitive bias vector from stroke analysis.
        word_bbox: Optional bounding box of the word in image coordinates.

    Returns:
        List of WordHypothesis objects, ranked by visual score (descending).
    """
    # Extract detailed character breakdown
    char_hyps = extract_char_hypotheses(logits, alphabet)

    # Decode using beam search
    if stroke_bias is not None:
        raw_hypotheses = beam_search_with_stroke_prior(
            logits, stroke_bias, alphabet, beam_width
        )
    else:
        raw_hypotheses = beam_search_ctc(logits, alphabet, beam_width)

    if not raw_hypotheses:
        return [WordHypothesis(
            text="???",
            visual_score=float('-inf'),
            bbox=word_bbox,
            char_hypotheses=char_hyps,
        )]

    # Build WordHypothesis objects
    word_hyps = []
    for i, (text, log_prob) in enumerate(raw_hypotheses):
        text_clean = text.strip()
        if char_hyps and i == 0:
            conf = float(np.mean([c.confidence for c in char_hyps])) if char_hyps else 0.5
        else:
            seq_len = max(len(text_clean), 1)
            conf = float(np.clip(np.exp(log_prob / seq_len), 0.01, 0.99))

        hyp = WordHypothesis(
            text=text_clean,
            visual_score=float(log_prob),
            confidence=conf,
            bbox=word_bbox,
            char_hypotheses=char_hyps if i == 0 else [],
        )
        word_hyps.append(hyp)

    # Attach alternatives to the top hypothesis
    if len(word_hyps) > 1:
        word_hyps[0].alternatives = word_hyps[1:]

    return word_hyps

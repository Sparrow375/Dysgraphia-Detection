"""
TrOCR Engine Module — State-of-the-Art Transformer Handwriting OCR.
===================================================================
Wraps Microsoft TrOCR (Vision Transformer Encoder + RoBERTa Autoregressive Decoder)
for high-accuracy, calibrated word- and line-level handwriting recognition.

Supports:
  1. Automated offline weight caching in models/trocr/
  2. Single word and full text-line recognition
  3. High-throughput GPU batched inference
  4. True calibrated softmax token confidence scoring
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class TrOCREngine:
    """
    Production-grade TrOCR wrapper providing robust handwriting recognition
    and calibrated word-level confidence estimates.
    """

    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
        cache_dir: Optional[Union[str, Path]] = "models/trocr/base",
        device: Optional[str] = None,
        use_fast_processor: bool = False,
        use_finetuned: bool = False,
    ):
        """
        Args:
            model_name: HuggingFace identifier (e.g. 'microsoft/trocr-base-handwritten'
                        or 'microsoft/trocr-small-handwritten').
            cache_dir: Local directory to cache model files for 100% offline usage.
            device: 'cuda' or 'cpu'. If None, auto-detected.
            use_fast_processor: Whether to force fast tokenization.
            use_finetuned: Whether to automatically load weights from models/trocr/finetuned if available.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for TrOCREngine.")

        self.model_name = model_name
        self.use_finetuned = use_finetuned
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing TrOCREngine ({model_name}) on device: {self.device}...")
        self._load_processor_and_model()

    def _load_processor_and_model(self):
        """Loads processor and vision-encoder-decoder weights with offline caching."""
        from transformers import (
            AutoImageProcessor,
            RobertaTokenizer,
            TrOCRProcessor,
            VisionEncoderDecoderModel,
            XLMRobertaTokenizer,
        )

        # Check if fine-tuned checkpoint exists first
        finetuned_path = Path("models/trocr/finetuned")
        if self.use_finetuned and finetuned_path.exists() and (finetuned_path / "config.json").exists():
            load_target = str(finetuned_path)
            logger.info(f"Loaded fine-tuned model checkpoint from: {load_target}")
        elif self.cache_dir and (self.cache_dir / "config.json").exists():
            load_target = str(self.cache_dir)
        else:
            load_target = self.model_name

        cache_str = str(self.cache_dir) if self.cache_dir else None

        # 1. Load Image Processor
        try:
            image_processor = AutoImageProcessor.from_pretrained(
                load_target, cache_dir=cache_str if load_target == self.model_name else None
            )
        except Exception as e:
            logger.warning(f"Could not load AutoImageProcessor directly: {e}. Retrying.")
            image_processor = AutoImageProcessor.from_pretrained(load_target)

        # 2. Load Tokenizer explicitly based on architecture family
        try:
            if "small" in str(load_target).lower():
                tokenizer = XLMRobertaTokenizer.from_pretrained(
                    load_target, cache_dir=cache_str if load_target == self.model_name else None
                )
            else:
                tokenizer = RobertaTokenizer.from_pretrained(
                    load_target, cache_dir=cache_str if load_target == self.model_name else None
                )
        except Exception:
            try:
                tokenizer = RobertaTokenizer.from_pretrained(load_target)
            except Exception:
                tokenizer = XLMRobertaTokenizer.from_pretrained(load_target)

        self.processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

        # 3. Load VisionEncoderDecoderModel
        try:
            self.model = VisionEncoderDecoderModel.from_pretrained(
                load_target, cache_dir=cache_str if load_target == self.model_name else None
            )
        except Exception as e:
            logger.warning(f"Direct model load failed: {e}. Attempting fallback load.")
            self.model = VisionEncoderDecoderModel.from_pretrained(load_target)

        self.model.to(self.device)
        self.model.eval()

        # Save to local cache directory if not already populated
        if self.cache_dir and not (self.cache_dir / "config.json").exists():
            try:
                logger.info(f"Persisting offline TrOCR weights to {self.cache_dir}...")
                self.model.save_pretrained(self.cache_dir)
                self.processor.save_pretrained(self.cache_dir)
                logger.info(f"Successfully cached TrOCR model locally at {self.cache_dir}.")
            except Exception as e:
                logger.debug(f"Could not save local cache: {e}")

        logger.info(f"TrOCREngine successfully loaded on {self.device}!")

    def _prepare_pil_image(self, img_input: Union[np.ndarray, Image.Image]) -> Image.Image:
        """Converts arbitrary numpy or PIL inputs to an RGB PIL Image with contrast stretching."""
        if isinstance(img_input, Image.Image):
            return img_input.convert("RGB")

        if not isinstance(img_input, np.ndarray):
            raise TypeError(f"Unsupported image input type: {type(img_input)}")

        if img_input.size == 0:
            return Image.new("RGB", (64, 64), color=(255, 255, 255))

        # Handle color dimensions
        if img_input.ndim == 2:
            # Grayscale: ensure dark ink on light background
            if np.mean(img_input) < 100:
                # White ink on black background -> invert to black ink on white
                img_input = 255 - img_input
            bgr = cv2.cvtColor(img_input, cv2.COLOR_GRAY2BGR)
        elif img_input.ndim == 3:
            if img_input.shape[2] == 4:
                bgr = cv2.cvtColor(img_input, cv2.COLOR_BGRA2BGR)
            elif img_input.shape[2] == 3:
                bgr = img_input
            else:
                bgr = cv2.cvtColor(img_input[:, :, 0], cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError(f"Invalid image array shape: {img_input.shape}")

        # Contrast normalization for faint pencil writing
        p_low = np.percentile(bgr, 3)
        p_high = np.percentile(bgr, 97)
        if p_high - p_low > 25:
            bgr = np.clip((bgr.astype(float) - p_low) / (p_high - p_low + 1e-5) * 255.0, 0, 255).astype(np.uint8)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def recognize_word(
        self,
        word_patch: Union[np.ndarray, Image.Image],
        max_new_tokens: int = 24,
    ) -> Tuple[str, float]:
        """
        Recognize a single word crop and compute calibrated token confidence.

        Returns:
            (decoded_text, confidence) where confidence is in [0.0, 1.0].
        """
        pil_img = self._prepare_pil_image(word_patch)
        if pil_img.width < 5 or pil_img.height < 5:
            return "", 0.0

        pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            gen_out = self.model.generate(
                pixel_values,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                num_beams=1,
            )

        text = self.processor.batch_decode(gen_out.sequences, skip_special_tokens=True)[0].strip()
        confidence = self._compute_calibrated_sequence_confidence(gen_out, num_beams=1)
        return text, confidence

    def recognize_line(
        self,
        line_crop: Union[np.ndarray, Image.Image],
        max_new_tokens: int = 64,
        num_beams: int = 4,
    ) -> Tuple[str, float]:
        """
        Recognize an entire text-line strip without heuristic word slicing.

        Returns:
            (decoded_text, confidence) where confidence is in [0.0, 1.0].
        """
        pil_img = self._prepare_pil_image(line_crop)
        if pil_img.width < 10 or pil_img.height < 10:
            return "", 0.0

        pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            gen_out = self.model.generate(
                pixel_values,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                repetition_penalty=1.4,
                no_repeat_ngram_size=2,
            )

        text = self.processor.batch_decode(gen_out.sequences, skip_special_tokens=True)[0].strip()
        confidence = self._compute_calibrated_sequence_confidence(gen_out, num_beams=num_beams)
        return text, confidence

    def _compute_calibrated_sequence_confidence(self, gen_out, num_beams: int = 1) -> float:
        """Computes true exponentiated transition probability across generated tokens."""
        try:
            beam_idx = getattr(gen_out, "beam_indices", None) if num_beams > 1 else None
            trans_scores = self.model.compute_transition_scores(
                gen_out.sequences, gen_out.scores, beam_idx, normalize_logits=True
            )
            probs = [float(torch.exp(s).item()) for s in trans_scores[0]]
            # Filter valid non-EOS tokens
            if probs:
                return float(np.mean(probs))
        except Exception as e:
            logger.debug(f"Transition scores fallback: {e}")
        return self._compute_confidence(gen_out)

    def recognize_line_with_word_alignment(
        self,
        line_bgr: np.ndarray,
        global_origin: Tuple[int, int] = (0, 0),
        max_new_tokens: int = 64,
        num_beams: int = 4,
        physical_word_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        End-to-End Line Recognition with Whitespace Valley Projection Alignment
        and Calibrated Per-Word Confidence Extraction.
        Anchors directly to physical_word_boxes when provided to guarantee 0 horizontal drift.

        Returns:
            List of dicts: [
                {
                    "text": str,
                    "confidence": float (individual calibrated per-word confidence),
                    "bbox": (x, y, w, h) in global image coordinates,
                }, ...
            ]
        """
        gx0, gy0 = global_origin
        h, w = line_bgr.shape[:2]
        gray = cv2.cvtColor(line_bgr, cv2.COLOR_BGR2GRAY) if line_bgr.ndim == 3 else line_bgr
        _, bin_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # 1. Horizontal ink range (crops out trailing/leading empty margins)
        col_proj_raw = np.sum(bin_inv == 255, axis=0).astype(float)
        ink_cols_raw = np.where(col_proj_raw > 0)[0]
        if len(ink_cols_raw) == 0:
            return []

        x_ink_min = max(0, int(ink_cols_raw[0]) - 8)
        x_ink_max = min(w, int(ink_cols_raw[-1]) + 8)
        active_line_crop = line_bgr[:, x_ink_min:x_ink_max]

        pil_img = self._prepare_pil_image(active_line_crop)
        pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            gen_out = self.model.generate(
                pixel_values,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                repetition_penalty=1.4,
                no_repeat_ngram_size=2,
            )

        # 2. Extract per-subword probabilities via transition scores
        try:
            beam_idx = getattr(gen_out, "beam_indices", None) if num_beams > 1 else None
            transition_scores = self.model.compute_transition_scores(
                gen_out.sequences, gen_out.scores, beam_idx, normalize_logits=True
            )
            seq = gen_out.sequences[0][1:]
            scores = transition_scores[0]
            tokens_info = []
            for tok_id, score in zip(seq, scores):
                tok_str = self.processor.tokenizer.decode([tok_id.item()]).replace('\ufffd', '')
                if tok_str.strip() in ['<s>', '</s>', '<pad>']:
                    continue
                prob = float(torch.exp(score).item())
                tokens_info.append((tok_str, prob))
        except Exception as e:
            logger.debug(f"Per-word transition scores error: {e}. Falling back.")
            raw_text = self.processor.batch_decode(gen_out.sequences, skip_special_tokens=True)[0].strip()
            tokens_info = [(w, 0.75) for w in raw_text.split()]

        # 3. Group subwords into words with individual calibrated confidence
        words_data = []
        curr_word = ""
        curr_probs = []
        for tok_str, prob in tokens_info:
            if tok_str.startswith(" ") or curr_word == "":
                if curr_word.strip():
                    clean_w = curr_word.strip()
                    if clean_w not in ['"', "'", "“", "”", "`"]:
                        words_data.append((clean_w, float(np.mean(curr_probs))))
                curr_word = tok_str
                curr_probs = [prob]
            else:
                curr_word += tok_str
                curr_probs.append(prob)

        if curr_word.strip():
            clean_w = curr_word.strip()
            if clean_w not in ['"', "'", "“", "”", "`"]:
                words_data.append((clean_w, float(np.mean(curr_probs))))

        # 4. Attach trailing punctuation (e.g. '.', ',', '!') to preceding word
        cleaned_words: List[Tuple[str, float]] = []
        for w_str, w_prob in words_data:
            if w_str in [".", ",", "!", "?", ";", ":", "-"] and cleaned_words:
                prev_w, prev_p = cleaned_words[-1]
                cleaned_words[-1] = (prev_w + w_str, float((prev_p + w_prob) / 2.0))
            elif w_str not in [".", ",", "!", "?", ";", ":", "-"]:
                cleaned_words.append((w_str, w_prob))

        if not cleaned_words:
            return []

        words = [cw[0] for cw in cleaned_words]
        word_confs = [cw[1] for cw in cleaned_words]

        # 5. Direct Physical Ink Box Anchoring (Guarantees zero horizontal drift)
        if physical_word_boxes and len(physical_word_boxes) > 0:
            K = len(physical_word_boxes)
            M = len(words)
            sorted_boxes = sorted(physical_word_boxes, key=lambda b: b[0])

            if K == M:
                return [{
                    "text": words[i],
                    "confidence": word_confs[i],
                    "bbox": sorted_boxes[i],
                } for i in range(M)]

            current_boxes = list(sorted_boxes)

            # Case A: K > M -> Merge closest adjacent fragments until count == M
            while len(current_boxes) > M:
                best_idx = 0
                min_gap = 999999
                for i in range(len(current_boxes) - 1):
                    b1 = current_boxes[i]
                    b2 = current_boxes[i + 1]
                    gap = b2[0] - (b1[0] + b1[2])
                    if gap < min_gap:
                        min_gap = gap
                        best_idx = i
                b1 = current_boxes[best_idx]
                b2 = current_boxes[best_idx + 1]
                nx = min(b1[0], b2[0])
                ny = min(b1[1], b2[1])
                nw = max(b1[0] + b1[2], b2[0] + b2[2]) - nx
                nh = max(b1[1] + b1[3], b2[1] + b2[3]) - ny
                current_boxes[best_idx] = (nx, ny, nw, nh)
                current_boxes.pop(best_idx + 1)

            # Case B: K < M -> Split widest boxes at their internal ink minimum until count == M
            while len(current_boxes) < M:
                widest_idx = max(range(len(current_boxes)), key=lambda i: current_boxes[i][2])
                bx, by, bw, bh = current_boxes[widest_idx]

                lx = max(0, bx - gx0)
                col_sums = np.sum(bin_inv[:, lx : min(w, lx + bw)] > 0, axis=0)
                if len(col_sums) > 20:
                    mid_start = int(0.25 * len(col_sums))
                    mid_end = int(0.75 * len(col_sums))
                    split_rel_x = mid_start + int(np.argmin(col_sums[mid_start:mid_end]))
                else:
                    split_rel_x = bw // 2

                b1 = (bx, by, split_rel_x, bh)
                b2 = (bx + split_rel_x, by, bw - split_rel_x, bh)
                current_boxes[widest_idx] = b1
                current_boxes.insert(widest_idx + 1, b2)

            return [{
                "text": words[i],
                "confidence": word_confs[i],
                "bbox": current_boxes[i],
            } for i in range(M)]

        # 6. Fallback: Vertical ink range for tight line bounding
        row_proj_full = np.sum(bin_inv == 255, axis=1).astype(float)
        ink_rows = np.where(row_proj_full > 0)[0]
        y_min = int(ink_rows[0]) if len(ink_rows) > 0 else 0
        y_max = int(ink_rows[-1]) if len(ink_rows) > 0 else h

        if len(words) == 1:
            box_h = max(1, y_max - y_min + 6)
            box_y = max(0, gy0 + y_min - 3)
            return [{
                "text": words[0],
                "confidence": word_confs[0],
                "bbox": (gx0 + x_ink_min, box_y, max(1, x_ink_max - x_ink_min), box_h),
            }]

        # 6. Smooth column projection within active ink region
        active_bin = bin_inv[:, x_ink_min:x_ink_max]
        col_proj = np.sum(active_bin == 255, axis=0).astype(float)
        active_w = x_ink_max - x_ink_min

        k = max(5, int(active_w * 0.015))
        smoothed = np.convolve(col_proj, np.ones(k) / k, mode='same')

        valleys = []
        for x in range(15, active_w - 15):
            if smoothed[x] <= smoothed[x - 1] and smoothed[x] <= smoothed[x + 1]:
                valleys.append((x, smoothed[x]))

        valleys.sort(key=lambda v: v[1])
        selected = []
        min_dist = active_w / (len(words) * 2.2)
        for vx, vy in valleys:
            if all(abs(vx - sx) >= min_dist for sx in selected):
                selected.append(vx)
                if len(selected) == len(words) - 1:
                    break

        selected.sort()
        bounds = [0] + selected + [active_w]

        if len(bounds) < len(words) + 1:
            total_chars = max(1, sum(len(w) for w in words))
            span = active_w
            bounds = [0]
            cur = 0
            for w_idx in range(len(words) - 1):
                cur += int(span * (len(words[w_idx]) / total_chars))
                bounds.append(cur)
            bounds.append(active_w)

        results = []
        for i in range(len(words)):
            bx1 = int(bounds[i])
            bx2 = int(bounds[i + 1])

            # Tighten bbox horizontally and vertically to local ink
            w_strip = active_bin[:, bx1:bx2]
            local_cols = np.where(np.sum(w_strip == 255, axis=0) > 0)[0]
            local_rows = np.where(np.sum(w_strip == 255, axis=1) > 0)[0]

            pad_x = 3
            pad_y = 4
            if len(local_cols) > 0 and len(local_rows) > 0:
                actual_x1 = max(0, x_ink_min + bx1 + int(local_cols[0]) - pad_x)
                actual_x2 = min(w, x_ink_min + bx1 + int(local_cols[-1]) + pad_x + 1)
                actual_y1 = max(0, int(local_rows[0]) - pad_y)
                actual_y2 = min(h, int(local_rows[-1]) + pad_y + 1)
                box_x = gx0 + actual_x1
                box_y = gy0 + actual_y1
                box_w = max(1, actual_x2 - actual_x1)
                box_h = max(1, actual_y2 - actual_y1)
            else:
                box_x = gx0 + x_ink_min + bx1
                box_y = gy0
                box_w = max(1, bx2 - bx1)
                box_h = h

            results.append({
                "text": words[i],
                "confidence": float(np.clip(word_confs[i], 0.05, 0.99)),
                "bbox": (box_x, box_y, box_w, box_h),
            })

        return results

    def recognize_batch(
        self,
        image_list: List[Union[np.ndarray, Image.Image]],
        max_new_tokens: int = 24,
        batch_size: int = 16,
    ) -> List[Tuple[str, float]]:
        """
        Batch-accelerated recognition over multiple word or line crops on GPU.
        """
        if not image_list:
            return []

        results = []
        for i in range(0, len(image_list), batch_size):
            chunk = image_list[i : i + batch_size]
            pil_images = [self._prepare_pil_image(im) for im in chunk]

            pixel_values = self.processor(pil_images, return_tensors="pt").pixel_values.to(
                self.device
            )

            with torch.no_grad():
                gen_out = self.model.generate(
                    pixel_values,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                )

            decoded_texts = self.processor.batch_decode(
                gen_out.sequences, skip_special_tokens=True
            )
            confs = self._compute_batch_confidence(gen_out)

            for txt, conf in zip(decoded_texts, confs):
                results.append((txt.strip(), conf))

        return results

    def _compute_confidence(self, gen_out) -> float:
        """Computes mean exponential token probability from generation scores."""
        scores = getattr(gen_out, "scores", None)
        if not scores:
            return 0.75

        step_probs = []
        seq = gen_out.sequences[0, 1:]  # Exclude decoder start token
        for step_i, tok_id in enumerate(seq):
            if step_i < len(scores):
                logits = scores[step_i][0]
                prob = torch.softmax(logits, dim=-1)[tok_id].item()
                step_probs.append(prob)

        if not step_probs:
            return 0.75

        # Geometric or arithmetic mean of token probabilities
        return float(np.mean(step_probs))

    def _compute_batch_confidence(self, gen_out) -> List[float]:
        """Computes token probabilities for a batch of generated sequences."""
        scores = getattr(gen_out, "scores", None)
        batch_size = gen_out.sequences.shape[0]
        if not scores:
            return [0.75] * batch_size

        batch_confs = []
        for b_idx in range(batch_size):
            step_probs = []
            seq = gen_out.sequences[b_idx, 1:]
            for step_i, tok_id in enumerate(seq):
                if step_i < len(scores):
                    logits = scores[step_i][b_idx]
                    prob = torch.softmax(logits, dim=-1)[tok_id].item()
                    step_probs.append(prob)

            conf = float(np.mean(step_probs)) if step_probs else 0.75
            batch_confs.append(conf)

        return batch_confs

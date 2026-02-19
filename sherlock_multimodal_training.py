"""Real multimodal training loop for Sherlock (text + optional image).

This module trains a lightweight multimodal classifier on JSONL training records.
Each record should contain:
- prompt: str
- completion: str
- metadata.source: str
- metadata.image_path: optional str (path to image)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from PIL import Image

    _HAS_PIL = True
except Exception:
    Image = None
    _HAS_PIL = False

_TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class Vocab:
    def __init__(self) -> None:
        self.stoi = {"<pad>": 0, "<unk>": 1}
        self.itos = ["<pad>", "<unk>"]

    def add_text(self, text: str) -> None:
        for tok in _tokenize(text):
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(tok, 1) for tok in _tokenize(text)]
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
        return ids[:max_len]


@dataclass
class Record:
    prompt: str
    completion: str
    source: str
    image_path: str | None


class SherlockMultimodalDataset(Dataset):
    def __init__(
        self,
        records: list[Record],
        vocab: Vocab,
        source_to_id: dict[str, int],
        *,
        max_text_len: int = 64,
        image_size: int = 64,
    ) -> None:
        self.records = records
        self.vocab = vocab
        self.source_to_id = source_to_id
        self.max_text_len = max_text_len
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, image_path: str | None) -> tuple[torch.Tensor, float]:
        if not image_path or not _HAS_PIL:
            return torch.zeros(3, self.image_size, self.image_size), 0.0

        path = Path(image_path)
        if not path.exists() or not path.is_file():
            return torch.zeros(3, self.image_size, self.image_size), 0.0

        try:
            with Image.open(path) as img:
                img = img.convert("RGB").resize((self.image_size, self.image_size))
                raw = torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(self.image_size, self.image_size, 3)
                arr = raw.permute(2, 0, 1) / 255.0
                return arr, 1.0
        except Exception:
            return torch.zeros(3, self.image_size, self.image_size), 0.0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        text = f"{rec.prompt} {rec.completion}"
        token_ids = self.vocab.encode(text, max_len=self.max_text_len)
        image, image_mask = self._load_image(rec.image_path)
        y = self.source_to_id[rec.source]

        return {
            "tokens": torch.tensor(token_ids, dtype=torch.long),
            "image": image,
            "image_mask": torch.tensor([image_mask], dtype=torch.float32),
            "label": torch.tensor(y, dtype=torch.long),
        }


class SherlockMultimodalNet(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, text_dim: int = 128, image_dim: int = 128) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, text_dim, padding_idx=0)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.image_proj = nn.Sequential(nn.Linear(64, image_dim), nn.ReLU(), nn.Dropout(0.1))

        fused_dim = text_dim + image_dim + 1
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, tokens: torch.Tensor, image: torch.Tensor, image_mask: torch.Tensor) -> torch.Tensor:
        emb = self.embed(tokens)
        pad_mask = (tokens != 0).unsqueeze(-1)
        denom = pad_mask.sum(dim=1).clamp(min=1)
        text_feat = (emb * pad_mask).sum(dim=1) / denom
        text_feat = self.text_proj(text_feat)

        image_feat = self.image_encoder(image).flatten(1)
        image_feat = self.image_proj(image_feat)

        fused = torch.cat([text_feat, image_feat, image_mask], dim=1)
        return self.classifier(fused)


def load_records(path: str | Path) -> list[Record]:
    recs: list[Record] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row.get("metadata", {})
        recs.append(
            Record(
                prompt=str(row.get("prompt", "")),
                completion=str(row.get("completion", "")),
                source=str(meta.get("source", "unknown")),
                image_path=(str(meta.get("image_path")) if meta.get("image_path") else None),
            )
        )
    return recs


def split_records(records: list[Record], val_ratio: float = 0.2, seed: int = 42) -> tuple[list[Record], list[Record]]:
    if not records:
        return [], []
    rnd = random.Random(seed)
    arr = records[:]
    rnd.shuffle(arr)
    n_val = max(1, int(len(arr) * val_ratio)) if len(arr) > 1 else 0
    return arr[n_val:], arr[:n_val]


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return float((pred == labels).float().mean().item())


def train_multimodal_model(
    jsonl_path: str | Path,
    *,
    output_dir: str | Path = "data/processed/mm_checkpoints",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str | None = None,
) -> dict[str, Any]:
    records = load_records(jsonl_path)
    if len(records) < 2:
        raise ValueError("Need at least 2 records for training")

    train_records, val_records = split_records(records)

    vocab = Vocab()
    sources = sorted({r.source for r in records})
    source_to_id = {src: idx for idx, src in enumerate(sources)}

    for r in records:
        vocab.add_text(r.prompt)
        vocab.add_text(r.completion)

    train_ds = SherlockMultimodalDataset(train_records, vocab, source_to_id)
    val_ds = SherlockMultimodalDataset(val_records, vocab, source_to_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if len(val_ds) > 0 else None

    use_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SherlockMultimodalNet(vocab_size=len(vocab.itos), num_classes=len(sources)).to(use_device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        steps = 0

        for batch in train_loader:
            tokens = batch["tokens"].to(use_device)
            image = batch["image"].to(use_device)
            image_mask = batch["image_mask"].to(use_device)
            labels = batch["label"].to(use_device)

            logits = model(tokens, image, image_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_acc += _accuracy(logits.detach(), labels)
            steps += 1

        train_loss = total_loss / max(1, steps)
        train_acc = total_acc / max(1, steps)

        val_loss = 0.0
        val_acc = 0.0
        val_steps = 0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    tokens = batch["tokens"].to(use_device)
                    image = batch["image"].to(use_device)
                    image_mask = batch["image_mask"].to(use_device)
                    labels = batch["label"].to(use_device)

                    logits = model(tokens, image, image_mask)
                    loss = criterion(logits, labels)

                    val_loss += float(loss.item())
                    val_acc += _accuracy(logits, labels)
                    val_steps += 1

        metrics = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": (val_loss / max(1, val_steps)) if val_steps else 0.0,
            "val_acc": (val_acc / max(1, val_steps)) if val_steps else 0.0,
        }
        history.append(metrics)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = out_dir / "sherlock_multimodal.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": vocab.stoi,
            "source_to_id": source_to_id,
            "history": history,
        },
        ckpt,
    )

    return {
        "checkpoint": str(ckpt),
        "num_records": len(records),
        "num_train": len(train_records),
        "num_val": len(val_records),
        "num_sources": len(sources),
        "history": history,
    }

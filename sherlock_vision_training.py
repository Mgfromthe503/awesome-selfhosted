"""Train Sherlock vision model from annotation JSONL."""

from __future__ import annotations

import json
from pathlib import Path
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


class VisionDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], label_to_id: dict[str, int], image_size: int = 96) -> None:
        self.rows = rows
        self.label_to_id = label_to_id
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        path = Path(str(row["image_path"]))
        label = self.label_to_id[str(row["label"])]

        if _HAS_PIL and path.exists():
            with Image.open(path) as img:
                img = img.convert("RGB")
                bbox = row.get("bbox", [])
                if isinstance(bbox, list) and len(bbox) == 4:
                    x, y, w, h = [int(max(0, v)) for v in bbox]
                    if w > 0 and h > 0:
                        img = img.crop((x, y, x + w, y + h))
                img = img.resize((self.image_size, self.image_size))
                arr = torch.from_numpy(__import__("numpy").array(img, dtype="float32"))
                x_tensor = arr.permute(2, 0, 1) / 255.0
        else:
            x_tensor = torch.zeros(3, self.image_size, self.image_size)

        return x_tensor, torch.tensor(label, dtype=torch.long)


class VisionNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, max(1, num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _load_annotations(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if ln.strip():
            rows.append(json.loads(ln))
    return rows


def train_sherlock_vision(
    annotations_path: str | Path = "data/images/annotations.jsonl",
    *,
    output_dir: str | Path = "data/processed/vision_checkpoints",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-3,
    device: str | None = None,
) -> dict[str, Any]:
    rows = _load_annotations(annotations_path)
    if not rows:
        raise ValueError("No annotation rows found")

    labels = sorted({str(r.get("label", "unknown")) for r in rows})
    label_to_id = {lbl: i for i, lbl in enumerate(labels)}

    train_rows = [r for r in rows if str(r.get("split", "train")) != "val"]
    val_rows = [r for r in rows if str(r.get("split", "train")) == "val"]

    train_ds = VisionDataset(train_rows, label_to_id)
    val_ds = VisionDataset(val_rows, label_to_id) if val_rows else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None

    use_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionNet(num_classes=len(labels)).to(use_device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    history = []
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_acc = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(use_device), y.to(use_device)
            logits = model(x)
            loss = criterion(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            tr_loss += float(loss.item())
            tr_acc += float((logits.argmax(1) == y).float().mean().item())
            n += 1

        val_loss = 0.0
        val_acc = 0.0
        vn = 0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(use_device), y.to(use_device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    val_loss += float(loss.item())
                    val_acc += float((logits.argmax(1) == y).float().mean().item())
                    vn += 1

        history.append(
            {
                "epoch": ep,
                "train_loss": tr_loss / max(1, n),
                "train_acc": tr_acc / max(1, n),
                "val_loss": val_loss / max(1, vn) if vn else 0.0,
                "val_acc": val_acc / max(1, vn) if vn else 0.0,
            }
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "sherlock_vision.pt"
    torch.save({"model_state": model.state_dict(), "labels": labels, "history": history}, ckpt)

    return {
        "checkpoint": str(ckpt),
        "num_rows": len(rows),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "num_labels": len(labels),
        "history": history,
    }


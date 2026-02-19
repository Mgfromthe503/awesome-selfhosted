# Benchmark V1 (Sherlock)

## Scope
Benchmark checkpoint for Sherlock as a multimodal detective AI (bioinformatics + AI evolution + MM language).

## Included Datasets
- `data/processed/alpha_training.jsonl`
- `data/processed/sherlock_snippet_training.jsonl`
- `data/processed/sherlock_training_combined.jsonl`

## Core Skills At This Checkpoint
- Pattern recognition from text records
- Numeric anomaly detection in MM scripts
- Emoji/symbolic interpretation using MM knowledge base
- Synthetic regression reasoning (Alpha Mind Gamma)
- Multimodal model training loop (text + optional image)

## New Training Loop
- Module: `sherlock_multimodal_training.py`
- CLI: `scripts/train_sherlock_multimodal.py`
- Objective: classify `metadata.source` using fused text and image features
- Output checkpoint: `data/processed/mm_checkpoints/sherlock_multimodal.pt`

## Snapshot Command
```powershell
python scripts/benchmark_snapshot.py
```

## Multimodal Train Command
```powershell
python scripts/train_sherlock_multimodal.py --data data/processed/sherlock_training_combined.jsonl --epochs 3
```

## Next Gaps
- Add labeled image detection dataset (legal/open-license only)
- Add object localization (bbox) head and mAP metrics
- Add TTS production backend (currently capability layer is mostly scaffolding)
- Add continual training registry/versioning

# Sherlock Capability Snapshot

Sherlock is configured as a multimodal detective AI focused on:
- bioinformatics
- AI evolution
- pattern recognition
- anomaly detection
- MM language (Hermetic-principle-inspired command language)

## Current capabilities
- Text classification and decisioning: `sherlock_ai_refactor.py`
- Emoji translation and symbolic reasoning: `sherlock_training_data.py`, `emoji_parser.py`
- MM language parse/execute: `mm_language.py`
- Anomaly detection primitives: `sherlock_training_data.py`, `mm_language.py`
- Synthetic predictive model + JSONL export: `alpha_mind_gamma_model.py`
- Unified dataset assembly: `sherlock_dataset_builder.py`
- Dataset and prediction evaluation: `sherlock_evaluation.py`
- Multimodal service wrappers:
  - TTS stub pipeline: `sherlock_multimodal.py`
  - Vision/image metadata extraction + training records: `sherlock_multimodal.py`

## Next capability upgrades
- Replace TTS stub with local engine backend (Piper/Coqui).
- Add vision captioning backend (LLaVA/BLIP2 via local inference service).
- Add MM language AST-to-action compiler for chained investigations.
- Add benchmark suite for bioinformatics and anomaly tasks.

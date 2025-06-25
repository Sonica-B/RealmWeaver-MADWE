# MADWE - Multi-Agent Diffusion World Engine

Real-time interactive game content generation using multi-agent diffusion models.

## Team Members
- **Ankit Gole**: Architecture & Integration Lead
- **Shreya Boyane**: AI Models & Generation Lead

## Setup Instructions

### Prerequisites
- Windows 10/11
- Python 3.13.5
- CUDA 12.1+ and compatible GPU drivers
- NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Sonica-B/RealmWeaver-MADWE.git
cd MADWE_PROJECT
```

2. Run setup script:
```bash
scripts\setup_environment.bat
```

3. Activate environment:
```bash
venv\Scripts\activate.bat
```

## Week 1 Tasks (Shreya)

### Day 1: Environment Setup
```bash
# Verify GPU setup and benchmark
python scripts/benchmark.py
```

### Day 2: Data Pipeline
```bash
# Generate synthetic data
python scripts/download_data.py

# Preprocess into train/val/test
python scripts/preprocess_data.py
```

### Day 3: LoRA Training
```bash
# Train LoRA for forest biome
python -m src.models.diffusion.lora_trainer --biome forest
```

## Project Structure
```
MADWE_PROJECT/
├── src/                    # Source code
│   ├── agents/            # Multi-agent system
│   ├── models/            # AI models
│   │   └── diffusion/     # Diffusion models & LoRA
│   └── utils/             # Utilities
├── data/                  # Data directory
│   ├── raw/              # Raw assets
│   ├── processed/        # Processed splits
│   └── models/           # Trained models
├── configs/              # Configuration files
├── scripts/              # Setup and utility scripts
└── notebooks/            # Experiments
```

## Key Features
- **LoRA Fine-tuning**: Biome-specific style adaptation
- **Real-time Generation**: Optimized for 20+ FPS
- **Multi-Agent System**: Coordinated content generation
- **Player Prediction**: Adaptive content pre-generation

## GPU Memory Optimization
- Mixed precision (fp16)
- Gradient checkpointing
- Batch size: 1 with accumulation
- LoRA rank: 8

## Dependencies
- PyTorch 2.7.1 with CUDA 12.1
- Diffusers 0.34.0
- Transformers 4.52.4
- Python 3.13.5

## Next Steps
1. Train LoRA adapters for all biomes
2. Implement multi-agent coordination
3. Add Unity integration
4. Optimize for real-time inference

## License
This project is for academic purposes.
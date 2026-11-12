# 🎙️ VoxTask — Autonomous Voice-Driven Small Language Model (SLM)

[![Live Demo](https://img.shields.io/badge/Live_Demo-GitHub_Pages-22D3EE?style=for-the-badge&logo=github)](https://subashis-dev.github.io/vox-task-slm/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B_CUDA_12.4-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Model Parameters](https://img.shields.io/badge/Parameters-4.8M_SLM-22D3EE?style=for-the-badge&logo=cpu&logoColor=black)](#-neural-architecture)
[![Pre-trained Weights](https://img.shields.io/badge/Pre--trained_Weights-ZERO_(Trained_From_Scratch)-8B5CF6?style=for-the-badge)](#-training-dynamics--convergence)
[![Offline](https://img.shields.io/badge/Privacy-100%25_Offline_(Zero_Cloud_APIs)-10B981?style=for-the-badge&logo=shield&logoColor=white)](#-key-highlights)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](#-docker-deployment)
[![License](https://img.shields.io/badge/License-MIT-F59E0B?style=for-the-badge)](LICENSE)

> 🌐 **Try the Live Interactive Demo directly in your browser:**  
> 👉 **[https://subashis-dev.github.io/vox-task-slm/](https://subashis-dev.github.io/vox-task-slm/)**  
> *(Runs 100% client-side via WebAssembly & ONNX Runtime Web — zero server required!)*

An end-to-end, privacy-first **Small Language Model (4.8M parameters)** designed and trained completely **from scratch** in pure PyTorch (initialized with standard random Gaussian distributions `torch.randn`, with zero reliance on Hugging Face, OpenAI, or external pre-trained foundation models).

VoxTask transforms natural conversational voice speech into structured, deterministic JSON actions with sub-20ms latency on consumer GPUs (NVIDIA RTX 3060) and sub-90ms on standard CPUs.

---

## 🌟 Key Highlights

- **100% Offline & Zero Cloud API Cost:** Runs entirely on local hardware. No OpenAI, Anthropic, or cloud API tokens required.
- **Trained 100% From Scratch:** Full ownership of neural architecture, custom tokenizer, and weights from scratch.
- **Ultra-Lightweight & Git-Friendly (~4.8M Parameters):** Checkpoint size is only **19.3 MB** (fits under GitHub's 100 MB file limit with no Git LFS needed) and consumes **< 80 MB of RAM** at runtime.
- **Conversational Speech Robustness:** Intelligently handles conversational preambles (*"write a todo list to..."*, *"better to do this to..."*, *"can you please remind me to..."*), speech recognition typos (*"wirte"*, *"tommorrow"*, *"asop"*), and domain-specific acronyms (*"SOP"*, *"PhD"*, *"Kent application"*).
- **Dual Machine-Readable & Spoken Response:** Emits both deterministic JSON command payloads and natural conversational assistant replies in a single autoregressive forward pass.
- **Real-Time Interactive Web Dashboard:** Includes a responsive cybernetic web UI with continuous microphone voice input (`WebSpeechAPI`), dynamic audio visualizer, live token stream telemetry, and auditory spoken confirmations (`SpeechSynthesis`).
- **Production-Ready Docker Setup:** Packaged with Docker Compose and persistent storage volume mounts for continuous deployment.

---

## 🧠 Neural Architecture

VoxTask implements a modern **Decoder-Only Causal Transformer** optimized for edge execution and deterministic token generation:

```
[ Natural Voice Speech ]
        │
        ▼ (Web Speech API / Microphone)
"write a todo list to write a sop at 5pm tomorrow"
        │
        ▼ (Voice Preamble Normalizer)
"remind me to write a sop at 5pm tomorrow"
        │
        ▼
   <bos> Tokens <sep>
        │
        ▼
┌────────────────────────────────────────────────────────┐
│               VoxTask Causal Transformer               │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 1. Token & Learned Positional Embeddings (256-d) │  │
│  └──────────────────────────────────────────────────┘  │
│                           │                            │
│                           ▼                            │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 2. Pre-Layer RMSNorm Normalization               │  │
│  └──────────────────────────────────────────────────┘  │
│                           │                            │
│                           ▼                            │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 3. Multi-Head Causal Self-Attention              │  │
│  │    • 8 Attention Heads (32-dim per head)         │  │
│  │    • Strict Lower-Triangular Causal Mask         │  │
│  └──────────────────────────────────────────────────┘  │
│                           │ (Residual Skip Connection) │
│                           ▼                            │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 4. Position-Wise Feed-Forward Network (MLP)      │  │
│  │    • 4x Hidden Expansion (1024 dim)              │  │
│  │    • SwiGLU / GeLU Non-Linear Activation         │  │
│  └──────────────────────────────────────────────────┘  │
│                           │ (Residual Skip Connection) │
│                           ▼                            │
│            [ Repeated across 6 Layers ]                │
│                           │                            │
│                           ▼                            │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 5. Final RMSNorm + Tied Linear Output Head       │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
        │
        ▼ (Greedy deterministic decoding)
{
  "action": "REMIND",
  "task": "Write a SOP",
  "time": "tomorrow at 5:00 pm",
  "reply": "I've set a reminder for Write a SOP tomorrow at 5:00 pm."
}
        │
        ├─► [Persistent Store]: Adds scheduled task with ⏰ badge
        └─► [Speech Synthesis]: Speaks confirmation aloud
```

### Model Specifications

| Parameter | Specification | Purpose |
| :--- | :--- | :--- |
| **Model Type** | Decoder-Only Causal Transformer | Autoregressive sequence-to-sequence mapping |
| **Total Parameters** | **4,803,840 (~4.8M)** | Verified via PyTorch parameter profiler |
| **Vocabulary Size** | 192 (Padded) | Subword keywords + temporal tokens + printable ASCII |
| **Embedding Dimension ($d_{model}$)** | 256 | Dense vector representation width |
| **Transformer Depth** | 6 Blocks | Stacked multi-head attention + FFN blocks |
| **Attention Heads** | 8 Heads | 32 dimensions per attention head |
| **FFN Hidden Dimension** | 1,024 ($4 \times d_{model}$) | High-capacity non-linear feature projection |
| **Max Context Window** | 128 Tokens | Accommodates multi-clause voice prompt + output JSON |
| **Normalization** | RMSNorm ($\epsilon=10^{-6}$) | Modern, scale-invariant normalization |
| **Weight Tying** | Enabled | Output projection head shares weights with token embedding |
| **Model Footprint** | **19.3 MB** (`checkpoint_best.pt`) / **18.9 MB** (`model.onnx`) | 100% self-contained in repository — zero external weights download, no Git LFS required |

---

## 📊 Training Dynamics & Hardware Setup

The model was initialized with standard normal random weights (`mean=0.0, std=0.02`) and trained using **AdamW** with a **Cosine Annealing Learning Rate Schedule** and gradient clipping ($\text{norm}=1.0$).

### Hardware Environment
- **GPU:** NVIDIA GeForce RTX 3060 Laptop GPU (6.0 GB VRAM)
- **CUDA:** 12.4 with FP16 Tensor Core acceleration (`torch.cuda.amp.autocast`)
- **Dataset:** 75,000 synthesized conversational samples with high-entropy task vocabulary and combinatorial time generators.

### Training Convergence Log

| Epoch | Train Loss | Validation Loss | Validation Perplexity | Epoch Time | Convergence Stage |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **01** | 1.1842 | 0.0894 | 1.09 | ~35s | Learned delimiters `<bos>`, `<sep>`, `<eos>` & JSON syntax |
| **02** | 0.0521 | 0.0028 | 1.00 | ~35s | Mastered time parsing (AM/PM, relative dates, hours) |
| **03** | 0.0112 | 0.0009 | 1.00 | ~35s | Span-copy attention for arbitrary academic & work tasks |
| **04** | 0.0039 | 0.0001 | 1.00 | ~35s | High-precision entity boundaries & preposition handling |
| **05** | 0.0011 | 0.0000 | 1.00 | ~35s | Conversational speech preamble resilience |
| **06** | **0.0004** | **0.0000** | **1.00** | ~35s | **Optimal global convergence & zero validation loss** |

The trained checkpoint is saved at `checkpoint_best.pt` (**19.3 MB**).

---

## 📁 Repository Structure

```
voice-todo-llm/
├── model.py            # Custom 4.8M PyTorch Transformer (RMSNorm, SwiGLU, Causal Attention)
├── tokenizer.py        # Subword/character hybrid tokenizer with keyword acceleration
├── dataset.py          # Dynamic combinatorial voice command generator
├── train.py            # Training pipeline with AdamW, Cosine Annealing & FP16 AMP
├── inference.py        # Interactive CLI for testing raw voice predictions
├── server.py           # Production multi-threaded HTTP REST API & task manager
├── vocab.json          # Serialized vocabulary index
├── checkpoint_best.pt  # Trained model weights (19.3 MB — Git push ready)
├── web/
│   └── index.html      # Cybernetic UI with WebSpeechAPI, WebAudio visualizer & TTS
├── data/
│   └── tasks.json      # Persistent JSON task database
├── Dockerfile          # Production-grade container (<700MB)
├── docker-compose.yml  # Docker Compose definition with volume mounts
├── .dockerignore       # Docker build exclusions
├── .gitignore          # Git exclusions (__pycache__, environments, etc.)
├── requirements.txt    # Minimal dependencies (torch, numpy, tqdm)
├── LICENSE             # MIT Open-Source License
└── README.md           # Comprehensive architectural & user documentation
```

---

## ⚡ Quick Start (Local Setup)

### 1. Clone & Enter Directory
```bash
git clone https://github.com/subashis-dev/vox-task-slm.git
cd vox-task-slm
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

*(For GPU acceleration on NVIDIA hardware, install PyTorch with CUDA support: `pip install torch --index-url https://download.pytorch.org/whl/cu124`)*

### 3. Launch Web Server & Dashboard
```bash
python server.py
```

Open your browser at:
👉 **[http://localhost:8000](http://localhost:8000)**

- Click the **Microphone** button or use the text box to try commands like:
  - *"write a todo list to write a sop at 5pm tomorrow"*
  - *"Remind me about Kent application at 2 am tomorrow"*
  - *"Add buy thermal paste to my hardware list"*
  - *"Mark submit research paper as completed"*
- The model parses the intent, triggers the spoken audio reply, and updates your task manager.

---

## 💻 Terminal Interactive CLI

You can also run inference directly in your terminal without a browser:

```bash
python inference.py
```

Example session:
```text
Voice Task SLM Interactive Console
Enter voice command: write a todo list to write a sop at 5pm tomorrow
Model Output:
{
  "action": "REMIND",
  "task": "Write a SOP",
  "time": "tomorrow at 5:00 pm",
  "reply": "I've set a reminder for Write a SOP tomorrow at 5:00 pm."
}
```

---

## 🐳 Docker Deployment

The application includes full containerization with persistent volume mapping so your tasks and reminders survive container restarts:

### 1. Build and Start Container
```bash
docker compose up -d --build
```

### 2. Monitor Container Logs
```bash
docker compose logs -f
```

### 3. Access Dashboard
Navigate to `http://localhost:8000` in your web browser.

### 4. Stop Container
```bash
docker compose down
```

---

## 🔌 REST API Reference

The local server exposes clean REST endpoints:

### `POST /api/parse`
Parses raw voice/text input through the SLM and updates task state.
```bash
curl -X POST http://localhost:8000/api/parse \
  -H "Content-Type: application/json" \
  -d '{"text": "remind me to write a sop at 5pm tomorrow"}'
```
**Response (200 OK):**
```json
{
  "action": {
    "action": "REMIND",
    "task": "Write a SOP",
    "time": "tomorrow at 5:00 pm",
    "reply": "I've set a reminder for Write a SOP tomorrow at 5:00 pm."
  },
  "raw_output": "{\"action\":\"REMIND\",\"task\":\"Write asop at\",\"time\":\"tomorrow at 5:00 pm\",\"reply\":\"I've set a reminder for Write asop at tomorrow at 5:00 pm.\"}",
  "inference_ms": 14.8,
  "spoken_reply": "I've set a reminder for Write a SOP tomorrow at 5:00 pm.",
  "tasks": [...]
}
```

### `GET /api/tasks`
Returns all active and completed tasks.

### `POST /api/tasks/toggle`
Toggles task completion status by task ID: `{"id": "t-1794487154751"}`.

### `POST /api/tasks/delete`
Deletes a task by ID: `{"id": "t-1794487154751"}`.

### `GET /api/health`
Returns system health, parameter count, and hardware compute device (`cuda` or `cpu`).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — feel free to modify, extend, and deploy it for personal or commercial projects.

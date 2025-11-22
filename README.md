# Qwen3 Invoice Extractor

> **Fine-tune a Small Language Model (SLM) for high-precision invoice extraction and structuring**

This repository provides a complete pipeline for fine-tuning **Qwen3 0.6B-Instruct**, a state-of-the-art Small Language Model, to extract structured data from invoices. The fine-tuned model is designed to support another project focused on invoice structuring and processing.

## 🎯 Purpose

This project fine-tunes a lightweight SLM (Small Language Model) to convert unstructured invoice text into structured JSON. The model is optimized for:

- **High accuracy** (target: 99%+) on invoice extraction tasks
- **Cost-effective deployment** on serverless infrastructure
- **Small model size** (<1GB) for fast inference
- **Structured output** that integrates seamlessly with invoice processing systems

The fine-tuned model serves as a critical component in a larger invoice structuring pipeline, enabling automated extraction of key fields like invoice numbers, dates, vendors, line items, and totals.

## 🏗️ Architecture Overview

```mermaid
graph TB
    subgraph "Training Pipeline (Local GPU)"
        A[Raw Invoice Datasets<br/>HuggingFace Hub] --> B[Phase 1: Data Preparation<br/>Download & Format]
        B --> C[ChatML Training Data<br/>train.jsonl]
        C --> D[Phase 2: Model Training<br/>Unsloth + LoRA]
        D --> E[Fine-tuned Qwen3 0.6B<br/>with LoRA Adapters]
        E --> F[Phase 3: Export<br/>Merge & Convert to FP16]
        F --> G[Model Artifacts<br/>model.tar.gz]
    end

    subgraph "Deployment Pipeline (AWS)"
        G --> H[S3 Bucket]
        H --> I[SageMaker Serverless<br/>Endpoint Deployment]
    end

    subgraph "Inference Pipeline (Production)"
        J[Invoice PDF] --> K[PDF Parser<br/>LangChain + OCR]
        K --> L[Raw Text Extraction]
        L --> I
        I --> M[Structured JSON<br/>Pydantic Validated]
        M --> N[Invoice Structuring<br/>System Integration]
    end

    style A fill:#b3e5fc,color:#000
    style B fill:#c5e1a5,color:#000
    style C fill:#ffccbc,color:#000
    style D fill:#ce93d8,color:#000
    style E fill:#90caf9,color:#000
    style F fill:#a5d6a7,color:#000
    style G fill:#ffb74d,color:#000
    style H fill:#81c784,color:#000
    style I fill:#64b5f6,color:#000
    style J fill:#ef5350,color:#fff
    style K fill:#ab47bc,color:#fff
    style L fill:#42a5f5,color:#000
    style M fill:#66bb6a,color:#000
    style N fill:#f48fb1,color:#000
```

## 📊 System Components

```mermaid
graph LR
    subgraph "Training Module"
        T1[Data Preparation<br/>Phase 1]
        T2[Model Training<br/>Phase 2]
        T3[Model Export<br/>Phase 3]
        T1 --> T2 --> T3
    end

    subgraph "Ingestion Module"
        I1[PDF Parser<br/>Native Text]
        I2[OCR Fallback<br/>Tesseract]
        I1 --> I2
    end

    subgraph "Inference Module"
        INF1[SageMaker Client]
        INF2[Schema Validator<br/>Pydantic]
        INF1 --> INF2
    end

    subgraph "Deployment Module"
        D1[SageMaker Deployer]
        D2[Serverless Config]
        D1 --> D2
    end

    T3 --> D1
    I2 --> INF1
    INF2 --> EXT[External Invoice<br/>Structuring System]

    style T1 fill:#81d4fa,color:#000
    style T2 fill:#a5d6a7,color:#000
    style T3 fill:#ffb74d,color:#000
    style I1 fill:#ba68c8,color:#fff
    style I2 fill:#64b5f6,color:#000
    style INF1 fill:#ef5350,color:#fff
    style INF2 fill:#26a69a,color:#000
    style D1 fill:#ffa726,color:#000
    style D2 fill:#ab47bc,color:#fff
    style EXT fill:#ec407a,color:#fff
```

## 🔄 Training Pipeline (3 Phases)

### Phase 1: Data Preparation

Downloads and formats invoice datasets from HuggingFace Hub into ChatML format.

```mermaid
sequenceDiagram
    participant HF as HuggingFace Hub
    participant Loader as Dataset Loader
    participant Adapter as Dataset Adapter
    participant Validator as Data Validator
    participant Output as train.jsonl

    HF->>Loader: Download Datasets
    Note over HF,Loader: mychen76/invoices-and-receipts_ocr_v1<br/>shubh303/Invoice-to-Json

    Loader->>Adapter: Extract Text/JSON Pairs
    Adapter->>Validator: Validate JSON Structure
    Validator->>Adapter: Filter Invalid Rows
    Adapter->>Output: Format as ChatML
    Note over Adapter,Output: System: "You are a strict invoice parser..."<br/>User: Raw invoice text<br/>Assistant: Structured JSON
```

**Key Features:**
- Downloads multiple invoice datasets from HuggingFace Hub
- Adapts different dataset formats to unified ChatML structure
- Validates JSON output quality
- Filters invalid or corrupted examples
- Outputs standardized `train.jsonl` file

### Phase 2: Model Training

Fine-tunes Qwen3 0.6B using Unsloth and LoRA (Low-Rank Adaptation).

```mermaid
graph TD
    A[Qwen3 0.6B-Instruct<br/>Base Model] --> B[Apply LoRA Adapters<br/>r=16, alpha=16]
    B --> C[Load Training Data<br/>train.jsonl]
    C --> D[SFT Trainer<br/>Supervised Fine-Tuning]
    D --> E[Training Loop<br/>60 steps, 2e-4 LR]
    E --> F[Fine-tuned Model<br/>with LoRA Weights]

    style A fill:#64b5f6,color:#000
    style B fill:#81c784,color:#000
    style C fill:#ffb74d,color:#000
    style D fill:#ba68c8,color:#fff
    style E fill:#ef5350,color:#fff
    style F fill:#26a69a,color:#000
```

**Training Configuration:**
- **Model**: Qwen3 0.6B-Instruct (released April 2025)
- **Method**: LoRA (Low-Rank Adaptation) with r=16
- **Quantization**: 4-bit during training (saves VRAM)
- **Sequence Length**: 2048 tokens
- **Training Steps**: 60 (fast convergence with Unsloth)
- **Hardware**: Local GPU with 16GB VRAM

### Phase 3: Model Export

Merges LoRA adapters into the base model and exports in FP16 format for SageMaker.

```mermaid
flowchart LR
    A[Fine-tuned Model<br/>Base + LoRA] --> B[Merge LoRA Adapters<br/>into Base Weights]
    B --> C[Convert to FP16<br/>Standard Precision]
    C --> D[Export Model Artifacts<br/>tokenizer + config]
    D --> E[Create model.tar.gz<br/>Ready for S3 Upload]

    style A fill:#ffb74d,color:#000
    style B fill:#81c784,color:#000
    style C fill:#64b5f6,color:#000
    style D fill:#ba68c8,color:#fff
    style E fill:#ef5350,color:#fff
```

**Export Details:**
- Merges LoRA weights into base model (no separate adapter files)
- Converts to FP16 for CPU inference compatibility
- Packages model, tokenizer, and configuration
- Creates tarball for SageMaker deployment

## 🚀 Deployment Architecture

```mermaid
graph TB
    subgraph "Local Environment"
        L1[Fine-tuned Model]
        L2[model.tar.gz]
        L1 --> L2
    end

    subgraph "AWS Cloud"
        S3[S3 Bucket]
        SM[SageMaker Serverless<br/>3GB Memory, CPU]
        EP[Inference Endpoint]

        L2 -->|Upload| S3
        S3 -->|Deploy| SM
        SM --> EP
    end

    subgraph "Client Application"
        CL1[Invoice PDF]
        CL2[PDF Parser]
        CL3[Extract Text]
        CL4[Invoke Endpoint]
        CL5[Structured JSON]

        CL1 --> CL2
        CL2 --> CL3
        CL3 --> CL4
        CL4 -->|HTTP Request| EP
        EP -->|JSON Response| CL5
    end

    CL5 --> EXT[External Invoice<br/>Structuring System]

    style L1 fill:#ffb74d,color:#000
    style L2 fill:#81c784,color:#000
    style S3 fill:#64b5f6,color:#000
    style SM fill:#ba68c8,color:#fff
    style EP fill:#26a69a,color:#000
    style CL1 fill:#ef5350,color:#fff
    style CL2 fill:#ab47bc,color:#fff
    style CL3 fill:#42a5f5,color:#000
    style CL4 fill:#66bb6a,color:#000
    style CL5 fill:#ffa726,color:#000
    style EXT fill:#ec407a,color:#fff
```

## 📦 Project Structure

```
qwen3-invoice-extractor/
├── src/
│   ├── training/           # Training pipeline (3 phases)
│   │   ├── phases/
│   │   │   ├── phase1_data_preparation.py
│   │   │   ├── phase2_model_training.py
│   │   │   └── phase3_export.py
│   │   ├── configs/       # Training & dataset configs
│   │   └── utils/         # Prompt formatting, validation
│   ├── ingestion/         # PDF parsing (LangChain + OCR)
│   ├── inference/         # SageMaker client & schema
│   └── deployment/        # SageMaker deployment scripts
├── scripts/
│   └── prepare_data.py    # CLI for data preparation
├── data/
│   └── train.jsonl        # Generated training data
└── docs/                  # Additional documentation
```

## 🛠️ Quick Start

### Prerequisites

- Python 3.12+
- Local GPU with 16GB VRAM (for training)
- AWS Account (for deployment)
- HuggingFace account (for dataset access)

### Installation

```bash
# Install dependencies
uv sync

# Install training dependencies (requires GPU)
uv sync --extra training
```

### Training Pipeline

```bash
# Phase 1: Prepare training data
python -m scripts.prepare_data --output data/train.jsonl

# Phase 2: Train the model (requires GPU)
python -m src.training.phases.phase2_model_training

# Phase 3: Export model for deployment
python -m src.training.phases.phase3_export
```

### Deployment

```bash
# Upload model to S3
aws s3 cp model.tar.gz s3://your-bucket/models/

# Deploy to SageMaker Serverless
python -m src.deployment.sagemaker_deployer
```

## 🔗 Integration with Invoice Structuring System

The fine-tuned model is designed to integrate with a larger invoice structuring and processing system:

```mermaid
graph LR
    A[Invoice PDF] --> B[This Repository<br/>Qwen3 Fine-tuned Model]
    B --> C[Structured JSON Output]
    C --> D[Invoice Structuring System<br/>External Project]
    D --> E[Business Logic<br/>Validation, Storage, Analytics]

    style A fill:#ef5350,color:#fff
    style B fill:#ffb74d,color:#000
    style C fill:#66bb6a,color:#000
    style D fill:#ec407a,color:#fff
    style E fill:#26a69a,color:#000
```

**Output Schema:**
```json
{
  "invoice_number": "INV-2024-001",
  "date": "2024-01-15",
  "vendor": "Acme Corporation",
  "total_amount": 1250.00,
  "line_items": [...],
  "tax": 100.00
}
```

The extracted structured data is validated using Pydantic schemas and can be directly consumed by downstream invoice processing workflows.

## 📈 Model Specifications

| Component | Specification |
|-----------|--------------|
| **Base Model** | Qwen3 0.6B-Instruct |
| **Model Size** | ~600M parameters, <1GB on disk |
| **Training Method** | LoRA (r=16, alpha=16) |
| **Quantization** | 4-bit during training, FP16 for inference |
| **Sequence Length** | 2048 tokens |
| **Deployment** | SageMaker Serverless (3GB memory) |
| **Inference Cost** | ~$0.00006/second |
| **Target Accuracy** | 99%+ on invoice extraction |

## 🧪 Data Sources

The model is trained on curated invoice datasets from HuggingFace Hub:

1. **mychen76/invoices-and-receipts_ocr_v1**
   - OCR text from real invoices
   - Structured JSON ground truth
   - Teaches model to handle OCR noise

2. **shubh303/Invoice-to-Json**
   - Question/answer format
   - Varied invoice layouts
   - Teaches model robustness

## 📝 License

See LICENSE file for details.

## 🙏 Acknowledgments

- **Qwen Team** for the Qwen3 model
- **Unsloth** for efficient fine-tuning framework
- **HuggingFace** for datasets and infrastructure

---

**Note**: This repository focuses solely on fine-tuning the SLM for invoice extraction. The fine-tuned model is intended to be used as a component in a larger invoice structuring and processing system.

High-Precision Invoice Extraction Pipeline (Hybrid Architecture)

1. Executive Summary

This project implements a hybrid AI pipeline for invoice extraction. Training occurs locally to leverage existing GPU resources, while inference is deployed to a serverless cloud environment to minimize operational costs. By fine-tuning the Qwen 3 0.6B model, we achieve high-accuracy extraction (target 99%) with a model small enough to run on the most cost-effective serverless tiers.

Component

Specification

Justification

Model

Qwen 3 0.6B-Instruct

State-of-the-art SLM (Small Language Model) released April 2025. High reasoning capability in <1GB size.

Training

Local GPU (16GB VRAM)

Free compute using local hardware. Unsloth QLoRA ensures fast convergence.

Deployment

SageMaker Serverless (3GB)

Zero idle cost. Scales to zero. Cost is approx. $0.00006/sec.

Ingestion

LangChain (PyPDF)

Standardizes input from diverse invoice formats.

2. System Architecture

graph TD
    subgraph Local Environment
    A[Raw SROIE Dataset] --> B[Unsloth Fine-Tuning Script]
    B --> C{Validation}
    C -->|Pass| D[Merge LoRA Adapters]
    D --> E[Save as model.tar.gz]
    end
    
    subgraph AWS Cloud
    E -->|Upload| F[S3 Bucket]
    F --> G[SageMaker Serverless Endpoint]
    end
    
    subgraph Inference Flow
    H[User Uploads PDF] --> I[LangChain Parser]
    I -->|Raw Text| G
    G -->|JSON Output| J[Pydantic Validator]
    J -->|Valid| K[Database]
    end


3. Phase 1: Data Ingestion (LangChain)

This script runs on the client-side (or lambda) to prepare the text before sending it to SageMaker.

Dependencies: langchain, pypdf, pdf2image, pytesseract

ingest.py

from langchain_community.document_loaders import PyPDFLoader
import pytesseract
from pdf2image import convert_from_path
import os

def parse_invoice(file_path):
    """
    Hybrid parser: Tries native text extraction first (fastest),
    falls back to OCR if the PDF is a scanned image.
    """
    # 1. Try Native Text Extraction (Fastest)
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        full_text = "\n".join([p.page_content for p in pages])
    except Exception as e:
        print(f"Native parse warning: {e}")
        full_text = ""
    
    # 2. OCR Fallback (If text is empty or too short)
    if len(full_text.strip()) < 50:
        print("⚠️ Scanned PDF detected. Switching to OCR...")
        try:
            images = convert_from_path(file_path)
            full_text = ""
            for img in images:
                full_text += pytesseract.image_to_string(img)
        except Exception as e:
            print(f"OCR failed: {e}")
            return None
            
    return full_text


4. Phase 2: Local Fine-Tuning (Unsloth)

Run this script on your local machine with the 16GB GPU. It fine-tunes the model and saves the artifacts in a format compatible with SageMaker.

Prerequisites: pip install unsloth "transformers>=4.40.0" trl

train_local_and_export.py

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch
import os
import shutil

# 1. Configuration
max_seq_length = 2048
dtype = None 
load_in_4bit = True # Train in 4-bit to save local memory

# 2. Load Qwen 3 0.6B
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3-0.6B-Instruct", 
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Apply LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
)

# 4. Formatting Function (ChatML)
alpaca_prompt = """<|im_start|>system
You are a strict invoice parser. Output strictly valid JSON.<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""

def formatting_prompts_func(examples):
    inputs = examples["text_input"]
    outputs = examples["json_output"]
    texts = []
    for input_text, output_text in zip(inputs, outputs):
        text = alpaca_prompt.format(input=input_text, output=output_text)
        texts.append(text)
    return { "text" : texts }

# 5. Load Dataset & Train
dataset = load_dataset("json", data_files="train.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit",
    ),
)

trainer.train()

# 6. Export for SageMaker (Critical Step)
# We merge LoRA into FP16 (standard precision) because SageMaker Serverless CPU 
# often struggles with 4-bit quantized inference without specific kernels.
export_dir = "qwen3_invoice_model"
print(f"Saving merged model to {export_dir}...")
model.save_pretrained_merged(export_dir, tokenizer, save_method = "merged_16bit")

# 7. Instructions for User
print(f"\n✅ Training Complete.")
print(f"Run this command in your terminal to prepare for upload:")
print(f"tar -czvf model.tar.gz -C {export_dir} .")


5. Phase 3: Cloud Deployment (SageMaker)

Once you have uploaded model.tar.gz to your S3 bucket, run this script to deploy.

Pricing Tier: We select 3072 MB (3 GB). This is sufficient for the 0.6B model (approx 1.5GB footprint) plus framework overhead.

deploy_sagemaker.py

import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serverless import ServerlessInferenceConfig

# Setup AWS Role
role = sagemaker.get_execution_role()
sess = sagemaker.Session()

# 1. Define Serverless Configuration
# 3072 MB = 3 GB RAM. Cost is approx $0.00006 per second.
serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=3072, 
    max_concurrency=5
)

# 2. Define Model from S3 Artifacts
huggingface_model = HuggingFaceModel(
    model_data='s3://your-bucket-name/model.tar.gz',  # <-- UPDATE THIS PATH
    role=role,
    transformers_version='4.37.0', 
    pytorch_version='2.1.0', 
    py_version='py310',
    env={
        'HF_MODEL_ID': 'Qwen/Qwen3-0.6B-Instruct', # Fallback ID if artifacts miss config
        'HF_TASK': 'text-generation',
        'SAGEMAKER_MODEL_SERVER_TIMEOUT': '3600',
        'SM_HP_MODEL_LOAD_IN_4BIT': 'False' # Force FP16/FP32 for CPU stability
    }
)

# 3. Deploy Endpoint
print("Deploying serverless endpoint... (This takes ~5 mins)")
predictor = huggingface_model.deploy(
    serverless_inference_config=serverless_config
)

print(f"Endpoint Deployed! Name: {predictor.endpoint_name}")


6. Phase 4: 99% Accuracy Protocol (Client)

This client script invokes the deployed SageMaker endpoint and validates the result.

pipeline_client.py

import boto3
import json
from pydantic import BaseModel, Field, validator
import datetime
from ingest import parse_invoice

# 1. SageMaker Runtime Client
client = boto3.client('sagemaker-runtime')
ENDPOINT_NAME = "huggingface-pytorch-inference-2025-..." # Update with your endpoint name

# 2. Define Strict Schema (The "Gatekeeper")
class InvoiceSchema(BaseModel):
    invoice_number: str = Field(..., description="Unique ID of invoice")
    total_amount: float = Field(..., description="Final total including tax")
    date: str = Field(..., description="YYYY-MM-DD format")
    vendor: str

    @validator('date')
    def validate_date_format(cls, v):
        try:
            datetime.datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError("Incorrect data format, should be YYYY-MM-DD")

def process_invoice(pdf_path):
    # A. Ingest
    print(f"Reading {pdf_path}...")
    raw_text = parse_invoice(pdf_path)
    if not raw_text: return "Failed to read PDF"

    # B. Prepare Prompt (ChatML format for Qwen)
    prompt = f"""<|im_start|>system
You are a strict invoice parser. Output strictly valid JSON.<|im_end|>
<|im_start|>user
{raw_text}<|im_end|>
<|im_start|>assistant
"""

    # C. Invoke SageMaker
    print("Invoking SageMaker Endpoint...")
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.1,
            "return_full_text": False
        }
    }

    response = client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    
    result_json = json.loads(response['Body'].read().decode())
    output_text = result_json[0]['generated_text']

    # D. Validation
    try:
        # Clean potential markdown fences if Qwen adds them
        clean_json = output_text.replace("```json", "").replace("```", "").strip()
        data_dict = json.loads(clean_json)
        
        # Pydantic Check
        invoice = InvoiceSchema(**data_dict)
        print("✅ SUCCESS: Valid Data Extracted")
        return invoice.json()
        
    except Exception as e:
        print(f"❌ FAILURE: Validation Error - {e}")
        return None

# Run
if __name__ == "__main__":
    result = process_invoice("test_invoice.pdf")
    print(result)

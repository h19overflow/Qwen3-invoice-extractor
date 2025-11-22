"""
Prompt templates for training and inference.

Role: Centralizes prompt formatting for consistency across pipeline.
"""

SYSTEM_PROMPT = "You are a strict invoice parser. Output strictly valid JSON."

CHATML_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{input}<|im_end|}
<|im_start|>assistant
{output}<|im_end|>"""


def format_training_example(input_text: str, output_text: str) -> str:
    """Format a single training example in ChatML format."""
    return CHATML_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        input=input_text,
        output=output_text
    )


def format_inference_prompt(input_text: str) -> str:
    """Format prompt for inference (no assistant response)."""
    return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
"""

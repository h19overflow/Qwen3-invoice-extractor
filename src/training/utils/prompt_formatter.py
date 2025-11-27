"""
Prompt formatting utilities for ChatML format.

Role: Centralizes prompt formatting for consistency across training and inference.
"""

# Enhanced system prompt with explicit schema definition
SYSTEM_PROMPT = """You are an invoice extraction AI. Extract invoice data into this exact JSON schema:

{"header": {"invoice_no": "string", "invoice_date": "string", "seller": "string", "client": "string", "seller_tax_id": "string", "client_tax_id": "string", "iban": "string"}, "items": [{"item_desc": "string", "item_qty": "string", "item_net_price": "string", "item_net_worth": "string", "item_vat": "string", "item_gross_worth": "string"}], "summary": {"total_net_worth": "string", "total_vat": "string", "total_gross_worth": "string"}}

Rules:
- Output ONLY valid JSON, no explanations
- Use exact field names from schema
- Keep all values as strings
- Include all items found"""

CHATML_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""


def format_training_example(input_text: str, output_text: str) -> str:
    """Format a single training example in ChatML format."""
    return CHATML_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        input=input_text,
        output=output_text
    )


def format_from_messages(messages: list[dict]) -> str:
    """
    Format ChatML messages list into training text.

    Expected format:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    """
    system_content = messages[0]["content"]
    user_content = messages[1]["content"]
    assistant_content = messages[2]["content"]

    return CHATML_TEMPLATE.format(
        system=system_content,
        input=user_content,
        output=assistant_content
    )


def format_inference_prompt(input_text: str) -> str:
    """Format prompt for inference (no assistant response)."""
    return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, pipeline

# Constants for model paths
ROBERTA_MODEL = "turalizada/AzBERTaContextualizedWordEmbeddingsinAzerbaijaniLanguage"
GPT2_MODEL = "turalizada/GPT2ContextualizedWordEmbeddinginAzerbaijaniLanguage"

def generate_text_roberta(sequence):
    """
    Generate text using the RoBERTa model trained for masked language modeling.
    Args:
        sequence (str): The text sequence with mask tokens for prediction.
    Returns:
        str: The text sequence with the mask filled.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
        model = AutoModelForMaskedLM.from_pretrained(ROBERTA_MODEL)
        fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
        sequence = sequence.replace("[MASK]", tokenizer.mask_token)  # Replace [MASK] with the correct mask token
        predictions = fill_mask(sequence)
        return predictions[0]['sequence'].replace(" " + tokenizer.mask_token + " ", tokenizer.mask_token).replace(tokenizer.mask_token, "")
    except Exception as e:
        raise Exception(f"Failed to generate text with mask with error: {str(e)}")

def generate_text_gpt_2(sequence):
    """
    Generate text using the GPT-2 model.
    Args:
        sequence (str): Input text sequence for generation.
    Returns:
        str: Generated text.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL)
        model = AutoModelForCausalLM.from_pretrained(GPT2_MODEL)
        inputs = tokenizer.encode(sequence, return_tensors='pt')
        outputs = model.generate(
            inputs,
            do_sample=True,
            max_length=100,
            temperature=0.8,
            pad_token_id=model.config.eos_token_id,
            top_k=100,
            top_p=0.95
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True).split('\n')[0]
    except Exception as e:
        raise Exception(f"Failed to generate text with GPT-2 with error: {str(e)}")

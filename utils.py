from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, pipeline

def load_model_and_tokenizer(model_name, model_type="RoBERTa"):
    """
    Loads a model and tokenizer from the Hugging Face Hub based on the model type.
    Args:
        model_name (str): The Hugging Face Hub identifier for the model.
        model_type (str): The type of model, e.g., "GPT2" or "RoBERTa".
    Returns:
        tokenizer: The loaded tokenizer.
        model: The loaded model.
    """
    try:
        if model_type == "GPT2":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:  # Default to RoBERTa
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        raise Exception(f"Failed to load model and tokenizer for {model_type} with error: {str(e)}")

def generate_text_with_mask(sequence, model_name):
    """
    Generate text using a model trained for masked language modeling.
    Args:
        sequence (str): The text sequence with mask tokens for prediction.
        model_name (str): The model identifier on Hugging Face Hub.
    Returns:
        str: The text sequence with the mask filled.
    """
    try:
        tokenizer, model = load_model_and_tokenizer(model_name, model_type="RoBERTa")
        fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
        sequence = sequence.replace("[MASK]", tokenizer.mask_token)  # Replace [MASK] with the correct mask token
        predictions = fill_mask(sequence)
        return predictions[0]['sequence'].replace(" " + tokenizer.mask_token + " ", tokenizer.mask_token).replace(tokenizer.mask_token, "")
    except Exception as e:
        raise Exception(f"Failed to generate text with mask with error: {str(e)}")

def generate_text_gpt_2(sequence, model_name):
    """
    Generate text using a GPT-2 model.
    Args:
        sequence (str): Input text sequence for generation.
        model_name (str): The model identifier on Hugging Face Hub.
    Returns:
        str: Generated text.
    """
    try:
        tokenizer, model = load_model_and_tokenizer(model_name, model_type="GPT2")
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

# Example usage:
model_gpt2 = "turalizada/GPT2ContextualizedWordEmbeddinginAzerbaijaniLanguage"
model_roberta = "turalizada/AzBERTaContextualizedWordEmbeddingsinAzerbaijaniLanguage"

sequence_roberta = "Here is an example sentence with a [MASK] to be predicted."
sequence_gpt2 = "Here is an example sentence to start generating text."

try:
    roberta_output = generate_text_with_mask(sequence_roberta, model_roberta)
    gpt2_output = generate_text_gpt_2(sequence_gpt2, model_gpt2)
    print("RoBERTa Output:", roberta_output)
    print("GPT-2 Output:", gpt2_output)
except Exception as e:
    print(f"An error occurred: {str(e)}")

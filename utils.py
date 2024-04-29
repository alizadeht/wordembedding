from transformers import GPT2LMHeadModel, GPT2Tokenizer

from transformers import RobertaForMaskedLM, RobertaTokenizer
from transformers import pipeline

def load_model(model_path, model_type="RoBERTa"):
    if model_type == "GPT2":
        model = GPT2LMHeadModel.from_pretrained(model_path)
    else:
        model = RobertaForMaskedLM.from_pretrained(model_path)
    return model

def load_tokenizer(tokenizer_path, model_type="RoBERTa"):
    if model_type == "GPT2":
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def generate_text_roberta(sequence, model_path="https://ikincibucket.s3.us-east-2.amazonaws.com/Models/AzBERTo", tokenizer_path="https://ikincibucket.s3.us-east-2.amazonaws.com/Models/AzBERTo", mask_token="<mask>"):
    model = load_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)

    fill_mask = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer
    )

    # Use the correct mask token from the tokenizer
    sequence = sequence.replace("[MASK]", tokenizer.mask_token)

    # Generate predictions
    predictions = fill_mask(sequence)

    # Get the most reliable prediction
    best_prediction = predictions[0]['sequence']  # The highest confidence prediction is always the first one

    # Optionally, you might want to inspect other high confidence predictions
    for i, prediction in enumerate(predictions):
        print(f"Rank {i+1} prediction: {prediction['sequence']} (Score: {prediction['score']})")

    # Clean the output
    clean_prediction = best_prediction.replace(" " + tokenizer.mask_token + " ", tokenizer.mask_token).replace(tokenizer.mask_token, "")
    
    return clean_prediction

def generate_text_gpt_2(sequence, model_path="/Users/turalalizada/Desktop/wordembedding/Models/GPT2", tokenizer_path="/Users/turalalizada/Desktop/wordembedding/Models/GPT2"):
    model = load_model(model_path, model_type="GPT2")
    tokenizer = load_tokenizer(tokenizer_path, model_type="GPT2")
    ids = tokenizer.encode(sequence, return_tensors='pt')

    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=100,
        temperature=0.8,
        pad_token_id=model.config.eos_token_id,
        top_k=100,
        top_p=0.95,
    )

    generated_text = tokenizer.decode(final_outputs[0], skip_special_tokens=True)
    return generated_text.split('\n')[0]

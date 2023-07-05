import os
import torch
import linecache
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GPT2LMHeadModel, 
    AutoModelForSeq2SeqLM, 
    GPT2Tokenizer
)

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = "cuda" if torch.cuda.is_available() else "cpu"
def main():
    # Load the pre-trained model and tokenizer
    """
    model_name: gpt2-large, facebook/xglm-1.7B
    """
    model_name = "facebook/xglm-1.7B" 
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    

    f_src, f_tgt = "../data/glue/train.source", "../data/glue/train.target"
    K = 8
    shot = []
    for k in range(1, K+2):
        shot.append((linecache.getline(f_src, k * 2), linecache.getline(f_tgt, k * 2)))
    
    prompt_template = "input: {} paraphrase: "
    input_text = ""
    for s in shot[:-1]:
        input_text += prompt_template.format(s[0])
        input_text += f"{s[1]} "
    input_text += prompt_template.format(shot[-1][0])
    batch = tokenizer(input_text, add_special_tokens=False, return_tensors="pt")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor): batch[k] = batch[k].to(device)
   
    # Generate output
    input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
    output_ids = model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        max_new_tokens=100 * K, 
        #min_length=20, 
        length_penalty=1.0, 
        #eos_token_id=tokenizer.eos_token_id, 
        #num_beams=5, 
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("output: ", output_text)
    print(f"expected: {shot[-1][1]}")
    import pdb 
    pdb.set_trace()
    


if __name__ == "__main__":
    main()
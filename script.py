from transformers import AutoTokenizer, pipeline
import json
import torch
from hf_ref import NewPhi3Config
from model import CustomedPhi3ForCausalLM
import time
import argparse

def load_data(file_path):
    model_inputs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line)
            model_inputs.append(json_obj)
    
    sorted_model_inputs = sorted(model_inputs, key=lambda inputs : len(inputs['message'][0]['content']))
    messages = [inputs['message'] for inputs in sorted_model_inputs]
    labels = [inputs['answer'] for inputs in sorted_model_inputs]
    return messages, labels

def main(file_path, base_path, batch_size, max_new_tokens):
    model_id = "microsoft/Phi-3-medium-4k-instruct"
    torch.random.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = NewPhi3Config(base_path=base_path)
    model = CustomedPhi3ForCausalLM(config)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size
    )

    generation_args = {
        "max_new_tokens": max_new_tokens,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    start = time.time()
    messages, labels = load_data(file_path= file_path)
    outs = pipe(messages, **generation_args)
    end = time.time()
    correct = 0
    for i, out in enumerate(outs):
        correct_answer = labels[i]
        answer = out[0]["generated_text"].lstrip().replace("\n","")
        if answer == correct_answer:
            correct += 1
        print(answer)

    print("===== Perf result =====")
    print("Elapsed_time: ", end-start)
    print(f"Correctness: {correct}/{len(labels)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--file_path', type=str, required=True, help="input JSONL file path를 넣어주세요")
    
    parser.add_argument('--base_path', type=str, required=True, help="model이 있는 경로를 넣어주세요 형식 : /home/base/path")
    
    parser.add_argument('--batch_size', type=int, required=True, default=100)
    parser.add_argument('--max_new_tokens', type=int, required=True, default=15)
    args = parser.parse_args()
    

    main(file_path=args.file_path, base_path=args.base_path,batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
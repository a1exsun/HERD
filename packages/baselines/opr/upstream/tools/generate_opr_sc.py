import argparse, json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from eval import overlong_filter


def calculate_avg_logprob(logprobs):
    total = 0.0
    count = 0
    for token_dict in logprobs:
        for logprob_obj in token_dict.values():
            total += logprob_obj.logprob
            count += 1
    return total / count if count > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="", type=str)
    parser.add_argument("--task-id", default=0, type=int, help="")
    parser.add_argument("--buffer-size", default=50, type=int)
    args = parser.parse_args()

    dataset_list = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path)

    replay_list = []

    for i in range(args.task_id + 1):

        with open(f"data/{dataset_list[i]}/train.jsonl") as f:
            
            data = [json.loads(line) for line in f]

            messages_list = [
                [
                    {
                        "role": "user",
                        "content": data[j]["prompt"]
                    },
                    {
                        "role": "assistant",
                        "content": data[j]["answer"]
                    }
                ]
                for j in range(len(data)) 
            ]
            
            overlong_index_list = overlong_filter(messages_list, tokenizer, 2048)

            query_list = [
                [
                    {
                        "role": "user",
                        "content": data[j]['prompt']
                    },
                ]
                for j in range(len(data)) if overlong_index_list[j]
            ]
        
        if args.task_id in [0, 1]:
            sampling_params = SamplingParams(temperature=0.1, max_tokens=1, logprobs=1)
        else:
            sampling_params = SamplingParams(temperature=0.1, max_tokens=512, logprobs=1)
        
        outputs = llm.chat(query_list, sampling_params)
        
        response_list = [output.outputs[0].text for output in outputs]
        
        log_prob_list = [calculate_avg_logprob(output.outputs[0].logprobs) for output in outputs]

        combined_list = [
            {
                "prompt": query_list[j][0]['content'],
                "answer": response_list[j],
                "logprob": log_prob_list[j]
            }
            for j in range(len(query_list))
        ]

        combined_list.sort(key=lambda x: x["logprob"], reverse=True)

        replay_list.extend(combined_list[:int(args.buffer_size / (args.task_id + 1))])

    if args.task_id < 7:
        output_file = f"data/{dataset_list[args.task_id + 1]}/buffer.jsonl"
        with open(output_file, "w") as f:
            for item in replay_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    

if __name__ == "__main__":
    main()
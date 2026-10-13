import argparse, json, re, os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from rouge_score import rouge_scorer
from fuzzywuzzy import fuzz
from evaluate import load


def overlong_filter(messages_list, tokenizer, max_input):
    text_list = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False) for messages in messages_list]
    len_list = [len(tokenizer(text).input_ids) for text in text_list]
    return [l <= max_input for l in len_list]


def eval_rougel(gold_answer_list, response_list):
    
    s = rouge_scorer.RougeScorer(['rougeL'])
    
    score_list = [
        s.score(g, r)['rougeL'].fmeasure * 100
        for g, r in zip(gold_answer_list, response_list)
    ]
    
    return sum(score_list) / len(score_list)



def eval_code(gold_answer_list, response_list):
    
    def postprocess(code):
        code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
        pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
        lits = re.findall(pattern, code)
        for lit in lits:
            code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
        return code
   
    r_l = [postprocess(resp) for resp in response_list]
    g_l = [postprocess(gt) for gt in gold_answer_list]
    
    score_list = [fuzz.ratio(r, g) for r, g in zip(r_l, g_l)]
    
    return sum(score_list) / len(score_list)


def eval_acc(gold_answer_list, response_list):

    score = 0

    for g, r in zip(gold_answer_list, response_list):
        if g[:1] == r[:1] and r != "":
            score += 1

    return (score / len(gold_answer_list)) * 100


def eval_math(gold_answer_list, response_list):

    score = 0

    for g, r in zip(gold_answer_list, response_list):

        answer = re.findall("(\\-?[0-9\\.\\,]+)", r)
        
        if len(answer) == 0:
            final_answer = None
        else:
            invalid_str = ["", "."]
            
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break

        if final_answer == g:
            score += 1
    
    return (score / len(gold_answer_list)) * 100


def eval_sari(gold_answer_list, response_list, simple_query_list):
    sari = load("sari")

    sources=[q.split('Paragraph:\n')[1].split('\n\nSimplification:')[0] for q in simple_query_list]

    predictions=response_list

    references=[[gold_answer] for gold_answer in gold_answer_list]

    sari_score = sari.compute(sources=sources, predictions=predictions, references=references)

    return sari_score['sari']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="baseline/replay/Qwen2.5/v0-20260128-014058/checkpoint-785", type=str)
    parser.add_argument("--task-id", default=0, type=int, help="")
    parser.add_argument("--temperature", default=0.1, type=float, help="")
    parser.add_argument("--max-input", default=2048, type=int, help="")
    parser.add_argument("--test-time", default=8, type=int, help="")
    args = parser.parse_args()

    dataset_list = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=8)

    conclusion = {'model_path': args.model_path}

    for i in range(args.task_id + 1):

        with open(f'data/{dataset_list[i]}/test.jsonl') as f:
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

            overlong_index_list = overlong_filter(messages_list, tokenizer, args.max_input)  # 获取超长prompt索引

            query_list = [
                [
                    {
                        "role": "user",
                        "content": data[j]["prompt"]
                    },
                ]
                for j in range(len(data)) if overlong_index_list[j]
            ]

            simple_query_list = [
                data[j]["prompt"]
                for j in range(len(data)) if overlong_index_list[j]
            ]

            gold_answer_list = [
                data[j]["answer"]
                for j in range(len(data)) if overlong_index_list[j]
            ]

        if args.task_id in [0, 1]:
            sampling_params = SamplingParams(temperature=args.temperature, max_tokens=1, logprobs=1)
        else:
            sampling_params = SamplingParams(temperature=args.temperature, max_tokens=512, logprobs=1)

        score_list = []

        for j in range(args.test_time):

            outputs = llm.chat(query_list, sampling_params)
            response_list = [output.outputs[0].text for output in outputs]

            if i in [0, 1, 4]:
                score_list.append(eval_acc(gold_answer_list, response_list))
            elif i == 2:
                score_list.append(eval_rougel(gold_answer_list, response_list))
            elif i == 3:
                score_list.append(eval_code(gold_answer_list, response_list))
            elif i in [5, 6]:
                score_list.append(eval_math(gold_answer_list, response_list))
            elif i == 7:
                score_list.append(eval_sari(gold_answer_list, response_list, simple_query_list))

        conclusion[dataset_list[i]] = sum(score_list) / len(score_list)

    with open(f"{os.path.dirname(args.model_path)}/conclusion.json", "w") as f:
        json.dump(conclusion, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
import argparse, json, re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from eval import overlong_filter
from rouge_score import rouge_scorer
from fuzzywuzzy import fuzz
from evaluate import load


def score_acc(gold, response):                                                                                                                                     
    if not response:                                                                                                                                               
        return 0                                                                                                                                                   
    return 100 if gold[:1] == response[:1] else 0


def score_rougel(gold, response):
    s = rouge_scorer.RougeScorer(['rougeL'])
    return s.score(gold, response)['rougeL'].fmeasure * 100


def score_code(gold, response):
    def postprocess(code):
        code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
        pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
        lits = re.findall(pattern, code)
        for lit in lits:
            code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
        return code
    return fuzz.ratio(postprocess(response), postprocess(gold))


def score_math(gold, response):
    answer = re.findall("(\\-?[0-9\\.\\,]+)", response)
    if len(answer) == 0:
        final_answer = None
    else:
        invalid_str = ["", "."]
        for final_answer in reversed(answer):
            if final_answer not in invalid_str:
                break
    return 100 if final_answer == gold else 0


def score_sari_batch(gold_list, response_list, simple_query_list):
    sari = load("/cpfs01/shared/public/users/randy.chen/on-policy-replay-v2/evaluate-0.4.6/metrics/sari/sari.py")
    score_list = []
    for g, r, q in zip(gold_list, response_list, simple_query_list):
        source = q.split('Paragraph:\n')[1].split('\n\nSimplification:')[0]
        result = sari.compute(sources=[source], predictions=[r], references=[[g]])
        score_list.append(result['sari'])
    return score_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="", type=str)
    parser.add_argument("--task-id", default=0, type=int, help="")
    parser.add_argument("--buffer-size", default=50, type=int)
    args = parser.parse_args()
    dataset_list = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=8)

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

            simple_query_list = [
                data[j]["prompt"]
                for j in range(len(data)) if overlong_index_list[j]
            ]

            gold_answer_list = [
                data[j]["answer"]
                for j in range(len(data)) if overlong_index_list[j]
            ]

        if args.task_id in [0, 1]:
            sampling_params = SamplingParams(temperature=0.1, max_tokens=1)
        else:
            sampling_params = SamplingParams(temperature=0.1, max_tokens=512)

        outputs = llm.chat(query_list, sampling_params)

        response_list = [output.outputs[0].text for output in outputs]

        if i in [0, 1, 4]:
            score_list = [score_acc(g, r) for g, r in zip(gold_answer_list, response_list)]
        elif i == 2:
            score_list = [score_rougel(g, r) for g, r in zip(gold_answer_list, response_list)]
        elif i == 3:
            score_list = [score_code(g, r) for g, r in zip(gold_answer_list, response_list)]
        elif i in [5, 6]:
            score_list = [score_math(g, r) for g, r in zip(gold_answer_list, response_list)]
        elif i == 7:
            score_list = score_sari_batch(gold_answer_list, response_list, simple_query_list)

        combined_list = [
            {
                "prompt": query_list[j][0]['content'],
                "answer": response_list[j],
                "score": score_list[j]
            }
            for j in range(len(query_list))
        ]

        combined_list.sort(key=lambda x: x["score"], reverse=True)

        replay_list.extend(combined_list[:int(args.buffer_size / (args.task_id + 1))])

    if args.task_id < 7:
        output_file = f"data/{dataset_list[args.task_id + 1]}/buffer.jsonl"
        with open(output_file, "w") as f:
            for item in replay_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
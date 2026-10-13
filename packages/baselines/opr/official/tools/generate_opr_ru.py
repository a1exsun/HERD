import argparse, json, re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from eval import overlong_filter
from rouge_score import rouge_scorer
from fuzzywuzzy import fuzz
from evaluate import load
from hero_evaluation.scoring import score_predictions  # [HERO] shared TRACE+17 scorer


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
    sari = load("sari")  # [HERO] was a local copy of evaluate-0.4.6's sari.py
    score_list = []
    for g, r, q in zip(gold_list, response_list, simple_query_list):
        source = q.split('Paragraph:\n')[1].split('\n\nSimplification:')[0]
        result = sari.compute(sources=[source], predictions=[r], references=[[g]])
        score_list.append(result['sari'])
    return score_list


def score_shared(name, keep_list, response_list):
    # [HERO] TRACE+17's SuperNI tasks have no upstream scorer: the shared evaluator's item score for the task's
    # metric, against data/{name}/gold.jsonl (the gold records of train.jsonl, in the same order).
    with open(f"data/{name}/gold.jsonl") as f:
        gold = [json.loads(line) for line in f]
    gold = [g for g, keep in zip(gold, keep_list) if keep]
    predictions = [{"id": g["id"], "prediction": r} for g, r in zip(gold, response_list)]
    return [item["score"] for item in score_predictions(gold[0]["metric"], predictions, gold)["itemScores"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="", type=str)
    parser.add_argument("--task-id", default=0, type=int, help="")
    parser.add_argument("--buffer-size", default=50, type=int)
    args = parser.parse_args()
    # [HERO] TRACE+17: the eight TRACE tasks in upstream order, then 17 SuperNI tasks
    dataset_list = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten",
                    "superni-024", "superni-074", "superni-195", "superni-511", "superni-1219", "superni-574",
                    "superni-192", "superni-177", "superni-064", "superni-346", "superni-108", "superni-111",
                    "superni-1415", "superni-1447", "superni-828", "superni-1728", "superni-217"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # [HERO] one GPU; context for a 16,384-token input plus 512 new tokens; no image/video inputs
    llm = LLM(model=args.model_path, tensor_parallel_size=1, max_model_len=16384 + 512,
              limit_mm_per_prompt={"image": 0, "video": 0})

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

            overlong_index_list = overlong_filter(messages_list, tokenizer, 16384)  # [HERO] protocol ceiling, was 2048

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

        outputs = llm.chat(query_list, sampling_params,
                           chat_template_kwargs={"enable_thinking": False})  # [HERO] protocol: thinking disabled

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
        else:  # [HERO] SuperNI tasks
            score_list = score_shared(dataset_list[i], overlong_index_list, response_list)

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

    if args.task_id < 24:  # [HERO] was 7: the last of 25 tasks builds no buffer
        output_file = f"data/{dataset_list[args.task_id + 1]}/buffer.jsonl"
        with open(output_file, "w") as f:
            for item in replay_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
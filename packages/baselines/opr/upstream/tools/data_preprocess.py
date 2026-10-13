import json, os, shutil


part_list = ['eval', 'test', 'train']
dataset_list = ['C-STANCE', 'FOMC', 'MeetingBank', 'Py150', 'ScienceQA', 'NumGLUE-cm', 'NumGLUE-ds', '20Minuten']

for d in dataset_list:
    for p in part_list:
        
        file_path = f'data/TRACE-Benchmark/LLM-CL-Benchmark_5000/{d}/{p}.json'
        with open(file_path, encoding='utf-8') as f:
            data = json.load(f)
        
        new_file_path = f'data/{d}/{p}.jsonl'
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        with open(new_file_path, 'w', encoding='utf-8') as f:
            for dialog in data:
                f.write(
                    json.dumps(
                        dialog,
                        ensure_ascii=False,
                    ) + '\n'
                )

shutil.rmtree('data/TRACE-Benchmark')
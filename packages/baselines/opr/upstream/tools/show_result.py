import os, argparse, json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output/qwen_opr_ru", type=str)
    args = parser.parse_args()

    task_list = [ f'{args.output_dir}/{name}' for name in os.listdir(args.output_dir)]
    task_list.sort()

    score_matrix = []

    dataset_list = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]


    for i, t in enumerate(task_list):

        score_list = []
       
        with open(f'{t}/conclusion.json') as f:
            conclusion = json.load(f)
        
        print(f'========== Task {i} ==========')

        for d in dataset_list:

            if conclusion.get(d) != None:

                print(f'{d}: {round(conclusion[d], 2)}')
                score_list.append(conclusion[d])

        score_matrix.append(score_list)

    print(f'========== Final Result ==========')

    overall_accuracy, bwt = [], []

    for i, num in enumerate(score_matrix[-1]):

        overall_accuracy.append(num)

        bwt.append(num - score_matrix[i][i])
    
    overall_accuracy = sum(overall_accuracy) / len(overall_accuracy)

    print(f'overall accuracy: {round(overall_accuracy, 2)}')

    bwt = sum(bwt) / len(bwt)
    
    print(f'backward transfer: {round(bwt, 2)}')


if __name__ == "__main__":
    main()
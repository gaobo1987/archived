if __name__ == '__main__':
    import json
    import numpy as np
    filename = 'qa_BM25_wikipedia_100_stride_50__1000_DPR_20__electra-base-squad2__squad-1000-formatted_1000.json'
    filename = 'qa_BM25_wikipedia_100_stride_50__1000_DPR_20__electra-base-squad2__nq-1000-formatted_1000.json'
    with open(filename, 'r') as f:
        items = json.load(f)
        print(len(items))
        count = 0
        f1s = []
        ems = []
        for item in items:
            if item['pred_answers'] and item['gold_answers']:
                count += 1
                f1s.append(item['f1'])
                ems.append(item['em'])

        print(count)
        print('f1 mean, std: ', np.mean(f1s), np.std(f1s))
        print('em mean, std: ', np.mean(ems), np.std(ems))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
from sklearn.metrics import classification_report
from tqdm import tqdm
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('--model')
    ap.add_argument('--tokenizer')
    ap.add_argument('--save')
    ap.add_argument('--eval_on', choices=['dev', 'test'], default='test')

    args = ap.parse_args()

    PATH2MODEL = args.model
    PATH2TOKENIZER = args.tokenizer

    assert not (PATH2MODEL is None or PATH2TOKENIZER is None), 'provide path to model/tokenizer'

    print(f'loading {args.eval_on} split')

    if args.eval_on == 'dev':
        ds = load_dataset("syedkhalid076/Sentiment-Analysis", split='validation')
    else:
        ds = load_dataset("syedkhalid076/Sentiment-Analysis", split='test')
    print('loaded')

    data_loader = DataLoader(ds, batch_size=64)

    id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
    label2id = {l: _id_ for _id_, l in id2label.items()}

    print('loading saved model and tokenizer')
    model = AutoModelForSequenceClassification.from_pretrained(PATH2MODEL).to(device)
    tokenizer = AutoTokenizer.from_pretrained(PATH2TOKENIZER)
    print('loaded')

    clf = pipeline('text-classification', model=model, tokenizer=tokenizer)

    Y_truth, Y_pred = [], []
    for batch in tqdm(data_loader, desc='Predictions'):
        texts, labels = batch['text'], batch['label']

        # [ {'label': 'Neutral', 'score': 0.9909084439277649},  {'label': 'Positive', 'score': 0.9909084439277649}]

        predictions_raw = clf(texts, batch_size=64)
        Y_pred.extend([label2id[p['label']] for p in predictions_raw])

        Y_truth.extend([label.item() for label in labels])

    assert len(Y_pred) == len(Y_truth)

    n = len(Y_truth)
    # micro precision, recall, f1
    tp = 0
    fp0, fp1, fp2 = 0, 0, 0
    fn0, fn1, fn2 = 0, 0, 0
    for i in range(n):
        if Y_truth[i] == 0:
            if Y_pred[i] == 0:
                tp += 1
            elif Y_pred[i] == 1:
                fn0 += 1
                fp1 += 1
            else:
                fn0 += 1
                fp2 += 1
        elif Y_truth[i] == 1:
            if Y_pred[i] == 1:
                tp += 1
            elif Y_pred[i] == 0:
                fn1 += 1
                fp0 += 1
            else:
                fn1 += 1
                fp2 += 1
        else:
            if Y_pred[i] == 2:
                tp += 1
            elif Y_pred[i] == 0:
                fn2 += 1
                fp0 += 1
            else:
                fn2 += 1
                fp1 += 1

    print(f'tp: {tp}, fp: {fp0 + fp1 + fp2}, fn: {fn0 + fn1 + fn2}')

    p_micro = tp / (tp + fp0 + fp1 + fp2)
    r_micro = tp / (tp + fn0 + fn1 + fn2)
    f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro)

    # print, save to file
    with open(args.save, 'wt', encoding='utf8') as fw:
        print(classification_report(Y_truth, Y_pred, digits=4))
        print(
            f'P-micro: {round(p_micro, 4) * 100}%, R-micro: {round(r_micro, 4) * 100}%, F1-micro: {round(f1_micro, 4) * 100}%')
        print(classification_report(Y_truth, Y_pred, digits=4), file=fw)
        print(
            f'P-micro: {round(p_micro, 4) * 100}%, R-micro: {round(r_micro, 4) * 100}%, F1-micro: {round(f1_micro, 4) * 100}%',
            file=fw)
    print(f'report saved to {args.save}')       

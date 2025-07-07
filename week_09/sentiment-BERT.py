from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name_or_path')
    ap.add_argument('--max_length', type=int, default=64)
    ap.add_argument('--output_dir')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=5e-5)

    args = ap.parse_args()

    ds = load_dataset("syedkhalid076/Sentiment-Analysis")

    print(ds)

    id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
    label2id = {l: _id_ for _id_, l in id2label.items()}

    print(f'labels2id: {label2id}')

    print('datasets format')
    print(ds['train'][0], ds['validation'][0], sep='\n')

    model_name = args.model_name_or_path

    try:
        # ModernBERT
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id),
                                                                   label2id=label2id,
                                                                   id2label=id2label)
        model.drop.p = 0.3
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id),
                                                                   label2id=label2id,
                                                                   id2label=id2label, hidden_dropout_prob=0.2,
                                                                   attention_probs_dropout_prob=0.2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    def tokenize(batch):
        inputs = tokenizer(batch['text'], truncation=True, padding='max_length', max_length=args.max_length,
                           return_tensors='pt')
        inputs['labels'] = batch['label']

        return inputs


    print('tokenizing data')
    ds_tokenized = ds.map(tokenize, batched=True)

    print(ds_tokenized)


    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=-1)

        assert len(predictions) == len(labels)

        predictions, labels = list(predictions), list(labels)

        print(classification_report(labels, predictions))

        p_micro = precision_score(labels, predictions, average='micro')
        r_micro = recall_score(labels, predictions, average='micro')
        f1_micro = f1_score(labels, predictions, average='micro')

        return {'P-micro': p_micro, 'R-micro': r_micro, 'F1-micro': f1_micro}


    train_args = TrainingArguments(output_dir=args.output_dir,
                                   num_train_epochs=args.epochs, learning_rate=args.lr,
                                   per_device_train_batch_size=args.batch_size,
                                   per_device_eval_batch_size=args.batch_size,
                                   eval_strategy='epoch', save_strategy='epoch',
                                   weight_decay=0.01,
                                   warmup_ratio=0.1,
                                   report_to='none',
                                   greater_is_better=True,
                                   load_best_model_at_end=True,
                                   metric_for_best_model='F1-micro')

    trainer = Trainer(model=model, processing_class=tokenizer, args=train_args, train_dataset=ds_tokenized['train'],
                      eval_dataset=ds_tokenized['validation'], compute_metrics=compute_metrics)

    trainer.train()

    model.save_pretrained(f'{args.output_dir}/model')
    tokenizer.save_pretrained(f'{args.output_dir}/tokenizer')

    print(f'model and tokenizer saved to {args.output_dir}')

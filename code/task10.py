import numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, pipeline)
from sklearn.metrics import accuracy_score, f1_score

ds = load_dataset("stanfordnlp/imdb")
split = ds["train"].train_test_split(test_size=0.1, seed=42)
train_ds, val_ds, test_ds = split["train"], split["test"], ds["test"]
print("train/val/test:", len(train_ds), len(val_ds), len(test_ds))

# Для быстрого прогона (около 3 минут) раскомментируйте:
# train_ds = train_ds.shuffle(seed=42).select(range(5000))
# val_ds = val_ds.select(range(1000))
# test_ds = test_ds.shuffle(seed=42).select(range(5000))

name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(name)
tok = lambda b: tokenizer(b["text"], truncation=True, max_length=256)
train_t = train_ds.map(tok, batched=True)
val_t = val_ds.map(tok, batched=True)
test_t = test_ds.map(tok, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    name, num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"}, label2id={"NEGATIVE": 0, "POSITIVE": 1})

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds), "f1": f1_score(p.label_ids, preds)}

args = TrainingArguments(
    output_dir="./distilbert-imdb", learning_rate=2e-5,
    per_device_train_batch_size=16, per_device_eval_batch_size=32,
    num_train_epochs=2, weight_decay=0.01,
    eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="f1",
    fp16=torch.cuda.is_available(), report_to="none")

trainer = Trainer(model=model, args=args, train_dataset=train_t, eval_dataset=val_t,
                  data_collator=DataCollatorWithPadding(tokenizer),
                  compute_metrics=compute_metrics)
trainer.train()

val = trainer.evaluate(val_t)
test = trainer.evaluate(test_t)

clf = pipeline("text-classification", model=trainer.model, tokenizer=tokenizer,
               device=0 if torch.cuda.is_available() else -1)
samples = [
    "A masterpiece! I was moved to tears by the final scene.",
    "Total garbage. The script is idiotic and the acting is wooden.",
    "Not bad, but I expected more from this director.",
    "I can't say I didn't enjoy it.",
    "Great, another sequel nobody asked for. Just what the world needed.",
]
with open("out/task10.txt", "w", encoding="utf-8") as f:
    f.write(f"train/val/test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}\n")
    f.write(f"VALIDATION: {val}\nTEST: {test}\n\nINFERENCE:\n")
    for s in samples:
        f.write(f"{s} -> {clf(s)}\n")
print(open("out/task10.txt", encoding="utf-8").read())

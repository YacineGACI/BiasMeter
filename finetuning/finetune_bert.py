from transformers import TrainingArguments
from adaptnlp import LMFineTuner

train_file = "corpora/SBIC.v2/SBIC.v2.trn.txt"
eval_file =  "corpora/SBIC.v2/SBIC.v2.dev.txt"

training_args = TrainingArguments(
    output_dir='finetuning/models/',
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    evaluate_during_training=True,
    logging_dir='finetuning/logs/',
    save_steps=2500,
    eval_steps=100,
    no_cuda=True
)

finetuner = LMFineTuner(model_name_or_path="bert-base-uncased")

finetuner.train(
    training_args=training_args,
    train_file=train_file,
    eval_file=eval_file,
    line_by_line=True,
    mlm=True,
    overwrite_cache=False
)

finetuner.evaluate()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

def classic_ml(train_dataset, test_dataset):
    train_df = train_dataset.to_pandas()
    test_df = test_dataset.to_pandas()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=19000)
    X_train = vectorizer.fit_transform(train_df["message_clean"])
    X_test = vectorizer.transform(test_df["message_clean"])
    y_train = train_df["is_toxic"]
    y_test = test_df["is_toxic"]

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=kf, scoring='f1_weighted', n_jobs=-1)

    print(f"10-fold cross-validation F1-scores: {cv_scores}")
    print(f"Mean CV F1-score: {np.mean(cv_scores)}")

    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-toxic", "Toxic"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig("toxic_clf/models/confusion_matrix.png")
    plt.close()

    return

def roberta(train_dataset, test_dataset):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def tokenize_function(examples):
        result = tokenizer(
            examples["message_clean"],
            padding="max_length",
            truncation=True,
            max_length=64 # с этим можно поиграться
        )
        result["labels"] = examples["is_toxic"]
        return result

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./roberta_toxicity",
        num_train_epochs=2,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        learning_rate=5e-5,
        metric_for_best_model="f1",
        save_total_limit=2
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    model.save_pretrained("toxic_clf/models/roberta-pretrainded")
    tokenizer.save_pretrained("toxic_clf/models/roberta-pretrainded")

    return


def classifier(train_dataset, test_dataset, model):
    if (model == "classic_ml"):
        classic_ml(train_dataset, test_dataset)
    elif (model == "roberta"):
        roberta(train_dataset, test_dataset)
    else:
        raise ValueError(f"Invalid model parameter: {model}")

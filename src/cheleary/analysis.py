import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def analyze(in_path, out_path):
    df = pd.read_csv(in_path)
    chemicals = df.iloc[0::3, :]
    prediction = df.iloc[1::3, :].astype("float").reset_index(drop=True)
    label = df.iloc[2::3, :].astype("float").reset_index(drop=True)
    threshold = 0.05
    label_bool = label.applymap(lambda x: x > (1 - threshold))
    prediction_bool = prediction.applymap(lambda x: x > (1 - threshold))
    results = pd.DataFrame(
        [
            dict(
                cls=c,
                precision=precision_score(
                    label_bool.iloc[:, i], prediction_bool.iloc[:, i]
                ),
                recall=recall_score(label_bool.iloc[:, i], prediction_bool.iloc[:, i]),
                f1=f1_score(label_bool.iloc[:, i], prediction_bool.iloc[:, i]),
            )
            for i, c in enumerate(label.columns.values)
        ]
    )
    results.to_csv(out_path)

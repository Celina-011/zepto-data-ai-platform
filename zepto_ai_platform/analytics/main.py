
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

OUT = Path(__file__).parent
sns.set_theme(style="whitegrid")


def load_once():
    df = sns.load_dataset("titanic")
    df.to_csv(OUT / "titanic.csv", index=False)
    return df


def clean_for_eda(df):
    out = df.copy()
    missing = out.isna().mean() * 100
    print("Missing percentages:\n", missing[missing > 0])

    for col in missing.index:
        pct = missing[col]
        if pct == 0:
            continue
        if pct < 5:
            out = out.dropna(subset=[col])
        elif pct <= 30:
            if pd.api.types.is_numeric_dtype(out[col]):
                out[col] = out[col].fillna(out[col].median())
            else:
                out[col] = out[col].fillna(out[col].mode()[0])
        else:
            out[col] = out[col].fillna("missing")

    return out


def eda(df):
    print(df.info())
    print(df.describe(include="all"))
    print("Shape:", df.shape)

    for col in ["age", "fare"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(df[col], kde=True, ax=axes[0])
        axes[0].set_title(f"{col} histogram")
        sns.boxplot(x=df[col], ax=axes[1])
        axes[1].set_title(f"{col} box plot")
        fig.tight_layout()
        fig.savefig(OUT / f"{col}_distribution.png")
        plt.close(fig)

        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        count = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
        print(f"{col} IQR outliers:", count)

    print("Fare mean:", df["fare"].mean())
    print("Fare median:", df["fare"].median())
    print("Fare mode:", df["fare"].mode().iloc[0])

    print("\nSurvival by sex:")
    print(df.groupby("sex")["survived"].mean())
    print("\nSurvival by pclass:")
    print(df.groupby("pclass")["survived"].mean())
    print("\nSurvival by sex+pclass:")
    print(df.groupby(["sex", "pclass"])["survived"].mean())

    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = df[corr_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Titanic numeric correlation matrix")
    plt.tight_layout()
    plt.savefig(OUT / "correlation_heatmap.png")
    plt.close()

    pairs = []
    for i, a in enumerate(corr_cols):
        for b in corr_cols[i + 1:]:
            pairs.append((abs(corr.loc[a, b]), a, b, corr.loc[a, b]))
    print("Two strongest correlations:")
    print(sorted(pairs, reverse=True)[:2])

    # Four multivariate charts.
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="sex", y="survived", hue="pclass")
    plt.title("Survival rate by sex and class")
    plt.tight_layout()
    plt.savefig(OUT / "survival_sex_class.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="pclass", y="fare", hue="survived")
    plt.title("Fare distribution by class and survival")
    plt.tight_layout()
    plt.savefig(OUT / "fare_class_survival.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="age", y="fare", hue="survived", alpha=0.6)
    plt.title("Age vs fare by survival")
    plt.tight_layout()
    plt.savefig(OUT / "age_fare_survival.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="pclass", hue="survived")
    plt.title("Passenger counts by class and survival")
    plt.tight_layout()
    plt.savefig(OUT / "class_survival_counts.png")
    plt.close()

    # EDA-only standardization check.
    for col in ["age", "fare"]:
        z = (df[col] - df[col].mean()) / df[col].std()
        print(f"{col} standardized mean/std:", z.mean(), z.std())


def modeling(df):
    target = "survived"
    features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    X = df[features].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    numeric = ["pclass", "age", "sibsp", "parch", "fare"]
    categorical = ["sex", "embarked"]

    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
    }

    comparison = []
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]

        comparison.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "auc": roc_auc_score(y_test, prob),
        })

        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name} confusion matrix")
        plt.tight_layout()
        plt.savefig(OUT / f"{name.lower().replace(' ', '_')}_confusion.png")
        plt.close()

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(OUT / "model_comparison.csv", index=False)
    print(comparison_df)

    # ROC curves.
    plt.figure(figsize=(8, 6))
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, prob):.3f})")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "roc_comparison.png")
    plt.close()

    # Decision tree visualization.
    tree_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(random_state=42, max_depth=4))
    ])
    tree_pipe.fit(X_train, y_train)
    feature_names = tree_pipe.named_steps["preprocessor"].get_feature_names_out()
    plt.figure(figsize=(20, 10))
    plot_tree(
        tree_pipe.named_steps["model"],
        feature_names=feature_names,
        class_names=["not survived", "survived"],
        filled=True,
        max_depth=4,
    )
    plt.tight_layout()
    plt.savefig(OUT / "decision_tree.png")
    plt.close()

    # Imbalance comparison using logistic regression.
    baseline = Pipeline([("preprocessor", preprocessor),
                         ("model", LogisticRegression(max_iter=2000))])
    balanced = Pipeline([("preprocessor", preprocessor),
                         ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])

    for label, pipe in [("baseline", baseline), ("balanced", balanced)]:
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        print(label, precision_score(y_test, pred), recall_score(y_test, pred), f1_score(y_test, pred))

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_train_t, y_train)
    smote_model = LogisticRegression(max_iter=2000)
    smote_model.fit(X_smote, y_smote)
    smote_pred = smote_model.predict(X_test_t)
    print("SMOTE", precision_score(y_test, smote_pred),
          recall_score(y_test, smote_pred), f1_score(y_test, smote_pred))

    # Random Forest GridSearch.
    rf_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=42, oob_score=True))
    ])
    grid = GridSearchCV(
        rf_pipe,
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
            "model__max_features": ["sqrt", "log2"],
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print("Best RF parameters:", grid.best_params_)
    print("Best CV score:", grid.best_score_)
    print("OOB score:", grid.best_estimator_.named_steps["model"].oob_score_)

    # Regression side-task.
    reg_features = ["survived", "pclass", "age", "sibsp", "parch"]
    reg_df = df[reg_features + ["fare"]].dropna()
    Xr = reg_df[reg_features]
    yr = reg_df["fare"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )
    reg = LinearRegression()
    reg.fit(Xr_train, yr_train)
    rp = reg.predict(Xr_test)
    mae = mean_absolute_error(yr_test, rp)
    rmse = np.sqrt(mean_squared_error(yr_test, rp))
    r2 = r2_score(yr_test, rp)
    n, p = Xr_test.shape
    adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)
    print("Regression MAE/RMSE/R2/Adjusted R2:", mae, rmse, r2, adj_r2)

    residuals = yr_test - rp
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=rp, y=residuals)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted fare")
    plt.ylabel("Residual")
    plt.title("Fare residual plot")
    plt.tight_layout()
    plt.savefig(OUT / "fare_residuals.png")
    plt.close()

    # Save complete fitted pipeline.
    full_pipeline = grid.best_estimator_
    joblib.dump(full_pipeline, OUT / "best_pipeline.joblib")

    loaded = joblib.load(OUT / "best_pipeline.joblib")
    sample_prediction = loaded.predict(X_test.head(1))
    print("Reloaded pipeline prediction:", sample_prediction)


def main():
    df = load_once()
    cleaned = clean_for_eda(df)
    eda(cleaned)
    modeling(cleaned)


if __name__ == "__main__":
    main()

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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
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
    """
    Load the Titanic dataset once and save an offline copy.
    """
    df = sns.load_dataset("titanic")
    df.to_csv(OUT / "titanic.csv", index=False)
    return df


def clean_for_eda(df):
    """
    Apply the capstone missing-value thresholds:
    <5%      -> drop rows
    5-30%    -> impute
    >30%     -> encode missing values explicitly
    """
    out = df.copy()

    missing = out.isna().mean() * 100

    print("Missing percentages:")
    print(missing[missing > 0])

    decisions = []

    for col in missing.index:
        pct = missing[col]

        if pct == 0:
            continue

        if pct < 5:
            out = out.dropna(subset=[col])
            decision = f"{col}: {pct:.2f}% missing -> dropped affected rows because missingness is below 5%."

        elif pct <= 30:
            if pd.api.types.is_numeric_dtype(out[col]):
                median_value = out[col].median()
                out[col] = out[col].fillna(median_value)
                decision = (
                    f"{col}: {pct:.2f}% missing -> median imputation "
                    f"using {median_value:.4f}."
                )
            else:
                mode_value = out[col].mode()[0]
                out[col] = out[col].fillna(mode_value)
                decision = (
                    f"{col}: {pct:.2f}% missing -> mode imputation "
                    f"using '{mode_value}'."
                )

        else:
            out[col] = out[col].astype("object").fillna("missing")
            decision = (
                f"{col}: {pct:.2f}% missing -> encoded missing values "
                f"as the explicit category 'missing' because missingness exceeds 30%."
            )

        decisions.append(decision)

    with open(OUT / "eda_summary.txt", "w", encoding="utf-8") as file:
        file.write("MISSING-VALUE DECISIONS\n")
        file.write("=" * 80 + "\n\n")

        for decision in decisions:
            file.write(decision + "\n")

    return out


def eda(df):
    print(df.info())
    print(df.describe(include="all"))
    print("Shape:", df.shape)

    # ---------------------------------------------------------
    # Age and fare distributions + IQR outliers
    # ---------------------------------------------------------
    outlier_results = {}

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

        count = (
            (df[col] < q1 - 1.5 * iqr)
            | (df[col] > q3 + 1.5 * iqr)
        ).sum()

        outlier_results[col] = int(count)

        print(f"{col} IQR outliers:", count)

    # ---------------------------------------------------------
    # Fare statistics + skewness
    # ---------------------------------------------------------
    fare_mean = df["fare"].mean()
    fare_median = df["fare"].median()
    fare_mode = df["fare"].mode().iloc[0]
    fare_skewness = df["fare"].skew()

    print("Fare mean:", fare_mean)
    print("Fare median:", fare_median)
    print("Fare mode:", fare_mode)
    print("Fare skewness:", fare_skewness)

    if fare_skewness > 0:
        skew_interpretation = (
            "Fare is positively/right skewed, meaning a relatively small "
            "number of passengers paid substantially higher fares."
        )
    elif fare_skewness < 0:
        skew_interpretation = (
            "Fare is negatively/left skewed, meaning lower values have "
            "the longer tail."
        )
    else:
        skew_interpretation = (
            "Fare is approximately symmetric because its skewness is close to zero."
        )

    # ---------------------------------------------------------
    # Survival rates using boolean masking
    # ---------------------------------------------------------
    female_rate = df.loc[df["sex"] == "female", "survived"].mean()
    male_rate = df.loc[df["sex"] == "male", "survived"].mean()

    pclass_rates = {}
    for pclass in sorted(df["pclass"].unique()):
        pclass_rates[pclass] = df.loc[
            df["pclass"] == pclass, "survived"
        ].mean()

    sex_class_rates = {}

    for sex in ["female", "male"]:
        for pclass in sorted(df["pclass"].unique()):
            mask = (df["sex"] == sex) & (df["pclass"] == pclass)
            sex_class_rates[f"{sex}, class {pclass}"] = df.loc[
                mask, "survived"
            ].mean()

    print("\nSurvival by sex:")
    print(pd.Series({
        "female": female_rate,
        "male": male_rate,
    }))

    print("\nSurvival by pclass:")
    print(pd.Series(pclass_rates))

    print("\nSurvival by sex+pclass:")
    print(pd.Series(sex_class_rates))

    # ---------------------------------------------------------
    # Exact six-column correlation matrix
    # ---------------------------------------------------------
    corr_cols = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
    ]

    corr = df[corr_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )
    plt.title("Titanic numeric correlation matrix")
    plt.tight_layout()
    plt.savefig(OUT / "correlation_heatmap.png")
    plt.close()

    pairs = []

    for i, a in enumerate(corr_cols):
        for b in corr_cols[i + 1:]:
            pairs.append(
                (
                    abs(corr.loc[a, b]),
                    a,
                    b,
                    corr.loc[a, b],
                )
            )

    strongest = sorted(pairs, reverse=True)[:2]

    print("Two strongest correlations:")
    print(strongest)

    # ---------------------------------------------------------
    # Four required multivariate charts
    # ---------------------------------------------------------

    # Chart 1
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="sex",
        y="survived",
        hue="pclass"
    )
    plt.title("Survival rate by sex and class")
    plt.tight_layout()
    plt.savefig(OUT / "survival_sex_class.png")
    plt.close()

    # Chart 2
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="pclass",
        y="fare",
        hue="survived"
    )
    plt.title("Fare distribution by class and survival")
    plt.tight_layout()
    plt.savefig(OUT / "fare_class_survival.png")
    plt.close()

    # Chart 3
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="age",
        y="fare",
        hue="survived",
        alpha=0.6
    )
    plt.title("Age vs fare by survival")
    plt.tight_layout()
    plt.savefig(OUT / "age_fare_survival.png")
    plt.close()

    # Chart 4
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=df,
        x="pclass",
        hue="survived"
    )
    plt.title("Passenger counts by class and survival")
    plt.tight_layout()
    plt.savefig(OUT / "class_survival_counts.png")
    plt.close()

    # ---------------------------------------------------------
    # EDA-only standardization check
    # ---------------------------------------------------------
    standardized_results = {}

    for col in ["age", "fare"]:
        z = (df[col] - df[col].mean()) / df[col].std()

        standardized_results[col] = {
            "mean": z.mean(),
            "std": z.std(),
        }

        print(
            f"{col} standardized mean/std:",
            z.mean(),
            z.std()
        )

    # ---------------------------------------------------------
    # Save EDA interpretation
    # ---------------------------------------------------------
    with open(OUT / "eda_interpretations.txt", "w", encoding="utf-8") as file:

        file.write("MODULE 2 EDA INTERPRETATIONS\n")
        file.write("=" * 80 + "\n\n")

        file.write("1. SURVIVAL RATE BY SEX AND CLASS\n")
        file.write("-" * 80 + "\n")
        file.write(
            f"Female passengers had a survival rate of {female_rate:.3f}, "
            f"while male passengers had a survival rate of {male_rate:.3f}. "
            f"Across the passenger classes, the survival rate generally varied "
            f"substantially by both sex and class.\n"
        )
        file.write(
            "The chart shows that passenger class and sex jointly provide "
            "useful information about survival outcomes. The differences are "
            "especially visible when comparing women and men within the same class.\n\n"
        )

        file.write("2. FARE DISTRIBUTION BY CLASS AND SURVIVAL\n")
        file.write("-" * 80 + "\n")
        file.write(
            "Fare distributions differ across passenger classes, with higher "
            "classes generally associated with higher fares. The boxplot also "
            "shows substantial variation and several high-fare observations.\n"
        )
        file.write(
            "Survival status creates additional differences within the fare "
            "distributions. This suggests that fare and passenger class contain "
            "related information about the survival outcome.\n\n"
        )

        file.write("3. AGE VS FARE BY SURVIVAL\n")
        file.write("-" * 80 + "\n")
        file.write(
            "The scatterplot shows that passengers span a wide range of ages "
            "and fares, while survival observations occur throughout this space. "
            "Higher fares are concentrated among a smaller subset of passengers.\n"
        )
        file.write(
            "The overlap between survived and non-survived observations indicates "
            "that age and fare alone do not perfectly separate the two outcomes. "
            "Their predictive usefulness therefore depends on their combination "
            "with other passenger characteristics.\n\n"
        )

        file.write("4. PASSENGER COUNTS BY CLASS AND SURVIVAL\n")
        file.write("-" * 80 + "\n")
        file.write(
            "The passenger counts show that the three classes contain different "
            "numbers of passengers. The survival and non-survival counts also "
            "differ across classes.\n"
        )
        file.write(
            "Class 3 contains many passengers and a large number of non-survivors, "
            "while the higher classes show different survival patterns. This makes "
            "pclass an important variable to retain during modeling.\n\n"
        )

        file.write("FARE DISTRIBUTION INTERPRETATION\n")
        file.write("-" * 80 + "\n")
        file.write(
            f"Fare mean = {fare_mean:.4f}\n"
            f"Fare median = {fare_median:.4f}\n"
            f"Fare mode = {fare_mode:.4f}\n"
            f"Fare skewness = {fare_skewness:.4f}\n\n"
        )
        file.write(skew_interpretation + "\n\n")

        file.write("IQR OUTLIER COUNTS\n")
        file.write("-" * 80 + "\n")
        file.write(
            f"Age IQR outliers: {outlier_results['age']}\n"
            f"Fare IQR outliers: {outlier_results['fare']}\n\n"
        )

        file.write("STRONGEST CORRELATIONS\n")
        file.write("-" * 80 + "\n")

        for value, a, b, signed_value in strongest:
            file.write(
                f"{a} vs {b}: correlation = {signed_value:.4f}\n"
            )

        file.write("\nEDA STANDARDIZATION CHECK\n")
        file.write("-" * 80 + "\n")

        for col, values in standardized_results.items():
            file.write(
                f"{col}: mean={values['mean']:.6f}, "
                f"std={values['std']:.6f}\n"
            )


def modeling(df):

    target = "survived"

    features = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked",
    ]

    X = df[features].copy()
    y = df[target].copy()

    # ---------------------------------------------------------
    # Stratified train/test split
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    numeric = [
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
    ]

    categorical = [
        "sex",
        "embarked",
    ]

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    ),
                ]),
                numeric,
            ),
            (
                "cat",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    ),
                ]),
                categorical,
            ),
        ]
    )

    # ---------------------------------------------------------
    # Classification models
    # ---------------------------------------------------------
    models = {
        "Logistic Regression":
            LogisticRegression(max_iter=2000),

        "Decision Tree":
            DecisionTreeClassifier(random_state=42),

        "Random Forest":
            RandomForestClassifier(random_state=42),
    }

    comparison = []

    for name, estimator in models.items():

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)

        prob = pipe.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        auc = roc_auc_score(y_test, prob)

        comparison.append({
            "metric_group": "classification",
            "model": name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        })

        # Confusion matrix
        cm = confusion_matrix(y_test, pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
        )

        plt.title(f"{name} confusion matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        plt.savefig(
            OUT / f"{name.lower().replace(' ', '_')}_confusion.png"
        )

        plt.close()

    classification_df = pd.DataFrame(comparison)

    classification_df.to_csv(
        OUT / "classification_model_comparison.csv",
        index=False,
    )

    print("\nClassification model comparison:")
    print(classification_df)

    # ---------------------------------------------------------
    # ROC curves
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))

    for name, estimator in models.items():

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        pipe.fit(X_train, y_train)

        prob = pipe.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(
            y_test,
            prob
        )

        auc = roc_auc_score(
            y_test,
            prob
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={auc:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
    )

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve comparison")
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        OUT / "roc_comparison.png"
    )

    plt.close()

    # ---------------------------------------------------------
    # Decision tree visualization
    # ---------------------------------------------------------
    tree_pipe = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            DecisionTreeClassifier(
                random_state=42,
                max_depth=4
            )
        ),
    ])

    tree_pipe.fit(
        X_train,
        y_train
    )

    feature_names = (
        tree_pipe
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    plt.figure(figsize=(20, 10))

    plot_tree(
        tree_pipe.named_steps["model"],
        feature_names=feature_names,
        class_names=[
            "not survived",
            "survived",
        ],
        filled=True,
        max_depth=4,
    )

    plt.tight_layout()

    plt.savefig(
        OUT / "decision_tree.png"
    )

    plt.close()

    # ---------------------------------------------------------
    # Class imbalance comparison
    # ---------------------------------------------------------
    imbalance_results = []

    baseline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=2000
            )
        ),
    ])

    balanced = Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        ),
    ])

    for label, pipe in [
        ("baseline", baseline),
        ("balanced", balanced),
    ]:

        pipe.fit(
            X_train,
            y_train
        )

        pred = pipe.predict(X_test)

        imbalance_results.append({
            "method": label,
            "accuracy": accuracy_score(
                y_test,
                pred
            ),
            "precision": precision_score(
                y_test,
                pred
            ),
            "recall": recall_score(
                y_test,
                pred
            ),
            "f1": f1_score(
                y_test,
                pred
            ),
        })

    # SMOTE only on training data
    X_train_t = preprocessor.fit_transform(
        X_train
    )

    X_test_t = preprocessor.transform(
        X_test
    )

    smote = SMOTE(
        random_state=42
    )

    X_smote, y_smote = smote.fit_resample(
        X_train_t,
        y_train
    )

    smote_model = LogisticRegression(
        max_iter=2000
    )

    smote_model.fit(
        X_smote,
        y_smote
    )

    smote_pred = smote_model.predict(
        X_test_t
    )

    imbalance_results.append({
        "method": "SMOTE",
        "accuracy": accuracy_score(
            y_test,
            smote_pred
        ),
        "precision": precision_score(
            y_test,
            smote_pred
        ),
        "recall": recall_score(
            y_test,
            smote_pred
        ),
        "f1": f1_score(
            y_test,
            smote_pred
        ),
    })

    imbalance_df = pd.DataFrame(
        imbalance_results
    )

    imbalance_df.to_csv(
        OUT / "imbalance_comparison.csv",
        index=False,
    )

    print("\nImbalance comparison:")
    print(imbalance_df)

    # ---------------------------------------------------------
    # Random Forest GridSearch
    # ---------------------------------------------------------
    rf_pipe = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                oob_score=True
            )
        ),
    ])

    grid = GridSearchCV(
        rf_pipe,
        {
            "model__n_estimators": [
                100,
                200
            ],
            "model__max_depth": [
                None,
                5,
                10
            ],
            "model__max_features": [
                "sqrt",
                "log2"
            ],
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )

    grid.fit(
        X_train,
        y_train
    )

    best_params = grid.best_params_
    best_cv_score = grid.best_score_
    oob_score = (
        grid.best_estimator_
        .named_steps["model"]
        .oob_score_
    )

    print(
        "Best RF parameters:",
        best_params
    )

    print(
        "Best CV score:",
        best_cv_score
    )

    print(
        "OOB score:",
        oob_score
    )

    # ---------------------------------------------------------
    # Regression side task: predict fare
    # ---------------------------------------------------------
    reg_features = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
    ]

    reg_df = df[
        reg_features + ["fare"]
    ].dropna()

    Xr = reg_df[reg_features]
    yr = reg_df["fare"]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr,
        yr,
        test_size=0.2,
        random_state=42,
    )

    reg = LinearRegression()

    reg.fit(
        Xr_train,
        yr_train
    )

    rp = reg.predict(
        Xr_test
    )

    mae = mean_absolute_error(
        yr_test,
        rp
    )

    rmse = np.sqrt(
        mean_squared_error(
            yr_test,
            rp
        )
    )

    r2 = r2_score(
        yr_test,
        rp
    )

    n, p = Xr_test.shape

    adj_r2 = (
        1
        - (1 - r2)
        * (n - 1)
        / (n - p - 1)
    )

    print(
        "Regression MAE/RMSE/R2/Adjusted R2:",
        mae,
        rmse,
        r2,
        adj_r2,
    )

    # ---------------------------------------------------------
    # Residual plot
    # ---------------------------------------------------------
    residuals = yr_test - rp

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        x=rp,
        y=residuals
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel("Predicted fare")
    plt.ylabel("Residual")
    plt.title("Fare residual plot")

    plt.tight_layout()

    plt.savefig(
        OUT / "fare_residuals.png"
    )

    plt.close()

    # Simple residual-spread diagnostic
    residual_df = pd.DataFrame({
        "predicted": rp,
        "residual": residuals,
    })

    residual_df["prediction_bin"] = pd.qcut(
        residual_df["predicted"],
        q=4,
        duplicates="drop"
    )

    residual_spread = (
        residual_df
        .groupby(
            "prediction_bin",
            observed=False
        )["residual"]
        .std()
    )

    spread_ratio = (
        residual_spread.max()
        / residual_spread.min()
    )

    if spread_ratio > 2:
        hetero_conclusion = (
            "The residual spread changes substantially across "
            "prediction ranges, suggesting possible heteroscedasticity."
        )
    else:
        hetero_conclusion = (
            "The residual spread is reasonably similar across "
            "prediction ranges, so strong heteroscedasticity is not evident "
            "from this diagnostic."
        )

    # ---------------------------------------------------------
    # Regression comparison
    # ---------------------------------------------------------
    regression_df = pd.DataFrame([{
        "metric_group": "regression",
        "model": "Linear Regression",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adj_r2,
    }])

    regression_df.to_csv(
        OUT / "regression_model_comparison.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Combined model comparison table
    # ---------------------------------------------------------
    combined_rows = []

    for row in comparison:
        combined_rows.append({
            "metric_group": "classification",
            "model": row["model"],
            "accuracy": row["accuracy"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
            "auc": row["auc"],
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "Adjusted_R2": np.nan,
        })

    combined_rows.append({
        "metric_group": "regression",
        "model": "Linear Regression",
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "auc": np.nan,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adj_r2,
    })

    combined_df = pd.DataFrame(
        combined_rows
    )

    combined_df.to_csv(
        OUT / "model_comparison.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Final recommendation
    # ---------------------------------------------------------
    best_classification = classification_df.loc[
        classification_df["f1"].idxmax()
    ]

    recommendation = (
        f"For the classification task, {best_classification['model']} "
        f"achieved the highest F1 score among the evaluated models "
        f"({best_classification['f1']:.3f}). "
        f"Its accuracy was {best_classification['accuracy']:.3f} "
        f"and ROC-AUC was {best_classification['auc']:.3f}. "
        f"The Random Forest GridSearch selected parameters "
        f"{best_params}, with a cross-validation F1 score of "
        f"{best_cv_score:.3f} and an OOB score of {oob_score:.3f}. "
        f"For the fare regression side-task, Linear Regression achieved "
        f"MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}, and "
        f"Adjusted R²={adj_r2:.3f}. "
        f"{hetero_conclusion}"
    )

    with open(
        OUT / "final_recommendation.txt",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "FINAL MODEL RECOMMENDATION\n"
        )

        file.write(
            "=" * 80 + "\n\n"
        )

        file.write(
            recommendation
        )

    print(
        "\nFinal recommendation:"
    )

    print(
        recommendation
    )

    # ---------------------------------------------------------
    # Save complete fitted pipeline
    # ---------------------------------------------------------
    full_pipeline = (
        grid.best_estimator_
    )

    joblib.dump(
        full_pipeline,
        OUT / "best_pipeline.joblib"
    )

    # Reload and predict raw test input
    loaded = joblib.load(
        OUT / "best_pipeline.joblib"
    )

    sample_prediction = loaded.predict(
        X_test.head(1)
    )

    print(
        "Reloaded pipeline prediction:",
        sample_prediction
    )


def main():
    df = load_once()

    cleaned = clean_for_eda(
        df
    )

    eda(
        cleaned
    )

    modeling(
        cleaned
    )


if __name__ == "__main__":
    main()

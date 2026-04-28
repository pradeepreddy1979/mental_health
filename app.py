import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

DATASET_PATH = "Mental_Health_Condition_Classification.csv"
BIPOLAR_TERMS = [
    "bipolar",
    "mania",
    "manic",
    "hypomania",
    "hypomanic",
    "mood swing",
    "mood swings",
    "mixed episode",
    "mixed state",
    "euphoria",
    "grandiosity",
    "racing thoughts",
    "quetiapine",
    "lithium",
    "lamotrigine",
    "vraylar",
    "reagila",
]


@st.cache_data
def load_dataset() -> pd.DataFrame:
    columns = pd.read_csv(DATASET_PATH, nrows=0).columns
    text_column = next((col for col in ["text", "statement"] if col in columns), None)
    label_column = next((col for col in ["status", "label", "condition"] if col in columns), None)

    if text_column is None or label_column is None:
        raise ValueError(
            "Dataset must contain a text column named 'text' or 'statement' "
            "and a label column named 'status', 'label', or 'condition'."
        )

    df = pd.read_csv(DATASET_PATH, usecols=[text_column, label_column]).dropna()
    df = df.rename(columns={text_column: "text", label_column: "status"})
    df["text"] = df["text"].astype(str).str.strip()
    df["status"] = df["status"].astype(str).str.strip().str.lower()
    return df[df["text"] != ""]


@st.cache_resource
def train_models() -> tuple[Pipeline, Pipeline, list[str]]:
    df = load_dataset()
    labels = sorted(df["status"].unique().tolist())

    multiclass_pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                stop_words="english",
                                ngram_range=(1, 2),
                                min_df=2,
                                max_features=40000,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=2,
                                max_features=15000,
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", LinearSVC(class_weight="balanced")),
        ]
    )
    multiclass_pipeline.fit(df["text"], df["status"])

    bipolar_targets = df["status"].eq("bipolar")
    bipolar_pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                stop_words="english",
                                ngram_range=(1, 2),
                                min_df=2,
                                max_features=30000,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=2,
                                max_features=12000,
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", LinearSVC(class_weight="balanced")),
        ]
    )
    bipolar_pipeline.fit(df["text"], bipolar_targets)
    return multiclass_pipeline, bipolar_pipeline, labels


def detect_condition(
    text: str,
    multiclass_model: Pipeline,
    bipolar_model: Pipeline,
) -> str:
    normalized_text = text.lower()
    multiclass_prediction = multiclass_model.predict([text])[0]
    bipolar_score = float(bipolar_model.decision_function([text])[0])
    matched_bipolar_terms = [term for term in BIPOLAR_TERMS if term in normalized_text]

    if multiclass_prediction == "bipolar":
        return "bipolar"

    # Strong bipolar markers should not fall through to nearby classes.
    if any(term in normalized_text for term in ["bipolar", "mania", "manic", "hypomania", "hypomanic"]):
        return "bipolar"

    if len(matched_bipolar_terms) >= 2 and bipolar_score > -0.25:
        return "bipolar"

    if bipolar_score > 0.2:
        return "bipolar"

    return multiclass_prediction


st.set_page_config(page_title="Mental Health Disorder Detection", page_icon="M")

st.title("Mental Health Disorder Detection")

multiclass_model, bipolar_model, conditions = train_models()
st.caption(f"Dataset labels: {', '.join(conditions)}")

text_input = st.text_area("Symptoms description:", height=150)

if st.button("Detect Disorder"):
    user_text = text_input.strip()
    if not user_text:
        st.warning("Please enter some text describing the symptoms.")
    else:
        predicted_label = detect_condition(user_text, multiclass_model, bipolar_model)

        st.success(f"Detected Disorder: {predicted_label}")

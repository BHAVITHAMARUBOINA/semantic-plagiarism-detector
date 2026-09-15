import streamlit as st
import pandas as pd
import plotly.express as px

from document_processor import extract_text
from detector import (
    calculate_similarity,
    exact_similarity,
    combined_score,
    classify_similarity
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SemanticShield",
    page_icon="🛡️",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    font-size: 42px;
    font-weight: 800;
}

.subtitle {
    font-size: 18px;
    color: #666;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="title">🛡️ SemanticShield</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Detect the meaning, not just the words.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# PROJECT DESCRIPTION
# --------------------------------------------------

st.info(
    "SemanticShield detects potential plagiarism by "
    "comparing the meaning of sentences using semantic "
    "embeddings, not just exact word matches."
)


# --------------------------------------------------
# FILE UPLOADERS
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("📄 Original Document")

    original_file = st.file_uploader(
        "Upload the original/reference document",
        type=["txt", "pdf", "docx"],
        key="original"
    )


with col2:

    st.subheader("📄 Suspected Document")

    suspected_file = st.file_uploader(
        "Upload the suspected document",
        type=["txt", "pdf", "docx"],
        key="suspected"
    )


st.write("")


# --------------------------------------------------
# ANALYZE BUTTON
# --------------------------------------------------

analyze = st.button(
    "🔍 ANALYZE DOCUMENTS",
    use_container_width=True
)


if analyze:

    if original_file is None or suspected_file is None:

        st.warning(
            "Please upload both documents before analysis."
        )

    else:

        with st.spinner(
            "🧠 SemanticShield is analyzing document meaning..."
        ):

            # Extract text
            original_text = extract_text(original_file)
            suspected_text = extract_text(suspected_file)

            # Calculate semantic similarity
            matches, semantic_score = calculate_similarity(
                original_text,
                suspected_text
            )

            # Calculate exact word similarity
            exact_score = exact_similarity(
                original_text,
                suspected_text
            )

            # Calculate combined score
            final_score = combined_score(
                semantic_score,
                exact_score
            )

            # Classify risk
            risk, explanation = classify_similarity(
                final_score
            )


        # --------------------------------------------------
        # RESULTS
        # --------------------------------------------------

        st.divider()

        st.header("📊 Analysis Results")


        metric1, metric2, metric3 = st.columns(3)


        with metric1:

            st.metric(
                "Semantic Similarity",
                f"{semantic_score * 100:.1f}%"
            )


        with metric2:

            st.metric(
                "Exact Word Similarity",
                f"{exact_score * 100:.1f}%"
            )


        with metric3:

            st.metric(
                "Overall Similarity",
                f"{final_score * 100:.1f}%"
            )


        # --------------------------------------------------
        # RISK LEVEL
        # --------------------------------------------------

        st.subheader("🚨 Risk Assessment")


        if risk == "HIGH":

            st.error(
                f"🔴 HIGH RISK — {explanation}"
            )

        elif risk == "MODERATE":

            st.warning(
                f"🟠 MODERATE RISK — {explanation}"
            )

        else:

            st.success(
                f"🟢 LOW RISK — {explanation}"
            )


        # --------------------------------------------------
        # VISUALIZATION
        # --------------------------------------------------

        st.subheader("📈 Similarity Breakdown")


        chart_data = pd.DataFrame({

            "Metric": [
                "Semantic Similarity",
                "Exact Word Similarity",
                "Overall Similarity"
            ],

            "Percentage": [
                semantic_score * 100,
                exact_score * 100,
                final_score * 100
            ]

        })


        fig = px.bar(
            chart_data,
            x="Metric",
            y="Percentage",
            range_y=[0, 100],
            text_auto=".1f"
        )


        fig.update_layout(
            yaxis_title="Similarity (%)",
            xaxis_title="",
            showlegend=False
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # --------------------------------------------------
        # MATCHING SENTENCES
        # --------------------------------------------------

        st.subheader("🔎 Potentially Matching Content")


        matching_data = []


        for match in matches:

            if match["score"] >= 0.60:

                matching_data.append({

                    "Similarity":
                        f'{match["score"] * 100:.1f}%',

                    "Suspected Sentence":
                        match["suspected"],

                    "Closest Original Sentence":
                        match["original"]

                })


        if matching_data:

            df = pd.DataFrame(matching_data)

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.success(
                "No strong semantic matches were detected."
            )


        # --------------------------------------------------
        # DISCLAIMER
        # --------------------------------------------------

        st.caption(
            "⚠️ Semantic similarity indicates potential overlap "
            "in meaning. It does not by itself prove plagiarism."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "SemanticShield • AI-powered semantic plagiarism detection"
)
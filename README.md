import streamlit as st
from transformers import pipeline

st.title("Emotion Detection from Social Media Text")
st.subheader("Classify text into emotions: Joy, Sadness, Anger, Fear, Surprise, Disgust")

st.write("Enter any sentence below and click 'Analyze Emotion' to get started.")


text_input = st.text_area("Enter Text", "")

if st.button("Analyze Emotion"):
    if text_input.strip():
        with st.spinner("Analyzing..."):
            classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)
            result = classifier(text_input)[0]
            result_sorted = sorted(result, key=lambda x: x['score'], reverse=True)
            top_emotion = result_sorted[0]
            st.success(f"**Predicted Emotion:** {top_emotion['label']} ({top_emotion['score']:.2f})")
            st.write("**Detailed Scores:**")
            for emotion in result_sorted:
                st.write(f"{emotion['label']}: {emotion['score']:.2f}")
    else:
        st.warning("Please enter some text.")


import streamlit as st
from generations.storm.storm import Storm

# Set the page title and icon
st.set_page_config(page_title="Generate Report", page_icon=":document:")

if "report_outline" not in st.session_state:
    st.session_state.report_outline = ""
if "report_content" not in st.session_state:
    st.session_state.report_content = ""

# A required topic input
topic = st.text_input("Topic", "Mushroom as a diabetes treatment")


if not topic:
    st.warning("Please enter a topic.")
    st.stop()

# Text area for outline of the report followed by a button to generate the outline
report_outline = st.text_area(
    "Report Outline", height=500, value=st.session_state.report_outline
)
if st.button("Generate Outline"):
    with st.spinner("Generating Outline"):
        outline = Storm().generate_outline(topic)
        # outline = "fewfwefwef"
        # Update the text area with the generated outline
        st.session_state.report_outline = outline
        st.experimental_rerun()

# Link to download pdf file at './article.pdf'
with open("article.pdf", "rb") as f:
    st.download_button("Download Report", f.read(), "article.pdf", "text/pdf")

st.markdown(st.session_state.report_content, unsafe_allow_html=True)
if st.button("Generate Report"):
    with st.spinner("Generating Report"):
        report = Storm().write_article(topic, report_outline)
        st.markdown(report, unsafe_allow_html=True)
        st.session_state.report_content = report

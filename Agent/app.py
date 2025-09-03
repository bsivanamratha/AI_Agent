# app.py
import streamlit as st
import json
from main import run_research, ResearchResponse

# --- Streamlit page setup ---
st.set_page_config(page_title="AI Research Assistant", page_icon="🧠", layout="wide")

st.title("🧠 AI Research Assistant")
st.write("Ask a question and I’ll research it for you using web + Wikipedia + local tools.")

# --- Input box ---
query = st.text_input("🔍 Enter your research topic/question:")

# --- Run Research ---
if st.button("Run Research"):
    if not query.strip():
        st.warning("⚠️ Please enter a query.")
    else:
        with st.spinner("🔎 Researching..."):
            try:
                response = run_research(query)

                # ✅ If parsed successfully into ResearchResponse
                if isinstance(response, ResearchResponse):
                    st.subheader("📌 Topic")
                    st.write(response.topic)

                    st.subheader("📝 Summary")
                    st.write(response.summary)

                    st.subheader("🔗 Sources")
                    for src in response.sources:
                        st.markdown(f"- [{src}]({src})" if src.startswith("http") else f"- {src}")

                    st.subheader("🛠️ Tools Used")
                    st.write(", ".join(response.tools_used))

                    st.success("✅ Research complete!")

                    st.download_button(
                        label="📥 Download JSON",
                        data=json.dumps(response.dict(), indent=2),
                        file_name="research_output.json",
                        mime="application/json"
                    )

                # ⚠️ If we got raw text back instead of structured JSON
                else:
                    st.subheader("⚠️ Raw Output (Unparsed)")
                    st.code(str(response))
                    st.warning("Could not format into structured JSON, showing raw output instead.")

            except Exception as e:
                st.error(f"❌ Research failed: {e}")

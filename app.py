import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi 
from langchain.schema import Document


# Streamlit app
st.set_page_config(page_title="Summarize Text from Youtube Videos or Websites")
st.title("Summarize Text from Youtube Videos or Websites")
st.subheader("Summarize URL")

# Get the Groq API key and url(YT & Website) field to be summarized
with st.sidebar:
    api_key = st.text_input("Enter Groq API Key", value="", type="password")
    
generic_url = st.text_input("Enter URL", label_visibility="collapsed")


# Groq Model and Prompt Template
llm = ChatGroq(api_key=api_key, model="gemma2-9b-it")
prompt_template = """
    Provide the summary of the following content in 300 words.
    Content:{text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])


# Helper to extract video ID from YouTube URL
def extract_video_id(url):
    from urllib.parse import urlparse, parse_qs
    parsed_url = urlparse(url)
    if 'youtube' in parsed_url.netloc:
        return parse_qs(parsed_url.query).get('v', [None])[0]
    elif 'youtu.be' in parsed_url.netloc:
        return parsed_url.path[1:]
    return None

if st.button("Summarize the Content from YT or Website"):
    if not api_key.strip() or not generic_url.strip():
        st.error("Please fill the credentials and URL.")
    
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")
    
    else:
        try:
            with st.spinner("Summarizing the content..."):
                docs = []

                # YouTube or Website Handling
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    video_id = extract_video_id(generic_url)
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                    transcript_text = " ".join([entry['text'] for entry in transcript_list])
                    docs.append(Document(page_content=transcript_text, metadata={"source": "YouTube"}))

                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False)
                    docs = loader.load()

                # Summarization
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success("✅ Summary generated successfully!")
                st.text_area("Summary", output_summary, height=300)

        except Exception as e:
            st.error("⚠️ An error occurred during summarization.")
            st.exception(e)

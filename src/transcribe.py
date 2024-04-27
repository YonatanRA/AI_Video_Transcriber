# libraries
from langchain_openai.chat_models import ChatOpenAI   
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from pytube import YouTube
import whisper
import tempfile
import argparse
from tools import logger

import os                           
from dotenv import load_dotenv 

# load api key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# argparsing
argparser = argparse.ArgumentParser(description="Video Transcriber")
argparser.add_argument("-u", "--url", type=str, help="YouTube video URL")
argparser.add_argument("-q", "--query", type=str, help="User query input")

parse_args = argparser.parse_args()

URL = parse_args.url
QUERY = parse_args.query

# transcription path
PATH_TXT = "../txt/video_transcription.txt"

def get_video_transcription() -> None:

    """
    Function to transcribe text from YouTube video
    """

    global URL, PATH_TXT

    youtube = YouTube(URL)
    audio = youtube.streams.filter(only_audio=True).first()

    logger.info("Loading Whisper Model.")
    whisper_model = whisper.load_model("large-v3")

    with tempfile.TemporaryDirectory() as dir_temporal:
        
        logger.info("Downloading audio from YouTube.")
        audio_file = audio.download(output_path=dir_temporal)
        
        logger.info('Transcribing...')
        transcription = whisper_model.transcribe(audio_file, fp16=False)["text"].strip()
        
        logger.info("Saving txt file...")
        with open(PATH_TXT, "w") as text_file:
            text_file.write(transcription)
        logger.info("Done!")


def get_response() -> str:

    """
    Fucntion to retrieve 10 bullet points from transcription
    """

    global OPENAI_API_KEY, PATH_TXT, QUERY

    logger.info("Loading transcription text.")
    with open(PATH_TXT, "r") as text_file:
        transcription = text_file.read()
        
    logger.info("Creating prompt template.")
    template = """
            Summarize in a list of 10 bullet points the 
            following context based on the following question:

            Context: {context}

            Question: {question}
            """


    prompt = ChatPromptTemplate.from_template(template)

    logger.info("Loading GPT4 Model.")
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4-turbo")

    parser = StrOutputParser()

    logger.info("Creating chain.")
    chain = prompt | model | parser

    logger.info('Answering...')
    response = chain.invoke({"context": transcription, "question": QUERY})
    
    return response.split("\n")


if __name__=="__main__":
    get_video_transcription()
    logger.info(get_response())

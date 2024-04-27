# libraries
from pytube import YouTube
import whisper
import tempfile
import argparse
from tools import logger


# argparsing
argparser = argparse.ArgumentParser(description="Video Transcriber")
argparser.add_argument("-u", "--url", type=str, help="YouTube video URL")

parse_args = argparser.parse_args()
URL = parse_args.url

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


if __name__=="__main__":
    get_video_transcription()

import openai
from groq import Groq
import requests




def printmultiplies(i):
    i=1
    while(i<11):
        print(i*5)
        i+=1



def p(printword):
    for i in printword:
        print(i)





def humandate(timestamp):
    # Import datetime module
    from datetime import datetime
    
    # Converting milliseconds to seconds
    timestamp_in_seconds = timestamp / 1000
    
    # Converting timestamp to date
    dt_object = datetime.fromtimestamp(timestamp_in_seconds)
    return f'Date and Time is: {dt_object}'






def chat_with_gpt4(prompt):
    """
    Sends a prompt to OpenAI's GPT-4 API and returns the response.

    Args:
        prompt (str): The user's input text to send to GPT-4.

    Returns:
        str: The response text from GPT-4.
    """
    # Replace with your OpenAI API key
    openai.api_key = "sk-WBCCVieT20QSsAw-lc38Ig3rlmwgkC1JeMqOON1q13T3BlbkFJ-Vbo1U-yXBz0wvQHT8v65p9ruZxPDG0hBC8OlTAu4A"

    try:
        # Make the API call
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are ChatGPT, a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract and return the response text
        return response['choices'][0]['message']['content'].strip()

    except Exception as e:
        return f"An error occurred: {e}"







def generate_chat_response(prompt):
    client = Groq(api_key="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")
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=False
    )
    return chat_completion.choices[0].message.content



#function interactes with groq cloud api to generate chat response(text to text model)





def transcribe_audio(file_path, language="en"):
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": "Bearer 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"}
    files = {"file": open(file_path, "rb")}
    data = {
        "model": "whisper-large-v3",
        "language": language,
        "response_format": "json",
        "temperature": 0
    }
    response = requests.post(url, headers=headers, files=files, data=data)
    return response.json()




#function interactes with groq cloud api to generate transcribed text(audio to text model)

##################################################################################



import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from pydub import AudioSegment
import threading
import requests
from groq import Groq

# Global flag to control recording
recording_flag = True
user_input_flag = None

# Function to capture user input asynchronously
def handle_user_input():
    global user_input_flag, recording_flag
    while recording_flag:
        user_input_flag = input("Press 2 to finish").strip()
        if user_input_flag == "2":
            break

# Function to record audio
def record_audio(file_path="user_recording.wav"):
    fs = 44100  # Sample rate
    print("Recording started. Press 2 to finish")

    # Start a thread to handle user input asynchronously
    input_thread = threading.Thread(target=handle_user_input)
    input_thread.start()

    # Record audio until a valid user input stops the recording
    global recording_flag, user_input_flag
    recording_flag = True
    audio_frames = []

    try:
        with sd.InputStream(samplerate=fs, channels=1, callback=lambda indata, frames, time, status: audio_frames.append(indata.copy())):
            while recording_flag:
                sd.sleep(1)  # Check every 1ms

                # Handle user inputs during recording
                if user_input_flag == "2":
                    print("Recording finished.")
                    recording_flag = False
    except Exception as e:
        print(f"An error occurred during recording: {e}")
        return None

    # Stop the input thread
    input_thread.join()

    if not audio_frames:
        print("No audio recorded.")
        return None

    # Convert the list of frames to a NumPy array
    audio_array = np.concatenate(audio_frames, axis=0)

    # Save the recording as a WAV file
    write(file_path, fs, audio_array)
    print(f"WAV file saved at {file_path}")

    # Convert WAV to MP3
    audio = AudioSegment.from_wav(file_path)
    mp3_file_path = file_path.replace(".wav", ".mp3")
    audio.export(mp3_file_path, format="mp3")
    print(f"Recording saved as {mp3_file_path}.")
    return mp3_file_path

# Function to transcribe the recorded audio
def transcribe_audio(file_path, language="en"):
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": "Bearer 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"}
    files = {"file": open(file_path, "rb")}
    data = {
        "model": "whisper-large-v3",
        "language": language,
        "response_format": "json",
        "temperature": 0
    }
    response = requests.post(url, headers=headers, files=files, data=data)
    return response.json()

# Function to generate a chat response from the transcribed text
def generate_chat_response(prompt):
    client = Groq(api_key="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")
    chat_completion = client.chat.completions.create(
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        model="llama3-8b-8192", temperature=0.5, max_tokens=1024, top_p=1, stop=None, stream=False
    )
    return chat_completion.choices[0].message.content

# Main function to start the program , it was earlier version in the developement as later it was transformed to 
# audio_transcription_chatbot() function


# if __name__ == "__main__":
#     start_input = input("Press 1 to start recording: ").strip()
#     if start_input == "1":
#         audio_file = record_audio()  # Start the recording

#         if audio_file:
#             # Transcribe the audio
#             transcription = transcribe_audio(audio_file)
#             if transcription.get("text"):
#                 chat_prompt = f"Transcribed text: {transcription['text']}\nBased on this, generate a response."
#                 chat_response = generate_chat_response(chat_prompt)
#                 print("Chatbot response:", chat_response)
#             else:
#                 print("Error in transcription:", transcription)
#     else:
#         print("Invalid input. Program terminated.")



#the intended uncommented above block is to capture audio from user through mic and give text response based on the audio
#transcription and chatbot response based on the transcription text
#the function audio_transcription_chatbot() is the final version of the above block of code 
#the above code combines between audio to text api and text to text api to generate a chatbot response based on the audio input








##################################################################################









def audio_transcription_chatbot():
    """
    Function to record audio, transcribe it, and generate a chatbot response based on the transcription.
    """
    start_input = input("Press 1 to start recording: ").strip()
    if start_input == "1":
        audio_file = record_audio()  # Start the recording
        
        if audio_file:
            # Transcribe the audio
            transcription = transcribe_audio(audio_file)
            if transcription.get("text"):
                chat_prompt = f"Transcribed text: {transcription['text']}\nBased on this, generate a response."
                chat_response = generate_chat_response(chat_prompt)
                print("Chatbot response:", chat_response)
            else:
                print("Error in transcription:", transcription)
        else:
            print("Audio recording failed.")
    else:
        print("Invalid input. Program terminated.")



#usuage in main example:

#audio_transcription_chatbot()

##################################################################################








def start_audiobot_program():
    """
    Starts the audio recording, transcription, and chatbot response generation process.
    """
    import pyttsx3

    start_input = input("Press 1 to start recording: ").strip()
    if start_input == "1":
        audio_file = record_audio()  # Start the recording

        if audio_file:
            # Transcribe the audio
            transcription = transcribe_audio(audio_file)
            if transcription.get("text"):
                chat_prompt = f"Transcribed text: {transcription['text']}\nBased on this, generate a response."
                chat_response = generate_chat_response(chat_prompt)
                print("Chatbot response:", chat_response)
                
                # Use text-to-speech to read the chatbot response
                engine = pyttsx3.init()
                engine.say(chat_response)
                engine.runAndWait()
            else:
                print("Error in transcription:", transcription)
    else:
        print("Invalid input. Program terminated.")


#usuage in main example:

#start_audiobot_program()

#the above code is developement of the audio_transcription_chatbot() function with the addition of text to speech
#user can hear the chatbot response through the speakers


##################################################################################


import os
import time
import requests
import json

api_key="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJBaG1lZCBNb2hhbWVkIiwiVXNlck5hbWUiOiJBaG1lZCBNb2hhbWVkIiwiQWNjb3VudCI6IiIsIlN1YmplY3RJRCI6IjE4ODAxNjEyMDY0MTM1NjIwMDYiLCJQaG9uZSI6IiIsIkdyb3VwSUQiOiIxODgwMTYxMjA2NDA5MzY3NzAyIiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiYWhtZWRtb2hhbWVkNzg1MTFAZ21haWwuY29tIiwiQ3JlYXRlVGltZSI6IjIwMjUtMDEtMTcgMTc6MzM6NDQiLCJUb2tlblR5cGUiOjEsImlzcyI6Im1pbmltYXgifQ.qKz5T1WVP3GnJDxo3EdaueD_lRd_B0JUkUz4enU8pWuLf3PYL2Y5FEk6ZvufA-Kxozk6U3S3eCPaHESZcWV6jGUhjBrxQI7xKiqvaw3q6hchigv8bK3s0f_SFVrub36VurEp2CQDWj-N7ujCuGyJ8-o__gcO1XL6hZXSWGksQU3Wy8CF8rrbjedbo5KEJo3EvRMD1Q3E2-eh9-k57xBfFKcBiH7DkfRLkiBWui11fflupeU0DjyzFGGsVD1b2UW39hzYVjiWnh2N_WgI6NNzY0yeEUcSkpYu1CO95KuPHYdoD-04p4S7Liyfpp-Uq6uKkcLmPxtCUIRXi0-EBYsxKg"


model = "video-01" 
output_file_name = "output.mp4" #Please enter the save path for the generated video here

def invoke_video_generation(prompt)->str:
    print("-----------------Submit video generation task-----------------")
    url = "https://api.minimaxi.chat/v1/video_generation"
    payload = json.dumps({
      "prompt": prompt,
      "model": model
    })
    headers = {
      'authorization': 'Bearer ' + api_key,
      'content-type': 'application/json',
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
    task_id = response.json()['task_id']
    print("Video generation task submitted successfully, task ID.："+task_id)
    return task_id

def query_video_generation(task_id: str):
    url = "https://api.minimaxi.chat/v1/query/video_generation?task_id="+task_id
    headers = {
      'authorization': 'Bearer ' + api_key
    }
    response = requests.request("GET", url, headers=headers)
    status = response.json()['status']
    if status == 'Queueing':
        print("...In the queue...")
        return "", 'Queueing'
    elif status == 'Processing':
        print("...Generating...")
        return "", 'Processing'
    elif status == 'Success':
        return response.json()['file_id'], "Finished"
    elif status == 'Fail':
        return "", "Fail"
    else:
        return "", "Unknown"


def fetch_video_result(file_id: str):
    print("---------------Video generated successfully, downloading now---------------")
    url = "https://api.minimaxi.chat/v1/files/retrieve?file_id="+file_id
    headers = {
        'authorization': 'Bearer '+api_key,
    }

    response = requests.request("GET", url, headers=headers)
    print(response.text)

    download_url = response.json()['file']['download_url']
    print("Video download link：" + download_url)
    with open(output_file_name, 'wb') as f:
        f.write(requests.get(download_url).content)
    print("THe video has been downloaded in："+os.getcwd()+'/'+output_file_name)

def process_video_generation(prompt):
    """
    Handles the entire video generation process, from task invocation to fetching the result.
    """
    task_id = invoke_video_generation(prompt)
    print("-----------------Video generation task submitted -----------------")
    
    while True:
        time.sleep(10)
        
        file_id, status = query_video_generation(task_id)
        
        if file_id:
            fetch_video_result(file_id)
            print("---------------Successful---------------")
            break
        elif status in ["Fail", "Unknown"]:
            print("---------------Failed---------------")
            break




#usuage in main example:

#if failed try again but if you recieved unsufficient balance make new account and get free credits and api
#process_video_generation(prompt)
#prompt = "basketball match"
    ######################################################################################################










import requests

def analyze_sentiment(text):
    """
    Analyzes the sentiment of a given text using the API Ninjas sentiment analysis API.

    Args:
        text (str): The text for which to analyze sentiment.

    Returns:
        dict: The response from the API containing sentiment analysis results if successful.
              If the request fails, prints an error message and returns None.
    """
    api_url = f'https://api.api-ninjas.com/v1/sentiment?text={text}'
    api_key = "1zkI4x/bof2YF43bkHK0pQ==MxzHZq0UtXhGqK54"
    try:
        response = requests.get(api_url, headers={'X-Api-Key': api_key})
        if response.status_code == requests.codes.ok:
            return response.json()  # Return the parsed JSON response
        else:
            print(f"Error: HTTP {response.status_code} - {response.text}")
            return None
    except requests.RequestException as e:
        print(f"Request Exception: {str(e)}")
        return None


#    text = "the product is bad and high-priced"
 #   sentiment_result = analyze_sentiment(text)




######################################################################################

def get_current_directory():
    """
    Get the current working directory and print it.
    """
    import pathlib
    import os

    myPath = pathlib.Path().resolve()  # Get the current working directory
    os.chdir(myPath)  # Change to that directory
    print(os.getcwd())  # Print the current working directory


def change_directory(new_path):
    import os
    os.chdir(new_path)  # Change to the specified directory
    print(os.getcwd())  # Print the current working directory


#########################################################################################

def remove_duplicate_error():
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'




#############################################################################################




import pandas as pd

def check_missing_values(file_name, column_name=None):
    """
    Load a JSONL file and check the percentage of missing values in a specified column or all columns.
    
    Parameters:
    - file_name (str): The name of the JSONL file to be read
    - column_name (str, optional): The name of the column to check for missing values. 
      If None, all columns will be checked. Defaults to None.
    
    Returns:
    - None: Prints the missing value percentage for the specified column(s)
    """
    df = pd.read_json(file_name, lines=True)
    
    if column_name is not None:
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the dataset")
        missing_percentage = df[column_name].isnull().mean()*100
        print(f"Missing values percentage in '{column_name}': {missing_percentage:.2f}%")
    else:
        for column in df.columns:
            missing_percentage = df[column].isnull().mean()*100
            print(f"Missing values percentage in '{column}': {missing_percentage:.2f}%")

# Example usage:
# Check all columns
#check_missing_values("meta_Software.jsonl")

# Check a specific column
#check_missing_values("meta_Software.jsonl", "price")


#########################################################################################
import pandas as pd

def check_missing_values_v2(file_name, column_name=None, chunksize=100000):
    """
    Load a JSONL file in chunks and check the percentage of missing values in a specified column or all columns.
    
    Parameters:
    - file_name (str): The name of the JSONL file to be read
    - column_name (str, optional): The name of the column to check for missing values. 
      If None, all columns will be checked. Defaults to None.
    - chunksize (int): Number of rows to process at a time. Adjust based on memory capacity.
    
    Returns:
    - None: Prints the missing value percentage for the specified column(s)
    """
    total_rows = 0
    missing_counts = {}

    # Read the file in chunks
    for chunk in pd.read_json(file_name, lines=True, chunksize=chunksize):
        total_rows += len(chunk)
        
        # Update missing counts for relevant columns
        if column_name:
            columns = [column_name]
        else:
            columns = chunk.columns
            
        for col in columns:
            missing = chunk[col].isnull().sum()
            missing_counts[col] = missing_counts.get(col, 0) + missing

    # Calculate and print missing percentages
    if column_name:
        print(f"Missing values percentage in '{column_name}': "
              f"{(missing_counts.get(column_name, 0) / total_rows * 100):.2f}%")
    else:
        for col in missing_counts:
            print(f"Missing values percentage in '{col}': "
                  f"{(missing_counts[col] / total_rows * 100):.2f}%")

# Example usage:
# Check all columns
#check_missing_values_pandas("meta_Books.jsonl")

# Check a specific column
#check_missing_values_pandas("meta_Books.jsonl", "price")

#########################################################################################


import os

def get_jsonl_files(path):
    """
    Returns a list of full paths to all .jsonl files in the specified directory.
    """
    jsonl_files = [
        os.path.join(path, f) for f in os.listdir(path)
        if f.endswith('.jsonl')
    ]
    return jsonl_files


#########################################################################################







def run_audio_chatbot(start_trigger='1', stop_trigger='2'):
    import sounddevice as sd
    from scipy.io.wavfile import write
    import numpy as np
    from pydub import AudioSegment
    import threading
    import requests
    from groq import Groq
    import pyttsx3

    fs = 44100  # Sample rate
    file_path = "user_recording.wav"
    mp3_path = file_path.replace(".wav", ".mp3")

    # Prompt user to start recording
    print(f"Press {start_trigger} to start recording...")
    start_input = input().strip()
    if start_input != start_trigger:
        print("Invalid start input. Exiting.")
        return

    recording_flag = True
    audio_frames = []

    # Function to handle stopping recording
    def handle_stop():
        nonlocal recording_flag
        print(f"\nPress {stop_trigger} to stop recording...")
        while True:
            user_input = input().strip()
            if user_input == stop_trigger:
                recording_flag = False
                break

    # Start stop trigger listener thread
    stop_thread = threading.Thread(target=handle_stop)
    stop_thread.start()

    # Start audio recording
    try:
        with sd.InputStream(samplerate=fs, channels=1, callback=lambda indata, frames, time, status: audio_frames.append(indata.copy())):
            while recording_flag:
                sd.sleep(1)  # Check every 1ms
    except Exception as e:
        print(f"Error during recording: {e}")
        return

    # Wait for stop thread to finish
    stop_thread.join()

    if not audio_frames:
        print("No audio recorded.")
        return

    # Save and convert audio files
    audio_array = np.concatenate(audio_frames, axis=0)
    write(file_path, fs, audio_array)
    audio = AudioSegment.from_wav(file_path)
    audio.export(mp3_path, format="mp3")

    # Transcribe audio using Groq API
    tr_api_key = "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"  # Replace with your key
    transcribe_url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {tr_api_key}"}
    files = {"file": open(mp3_path, "rb")}
    data = {"model": "whisper-large-v3", "language": "en"}
    response = requests.post(transcribe_url, headers=headers, files=files, data=data)
    transcription = response.json()

    if 'text' not in transcription:
        print("Transcription failed:", transcription.get("error", transcription))
        return

    # Generate chatbot response using Groq LLM
    chat_api_key = "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"  # Replace with your key
    client = Groq(api_key=chat_api_key)
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": transcription["text"]}],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=500
        )
        response_text = completion.choices[0].message.content
    except Exception as e:
        print("Chatbot response error:", e)
        return

    print("\nChatbot response:\n", response_text)
    # Speak response using text-to-speech
    engine = pyttsx3.init()
    engine.say(response_text)
    engine.runAndWait()

    

# Usage example (replace placeholders with your API keys)
 #   run_audio_chatbot(start_trigger='s', stop_trigger='x')




########################################################################################






def generate_chat_response_v2(prompt):
    from google import genai
    client = genai.Client(api_key="AIzaSyD6VxooY8ZAop729dNk0NRFTHxBX380Lr8")
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt
    )
    return response.text 

# Example usage:
#prompt = "What are the best ways to improve memory?"
#response = my_utils.generate_chat_response_v2(prompt)

##############################################################################################


import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from pydub import AudioSegment
import threading
import requests
import pyttsx3
from google import genai

def run_audio_chatbot_v2(start_trigger='1', stop_trigger='2'):
    fs = 44100  # Sample rate
    file_path = "user_recording.wav"
    mp3_path = file_path.replace(".wav", ".mp3")

    print(f"Press {start_trigger} to start recording...")
    start_input = input().strip()
    if start_input != start_trigger:
        print("Invalid start input. Exiting.")
        return

    recording_flag = True
    audio_frames = []

    def handle_stop():
        nonlocal recording_flag
        print(f"\nPress {stop_trigger} to stop recording...")
        while True:
            user_input = input().strip()
            if user_input == stop_trigger:
                recording_flag = False
                break

    stop_thread = threading.Thread(target=handle_stop)
    stop_thread.start()

    try:
        with sd.InputStream(samplerate=fs, channels=1, callback=lambda indata, frames, time, status: audio_frames.append(indata.copy())):
            while recording_flag:
                sd.sleep(1)
    except Exception as e:
        print(f"Error during recording: {e}")
        return

    stop_thread.join()

    if not audio_frames:
        print("No audio recorded.")
        return

    audio_array = np.concatenate(audio_frames, axis=0)
    write(file_path, fs, audio_array)
    audio = AudioSegment.from_wav(file_path)
    audio.export(mp3_path, format="mp3")

    # Transcribe using Groq API
    tr_api_key = "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" 
    transcribe_url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {tr_api_key}"}
    with open(mp3_path, "rb") as file:
        files = {"file": file}
        data = {"model": "whisper-large-v3", "language": "en"}
        response = requests.post(transcribe_url, headers=headers, files=files, data=data)
    
    transcription = response.json()
    if 'text' not in transcription:
        print("Transcription failed:", transcription.get("error", transcription))
        return
    
    print("\nTranscription:\n", transcription["text"])

    # Generate chatbot response using Gemini API
    chat_api_key = "AIzaSyD6VxooY8ZAop729dNk0NRFTHxBX380Lr8" 
    client = genai.Client(api_key=chat_api_key)
    
    try:
        gemini_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=transcription["text"]
        )
        
        # Extract the text correctly
        response_text = gemini_response.text if hasattr(gemini_response, "text") else str(gemini_response)
    
    except Exception as e:
        print("Chatbot response error:", e)
        return

    print("\nChatbot response:\n", response_text)

    # Speak response using text-to-speech
    engine = pyttsx3.init()

    # Prevent "run loop already started" error
    try:
        engine.endLoop()  # Safely stop any existing loop
    except:
        pass  # Ignore errors if no loop was running

    engine.say(response_text)
    engine.runAndWait()
    

# Usage example (replace placeholders with your API keys)
 #   run_audio_chatbot(start_trigger='s', stop_trigger='x')




########################################################################################




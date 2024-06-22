import os
import subprocess
import torch
import torchaudio
import librosa
import re
from pytube import YouTube
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf


def convert_audio_to_text(VIDEO_URL):
    # YouTube video URL
    # VIDEO_URL = 'https://www.youtube.com/watch?v=OwQhSUqF_bE'
    OUTPUT_DIR = 'content'  # Output directory for audio files
    
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Download audio
    yt = YouTube(VIDEO_URL)
    audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4').first()
    audio_file = audio_stream.download(output_path=OUTPUT_DIR, filename='ytaudio.mp4')
    
    # Convert to WAV using ffmpeg
    output_wav_file = os.path.join(OUTPUT_DIR, 'ytaudio.wav')
    ffmpeg_cmd = ['ffmpeg', '-i', os.path.join(OUTPUT_DIR, 'ytaudio.mp4'), '-y', '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', output_wav_file]
    subprocess.run(ffmpeg_cmd, check=True)
    
    # Check if the output file exists
    if os.path.exists(output_wav_file):
        print(f"Successfully converted {os.path.join(OUTPUT_DIR, 'ytaudio.mp4')} to {output_wav_file}")
    else:
        print(f"Failed to convert {os.path.join(OUTPUT_DIR, 'ytaudio.mp4')} to {output_wav_file}")
        exit(1)
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    # Load pre-trained model and processor
    model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    
    # Split audio into 30-second chunks and transcribe each chunk
    chunk_duration = 30  # Duration of each chunk in seconds
    stream = librosa.stream(
        output_wav_file,
        block_length=30,
        frame_length=16000,
        hop_length=16000
    )
    
    # Process each chunk
    audio_path = []
    for i, speech in enumerate(stream):
        chunk_file = os.path.join(OUTPUT_DIR, f'chunk_{i}.wav')
        sf.write(chunk_file, speech, 16000)
        audio_path.append(chunk_file)
    
    # Function to transcribe audio file
    def transcribe_audio(audio_file):
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_file)
        
        # Ensure single channel (if multi-channel)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        input_values = processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=sample_rate).input_values.to(device)
        
        with torch.no_grad():
            logits = model(input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        
        return transcription
    
    # Transcribe each chunk
    full_transcript = ""
    for chunk_file in audio_path:
        chunk_transcript = transcribe_audio(chunk_file)
        full_transcript += chunk_transcript + " "
    
    # print("Full Transcript:", full_transcript)
    
    # Create a regular expression to match words in the transcript
    word_regex = r"\b\w+\b"
    
    # Find all the words in the transcript
    words = re.findall(word_regex, full_transcript)
    
    # Calculate the total duration of the audio
    total_duration = sum(librosa.get_duration(filename=path) for path in audio_path)
    
    # Calculate the average duration of each word
    average_word_duration = total_duration / len(words)
    
    # Create a dictionary to store the clickable words and their corresponding timestamps
    clickable_words = {}
    current_time = 0
    wordsT = {}
    i = 0
    for word in words:
        # Create a clickable link for the word
        wordsT[i] = word
        clickable_words[i] = f"{VIDEO_URL}&t={int(current_time)}s"
        i += 1
        current_time += average_word_duration
        # print(word, current_time)
    
    # Print the clickable words as HTML links
    html_output = ""
    for i, link in clickable_words.items():
        html_output += f"<a href='{link}' target='_blank'>{wordsT[i]}</a> "
    
    # Print the HTML output
    # print("Clickable Words:", html_output)
    return html_output

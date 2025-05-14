# TranscribeSight Test Files

This directory contains test audio and reference transcription files for testing the TranscribeSight application.

## Usage

Upload both the audio files and their corresponding reference text files to the application to evaluate and compare the performance of different speech-to-text services.

## File Format

1. Audio files should be in WAV format
2. Reference files should be plain text files with the same base name as the audio files
   
   Example:
   - audio_sample1.wav
   - audio_sample1.txt

## Sample Files

This directory includes sample audio files with different speech patterns:
- Clear speech
- Accented speech
- Speech with background noise
- Technical terminology

Each audio file has a corresponding reference text file that contains the actual transcription of the speech.

## Creating Your Own Test Files

To create your own test files:

1. Record a WAV file (16kHz mono is recommended)
2. Create a text file with the exact transcription of the speech
3. Ensure both files have the same base name
4. Upload both files to the application

## Notes

- For best results, reference transcriptions should be as accurate as possible
- Transcriptions should include punctuation if you want to evaluate punctuation accuracy
- Consider including files with different audio quality, speaker accents, and domain-specific vocabulary to get a comprehensive evaluation
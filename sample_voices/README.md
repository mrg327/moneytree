# Voice Reference Setup Instructions

## Option 1: Download Mozilla Common Voice
1. Visit: https://commonvoice.mozilla.org/datasets
2. Accept the license terms
3. Download the English dataset (or preferred language)
4. Extract the .tar.gz file
5. Copy interesting clips from the 'clips' folder to this directory

## Option 2: Use LibriVox Audiobooks
1. Visit: https://librivox.org/
2. Find an audiobook with a voice you like
3. Download MP3 chapters
4. Run: python extract_voice_samples.py audiobook.mp3
5. Use the extracted samples from 'voice_samples' folder

## Option 3: Record Your Own Voice
1. Use the provided recording_script.txt
2. Record in a quiet environment
3. Speak clearly and consistently
4. Save as high-quality WAV file
5. Aim for 10-20 seconds of clean speech

## Testing Your Voice Reference
python demo_voice_cloning.py --analyze-voice your_reference.wav
python demo_voice_cloning.py "Test message" --reference-audio your_reference.wav

## Voice Quality Tips
- Use 6+ seconds of clean speech
- Avoid background noise or music
- Maintain consistent volume and pace
- Single speaker only (no conversations)
- Clear pronunciation and articulation
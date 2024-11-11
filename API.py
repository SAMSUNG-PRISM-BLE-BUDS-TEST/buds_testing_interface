from flask import Flask, jsonify, request
import sounddevice as sd
import wavio as wv
import librosa
import numpy as np
import pydub
import matplotlib.pyplot as plt

app = Flask(__name__)

RECORDING_FILE1 = "recording1.wav"
RECORDING_FILE2 = "recording2.wav"
FREQUENCY = 16000
CHANNELS = 2

# Default weights for MOS calculation
default_weights = {'snr': 0.2,'thd': 0.25,'frd': 0.15,'loudness': 0.2,'sharpness': 0.2}

def load_audio(file_path: str) -> tuple:
    """Load audio file"""
    return librosa.load(file_path)

def record_audio(duration: int, device1_idx: int, device2_idx: int) -> None:
    """Record audio from two devices simultaneously"""
    samplerate = FREQUENCY
    channels = CHANNELS

    with sd.InputStream(device=(device1_idx, device2_idx), samplerate=samplerate, channels=channels) as stream:
        recording = np.empty((int(duration * samplerate), channels))
        for i in range(int(duration * samplerate // 1024)):
            recording[i*1024:(i+1)*1024] = stream.read(1024)

    wv.write(RECORDING_FILE1, recording[:, 0], samplerate)
    wv.write(RECORDING_FILE2, recording[:, 1], samplerate)

def play_recorded_file(file_path: str) -> None:
    """Play recorded audio file"""
    try:
        audio = pydub.AudioSegment.from_file(file_path, format="wav")
        fs = audio.frame_rate
        array = np.array(audio.get_array_of_samples())
        sd.play(array, fs)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio: {e}")

def analyze_snr(audio: np.ndarray, sr: int) -> float:
    """Calculate signal-to-noise ratio"""
    array = np.array(audio)
    snr = 20 * np.log10(np.max(np.abs(array)) / np.mean(np.abs(array)))
    return round(snr, 3)

def analyze_thd(audio: np.ndarray, sr: int) -> float:
    """Calculate total harmonic distortion"""
    window = np.hanning(len(audio))
    audio_windowed = audio * window
    pad_length = 1024
    audio_padded = np.pad(audio_windowed, (0, pad_length), mode='constant')
    fft = np.fft.fft(audio_padded)
    harmonics = []
    for i in range(2, 11):
        harmonic_freq = i * sr / len(fft)
        harmonic_idx = min(int(harmonic_freq * len(fft)), len(fft) - 1)
        harmonic_amp = np.abs(fft[harmonic_idx])
        harmonics.append(harmonic_amp)

    harmonic_amplitudes = np.interp(np.arange(len(harmonics)), np.arange(len(harmonics)), harmonics)
    fundamental_amp = np.abs(fft[int(sr / len(fft))])
    thd = np.sqrt(np.sum([h**2 for h in harmonic_amplitudes])) / fundamental_amp
    thd_db = 20 * np.log10(thd)
    return round(thd_db, 3)

def analyze_frd(audio: np.ndarray, sr: int) -> float:
    """Calculate frequency response deviation"""
    freq = np.fft.fftfreq(len(audio), d=1.0/sr)
    magnitude = np.abs(np.fft.fft(audio))
    ideal_response = np.ones_like(magnitude)  
    deviation = np.sqrt(np.mean((magnitude - ideal_response) ** 2)) / np.max(ideal_response)
    return round(deviation, 3)

def calculate_loudness(audio_data: np.ndarray) -> float:
    """Calculate loudness based on ITU-R BS.1770 standard"""
    rms = np.sqrt(np.mean(audio_data ** 2))
    loudness = 20 * np.log10(rms) - 0.691  
    return round(loudness, 3)

def calculate_sharpness(audio_data: np.ndarray, sample_rate: int) -> float:
    """Calculate sharpness as weighted sum of high-frequency content"""
    n = len(audio_data)
    freqs = np.fft.fftfreq(n, d=1/sample_rate)
    magnitudes = np.abs(np.fft.fft(audio_data))

    cutoff_idx = np.argmin(np.abs(freqs - 8000))  
    weights = np.where(freqs > 0, 1.5 * (freqs / (freqs + 20.6)) ** 2, 0)
    sharpness = np.sum(weights[cutoff_idx:] * magnitudes[cutoff_idx:]) / np.sum(magnitudes)
    return round(sharpness, 3)

def calculate_mos(snr, thd, frd, loudness, sharpness, weights):
    """Calculate Mean Opinion Score (MOS)"""
    mos = (weights['snr'] * (snr / 30) +  weights['thd'] * (1 - (thd / 100)) +  weights['frd'] * (1 - frd) +  weights['loudness'] * (loudness / 100) +  weights['sharpness'] * sharpness)
    return round(mos * 5, 2)  


def save_waveform(file_path: str) -> None:
    """Save waveform plot of an audio file"""
    audio, sr = load_audio(file_path)
    time = np.arange(len(audio)) / sr
    plt.figure(figsize=(12, 6))
    plt.plot(time, audio)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.savefig(f'waveform_{file_path.split(".")[0]}.png', bbox_inches='tight')


def get_device_index() -> dict:
    """Get dictionary of device names and indices"""
    devices = sd.query_devices()
    device_indices = {}
    for i, device in enumerate(devices):
        device_indices[device['name']] = i
    return device_indices


@app.route('/record', methods=['POST'])
def record():
    """Record audio from two devices"""
    data = request.json
    duration = data['duration']
    device1_name = data['device1_name']
    device2_name = data['device2_name']

    device_indices = get_device_index()
    device1_idx = device_indices.get(device1_name)
    device2_idx = device_indices.get(device2_name)

    if device1_idx is None or device2_idx is None:
        return jsonify({'error': 'Device not found'}), 400

    try:
        record_audio(duration, device1_idx, device2_idx)
        save_waveform(RECORDING_FILE1)
        save_waveform(RECORDING_FILE2)
        return jsonify({'message': 'Recording successful'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/play', methods=['POST'])
def play():
    """Play recorded audio file"""
    data = request.json
    file_to_play = data['file']

    if file_to_play not in [RECORDING_FILE1, RECORDING_FILE2]:
        return jsonify({'error': 'Invalid file'}), 400

    try:
        play_recorded_file(file_to_play)
        return jsonify({'message': 'Playback successful'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/save-waveform', methods=['POST'])
def save_waveform_endpoint():
    """Save waveform of recorded audio file"""
    data = request.json
    file_to_save = data['file']

    if file_to_save not in [RECORDING_FILE1, RECORDING_FILE2]:
        return jsonify({'error': 'Invalid file'}), 400

    try:
        save_waveform(file_to_save)
        return jsonify({'message': f'Waveform saved for {file_to_save}'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/metrics', methods=['GET'])
def metrics():
    """Get metrics for recorded audio files"""
    metrics = []
    for file in [RECORDING_FILE1, RECORDING_FILE2]:
        audio, sr = load_audio(file)
        snr = analyze_snr(audio, sr)
        thd = analyze_thd(audio, sr)
        frd = analyze_frd(audio, sr)
        loudness = calculate_loudness(audio)
        sharpness = calculate_sharpness(audio, sr)
        mos = calculate_mos(snr, thd, frd, loudness, sharpness, default_weights)
        metrics.append({'file': file,'snr(dB)': float(round(snr, 3)),'thd(dB)': float(round(thd, 3)),'frd(No Unit)': float(round(frd, 3)),'loudness(dB)': float(round(loudness, 3)),'sharpness(No Unit)': float(round(sharpness, 3)),'mos': mos})
    return jsonify(metrics), 200


@app.route('/devices', methods=['GET'])
def devices():
    """Get list of available devices"""
    device_indices = get_device_index()
    return jsonify(device_indices), 200


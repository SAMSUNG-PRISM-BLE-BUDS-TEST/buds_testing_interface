from flask import Flask, jsonify, request
import sounddevice as sd
import wavio as wv
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

app = Flask(__name__)


RECORDING_FILE1 = "RIGHT.wav"
RECORDING_FILE2 = "LEFT.wav"

default_weights = {'snr': 0.2, 'thd': 0.25, 'frd': 0.15, 'loudness': 0.2, 'sharpness': 0.2}

def load_audio(file_path: str) -> tuple:
    """Load audio file"""
    return librosa.load(file_path, sr=None)

def record_audio(duration: int, device1_idx: int, device2_idx: int,
                 frequency1: int = 16000, frequency2: int = 16000,
                 channels1: int = 2, channels2: int = 2) -> None:
    """Record audio from two devices with separate frequency and channel settings."""
    try:
        # Allocate buffers for each device's recording
        recording1 = np.empty((int(duration * frequency1), channels1), dtype=np.float32)
        recording2 = np.empty((int(duration * frequency2), channels2), dtype=np.float32)

        # Create input streams for both devices
        with sd.InputStream(device=device1_idx, samplerate=frequency1, channels=channels1) as stream1, \
             sd.InputStream(device=device2_idx, samplerate=frequency2, channels=channels2) as stream2:

            # Record audio from both devices
            for i in range(0, int(duration * min(frequency1, frequency2)), min(frequency1, frequency2) // 10):
                if i < len(recording1):
                    recording1[i:i + frequency1 // 10] = stream1.read(frequency1 // 10)[0]
                if i < len(recording2):
                    recording2[i:i + frequency2 // 10] = stream2.read(frequency2 // 10)[0]

        # Normalize and convert to int16 for both recordings
        recording1 = np.int16(recording1 / np.max(np.abs(recording1)) * 32767)
        recording2 = np.int16(recording2 / np.max(np.abs(recording2)) * 32767)

        # Save recordings as WAV files
        wv.write(RECORDING_FILE1, recording1, frequency1, sampwidth=2)
        wv.write(RECORDING_FILE2, recording2, frequency2, sampwidth=2)

    except Exception as e:
        print(f"Error recording audio: {e}")
        raise

def analyze_snr(audio: np.ndarray, sr: int) -> float:
    """Calculate signal-to-noise ratio using spectral analysis"""
    spectrum = np.abs(np.fft.fft(audio))
    signal_power = np.sum(spectrum[:len(spectrum)//2]**2)
    noise_power = np.sum(spectrum[len(spectrum)//2:]**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return round(snr, 3)

def analyze_thd(audio: np.ndarray, sr: int) -> float:
    """Calculate total harmonic distortion using FFT"""
    fft = np.fft.fft(audio)
    fundamental_idx = np.argmax(np.abs(fft[:len(fft)//2]))
    harmonics = [np.abs(fft[(fundamental_idx * i) % len(fft)]) for i in range(2, 6)]
    thd = np.sqrt(sum(h**2 for h in harmonics)) / np.abs(fft[fundamental_idx])
    thd_db = 20 * np.log10(thd)
    return round(thd_db, 3)

def analyze_frd(audio: np.ndarray, sr: int) -> float:
    """Calculate frequency response deviation"""
    spectrum = np.abs(np.fft.fft(audio))
    target_response = np.linspace(1, 0, len(spectrum)//2)
    deviation = np.sqrt(np.mean((spectrum[:len(target_response)] - target_response)**2))
    return round(deviation, 3)

def calculate_loudness(audio_data: np.ndarray) -> float:
    """Calculate loudness using RMS"""
    rms = np.sqrt(np.mean(audio_data ** 2))
    loudness = 20 * np.log10(rms + 1e-9)
    return round(loudness, 3)

def calculate_sharpness(audio_data: np.ndarray, sr: int) -> float:
    """Calculate sharpness using high-frequency emphasis"""
    fft = np.abs(np.fft.fft(audio_data))
    freqs = np.fft.fftfreq(len(audio_data), 1/sr)
    sharpness = np.sum(freqs * fft) / np.sum(fft)
    return round(sharpness, 3)

def calculate_mos(snr, thd, frd, loudness, sharpness, weights):
    """Calculate Mean Opinion Score (MOS)"""
    mos = (weights['snr'] * (snr / 30) +
           weights['thd'] * (1 - abs(thd / 100)) +
           weights['frd'] * (1 - frd) +
           weights['loudness'] * (loudness / 100) +
           weights['sharpness'] * sharpness)
    return round(mos * 5, 2)

@app.route('/record', methods=['POST'])
def record():
    """Record audio from two devices with separate frequency and channel settings."""
    data = request.json

    # Get user inputs with default values
    duration = data.get('duration', 5)  # Default to 5 seconds
    device1_name = data.get('device1_name')
    device2_name = data.get('device2_name')
    frequency1 = data.get('frequency1', 16000)  # Default to 16kHz for device 1
    frequency2 = data.get('frequency2', 16000)  # Default to 16kHz for device 2
    channels1 = data.get('channels1', 2)  # Default to 2 (stereo) for device 1
    channels2 = data.get('channels2', 2)  # Default to 2 (stereo) for device 2

    # Validate device indices
    device_indices = {device['name']: i for i, device in enumerate(sd.query_devices())}
    device1_idx = device_indices.get(device1_name)
    device2_idx = device_indices.get(device2_name)

    if device1_idx is None or device2_idx is None:
        return jsonify({'error': 'Device not found'}), 400

    # Record audio
    try:
        record_audio(duration, device1_idx, device2_idx, frequency1, frequency2, channels1, channels2)
        return jsonify({'message': 'Recording successful'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/metrics', methods=['GET'])
def metrics():
    """Get metrics for recorded audio files"""
    metrics = []
    for file in [RECORDING_FILE1, RECORDING_FILE2]:
        try:
            audio, sr = load_audio(file)
            snr = float(analyze_snr(audio, sr))  # Convert numpy.float32 to float
            thd = float(analyze_thd(audio, sr))
            frd = float(analyze_frd(audio, sr))
            loudness = float(calculate_loudness(audio))
            sharpness = float(calculate_sharpness(audio, sr))
            mos = calculate_mos(snr, thd, frd, loudness, sharpness, default_weights)
            
            metrics.append({
                'file': file,
                'snr(dB)': snr,
                'thd(dB)': thd,
                'frd(No Unit)': frd,
                'loudness(dB)': loudness,
                'sharpness(No Unit)': sharpness,
                'mos': mos
            })
        except Exception as e:
            metrics.append({'file': file, 'error': str(e)})

    return jsonify(metrics), 200


@app.route('/devices', methods=['GET'])
def devices():
    """Get list of available devices"""
    devices = sd.query_devices()
    return jsonify({device['name']: i for i, device in enumerate(devices)}), 200

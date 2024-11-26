from flask import Flask, render_template, redirect, url_for,jsonify, send_file,request
from mobile_automate import * 
from datetime import datetime
from fpdf import FPDF
import webview
import threading 
import signal,requests,os
import sounddevice as sd
import wavio as wv
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

app = Flask(__name__)
#change paniko da
connected = True  
stbg,dtbg,ttbg="white","white","white"
stc,dtc,ttc="black","black","black"
sn,ss,volume,callclr,calltxt="","","","",""
sinfo=0
stw,dtw,ttw="Not yet Tested","Not yet Tested","Not yet Tested"
d=""
vt,snt,sst,calt=1,"","",""
stcl=0
#log events array
le=[]
@app.route('/', methods=["GET","POST"])
def main_func():    
    a=ble_status()
    if (a=="Device is not connect"):
        bgclr="red"
    else:
        connected=True
        tsp = datetime.now()
        tsp = tsp.strftime('%d-%m-%Y %H:%M:%S')
        le.append([tsp,"BLE DEVICE CONNECTED"])
        bgclr="green"
    return render_template("index.html", sts=a,bgclr=bgclr)

@app.route('/test', methods=["GET","POST"])
def test():
    global vt,snt,sst,calt,stcl,le
    if len(le)==0:
        le=[""]
    if connected:
        sn=get_metadata()
        call=call_status()
        ss=sn[0]
        sn=sn[1].title()
        volume=get_volume()[0]
        if vt!=volume:
            tsp = datetime.now()
            tsp = tsp.strftime('%d-%m-%Y %H:%M:%S')
            le.append([tsp,"VOLUME CHANGED TO "+str(vt)])
            vt=volume
            
        if snt!=sn:
            tsp = datetime.now()
            tsp = tsp.strftime('%d-%m-%Y %H:%M:%S')
            le.append([tsp,"SONG CHANGED TO "+str(sn)])
            snt=sn
            
        if sst!=ss:
            tsp = datetime.now()
            tsp = tsp.strftime('%d-%m-%Y %H:%M:%S')
            le.append([tsp,"SONG IS "+str(ss)])
            sst=ss
        
        if calt != call:
            tsp = datetime.now()
            tsp = tsp.strftime('%d-%m-%Y %H:%M:%S')
            le.append([tsp,call])
            calt=call
        
        if stcl==2:
            tsp = datetime.now()
            tsp = tsp.strftime('%d-%m-%Y %H:%M:%S')
            le.append([tsp,"PHONE CALL IS REJECTED"])
            
        if call == "INCOMING CALL DETECETED": 
            stcl=1    
            callclr="red"
            calltxt="white"
        elif call == "No call is detected":
            if stcl==1:
                stcl=2
            else:
                stcl=0
            callclr ="white"
            calltxt="black"
        elif call=="phone is currently on a call":
            stcl=0
            callclr="green"
            calltxt="white"
        elif call == "The call is on hold":
            callclr="yellow"
            calltxt="black"
        else:
            callclr="blue"
            calltxt="white"
            
        if sn!="":
            sinfo=1
        return render_template('mobile_test.html',ss=ss,sn=sn,volume=volume,stbg=stbg,dtbg=dtbg,ttbg=ttbg,stc=stc,dtc=dtc,ttc=ttc,stw=stw,dtw=dtw,ttw=ttw,callsts=call,callclr=callclr,calltxt=calltxt)  
    else:
        return redirect(url_for('main_func'))
    
@app.route('/touchtest', methods=["GET","POST"])
def touchtest():
    sttest()
    dttest()
    tttest()
    return render_template('mobile_test.html',ss=ss,sn=sn,volume=volume,stbg=stbg,dtbg=dtbg,ttbg=ttbg,stc=stc,dtc=dtc,ttc=ttc,stw=stw,dtw=dtw,ttw=ttw,callsts=call,callclr=callclr,calltxt=calltxt)  

@app.route('/sttest', methods=["GET","POST"])
def sttest():
    a=single_tap()
    if (a=="Single tap is not working"):
        stbg="red"
        stc="white"
        stw="NOT WORKING"
        dtw="TESTING"
    elif (a=="Unable to fetch song info" or sinfo == 0):
        return render_template('mobile_test.html',ss=ss,sn=sn,volume=volume,stbg=stbg,dtbg=dtbg,ttbg=ttbg,stc=stc,dtc=dtc,ttc=ttc,stw=stw,dtw=dtw,ttw=ttw,callsts=call,callclr=callclr,calltxt=calltxt)  
    elif(a=="Single tap is working"):
        stbg="green"
        stc="white"
        stw="WORKING"
        dtw="TESTING"
    return render_template('mobile_test.html',ss=ss,sn=sn,volume=volume,stbg=stbg,dtbg=dtbg,ttbg=ttbg,stc=stc,dtc=dtc,ttc=ttc,stw=stw,dtw=dtw,ttw=ttw,callsts=call,callclr=callclr,calltxt=calltxt)  
    

@app.route('/dttest', methods=["GET","POST"])
def dttest():
    a=double_tap_music()
    if (a=="Double tap is not working"):
        dtbg="red"
        dtc="white"
        dtw="NOT WORKING"
        ttw="TESTING"
    elif (a=="Unable to fetch song info" or sinfo == 0):
        return render_template('mobile_test.html',ss=ss,sn=sn,volume=volume,stbg=stbg,dtbg=dtbg,ttbg=ttbg,stc=stc,dtc=dtc,ttc=ttc,stw=stw,dtw=dtw,ttw=ttw,callsts=call,callclr=callclr,calltxt=calltxt)  
    else:
        dtbg="green"
        dtc="white"
        dtw="WORKING"
        ttw="TESTING"
    return render_template('mobile_test.html',ss=ss,sn=sn,volume=volume,stbg=stbg,dtbg=dtbg,ttbg=ttbg,stc=stc,dtc=dtc,ttc=ttc,stw=stw,dtw=dtw,ttw=ttw,callsts=call,callclr=callclr,calltxt=calltxt)  


@app.route('/tttest', methods=["GET","POST"])
def tttest():
    a=triple_tap_music()
    if (a=="Triple tap is not working"):
        ttbg="red"
        ttc="white"
        ttw="WORKING"
    elif (a=="Unable to fetch song info" or sinfo == 0):
        return render_template('mobile_test.html',ss=ss,sn=sn,volume=volume,stbg=stbg,dtbg=dtbg,ttbg=ttbg,stc=stc,dtc=dtc,ttc=ttc,stw=stw,dtw=dtw,ttw=ttw,callsts=call,callclr=callclr,calltxt=calltxt)  
    else:
        ttbg="green"
        ttc="white"
        ttw="WORKING"
    return render_template('mobile_test.html',ss=ss,sn=sn,volume=volume,stbg=stbg,dtbg=dtbg,ttbg=ttbg,stc=stc,dtc=dtc,ttc=ttc,stw=stw,dtw=dtw,ttw=ttw,callsts=call,callclr=callclr,calltxt=calltxt)  
    


@app.route('/get_metadata_value')
def get_metadata_value():
    metadata = get_metadata()
    volume=get_volume()
    call=call_status()
    # print(call,metadata)
    return jsonify({'status': metadata[0], 'song_name': metadata[1], 'volume':volume,'calls':call})



@app.route('/get_connection_value')
def get_connection_value():
    a = ble_status()
    return jsonify({'status': a})

@app.route('/event_log')
def event_log():
    if (len(le)):
        eventar=le
    else:
        eventar=[[" ","NO EVENTS YET"]]
    
    data = eventar
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    col_widths = [60, 90, 40]
    # Create table header
    for i, col in enumerate(data[0]):
        pdf.cell(col_widths[i], 10, col, 1)
    pdf.ln()
    for row in data[1:]:
        for i, col in enumerate(row):
            pdf.cell(col_widths[i], 10, str(col), 1)
        pdf.ln()
    pdf.output('log_file.pdf')
    
    return render_template("event_log.html",le=eventar)
 
 
@app.route('/download_log_file')
def download_log_file():
    # Serve the PDF file for download
    return send_file('log_file.pdf', as_attachment=True)

class Api:
    def close_window(self):
        webview.windows[0].destroy()  
    
    def minimize_window(self):
        webview.windows[0].minimize() 
    
    def maximize_window(self):
        if not webview.windows[0].fullscreen:
            webview.windows[0].toggle_fullscreen() 
        else:
            webview.windows[0].toggle_fullscreen()

@app.route('/shutdown', methods=['POST'])
def shutdown():
    os.kill(os.getpid(), signal.SIGINT)  # Terminate Flask server process
    return "Server shutting down..."

def on_window_close():
    try:
        requests.post('http://127.0.0.1:5000/shutdown')
    except requests.exceptions.RequestException:
        pass  # Ignore errors if server is already down
    
    
#AUDIO QUALITY TESTING

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
    mos = (weights['snr'] * (snr / 30) +
           weights['thd'] * (1 - abs(thd / 100)) +
           weights['frd'] * (1 - frd) +
           weights['loudness'] * (loudness / 100) +
           weights['sharpness'] * sharpness)
    return round(mos * 5, 2)

@app.route('/record', methods=['GET','POST'])
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
        erf="ERROR DEVICE NOT FOUND"
        #return jsonify({'error': 'Device not found'}), 400
        return render_template("audio.html",erf=erf,devices=d)

    # Record audio
    try:
        record_audio(duration, device1_idx, device2_idx, frequency1, frequency2, channels1, channels2)
        #return jsonify({'message': 'Recording successful'}), 200
        erf="RECORDING SUCCESSFUL"
        return render_template("audio.html",devices=d,erf=erf)

    except Exception as e:
        erf=str(e)
        return render_template("audio.html",devices=d,erf=erf)
        #return jsonify({'error': str(e)}), 400

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

    return render_template("metrics_out.html")

@app.route('/devices', methods=['GET'])
def devices():
    """Get list of available devices"""
    devices = sd.query_devices()
    return devices

@app.route('/audio',method=['GET'])
def audio():
    d=devices()
    erf=""
    return render_template("audio.html",devices=d,erf=erf)
def run_flask():
    app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
    threading.Thread(target=run_flask).start()
    api = Api()  
    window=webview.create_window('Mobile Automation App', 'http://127.0.0.1:5000', js_api=api, frameless=True)
    
    window.events.closed += on_window_close
    webview.start()
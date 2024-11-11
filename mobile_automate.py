import subprocess
import time

call,call_ans=0,0

def remote_connect():
    ip_addrss=input("Enter the ip address: ")
    subprocess.run(["adb", "tcpip", "5555"], stdout=subprocess.PIPE)
    time.sleep(1)
    a=str(ip_addrss)+":5555"
    subprocess.run(["adb", "connect",a], stdout=subprocess.PIPE)
 
def get_metadata():
    result = subprocess.run(["adb", "shell", "dumpsys", "media_session"], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    if ("description=" in output and "state=PlaybackState" in output):
        metadata = output.split("description=")[1].split("\n")[0].strip()
        metadata=metadata.split(",")[0]
        current_state=output.split("state=PlaybackState {state=")[1][0]
        if current_state == "1":
            song_status = "STOPPED"
        elif current_state == "2":
            song_status = "PAUSED"
        elif current_state == "3":
            song_status = "PLAYING"
        else:
            return ("No song found")
        
    else:
        return ("No song found.")
    return [song_status,metadata]
        
def get_volume():
    volume,k=0,0
    result = subprocess.run(["adb", "shell", "dumpsys", "audio"], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    a1=output.split("-")
    for i,j in enumerate(a1):
        if ("STREAM_MUSIC" in j and  "streamVolume:" in j):
            for k in j.split():
                if "streamVolume:" in k:
                    volume=k.split(":")[1]
    return [volume,k]

def call_status():
    global call,call_ans
    result = subprocess.run(["adb", "shell", "dumpsys", "telephony.registry"], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    phone_data0=output.split("Phone Id=1")[0]
    phone_data0=phone_data0.split("\r\n")
    mCallState= phone_data0[2].strip()
    mRingingCallState=phone_data0[3]
    # print(mCallState)
    # print(phone_data0[13]) - To detect snr values
    if (mCallState== "mCallState=1"):
        call_state="INCOMING CALL DETECETED"
        call=1
        call_ans=0
    elif (mCallState== "mCallState=2"):
        call_state="phone is currently on a call"
        call_ans=1
    elif (mCallState=="mCallState=3"):
        call_state="phone is currently dialing"
    elif (mCallState=="mCallState=4"):
        call_state="The call is on hold"
    else:
        call_state="No call is detected"
    return call_state
    
def call_busy():
    if (call and call_ans):
        return ("The Call is Answered")
    elif (call==1 and call_ans==0):
        return ("The Call is Rejected")
    else:
        return ("No Call Detected")

def single_tap():
    sn= get_metadata()
    if "No" not in sn[0]:
        ss=sn[1]
        sn=sn[0]
        print("Perform single tap and press enter")
        input()
        sn1= get_metadata()
        if "No" not in sn1[0]:
            ss1=sn1[1]
            sn1=sn1[0]
            if ss1!=ss:
                return ("Single tap is working")
            else:
                return("Single tap is not working")
        else:
            return("Unable to fetch song info")
    else:
        return("Unable to fetch song info")
    
def double_tap_music():
    sn= get_metadata()
    if "No" not in sn[0]:
        ss=sn[1]
        sn=sn[0]
        print("Perform double tap and press enter")
        input()
        sn1= get_metadata()
        if "No" not in sn1[0]:
            ss1=sn1[1]
            sn1=sn1[0]
            if sn1!=sn:
                return ("Double tap is working")
            else:
                return("Double tap is not working")
        else:
            return("Unable to fetch song info")
    else:
        return("Unable to fetch song info")


def double_tap_call():
    print("Perform incoming call and press Enter")
    input()
    cs=call_status()
    if (cs=="INCOMING CALL DETECETED"):
        print("Perform Double tap and press Enter")
        input()
        cs=call_status()
        if (cs!="phone is currently on a call"):
           return("Double tap not working")
        
        print("Perform Double tap and press Enter")
        input()
        cs=call_status()
        if (cs!="No call is detected"):
           return("Double tap not working")
        return ("Double tap not working")
    
def triple_tap_music():
    sn= get_metadata()
    if "No" not in sn[0]:
        ss=sn[1]
        sn=sn[0]
        print("Perform triple tap and press enter")
        input()
        sn1= get_metadata()
        if "No" not in sn1[0]:
            ss1=sn1[1]
            sn1=sn1[0]
            if sn1!=sn:
                return ("Triple tap is working")
            else:
                return("Triple tap is not working")
        else:
            return("Unable to fetch song info")
    else:
        return("Unable to fetch song info")
    
def ble_status():
    result = subprocess.run(["adb", "shell", "dumpsys", "bluetooth_manager"], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    s=get_volume()
    if s[1]!="speaker" and "mCurrentState: Connected" in output:
        return (f"Device is connected ")
    else:
        return ("Device is not connect")


def volume_up():
    subprocess.run(["adb", "shell", "input" ,"keyevent" ,"24"], stdout=subprocess.PIPE)

def volume_down():
    subprocess.run(["adb", "shell", "input" ,"keyevent" ,"25"], stdout=subprocess.PIPE)


def mute():
    subprocess.run(["adb", "shell", "input" ,"keyevent" ,"164"], stdout=subprocess.PIPE)

def pause_play():
    subprocess.run(["adb", "shell", "input" ,"keyevent" ,"85"], stdout=subprocess.PIPE)


def music_pre():
    subprocess.run(["adb", "shell", "input" ,"keyevent" ,"88"], stdout=subprocess.PIPE)


def music_nxt():
    subprocess.run(["adb", "shell", "input" ,"keyevent" ,"87"], stdout=subprocess.PIPE)



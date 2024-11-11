from flask import Flask, render_template, redirect, url_for,jsonify, send_file
from mobile_automate import * 
from datetime import datetime
from fpdf import FPDF

app = Flask(__name__)
#change paniko da
connected = True  
stbg,dtbg,ttbg="white","white","white"
stc,dtc,ttc="black","black","black"
sn,ss,volume,callclr,calltxt="","","","",""
sinfo=0
stw,dtw,ttw="Not yet Tested","Not yet Tested","Not yet Tested"

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
    global vt,snt,sst,calt,stcl
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

if __name__ == '__main__':
    app.run(debug=True)  
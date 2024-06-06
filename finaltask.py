import time as tim
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as plotly
import plotly.graph_objects as go
import streamlit as st
import streamlit_vertical_slider as svs
from scipy.io.wavfile import write
from streamlit_option_menu import option_menu
matplotlib.use('Agg')

#..............................................Front Page.....................................................................................

st.set_page_config(page_title="Equalizer", page_icon=":level_slider:",layout="wide")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;} 2q2
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

button_style = """
        <style>
        .stButton > button {
            width: 90px;
            height: 35px;
        }
        </style>
        """
st.markdown(button_style, unsafe_allow_html=True)

with open("style.css") as source_des:
    st.markdown(f"""<style>{source_des.read()}</style>""", unsafe_allow_html=True)


if 'i' not in st.session_state:
    st.session_state['i'] = 0



file=st.sidebar.file_uploader("")       # To Upload Files
timesketch_col,  inversesketch_col = st.columns(2, gap="small")             # To Show Figures next_to_each_other

# Options
with st.sidebar:                        
    choise = option_menu(
        menu_title = " Main Menu ",
        options = ["Sin_Wave","Medical_Signal","Audio_Frequency","Music_Instruments","Vowels","Animals"]
    )

inverse_btn=st.sidebar.button("Apply")                         # Inverse Button
st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:0px;}</style>', unsafe_allow_html=True) 
normal_col,dynamic_col=st.sidebar.columns(2, gap="small")      # To Show checkboxes next_to_each_other
normal=normal_col.checkbox("Normal_Figures")                   # To Show Normal Figures
dynamic=dynamic_col.checkbox("Dynamic_Figures")                # To Show Dynamic Figures
Spectrogram=st.sidebar.checkbox("Spectrogram")                 # To Show Spectrogram
audio_col,invaudio_col=st.sidebar.columns(2,gap="small")       # To Show audio and inverse_audio next_to_each_other

#.......................................................Functions..................................................................

# PLot Function
def sketch(x_axis,y_axis,x_label, y_label, title):
        figure = plotly.line()
        figure.add_scatter(x=x_axis, y=y_axis,mode='lines',name=title,line=dict(color='red'))
        figure.update_layout(width=500, height=400,template='simple_white',yaxis_title= y_label,xaxis_title=x_label,hovermode="x")
        st.plotly_chart(figure,use_container_width=True)

# Function to make  Dynamic chart
def make_chart(df, yaxis_min, yaxis_max):
    fig = go.Figure(layout_yaxis_range=[yaxis_min, yaxis_max])
    fig.add_trace(go.Line(x=df['x_axis'], y=df['y_axis'],name="Dynamic Figure",line=dict(color='red')))
    fig.update_layout(width=500, height=400,template='simple_white',yaxis_title="Amplitude (V)",xaxis_title="Time (Sec)",hovermode="x")
    st.plotly_chart(fig,use_container_width=True)

# Dynamic Figure
def dynamic(df):
    plot_spot=st.empty()
    ymax = max(df["y_axis"])
    ymin = min(df["y_axis"])
    for st.session_state['i'] in range(0,len(df)):
        df_tmp=df.iloc[st.session_state['i']:st.session_state['i']+100,:]
        with plot_spot:
            make_chart(df_tmp, ymin, ymax)
        tim.sleep(0.0000000001) 
    
# Plot Spectrogram
def plot_spectrogram(signal, sample_rate):
    fig2 = plt.figure(figsize=(5, 2))
    plt.specgram(signal, Fs=sample_rate)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (sec)")
    plt.colorbar()
    st.pyplot(fig2)

# Sliders function
def sliders(no_col,writes=[]):
    sliders = []
    columns = st.columns(no_col,gap='small')
    for column  in columns:
        with column:
            slider = svs.vertical_slider(key=f"slider{columns.index(column)}",default_value=1,step=0.5,min_value=0,max_value=10,
                                        thumb_color="black", slider_color="red", track_color="grey")
            if slider == None:
                slider  = 1 
            sliders.append(slider) 
    for i in range(0, no_col):
        with columns[i]:
            slider_val = writes[i]
            st.write(f" { slider_val }")
    return sliders    

# Fourier Transform
def get_freq(magnitude=[],time=[],sample_rate=0):
    magnitude=np.fft.rfft(magnitude)
    n_samples = len(magnitude)

    if sample_rate==0:
        return  magnitude
    else :
        duration =n_samples/sample_rate
        return duration, magnitude
         
# Inverse Fourier Transform
def get_inverse(frequency):
    inverse=np.fft.irfft(frequency)
    return inverse

# Upload CSV File
def upload_Signal():
    if file :
        File = pd.read_csv(file)  
        signal_time= File[File.columns[0]].to_numpy()
        signal_magnitude= File[File.columns[1]].to_numpy()
        sample_period = (signal_time[1]-signal_time[0])
        sample_rate = 1/sample_period
        with timesketch_col:
            if normal:
                sketch(signal_time,signal_magnitude,"Time (Sec)","Amplitude (V)","Time_Domain")
            elif Spectrogram:
                plot_spectrogram(signal_magnitude, sample_rate )
            elif dynamic:
                df=pd.DataFrame({"x_axis":signal_time,"y_axis":signal_magnitude})
                dynamic(df)
        return signal_magnitude,signal_time,sample_rate  
 
# Upload Audio WAV File
def upload_audio():
    if file:
        audio_col.audio(file, format= 'audio/wav')
        magnitude , sample_rate = librosa.load(file)
        length = magnitude.shape[0] / sample_rate
        time = np.linspace(0, length, magnitude.shape[0])  
        with timesketch_col:
            if normal:
                sketch(time,magnitude,"Time (Sec)","Amplitude (V)","Time_Domain")
            elif Spectrogram:
                plot_spectrogram(magnitude, sample_rate )
            elif dynamic:
                df=pd.DataFrame({"x_axis":time,"y_axis":magnitude})
                dynamic(df) 
        return magnitude ,  sample_rate, time

# Changing
def change(mag, ranges,sliders_value,sample_rate=0,time=[]):
    if  sample_rate==0:
        magnitude=get_freq(mag,time)
        for i, j in zip(range(len(sliders_value)),range(len(ranges))):
            magnitude[ranges[j]:ranges[j+1]]*=sliders_value[i]
    else:
        duration, magnitude =get_freq(magnitude=mag,sample_rate=sample_rate)
        for i, j in zip(range(len(sliders_value)),range(0,len(ranges),2)):
            magnitude[int(duration*ranges[j]):int(duration*ranges[j+1])]*=sliders_value[i]
    return magnitude

# Sketch Inverse Signal
def Draw_inverse(inverse,time,sample_rate):
    with inversesketch_col:
        if normal:
            sketch(time,inverse,"Time (Sec)","Amplitude (V)","Inverse_Transform")
        elif Spectrogram:
            plot_spectrogram(inverse, sample_rate)
        elif dynamic:
            df=pd.DataFrame({"x_axis":time,"y_axis":inverse})
            dynamic(df) 

# Return to audio    
def return_audio(inverse,sample_rate):
    norm=np.int16(inverse*(32767/inverse.max()))            # Data of inverse in 16bits
    write('Edited_audio.wav' , round(sample_rate ), norm)   
    invaudio_col.audio('Edited_audio.wav' , format= 'audio/wav')

#................................................ Sin Signal Option  .....................................................................

if choise== "Sin_Wave" :
    writes=[" 0 : 10 "," 10 : 20 "," 20 : 30"," 30 : 40"," 40 : 50 "," 50 : 60 "," 60 : 70 "," 70 : 80"," 80 : 90"," 90 : 100 "]
    ranges=[0,10,20,30,40,50,60,70,80,90,100]
    slider_value=sliders(no_col=10,writes=writes)
    if file:
        mag,time,sample_rate=upload_Signal()
        power=change(mag=mag,ranges=ranges,sliders_value=slider_value,time=time)
        if inverse_btn:
            inverse = get_inverse(power)
            Draw_inverse(inverse,time,sample_rate)
# ................................................ Medical Signal Option........................................................................

elif choise== "Medical_Signal":
    writes=[" Bradycardia "," Normal_Range "," Atrial_Tachycardia "," Atrial_Flutter  "," Atrial_Fibrillation "]
    ranges=[0,60,90,250,300,600]
    slider_value=sliders(no_col=5,writes=writes)
    if file:
        mag,time,sample_rate=upload_Signal()
        power=change(mag=mag,ranges=ranges,sliders_value=slider_value,time=time)
        if inverse_btn:
            inverse = get_inverse(power)
            Draw_inverse(inverse,time,sample_rate)
#..............................................Vowels Option .............................................................................
    
elif choise=="Vowels":
    writes=[" Letter S "," Letter R ", " Letter SH "," Letter O "]
    ranges=[4433,8615,600,3000,2000,7000,150,1000]
    slider_value=sliders(no_col=4,writes=writes)
    if file:
        mag , sample_rate, time=upload_audio()
        power=change(mag=mag,ranges=ranges,sliders_value=slider_value,sample_rate=sample_rate)
        if inverse_btn:
            inverse=get_inverse(power)               
            Draw_inverse(inverse,time,sample_rate)
            return_audio(inverse,sample_rate)
#.................................................. Music Instruments Option.....................................................................

elif choise=="Music_Instruments":
    writes=[" Drums "," Acoustic Guitar "," Cymbals "]
    ranges=[0,1000,3000,6000,9000,20000]
    slider_value=sliders(no_col=3,writes=writes)
    if file:
        mag , sample_rate, time=upload_audio()
        power=change(mag=mag,ranges=ranges,sliders_value=slider_value,sample_rate=sample_rate)
    if inverse_btn:
        inverse = get_inverse(power)    
        Draw_inverse(inverse,time,sample_rate)
        return_audio(inverse,sample_rate)
#.....................................................Animals Option........................................................................

elif choise=="Animals":
    writes=[" Dog "," Chick "]
    ranges=[10,2000,3000,10000]
    slider_value=sliders(no_col=2,writes=writes)
    if file:
        mag , sample_rate, time=upload_audio()
        power=change(mag=mag,ranges=ranges,sliders_value=slider_value,sample_rate=sample_rate)
    if inverse_btn:
        inverse = get_inverse(power)      
        Draw_inverse(inverse,time,sample_rate)
        return_audio(inverse,sample_rate)
#...................................................Audio Frequency Option..................................................................

elif choise=="Audio_Frequency":
    pitch=st.slider(" Frequency Rate ", min_value=-20,max_value=20,value=0)
    if file:
        magnitude, sample_rate, time= upload_audio()
        update=librosa.effects.pitch_shift(magnitude,sr=sample_rate,n_steps=pitch)
        if inverse_btn:
            Draw_inverse(update,time,sample_rate)
            return_audio(update,sample_rate)
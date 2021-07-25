import numpy as np
import pretty_midi
import random
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from progress.bar import Bar
import sys

musician=load_model("./model/musicianAi.h5")
musician.load_weights("./model/weights/musicianAi_weights-Yedek-3.h5")

noteValues=[1/256,3/512,7/1024,15/2048,1/128,3/256,7/512,15/1024,1/64,3/128,7/256,15/512,1/32,3/64,7/128,15/256,1/16,3/32,7/64,15/128,1/8,3/16,7/32,15/64,1/4,3/8,7/16,15/32,1/2,3/4,7/8,15/16,1.0,3/2,7/4,15/8,2.0,3.0,3.5,3.75]
note_sequences=[]
vel_sequences=[]
nVal_sequences=[]

midi=pretty_midi.PrettyMIDI()
piano=pretty_midi.Instrument(program=0)

# print(note_sequence)
tempo=120
length=300
try:
    tempo=int(sys.argv[1])
except:
    pass

try:
    length=int(sys.argv[2])
except:
    pass

    

def get_datas():
    
    files = os.listdir("./classical-piano-type0/")
#     global noteTypes
#     global noteValues

    notes=[]
    # offsets=[]
    durations=[]
    velocities=[]

    for file in files:
        midi=pretty_midi.PrettyMIDI(os.path.join("./classical-piano-type0/",file))
        beats=midi.get_beats()
        tempos=midi.get_tempo_changes()
        tempoChanges=list(tempos[0])
        tempo_ticks=list(tempos[1])
        tempoChanges.reverse()
        for instrument in midi.instruments:
            s=0
            tindex=0
            for note in instrument.notes:
                for i,tempo in enumerate(tempoChanges):
                    if note.start>=tempo:
                        tindex=i
                        break
        #         tindex=tempoChanges.index(note.start)
                velocities.append(note.velocity)
                notes.append(note.pitch)
                durations.append(round((note.end-note.start)/(60/tempo_ticks[tindex]),5))

            for s in range(len(durations)):
                absolute_difference_function = lambda list_value : abs(list_value - durations[s])
                closest_value = min(noteValues, key=absolute_difference_function)
                durations[s]=noteValues[noteValues.index(closest_value)]
                durations[s]=noteValues.index(durations[s])

        #     print(len(durations))  ##durations OK
        #     print(len(notes))  ##notes OK
        #     print(len(velocities))  ##velocities OK
            return notes,durations,velocities
def prepare_sequences(notes,n_vocab):
    s_length=100
    
    note_input=[]
    note_output=[]
    
    for i in range(0, len(notes) - s_length, 1):
        note_input.append(notes[i:i+s_length])
        n_ouput=[0 for s in range(n_vocab)]
        n_ouput[notes[i+s_length]]=1
        note_output.append(n_ouput)
    n_pattern=len(note_input)
    
    note_input=np.reshape(note_input, (n_pattern,s_length,1))
    note_input=note_input/float(n_vocab)
    
    note_output=np.reshape(note_output,(n_pattern,n_vocab))
    return note_input
    
def generate_firsNote():
    
    global noteValues,nVal_sequence,note_sequence,vel_sequence
    
    note=random.randint(0,106)+21
    note_sequence[-1]=[note/127]
    velocity=random.randint(0,106)+21
    vel_sequence[-1]=[velocity/127]
    noteValue=random.randint(0,len(noteValues)-16)+16
    nVal_sequence[-1]=[noteValue/len(noteValues)]
    
def create_times(noteValue,lastEnd):
    global tempo
    start=lastEnd
    end=start+noteValue*(60/tempo)
    return start,end
    
def composeMusic():
    start=0
    end=0
    global note_sequence,vel_sequence,nVal_sequence,piano,length
    notes,durations,velocities=get_datas()
    note_sequences=prepare_sequences(notes,127)
    vel_sequences=prepare_sequences(velocities,127)
    nVal_sequences=prepare_sequences(durations,len(noteValues))
    note_sequence=[[.0] for i in range(100)]
    vel_sequence=[[.0]for i in range(100)]
    nVal_sequence=[[.0] for i in range(100)]
    generate_firsNote()
#     start1=random.randint(0,len(note_sequences)-1)
#     print(len(note_sequences[start1]))
    note_sequence=np.reshape(note_sequence,(-1,100,1)) 
#     note_sequence=np.reshape(note_sequences[start1],(-1,100,1))
#     start2=random.randint(0,len(vel_sequences)-1)
    vel_sequence=np.reshape(vel_sequence,(-1,100,1)) 
#     vel_sequence=np.reshape(vel_sequences[start2],(-1,100,1))
#     start3=random.randint(0,len(nVal_sequences)-1)
    nVal_sequence=np.reshape(nVal_sequence,(-1,100,1))
#     nVal_sequence=np.reshape(nVal_sequences[start3],(-1,100,1))
    bar=Bar("Besteleniyor...", max=length)
    for i in range(length):
        newNote=musician.predict([note_sequence, vel_sequence,nVal_sequence])
#         note=list(newNote[0][0])
#         vel=list(newNote[1][0])
#         nVal=list(newNote[2][0])
        dices=[]
        indexes=[]
        altIndexes=[]
        for a in range(3):
#             idxes=np.argsort(newNote[a][0])[-6:]
            idx=newNote[a][0].argmax()
            altIndexes.append(idx)
#             indexes.append(idxes)
#             dices.append(newNote[a][0][idxes])
#             dices[a]=np.exp(dices[a])
#             dices[a]=dices[a]/np.sum(dices[a])
#         dice=np.random.multinomial(1,dices[0])
#         dice=newNote[0][0]
#         offset=np.argmax(dice)
        note=altIndexes[0]
#         note=indexes[0][offset]#+offset
#         dice=np.random.multinomial(1,dices[1])
#         dice=newNote[1][0]
#         offset=np.argmax(dice)
        vel=altIndexes[1]
#         vel=indexes[1][offset]#+offset
#         dice=np.random.multinomial(1,dices[2])
#         dice=newNote[2][0]
#         offset=np.argmax(dice)
#         nVal=indexes[2][offset]#+offset
        nVal=altIndexes[2]
#         print("note=",note)
#         print("velocity=",vel)
#         print("note Value=",nVal)
#         print("%i. note"%i)
        note_sequence[0][0:99]=note_sequence[0][1:100]
        note_sequence[0][-1]=[note/127]
        vel_sequence[0][0:99]=vel_sequence[0][1:100]
        vel_sequence[0][-1]=[vel/127]
        nVal_sequence[0][0:99]=nVal_sequence[0][1:100]
        nVal_sequence[0][-1]=[nVal/len(noteValues)]
#         start+=0.1
        start,end=create_times(noteValues[nVal],end)
        notetoplay=pretty_midi.Note(start=start,end=end,pitch=note,velocity=vel)
        piano.notes.append(notetoplay)
        bar.next()
#         print(note_sequence)
 

composeMusic()
midi.instruments.append(piano)
midi.write("./midi/composed.midi")
print("\ntamamlandı!")
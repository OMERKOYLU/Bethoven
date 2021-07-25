import pretty_midi
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Activation, BatchNormalization, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import math

directory="./classical-piano-type0/"
files=os.listdir(directory)
# files=["debussy_cc_1.mid"]

pitchTypes=[i for i in range(128)]
velocityTypes=[i for i in range(128)]
noteTypes=[1/256,3/512,7/1024,15/2048,1/128,3/256,7/512,15/1024,1/64,3/128,7/256,15/512,1/32,3/64,7/128,15/256,1/16,3/32,7/64,15/128,1/8,3/16,7/32,15/64,1/4,3/8,7/16,15/32,1/2,3/4,7/8,15/16,1.0,3/2,7/4,15/8,2.0,3.0,3.5,3.75]
noteValues=["1/256","3/512","7/1024","15/2048","1/128","3/256","7/512","15/1024","1/64","3/128","7/256","15/512","1/32","3/64","7/128","15/256","1/16","3/32","7/64","15/128","1/8","3/16","7/32","15/64","1/4","3/8","7/16","15/32","1/2","3/4","7/8","15/16","1.0","3/2","7/4","15/8","2.0","3.0","7/2","15/4"]
noteValuesNormalized=[i for i in range(len(noteValues))]
n_vocab_notes=len(pitchTypes)
n_vocab_noteValues=len(noteValues)
n_vocab_velocity=len(velocityTypes)
n_vocab_durations=len(noteValues)


def prepare_data(midi:pretty_midi.PrettyMIDI):
    
    global noteTypes
    global noteValues
    
    beats=midi.get_beats()
    tempos=midi.get_tempo_changes()
    tempoChanges=list(tempos[0])
    tempo_ticks=list(tempos[1])

    notes=[]
    # offsets=[]
    durations=[]
    velocities=[]
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
        closest_value = min(noteTypes, key=absolute_difference_function)
        durations[s]=noteValues[noteTypes.index(closest_value)]
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
    return note_input,note_output

def createNetwork(notes,durations,velocities):
    
    global n_vocab_durations,n_vocab_notes,n_vocab_velocity
    
    note_input_layer=Input(shape=(notes.shape[1],notes.shape[2]))
    input_notes=LSTM(
        256,
        input_shape=(notes.shape[1], notes.shape[2]),
        return_sequences=True
    )(note_input_layer)
    input_notes=Dropout(0.2)(input_notes)
    
    input_velocity_layer=Input(shape=(notes.shape[1],notes.shape[2]))
    input_velocities=LSTM(
        256,
        input_shape=(notes.shape[1], notes.shape[2]),
        return_sequences=True
    )(input_velocity_layer)
    input_velocities=Dropout(0.2)(input_velocities)
    
    input_duration_layer=Input(shape=(notes.shape[1],notes.shape[2]))
    input_durations=LSTM(
        256,
        input_shape=(notes.shape[1], notes.shape[2]),
        return_sequences=True
    )(input_duration_layer)
    input_durations=Dropout(0.2)(input_durations)
    
    inputs = concatenate([input_notes, input_velocities, input_durations])
    x=LSTM(512, return_sequences=True)(inputs)
    x=Dropout(0.3)(x)
    x=LSTM(512)(x)
    x=BatchNormalization()(x)
    x=Dropout(0.3)(x)
    x=Dense(256,activation="relu")(x)
    
    outputNotes = Dense(128, activation='relu')(x)
    outputNotes = BatchNormalization()(outputNotes)
    outputNotes = Dropout(0.3)(outputNotes)
    outputNotes = Dense(n_vocab_notes, activation='softmax', name="Note")(outputNotes)
    
    # Branch of the network that classifies the note offset
    outputVelocities = Dense(128, activation='relu')(x)
    outputVelocities = BatchNormalization()(outputVelocities)
    outputVelocities = Dropout(0.3)(outputVelocities)
    outputVelocities = Dense(n_vocab_velocity, activation='softmax', name="Velocity")(outputVelocities)
    
    # Branch of the network that classifies the note duration
    outputDurations = Dense(n_vocab_durations, activation='relu')(x)
    outputDurations = BatchNormalization()(outputDurations)
    outputDurations = Dropout(0.3)(outputDurations)
    outputDurations = Dense(n_vocab_durations, activation='softmax', name="Duration")(outputDurations)
    
    model = Model(inputs=[note_input_layer, input_velocity_layer, input_duration_layer], outputs=[outputNotes, outputVelocities, outputDurations])
    
    #Adam seems to be faster than RMSProp and learns better too 
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

note_datas=[]
note_Y=[]
duration_datas=[]
duration_Y=[]
velocity_datas=[]
velocity_Y=[]

for file in files:
    if file[-3:]=="mid":
        print(file)
        midi=pretty_midi.PrettyMIDI(os.path.join(directory,file))
        notes, durations, velocities = prepare_data(midi)
        note_input, note_output=prepare_sequences(notes,n_vocab_notes)
        dur_input,dur_output=prepare_sequences(durations,n_vocab_noteValues)
        vel_input,vel_output=prepare_sequences(velocities,n_vocab_velocity)
        for n in note_input: note_datas.append(n)
        for n in dur_input: duration_datas.append(n)
        for n in vel_input: velocity_datas.append(n)
        for n in note_output: note_Y.append(n)
        for n in dur_output: duration_Y.append(n)
        for n in vel_output: velocity_Y.append(n)

note_datas=np.reshape(note_datas,(-1,100,1))
duration_datas=np.reshape(duration_datas,(-1,100,1))
velocity_datas=np.reshape(velocity_datas,(-1,100,1))

note_Y=np.reshape(note_Y,(-1,n_vocab_notes))
velocity_Y=np.reshape(velocity_Y,(-1,n_vocab_velocity))
duration_Y=np.reshape(duration_Y,(-1,n_vocab_durations))

model=createNetwork(note_datas,duration_datas,velocity_datas)
model.load_weights("./model/weights/musicianAi_weights.h5")
print(note_datas.shape)
model.fit(x=[note_datas,velocity_datas,duration_datas], y=[note_Y,velocity_Y,duration_Y],epochs=10)
model.save("./model/musicianAi.h5")
model.save_weights("./model/weights/musicianAi_weights.h5")


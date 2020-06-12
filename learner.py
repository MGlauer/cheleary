from itertools import chain
import pickle
from pysmiles.read_smiles import _tokenize, TokenType
import pandas
import numpy as np
import re
import multiprocessing as mp
  # elements + bonds + ringnumbers

atom_chars = sorted(['C@@H', 'C@H', 'N@H+', 'N', 'c', 'n', 'CH', 'O', 'C', 'P', 'S', 'Cl', 'nH', 's', 'Br', 'o', 'I', 'H', '*', 'F', 'Ca', 'Al', 'OH', 'Na', 'NH', 'Se', 'Co', 'Hg', 'As', 'Mg', 'Cu', 'Si', 'Au', 'Tc', 'B', 'Fe', 'Ge', 'Sm', 'Ru', 'V', 'Mo', 'He', 'Sb', 'Yb', 'Gd', 'Li', 'Cr', 'Ag', 'Fr', 'Ba', 'Pb', 'Y', 'Sr', 'Ga', 'Eu', 'Mn', 'Os', 'Tl', 'In', 'Sn', 'Ir', 'La', 'Lu', 'Cs', 'Ce', 'W', 'Zn', 'Be', 'Bi', 'U', 'Ni', 'Ho', 'Pt', 'Rb', 'K', 'SeH', 'TeH', 'Te', 'At', 'Re', 'Ra', 'Ti', 'SiH', 'se', 'pH', 'te', 'Ar', 'Xe', 'Kr', 'Cd', 'Pd', 'Rh', 'cH', 'p', 'Ne', 'Rn', 'LiH', 'Zr', 'AsH', 'Pr', 'Po', 'Tb'], key=lambda x: -len(x))
modifier_chars = ['+', '++', '-', '--', '@', '@@', '@+', '@@+']
bond_chars = [".", "-", "=", "#", "$", ":", "/", "\\"]
branch_chars = ["(", ")"]
mults = 6
isotopes = ['1', '123', '125', '129', '13', '131', '14', '15', '18', '197', '2', '201', '203', '223', '3', '4', '51', '63', '67', '75', '9', '99']
s = set()



input_lenth = len(atom_chars) + len(modifier_chars) + len(isotopes) + 2*mults + len(bond_chars) + 36 + len(branch_chars)
print(input_lenth)


import tensorflow as tf

def encode(token):
    v = np.zeros(input_lenth, dtype="i")
    t, x = token
    indices = []
    offset = 0

    if t == TokenType.ATOM:
        regex = re.compile(rf"^(?P<isotopes>{'|'.join(map(re.escape, isotopes))})?(?P<atom>{'|'.join(map(re.escape, atom_chars))})(?P<multiplicity_0>[1-{mults}])?(?P<modifier>{'|'.join(map(re.escape, modifier_chars))})?(?P<multiplicity_1>[1-{mults}])?$")
        for y in ["[", "]"]:
            x = x.replace(y, "")
        match = regex.match(x)
        if match is not None:
            offset = 0
            if match.group("atom"):
                indices.append(atom_chars.index(match.group("atom")))
            offset += len(atom_chars)
            if match.group("modifier"):
                indices.append(modifier_chars.index(match.group("modifier"))+offset)
            offset += len(modifier_chars)
            if match.group("isotopes"):
                indices.append(isotopes.index(match.group("isotopes"))+offset)
            offset += len(isotopes)
            for i in range(0, 2):
                if match.group("multiplicity_" + str(i)):
                    indices.append(int(match.group("multiplicity_"+str(i)))+offset+mults*i)
        else:
            raise Exception("Could not encode atom", x)
    else:
        offset += len(atom_chars) + len(modifier_chars) + len(isotopes) + 2*mults
        if t == TokenType.BOND_TYPE or t == TokenType.EZSTEREO:
            indices.append(bond_chars.index(x))

        else:
            offset += len(bond_chars)
            if t == TokenType.RING_NUM:
                indices.append(x + offset - 1)
            else:
                offset += 36
                if t == TokenType.BRANCH_START or t == TokenType.BRANCH_END:
                    indices.append(branch_chars.index(x))
    if not indices:
        raise Exception("Could not encode", x)
    for index in indices:
        v[index]=1
    return v

def lstm_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(100, activation='relu', input_shape=input_shape, name="forward"))
    model.add(tf.keras.layers.RepeatVector(max_len))
    model.add(tf.keras.layers.LSTM(input_lenth, activation='relu', return_sequences=True, name="backwards"))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_lenth)))
    model.compile(optimizer='adam', loss='mse')
    return model

def cnn_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(100, input_shape=input_shape))
    model.add(tf.keras.layers.Dense(input_shape[0]))
    model.compile(optimizer='adam', loss='mse')
    return model

def encode_smiles(row):
    return [encode(smiles) for smiles in _tokenize(row[1][2])]


def generate():
    with open('data/chemdata500x100x1024.pkl', 'rb') as output:
        # pickle.dump(chemdata,output)
        chemdata = pickle.load(output)

    with mp.Pool(mp.cpu_count() - 2) as pool:
        for result in pool.imap(encode_smiles, chemdata.iterrows()):
            yield result

ds = tf.keras.preprocessing.sequence.pad_sequences(list(generate()))



max_len = len(ds[0])
ds = ds.reshape((len(ds), input_lenth*max_len))
#model = lstm_model((max_len, input_lenth))
model = cnn_model((max_len* input_lenth,))

tf.keras.utils.plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
# fit model
model.fit(ds, ds, epochs=300)
model.save("out")

# demonstrate recreation
#yhat = model.predict(chemdata["encoding"], verbose=0)
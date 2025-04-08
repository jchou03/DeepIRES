
import tensorflow as tf
import keras as tfk
tfkl = tf.keras.layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tfa_spectral import SpectralNormalization as SN


nucleotides = {'A':0,'G':1,'C':2,'T':3,'a':0,'g':1,'c':2,'t':3}
letters = ['A','G','C','T']


#-----------------------------------------------    
#One Hot Encoding

# Since sequences are different lengths, this will also zero pad the shorter ones

def one_hot_encode(seqs):
    full_set = seqs.sum(axis=1).astype(str)
    seqs = full_set.apply(lambda x: pd.Series(list(x))).to_numpy()

    #seqs is shape (nseqs,seq_len)
    nseqs = seqs.shape[0]
    seq_len = seqs.shape[1]
    out = np.zeros(shape = (nseqs,4,seq_len))
    for i in range(nseqs):
        for j in range(seq_len):
            #Nan catch
            if type(seqs[i,j]) is float:
                pass
            #Catch for a typo with spaces in line 26 of the natural terminator dataset
            elif seqs[i,j] == ' ':
                seqs[i,:] = np.roll(seqs[i,:],shift=-1,axis=0)
                seqs[i,-1] = np.nan
            else:
                if seqs[i,j] == 'Z':
                    pass
                else:
                    idx = nucleotides[seqs[i,j]]
                    out[i,idx,j] = 1
    return out

#-----------------------------------------------
def prototype_ppms_fast(seqs):
    if seqs.shape[0] is None:
        return None
    else:
        output = np.zeros(shape=(seqs.shape[0],seqs.shape[2],seqs.shape[2]))
        for i in range(seqs.shape[0]):
            output[i,:,:] = contact_map(seqs[i,:,:])
        return output

#-----------------------------------------------
def contact_map(seq):
    
    #computes the structure array
    
    #seq is shape 4, nt
    seq_len = seq.shape[1]
    
    output = np.zeros(shape = (seq_len,seq_len)) #increasing dimension to 2 for G-U pair calculation
    
    gu_output = np.zeros(shape = (seq_len,seq_len))
    
    #Find the indices of each nucleotide
    As = np.where(np.argmax(seq,axis=0)==0)
    Gs = np.where(np.argmax(seq,axis=0)==1)
    Cs  = np.where(np.argmax(seq,axis=0)==2)
    Ts = np.where(np.argmax(seq,axis=0)==3)
    
    
    #Make it so now every G-C pair is represented as a 3
    output[Gs,:] = 1
    output[:,Cs]+=2
    output[output<3] = 0
    
    output[Cs,:] = 1
    output[:,Gs] +=2
    output[output<3] = 0
    
    
    #Repeat for every A-U base pair
    output[As,:] = 1
    output[:,Ts] += 1
    
    output[output< 2] = 0
    

    output[Ts,:] = 1
    output[:,As] +=1
    output[output<2] = 0
    
    #Repeat for every G-U base pair
    gu_output[Gs,:] = 1
    gu_output[:,Ts] += 1
    
    gu_output[gu_output< 2] = 0
    

    gu_output[Ts,:] = 1
    gu_output[:,Gs] +=1
    gu_output[gu_output<2] = 0
    

    
#     #Now we need to divide by the distance between the nucleotides
#     where_vals_gc = np.where(output==3)
#     where_vals_at = np.where(output==2)
    
#     #dividing base pairs by distance without for loops
#     distance_vec_gc = np.abs(where_vals_gc[0] - where_vals_gc[1])
#     # output[where_vals_gc] =1 
#     output[where_vals_gc] /= distance_vec_gc
    
#     distance_vec_at = np.abs(where_vals_at[0] - where_vals_at[1])
#     output[where_vals_at] /= distance_vec_at
    
    
#     where_vals_gu = np.where(gu_output==2)
#     distance_vec_gu = np.abs(where_vals_gu[0] - where_vals_gu[1])
#     gu_output[where_vals_gu] /= distance_vec_gu


    
    # return output,gu_output
    
    output +=gu_output
    
#     # row_sums = output.sum(axis=1)
#     # new_matrix = output / row_sums[:, np.newaxis]
#     # #Nan catch for columns with no binding partners
#     # new_matrix = np.nan_to_num(new_matrix)

    return output # Just do the binarization


def create_SANDSTORM(seq_len=60,ppm_len=60,latent_dim=128,internal_activation='relu',output_activation='linear',output_nodes=1,output_units=1,kernel_1_size=[4,18],kernel_2_size=[4,9],kernel_3_size=[4,3]):
    # seq_len is the length of the sequences
    # Latent dim controls the number of filters
    # internal activation is a tensorflow activation function, applied to the hiden layers
    # output activation is the activation function of the model output
    # output nodes is the number of different output channels the model will have (e.g. 1 for UTR prediciton, 2 for Toehold Prediction (on, off)
    # output units is the number of units in the output layer, 1 for regression problems and N for classification with N different classes 
    
    #Define the Inputs
    input_seqs = tfk.Input(shape=(4,seq_len,1))
    input_probs = tfk.Input(shape=(ppm_len,ppm_len,1))


    # Predictive Model Definition
    #Sequence Branch
    y = SN(tfkl.Conv2D(latent_dim/4, kernel_1_size, strides=(4, 1), padding="same",activation=internal_activation))(input_seqs)
    y = tfkl.BatchNormalization()(y)
    # y = layers.SpatialDropout2D(0.2)(y)
    y = SN(tfkl.Conv2D(latent_dim/8, kernel_2_size, strides=(4, 1), padding="same",activation=internal_activation))(y)
    # y = layers.BatchNormalization()(y)
    y = SN(tfkl.Conv2D(latent_dim/16,kernel_3_size,strides=(4,1),padding='same',activation=internal_activation))(y)
    # y = layers.BatchNormalization()(y)
    y = tfkl.Flatten()(y)

    #PPM Branch    
    x = SN(tfkl.Conv2D(latent_dim/4, (9,9), strides=(ppm_len, 1), padding="same",activation=internal_activation))(input_probs)
    # x = layers.BatchNormalization()(x)
    x = tfkl.SpatialDropout2D(0.2)(x)
    x = SN(tfkl.Conv2D(latent_dim/8, (5,5), strides=(ppm_len, 1), padding="same",activation=internal_activation))(x)
    # x = layers.BatchNormalization()(x)
    x = SN(tfkl.Conv2D(latent_dim/16, (3,3),strides=(ppm_len,1),padding='same',activation=internal_activation))(x)
    x = tfkl.GlobalMaxPooling2D()(x)


    x = tfkl.Flatten()(x)

    # z = CrossAttentionLayer()([x,y])


    #Combine the two
    z = tfkl.Concatenate()([x,y])
    z = tfkl.Dense(16,activation=internal_activation)(z)
    z = tfkl.Dense(8,activation=internal_activation)(z)
    z = tfkl.Dense(4,activation=internal_activation)(z)
	
    
    output_lst = []
    for i in range(output_nodes):
        tmp = tfkl.Dense(output_units,activation=output_activation,name='prediction_output_%s'%i)(z)
        output_lst.append(tmp)


    output_model  = tfk.Model(inputs=[input_seqs,input_probs],outputs=output_lst,name='joint_model')
    return output_model
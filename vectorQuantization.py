import numpy as np
import numpy.matlib as ml
import wave
import struct
import pickle
# from sound import *


class vectorQuant:
    '''
    An implementation of vector quantization 'K-Means' for audio wav file quantization
    '''

    def __init__(self,trainFile,eps,dim):
        '''
        :param trainFile:  audio file with similar characterstics to the files that will be quantized
        :param eps: distance thresold between two iterations for computing the codeBook model
        :param dim:   two elemetn array of the dimensions of codeBook, first element is the number of codeBook entries and the second is the size of the entry 
        :param outBin: output binary file of quantized indices, codeBook, and sampling rate.
        :param outWav:   output wav file
        '''
        self.trainFile = wave.open(trainFile, 'r') 
        self.samplingRateTrain = self.trainFile.getframerate() # get sampling rate
        self.nChannels = self.trainFile.getnchannels() # get number of channels
        self.nFramesTrain = self.trainFile.getnframes() # get number of frames in the audio file
        self.width = self.trainFile.getsampwidth()
        data=self.trainFile.readframes(self.nFramesTrain) # temp storage for frames
        self.framesTrain =  np.array(struct.unpack('h' *(self.nFramesTrain*self.width) ,data)).reshape(-1,self.nChannels) # get the audio frames
        self.dim = dim
        self.eps = eps



    def createModel(self,livePlot=False):
        '''
        Create a codeBook model for quantizing based on training file, eps, dim.
        livePlot: if set to True and codebook 2nd dimension is 2, a live plot is created and updated with each iteration with the codeBook values and the audio frames 
        '''
        reshapedframesTrain = self.framesTrain[:int(len(self.framesTrain.flat)/self.dim[1])*self.dim[1]].reshape(-1,self.dim[1])/np.amax(self.framesTrain) # reshape frames based on the codeBook dimensions


        self.codeBook = np.random.uniform(np.amin(reshapedframesTrain), np.amax(reshapedframesTrain), size=(self.dim)) # random initialization for codebook
        # cbIdx = np.argsort(np.sum(iniCodeBook**2,axis=1))
        # self.codeBook = iniCodeBook[cbIdx]
        # dist = np.zeros([np.shape(reshapedframesTrain)[0],np.shape(self.codeBook)[0]]) #empty array to store distance between each vector and all elements in codebook
        teps = 1e10
        if livePlot and self.dim[1] == 2: # if live plotting is enabeled prepare the figure
            import matplotlib.pyplot as plt
            plt.ion()
            fig, ax = plt.subplots()
            ax.scatter(reshapedframesTrain[:,0][::10],reshapedframesTrain[:,1][::10] ,color='cyan', label='Audio frames')
            sp = ax.scatter(self.codeBook[:,0],self.codeBook[:,1],color='b', label='CodeBook for quantization')
            plt.draw()
            plt.legend()
        while teps>self.eps:
            diffToCB =  reshapedframesTrain - np.rot90(ml.repmat(self.codeBook,reshapedframesTrain.shape[0],1).reshape(-1,self.dim[0],self.dim[1]),axes=(1,0)) # subtract each audio frames from all the entries of the codeBook
            ind = np.argmin(np.sqrt(np.einsum('ijk,ijk->ij', diffToCB, diffToCB)) , axis=0) # get the index of the codebook entry with minumy distance to each audio frame

            tempCB = np.array([np.nan_to_num(np.mean(reshapedframesTrain[ind==i],axis=0)) for i in range(self.dim[0])]) # create a new codebook with new entries based on the mean value of all frames that belong to a specific entry in the old codebook
            
            teps= np.sqrt(np.sum((tempCB-self.codeBook)**2)) # get the distance between new and old codebook
            print ('eps is', teps)
            self.codeBook=tempCB.copy() # assign the new values to the codeBook

            if livePlot and self.dim[1] == 2: # upfate plot
                sp.set_offsets(self.codeBook)
                fig.canvas.draw_idle()
                plt.pause(0.0001)


    def applyVectorQ(self,audioFile,binFile=None , wavFile=None):
        '''
        quantize audio file based on the trained model "codeBook"
        audioFile: audio file to be quantized
        binFile: if set to a binary file name, the sampling rate, codeBook, and quantized indices are writen to a binary file
        wavFile: if set to a wav file name, the decoded frames are writen to a wav file 
        '''
        audioFile = wave.open(audioFile, 'r') 
        self.samplingRate = audioFile.getframerate() # get the samplingRate of input audio file
        nChannels = audioFile.getnchannels() # get number of channels
        width = audioFile.getsampwidth()
        nFrames = audioFile.getnframes() # get frames count
        data=audioFile.readframes(nFrames) # temp storage for frames
        frames =  np.array(struct.unpack('h' *(nFrames*width) ,data)).reshape(-1,nChannels) # get the audio frames
            
        reshapedFrames = frames[:int(len(frames.flat)/self.dim[1])*self.dim[1]].reshape(-1,self.dim[1])/np.amax(frames) #reshape frames based on the desired codebook shape

        # dist2 = np.zeros([nFrames,self.dim[0]])

        diffToCB =  reshapedFrames - np.rot90(ml.repmat(self.codeBook,reshapedFrames.shape[0],1).reshape(-1,self.dim[0],self.dim[1]),axes=(1,0)) # subtract each audio frames from all the entries of the codeBook
        self.ind = np.argmin(np.sqrt(np.einsum('ijk,ijk->ij', diffToCB, diffToCB)) , axis=0) # get the index of the codebook entry with minmum distance to each audio frame
        if self.dim[0]>255: # reduce the memory used by indices
            self.ind = self.ind.astype(np.uint8)
        else:
            self.ind = self.ind.astype(np.uint16)
        self.outputDecoded=self.codeBook[self.ind].reshape(frames.shape) * np.amax(frames) # restore the quatized and coded frames
        if binFile: # save indices, sampling rate, and codebook to binary file
            bFile = open(binFile, 'wb')
            binData = (self.samplingRate , self.codeBook , self.ind)
            pickle.dump(binData,bFile ) 
            bFile.close()
        if wavFile: # save decoded frames to wav file
            snd = self.outputDecoded.flatten().astype(int)
            length=len(snd)
            
            wf = wave.open(wavFile, 'wb')
            wf.setnchannels(nChannels)
            wf.setsampwidth(width)
            wf.setframerate(self.samplingRate)
            data=struct.pack( 'h' * length, *snd )
            wf.writeframes(data)
            wf.close()
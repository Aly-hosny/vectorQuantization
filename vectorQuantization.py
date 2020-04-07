import numpy as np
import numpy.matlib as ml
import wave
import struct
# from sound import *


class vectorQuant:
    '''
    An implementation of vector quantization 'K-Means' for audio wav file quantization
    '''

    def __init__(self,trainFile,eps,dim,outBin,outWav):
        '''
        :param trainFile:  audio file with similar characterstics to the files that will be quantized
        :param eps: distance thresold between two iterations for computing the codeBook model
        :param dim:   two elemetn array of the dimensions of codeBook, first element is the number of codeBook entries and the second is the size of the entry 
        :param outBin: output binary file of quantized indices, codeBook, and sampling rate.
        :param outWav:   output wav file
        '''
        self.trainFile = wave.open(trainFile, 'r') 
        self.samplingRateTrain = self.trainFile.getframerate() # get sampling rate
        self.signalModeTrain = 'stereo' if self.trainFile.getnchannels()==2 else 'mono' # get mode
        self.nFramesTrain = self.trainFile.getnframes() # get number of frames in the audio file
        self._pck = "<2h" if self.signalModeTrain == 'stereo' else "<h"
        self.framesTrain =  np.array([struct.unpack(self._pck ,self.trainFile.readframes(1)) for i in range(self.nFramesTrain)],dtype=float) # get the audio frames
        self.dim = dim
        self.eps = eps
        # self.normFactor = np.amax(self.framesTrain)
        self.outBin = outBin
        self.outWav = outWav


    def createModel(self,livePlot=False):
        '''
        Create a codeBook model for quantizing based on training file, eps, dim.
        '''
        reshapedframesTrain = self.framesTrain[:int(len(self.framesTrain.flat)/self.dim[1])*self.dim[1]].reshape(-1,self.dim[1])/np.amax(self.framesTrain) # reshape frames based on the codeBook dimensions


        self.codeBook = np.random.uniform(np.amin(reshapedframesTrain), np.amax(reshapedframesTrain), size=(self.dim)) # random initialization for codebook
        # cbIdx = np.argsort(np.sum(iniCodeBook**2,axis=1))
        # self.codeBook = iniCodeBook[cbIdx]
        # dist = np.zeros([np.shape(reshapedframesTrain)[0],np.shape(self.codeBook)[0]]) #empty array to store distance between each vector and all elements in codebook
        teps = 1e10
        if livePlot: # if live plotting is enabeled prepare the figure
            import matplotlib.pyplot as plt
            plt.ion()
            fig, ax = plt.subplots()
            ax.scatter(reshapedframesTrain[:,0],reshapedframesTrain[:,1])
            sp = ax.scatter(self.codeBook[:,0],self.codeBook[:,1])
            plt.draw()

        while teps>self.eps:
            diffToCB =  reshapedframesTrain - np.rot90(ml.repmat(self.codeBook,reshapedframesTrain.shape[0],1).reshape(-1,self.dim[0],self.dim[1]),axes=(1,0)) # subtract each audio frames from all the entries of the codeBook
            ind = np.argmin(np.sqrt(np.einsum('ijk,ijk->ij', diffToCB, diffToCB)) , axis=0) # get the index of the codebook entry with minumy distance to each audio frame

            tempCB = np.array([np.nan_to_num(np.mean(reshapedframesTrain[ind==i],axis=0)) for i in range(self.dim[0])]) # create a new codebook with new entries based on the mean value of all frames that belong to a specific entry in the old codebook
            
            teps= np.sqrt(np.sum((tempCB-self.codeBook)**2)) # get the distance between new and old codebook
            print ('eps is', teps)
            self.codeBook=tempCB.copy() # assign the new values to the codeBook

            if livePlot: # upfate plot
                sp.set_offsets(self.codeBook)
                fig.canvas.draw_idle()
                plt.pause(0.0001)


    def applyQuantization(self,audioFile):
        audioFile = wave.open(audioFile, 'r') 
        self.samplingRate = audioFile.getframerate() # get the samplingRate of input audio file
        mode = 'stereo' if audioFile.getnchannels()==2 else 'mono' # get mode
        nFrames = audioFile.getnframes() # get frames count
        pck = "<2h" if mode == 'stereo' else "<h"
        frames =  np.array([struct.unpack(pck ,audioFile.readframes(1)) for i in range(nFrames)]) # get frames
     
        reshapedFrames = frames[:int(len(frames.flat)/self.dim[1])*self.dim[1]].reshape(-1,self.dim[1])/np.amax(frames) #reshape frames based on the desired codebook shape

        # dist2 = np.zeros([nFrames,self.dim[0]])

        diffToCB =  reshapedFrames - np.rot90(ml.repmat(self.codeBook,reshapedFrames.shape[0],1).reshape(-1,self.dim[0],self.dim[1]),axes=(1,0)) # subtract each audio frames from all the entries of the codeBook
        self.ind = np.argmin(np.sqrt(np.einsum('ijk,ijk->ij', diffToCB, diffToCB)) , axis=0) # get the index of the codebook entry with minmum distance to each audio frame

        self.outputDecoded=self.codeBook[self.ind].reshape(frames.shape) * np.amax(frames)
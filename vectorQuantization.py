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


    def createModel(self):
        '''
        Create a codeBook model for quantizing based on training file, eps, dim.
        '''
        reshapedframesTrain = self.framesTrain[:int(len(self.framesTrain.flat)/self.dim[1])*self.dim[1]].reshape(-1,self.dim[1]) # reshape frames based on the codeBook dimensions


        self.codeBook = np.random.normal(np.mean(reshapedframesTrain), np.std(reshapedframesTrain), size=(self.dim)) # random initialization for codebook
        # cbIdx = np.argsort(np.sum(iniCodeBook**2,axis=1))
        # self.codeBook = iniCodeBook[cbIdx]
        # dist = np.zeros([np.shape(reshapedframesTrain)[0],np.shape(self.codeBook)[0]]) #empty array to store distance between each vector and all elements in codebook
        teps = 1e10
        while teps>self.eps:
            ### old attempts for ind 'slower'
            # repeatedCB =  np.rot90(ml.repmat(codeBook,reshapedframesTrain.shape[0],1).reshape(-1,16,2),axes=(1,0))
            # ind = np.argmin((np.sum((reshapedframesTrain - repeatedCB)**2,axis=2)**0.5) , axis=0)

            repeatedCB =  reshapedframesTrain - np.rot90(ml.repmat(self.codeBook,reshapedframesTrain.shape[0],1).reshape(-1,16,2),axes=(1,0))
            ind = np.argmin(np.sqrt(np.einsum('ijk,ijk->ij', repeatedCB, repeatedCB)) , axis=0)

            # for i in range(self.dim[0]):
            #     difference = reshapedframesTrain - ml.repmat(self.codeBook[i,:],np.shape(reshapedframesTrain)[0],1)
            #     dist[:,i] = np.sqrt(np.sum(difference**2,1)) 
            # ind = np.argmin(dist,axis=1)

            # ind = np.digitize(np.sum(reshapedframesTrain**2,1) , np.sum(self.codeBook**2,1))

            tempCB = np.array([np.nan_to_num(np.mean(reshapedframesTrain[ind==i],axis=0)) for i in range(self.dim[0])])
            # tempIdx = np.argsort(np.sum(tempCB**2,axis=1))
            # tempCB = tempCB[tempIdx]
            
            teps= np.sqrt(np.sum((tempCB-self.codeBook)**2))
            print ('eps is', teps)
            self.codeBook=tempCB.copy()
################################################################################## Add plot ##################################################################################
    def applyQuantization(self,audioFile):
        audioFile = wave.open(audioFile, 'r') 
        self.samplingRate = audioFile.getframerate()
        mode = 'stereo' if audioFile.getnchannels()==2 else 'mono'
        nFrames = audioFile.getnframes()
        pck = "<2h" if mode == 'stereo' else "<h"
        frames =  np.array([struct.unpack(pck ,audioFile.readframes(1)) for i in range(nFrames)])
     
        reshapedFrames = frames[:int(len(frames.flat)/self.dim[1])*self.dim[1]].reshape(-1,self.dim[1])

        dist2 = np.zeros([nFrames,self.dim[0]])
        for i in range(self.dim[0]):
            difference = reshapedFrames - ml.repmat(self.codeBook[i,:],np.shape(reshapedFrames)[0],1)
            dist2[:,i] = np.sqrt(np.sum(difference**2,1))
        self.ind = np.argmin(dist2,axis=1)
        self.outputDecoded=self.codeBook[self.ind].reshape(frames.shape)
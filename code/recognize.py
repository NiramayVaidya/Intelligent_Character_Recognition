from __future__ import division
from __future__ import print_function

import sys
import argparse
import cv2
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
import os
import tensorflow as tf
import re

class FilePaths:
    "filenames and paths to data"
    fnCharList = 'model/charList.txt'
    fnAccuracy = 'model/accuracy.txt'
    fnTrain = 'data/'
    fnInfer = 'data/test.png'
    # fnInfer = '../../testing/generated_words/para_6_with_gaps/line_1_word_1.jpg'
    fnCorpus = '/data/corpus.txt'

def train(model, loader):
    "train NN"
    epoch = 0 # number of training epochs since start
    bestCharErrorRate = float('inf') # best valdiation character error rate
    noImprovementSince = 0 # number of epochs no improvement of character error rate occured
    earlyStopping = 5 # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)

        # validate
        charErrorRate = validate(model, loader)

        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
            break

def validate(model, loader):
    "validate NN"
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0],'/', iterInfo[1])
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')

    # print validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
    return charErrorRate

def infer(model, fnImg):
    "recognize text in image provided by file path"
    img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    # print('Recognized:', '"' + recognized[0] + '"')
    # print('Probability:', probability[0])
    return recognized[0]

def recog(folder):
    tf.logging.set_verbosity(tf.logging.ERROR)

    decoderType = DecoderType.WordBeamSearch

    print(open(FilePaths.fnAccuracy).read())
    model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True)

    output = ''
    check_location = 'generated_words/' + folder + '/'
    with open('output.txt', 'w') as f:
        match_with_1 = re.compile(r'_\d*_')
        match_with_2 = re.compile(r'_\d*\.')
        files = os.listdir(check_location)
        files.sort(key = lambda x :
            (int(match_with_1.search(x).group()[1:-1]),
                int(match_with_2.search(x).group()[1:-1])))
        for filename in files:
            if filename.endswith('jpg'):
                word = infer(model, check_location + filename)
                f.write(word + ' ')
                output += word + ' '
    return output

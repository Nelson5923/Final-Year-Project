from gensim.models import word2vec
import numpy as np
import re
import tensorflow as tf
from tensorflow.contrib import learn

def Preprocessor(CorpusPath,LabelPath, ModelPath):

    # Cleanse the Corpus & Label

    RawCorpus = list(open(CorpusPath, "r", encoding='utf-8').readlines())
    RawCorpus = [s.strip() for s in RawCorpus]
    RawCorpus = [re.sub(' {2,}', ' ', s) for s in RawCorpus]

    RawLabel = list(open(LabelPath, "r", encoding='utf-8').readlines())
    RawLabel = [s.strip() for s in RawLabel]
    RawLabel = np.asarray([[0,1] if (Label == '1') else [1,0] for Label in RawLabel])

    print("Number of Sentence: " + str(len(RawCorpus)))
    print("Number of Label:" + str(len(RawLabel)))

    # Generate Vocabulary List for Word2Vec Pretrain Vector

    Embedding = []
    Vocab = []
    model = word2vec.Word2Vec.load(ModelPath)

    for idx, key in enumerate(model.wv.vocab):
        Vocab.append(key)

    print("VocabSize: " + str(len(Vocab)))

    # Input Vocabulary to VocabProcessor for Generating Word Index
    # Vocab Processor will perform Zero Padding

    MaxDocLength = 39 # max([len(s.split(" ")) for s in RawCorpus])
    VocabPro = learn.preprocessing.VocabularyProcessor(MaxDocLength)
    Pretrain = VocabPro.fit(Vocab)
    RawCorpus = np.array(list(Pretrain.transform(RawCorpus)))

    print("Vocab Processor:" + str(Pretrain.vocabulary_._mapping))

    # Create a Word Embedding Matrix with the Vocabulary List of VocabProcessor

    for idx, key in enumerate(Pretrain.vocabulary_._mapping):
        if key not in Vocab:
            Embedding.append([0] * 250)
        else:
            Embedding.append(model.wv[key])
            print(str(idx) + ":" + key)

    Embedding = np.asarray(Embedding)

    print("EmbeddingDim: " + str(Embedding.shape))
    print("Vocabulary Size: {:d}".format(len(Pretrain.vocabulary_)))

    return RawCorpus, RawLabel, Embedding, Pretrain

def DataSplit(RawCorpus, RawLabel, SplitRatio):

    # Shuffle Data

    np.random.seed(10)
    ShuffledIndices = np.random.permutation(np.arange(len(RawLabel)))
    ShuffledCorpus = RawCorpus[ShuffledIndices]
    ShuffledLabel = RawLabel[ShuffledIndices]

    # Split Data

    SplitIndex = -1 * int(SplitRatio * float(len(RawLabel)))
    TrainCorpus, TestCorpus = ShuffledCorpus[:SplitIndex], ShuffledCorpus[SplitIndex:]
    TrainLabel, TestLabel = ShuffledLabel[:SplitIndex], ShuffledLabel[SplitIndex:]

    del RawCorpus, RawLabel, ShuffledCorpus, ShuffledLabel

    print("Train/Test Split: {:d}/{:d}".format(len(TrainLabel), len(TestLabel)))

    return TrainCorpus, TrainLabel, TestCorpus, TestLabel

def BatchIterator(TrainData, BatchSize, EpochsNum, Shuffle=True):

    TrainData = np.array(TrainData)
    DataSize = len(TrainData)
    BatchNum = int((len(TrainData)-1)/BatchSize) + 1

    for Epoch in range(EpochsNum):

        if Shuffle:
            ShuffleIndices = np.random.permutation(np.arange(DataSize))
            ShuffleData = TrainData[ShuffleIndices]
        else:
            ShuffleData = TrainData

        for s in range(BatchNum):
            i = s * BatchSize
            j = min((s + 1) * BatchSize, DataSize)
            yield ShuffleData[i:j]
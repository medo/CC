#!/usr/bin/python

import numpy as np
import getopt, sys, os, traceback
import cv2

from FeatureExtractorFactory import FeatureExtractorFactory
from Image import Image
from KMeanCluster import KMeanCluster
from SIFTManager import SIFTManager
from HistogramCalculator import HistogramCalculator
from ClassifierFactory import ClassifierFactory
from CategoriesManager import CategoriesManager
from EvenImageDivider import EvenImageDivider
from BagOfWordsVectorCalculator import BagOfWordsVectorCalculator
from AveragePrecisionCalculator import AveragePrecisionCalculator

# Private helper functions

def __check_dir_condition(path):
    if not os.path.isdir(path):
        print("%s: No such directory" % (path)) 
        sys.exit(2)
        
def __check_file_condition(file):
    if not os.path.isfile(file):
        print("%s: No such file" % (file)) 
        sys.exit(2)
        
def __check_label_existence(label_name):
    label_number = classesHashtable.getClassNumber(str(label_name))
    if label_number == None:
        print ("Label %s is not trained in our database" % label_name)
    return label_number
    
def __from_array_to_matrix(array_data):
    return np.matrix(array_data).astype('float32')

def __get_image_features(img_file):
    extractor = FeatureExtractorFactory.newInstance(Image.from_local_directory(img_file),False)
    return extractor.extract_feature_vector()

def __get_image_features_memory(img):
    extractor = FeatureExtractorFactory.newInstance(img,True)
    return extractor.extract_feature_vector()

def __load_image(img):
    return cv2.imread(img)

def __belongs_to_class(instance,correctClass):
    return instance.__class__.__name__ == correctClass            

def __init_histogram_calculator(vocab_file):
    print ("Loading vocabulary from: %s" % vocab_file)
    vocab = SIFTManager.load_from_local_file(vocab_file)
    global histCalculator
    histCalculator = HistogramCalculator(vocab)

def __init_bow_vector_calculator():
    global bowCalculator 
    bowCalculator = BagOfWordsVectorCalculator()

def __load_classifier(classifier_file):
    print ("Loading classifier from: %s" % classifier_file)
    global classifier 
    classifier = ClassifierFactory.createClassifier()
    classifier.load(classifier_file)

def __load_category_dictionary(dictionary_file):
    print ("Loading Classes Hashtable from: %s" % dictionary_file)
    global classesHashtable
    classesHashtable = CategoriesManager()
    classesHashtable.loadFromFile(dictionary_file)

def __init_average_precision_calculator(path,dictionary_file):
    global averagePrecisionCalc
    averagePrecisionCalc = AveragePrecisionCalculator(path,dictionary_file)
    averagePrecisionCalc.generate_binary_labels()

def __create_and_train_classifier():
    global classifier
    classifier = ClassifierFactory.createClassifier()
    if __belongs_to_class(classifier,"SVMClassifierScikit"):
        classifier.setTrainingData(trainingDataMat)
        classifier.setTrainingLabels(trainingLabelsMat)
    else:
        classifier.setTrainingData(__from_array_to_matrix(trainingDataMat))
        classifier.setTrainingLabels(__from_array_to_matrix(trainingLabelsMat))
        classifier.train()

def __save_classifier(output_file):
    print ("Saving Classifier in: %s" % output_file)
    classifier.save(output_file)

def __save_categories_dictionary(output_file):
    print ("Saving Dictionary in: %s" % output_file)        
    classesHashtable.saveToFile(output_file)


# Main functions

def vocabulary(path, output_file):
    __check_dir_condition(path)
    
    count = 0
    cluster = KMeanCluster(100)
    for i in os.listdir(path):
        if i.endswith(".jpg") or i.endswith(".png"):
            try:
                print i
                count += 1
                imgfile = "%s/%s" % (path, i)
                vector = __get_image_features(imgfile)
                cluster.add_to_cluster(vector)
            except Exception, Argument:
                print "Exception happened: ", Argument 

    if count == 0:
        print ("%s contains no png/jpg images" % (path))
        return

    result = cluster.cluster()
    SIFTManager.save_to_local_file(result, output_file)


def evaluating(path, vocab_file, classifier_file, dictionary_file):
    __check_dir_condition(path)
    __check_file_condition(vocab_file)
    __check_file_condition(classifier_file)
    __check_file_condition(dictionary_file)
    __init_bow_vector_calculator()
    __init_histogram_calculator(vocab_file)
    __load_category_dictionary(dictionary_file)
    __init_average_precision_calculator(path,dictionary_file)  
    __load_classifier(classifier_file)    

    for d in os.listdir(path):
        if d.startswith("."):
            continue
        subdir = ("%s/%s" % (path, d))
        averagePrecisionCalc.add_evaluated_category_name(d)
        if os.path.isdir(subdir):
            print ("Evaluating label '%s'" % d)
            wrongPredictions = 0
            totalPredictions = 0
            label = __check_label_existence(d)
            for f in os.listdir(subdir):
                print f
                if f.endswith(".jpg") or f.endswith(".png"):
                    try:
                        print f
                        imgfile = "%s/%s" % (subdir, f)
                        image = __load_image(imgfile)
                        dividedImage=EvenImageDivider(image,4)
                        for i in xrange(1,(dividedImage.n + 1)):
                            sectorOfFeatures = __get_image_features_memory(dividedImage.divider(i))
                            bow = histCalculator.hist(sectorOfFeatures)
                            bowCalculator.createMergedBow(bow)
                        bow = __from_array_to_matrix(bowCalculator.getMergedBow())
                        totalPredictions += 1
                        correctResponse = classifier.evaluateData(bow, label)
                        confidenceScore = classifier.calculateScore(bow)
                        confidenceScore = confidenceScore[0][classifier.predict(bow)]
                        averagePrecisionCalc.generate_score_list(confidenceScore)
                        if not correctResponse:
                            wrongPredictions += 1
                        bowCalculator.emptyMergedBow()
                    except Exception, Argument:
                        print "Exception happened: ", Argument
                        traceback.print_stack()
            
            print ("Label %s results:\n%d were wrongly predicted from %d" % (d, wrongPredictions, totalPredictions))
    
    print ("Final results:\n%d were wrongly predicted from %d" % (classifier.getErrorCount(), classifier.getEvaluationsCount()))

    averagePrecisionCalc.generate_tuples_list()
    averagePrecisionCalc.split_tuples_list_per_class()
    for i in range(0,averagePrecisionCalc.get_evaluated_categories_count()):
        specific = averagePrecisionCalc.get_specific_tuples_list(i)
        y_true = averagePrecisionCalc.extract_y_true_from_specific_tuples_list(specific)
        y_score = averagePrecisionCalc.extract_score_from_specific_tuples_list(specific)
        print "Average Precision Score for class '%s' = " % (averagePrecisionCalc.get_evaluated_category_names())[i], averagePrecisionCalc.calculate_average_precision_score(y_true,y_score)
    print "The Mean Average Precision (MAP) = ", averagePrecisionCalc.calculate_map()
    

def training(path, output_file, vocab_file, dictionary_output_file):
    __check_dir_condition(path)
    __check_file_condition(vocab_file)
    
    __init_histogram_calculator(vocab_file)
    __init_bow_vector_calculator()
    
    label = 0
    labelsVector = None
    global classesHashtable 
    classesHashtable = CategoriesManager()
    
    for d in os.listdir(path):
        subdir = ("%s/%s" % (path, d))
        if os.path.isdir(subdir):
            print ("Training label '%s'" % d)
            classesHashtable.addClass(label, d)
            correctLabel = classesHashtable.getClassNumber(d)
            
            for f in os.listdir(subdir):
                if f.endswith(".jpg") or f.endswith(".png"):
                    try:
                        print f
                        imgfile = "%s/%s" % (subdir, f)
                        image = __load_image(imgfile)

                        dividedImage=EvenImageDivider(image,4)
                        for i in xrange(1,(dividedImage.n + 1)):
                            sectorOfFeatures = __get_image_features_memory(dividedImage.divider(i))
                            bow = histCalculator.hist(sectorOfFeatures)
                            bowCalculator.createMergedBow(bow)
            
                        bow = __from_array_to_matrix(bowCalculator.getMergedBow())
                        bowCalculator.createBowVector(bow)

                        if labelsVector == None:
                            labelsVector = np.array(correctLabel)
                        else:
                            labelsVector = np.insert(labelsVector, labelsVector.size, correctLabel)

                        bowCalculator.emptyMergedBow()
                    except Exception, Argument:
                        print "Exception happened: ", Argument
                        traceback.print_stack()
            
            if label == correctLabel:
                label += 1
    try:
        print "Training Classifier"
        
        global trainingDataMat
        trainingDataMat = __from_array_to_matrix(bowCalculator.getBowVector()) 
        global trainingLabelsMat
        trainingLabelsMat = __from_array_to_matrix(labelsVector)

        print ("trainingDataMat", trainingDataMat)
        print ("trainingLabelsMat", trainingLabelsMat)
               
        __create_and_train_classifier()   
        __save_classifier(output_file)
        __save_categories_dictionary(dictionary_output_file)

    except Exception, Argument:
        print "Exception happened: ", Argument
        traceback.print_stack()


def main(args):
    try:
        optlist, args = getopt.getopt(args, 'v:o:t:r:d:e:c:s:')
        optlist = dict(optlist)
        output_file = "vocab/vocab.sift"
        if "-o" in optlist:
            output_file = optlist["-o"]
        for opt, arg in optlist.iteritems():
            if opt == '-t':
                if "-r" not in optlist or "-d" not in optlist:
                    print "Usage: -t <training_dir> -r <reference_vocab> -d <dictionary_output>"
                    sys.exit(2)

                training(arg, output_file, optlist['-r'], optlist['-d'])
                sys.exit()
                
            if opt == '-v':
                vocabulary(arg, output_file)
                sys.exit()
                
            if opt == '-e':
                if "-r" not in optlist or "-c" not in optlist or "-d" not in optlist:
                    print "Usage: -e <evaluating_dir> -r <reference_vocab> -c <reference_classifier> -d <reference_dictionary>"
                    sys.exit(2)
                
                evaluating(arg, optlist['-r'], optlist['-c'], optlist['-d'])
                sys.exit()

    except getopt.GetoptError, e:
        print str(e)
        sys.exit(2)
    
    
if __name__ == "__main__":
    main(sys.argv[1:])

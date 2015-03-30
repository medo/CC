#!/usr/bin/python

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np
import getopt, sys, os, traceback
import cv2, csv, subprocess

from FeatureExtractorFactory import FeatureExtractorFactory
from Image import Image
from KMeanCluster import KMeanCluster
from SIFTManager import SIFTManager
from HistogramCalculator import HistogramCalculator
from ClassifierFactory import ClassifierFactory
from CategoriesManager import CategoriesManager
from EvenImageDivider import EvenImageDivider
from BagOfWordsVectorCalculator import BagOfWordsVectorCalculator

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
                
def __init_histogram_calculator(vocab_file):
    print ("Loading vocabulary from: %s" % vocab_file)
    vocab = SIFTManager.load_from_local_file(vocab_file)
    global histCalculator
    histCalculator = HistogramCalculator(vocab)

def __init_bow_vector_calculator():
    global bowCalculator 
    bowCalculator = BagOfWordsVectorCalculator()

def __calculate_merged_histogram(image,numberOfSectors):
    dividedImage=EvenImageDivider(image,numberOfSectors)
    for i in xrange(1,(dividedImage.n + 1)):
        sectorOfFeatures = __get_image_features_memory(dividedImage.divider(i))
        bow = histCalculator.hist(sectorOfFeatures)
        bowCalculator.createMergedBow(bow)

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

def __create_and_train_classifier():
    global classifier
    classifier = ClassifierFactory.createClassifier()
    classifier.setTrainingData(trainingDataMat)
    classifier.setTrainingLabels(trainingLabelsMat)
    classifier.train()

def __save_classifier(output_file):
    print ("Saving Classifier in: %s" % output_file)
    classifier.save(output_file)

def __save_categories_dictionary(output_file):
    print ("Saving Dictionary in: %s" % output_file)        
    classesHashtable.saveToFile(output_file)

def __get_image_features_and_cluster_from_csv(path,fileName):
    with open(("%s/%s" % (path, fileName))) as fileReader:
        reader = csv.reader(fileReader, delimiter=' ')
        for i in reader:
            imgfile = i[0].split("/")
            imgfile = imgfile[len(imgfile)-1]
            print imgfile
            vector = __get_image_features(i[0])
            cluster.add_to_cluster(vector)

def __load_image_from_csv():
    pass

def __get_image_name_from_path(path):
    imgName = path.split("/")
    return imgName[len(imgName)-1]

def __get_category_name_from_file_name(file_name):
    categoryName = file_name.split("_")
    categoryName = categoryName[1].split(".csv")[0]
    return categoryName

# Main functions

def vocabulary(path, output_file):
    __check_dir_condition(path)
    
    count = 0
    global cluster
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

        if i.endswith(".csv"):
            try:
                count += 1
                __get_image_features_and_cluster_from_csv(path,i)
            except Exception, Argument:
                print "Exception happened: ", Argument
    #if count == 0:
        #print ("%s contains no png/jpg images" % (path))
        #return

    result = cluster.cluster()
    SIFTManager.save_to_local_file(result, output_file)


def evaluating(path, vocab_file, classifier_file, dictionary_file):
    __check_dir_condition(path)
    __check_file_condition(vocab_file)
    __check_file_condition(classifier_file)
    __check_file_condition(dictionary_file)
    __init_bow_vector_calculator()
    __init_histogram_calculator(vocab_file)  
    __load_classifier(classifier_file)    
    __load_category_dictionary(dictionary_file)
    bowVector = None
    for d in os.listdir(path):
        if d.startswith("."):
            continue
        subdir = ("%s/%s" % (path, d))
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
                        if not correctResponse:
                            wrongPredictions += 1
                        bowCalculator.emptyMergedBow()
                    except Exception, Argument:
                        print "Exception happened: ", Argument
                        traceback.print_stack()
            
            print ("Label %s results:\n%d were wrongly predicted from %d" % (d, wrongPredictions, totalPredictions))
        else:

            break
    for f in os.listdir(path):
        if f.endswith(".csv"):
            try:
                categoryName = __get_category_name_from_file_name(f)
                print ("Evaluating label '%s'" % categoryName)
                wrongPredictions = 0
                totalPredictions = 0
                label = __check_label_existence(categoryName)
                with open(("%s/%s" % (path, f))) as fileReader:
                    reader = csv.reader(fileReader, delimiter=' ')
                    for i in reader:
                        imgName = __get_image_name_from_path(i[0])
                        print imgName
                        image = __load_image(i[0])
                        __calculate_merged_histogram(image,4)
                        bow = __from_array_to_matrix(bowCalculator.getMergedBow())
                        totalPredictions += 1
                        correctResponse = classifier.evaluateData(bow, label)
                        #print "Confidence score = is", classifier.predict(bow, True)
                        #print "Are they equivalent", classifier.predict(bow,True)== classifier.predict(bow)
                        if not correctResponse:
                            wrongPredictions += 1
                        bowCalculator.emptyMergedBow()

            except Exception, Argument:
                print "Exception happened: ", Argument
                traceback.print_stack()
                print ("Label %s results:\n%d were wrongly predicted from %d" % (d, wrongPredictions, totalPredictions))


    
    print ("Final results:\n%d were wrongly predicted from %d" % (classifier.getErrorCount(), classifier.getEvaluationsCount()))
    

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
        if d.startswith("."):
            continue
        subdir = ("%s/%s" % (path, d))
        if os.path.isdir(subdir):
            print "You're training using a regular dataset"
            print ("Training label '%s'" % d)
            classesHashtable.addClass(label, d)
            correctLabel = classesHashtable.getClassNumber(d)
            for f in os.listdir(subdir):
                if f.endswith(".jpg") or f.endswith(".png"):
                    try:
                        print f
                        imgfile = "%s/%s" % (subdir, f)
                        image = __load_image(imgfile)

                        __calculate_merged_histogram(image,4)
            
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

        else:
            print "You're training using a Pascal dataset"
            break

    classesHashtable.loadFromFile(dictionary_output_file)
    for f in os.listdir(path):
        if f.endswith(".csv"):
            try:
                categoryName = __get_category_name_from_file_name(f)
                print ("Training label '%s'" % categoryName)
                correctLabel = classesHashtable.getClassNumber(categoryName)

                with open(("%s/%s" % (path, f))) as fileReader:
                    reader = csv.reader(fileReader, delimiter=' ')
                    for i in reader:
                        imgName = __get_image_name_from_path(i[0])
                        print imgName
                        image = __load_image(i[0])                      
                        __calculate_merged_histogram(image,4)
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

def get_precision_scores(path, vocab_file, classifier_file, dictionary_file):
    __check_dir_condition(path)
    __check_file_condition(vocab_file)
    __check_file_condition(classifier_file)
    __check_file_condition(dictionary_file)
    
    __init_histogram_calculator(vocab_file)  
    __load_classifier(classifier_file)    
    __load_category_dictionary(dictionary_file)
    
    categories_number = classesHashtable.getCategoriesCount()
    true_values_arr = [None] * (categories_number + 1)
    score_values_arr = [None] * (categories_number + 1)
    
    for d in os.listdir(path):
        subdir = ("%s/%s" % (path, d))
        if os.path.isdir(subdir):
            print ("\n\nEvaluating label '%s'" % d)
            label = __check_label_existence(d)
            if label == None:
                continue
            
            for f in os.listdir(subdir):
                if f.endswith(".jpg") or f.endswith(".png"):
                    try:
                        print "\n" + f
                        imgfile = "%s/%s" % (subdir, f)
                        vector = __get_image_features(imgfile)
                        bow = histCalculator.hist(vector)
                        bow = __from_array_to_matrix(bow)
                        res = classifier.predict(bow)
                        
                        label = int(round(label))
                        res = int(round(res))
                        
                        for i in range(categories_number):
                            res_value = 1 if i == res else 0
                            label_value = 1 if i == label else 0
                            
                            if score_values_arr[i] == None:
                                score_values_arr[i] = np.array(res_value)
                                true_values_arr[i] = np.array(label_value)
                            else:
                                score_values_arr[i] = np.insert(score_values_arr[i], score_values_arr[i].size, res_value)
                                true_values_arr[i] = np.insert(true_values_arr[i], true_values_arr[i].size, label_value)
                             
                        if score_values_arr[categories_number] == None:
                            score_values_arr[categories_number] = np.array(res)
                            true_values_arr[categories_number] = np.array(label)
                        else:
                            score_values_arr[categories_number] = np.insert(score_values_arr[categories_number], 
                                                                            score_values_arr[categories_number].size, res)
                            true_values_arr[categories_number] = np.insert(true_values_arr[categories_number], 
                                                                            true_values_arr[categories_number].size, label)
                                                    
                    except Exception, Argument:
                        print "Exception happened: ", Argument
                        traceback.print_stack()
    
    try:
        classes_list = range(categories_number)
        print "\nThe total average precision score: "
        print str(average_precision_score(label_binarize(true_values_arr[categories_number], classes_list),
                                          label_binarize(score_values_arr[categories_number], classes_list)))
        
        for i in range(categories_number):
            print str(classesHashtable.getClassName(i)) + " precision: "
            print average_precision_score(true_values_arr[i], score_values_arr[i])
            
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

            if opt == '-s':
                if "-r" not in optlist or "-c" not in optlist or "-d" not in optlist:
                    print "Usage: -s <evaluating_dir> -r <reference_vocab> -c <reference_classifier> -d <reference_dictionary>"
                    sys.exit(2)
                
                get_precision_scores(arg, optlist['-r'], optlist['-c'], optlist['-d'])
                sys.exit()

                        
    except getopt.GetoptError, e:
        print str(e)
        sys.exit(2)
    
    
if __name__ == "__main__":
    main(sys.argv[1:])

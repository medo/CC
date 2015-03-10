#!/usr/bin/python

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
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

def __get_image_dense_features(img):
    extractor = FeatureExtractorFactory.newInstance(img,True)
    return extractor.create_dense_descriptor()

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
                image = cv2.imread(imgfile,0)
                vector = __get_image_dense_features(image)
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
    
    __init_histogram_calculator(vocab_file)  
    __load_classifier(classifier_file)    
    __load_category_dictionary(dictionary_file)
    for d in os.listdir(path):
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
                        vector = __get_image_features(imgfile)
                        bow = histCalculator.hist(vector)
                        imgfile = "%s/%s" % (subdir, f)
                        image = __load_image(imgfile)
                        rows = len(image)
                        cols = len(image[0])
                        if rows%2==0 and cols%2==0:
                            quad1= image[:(rows/2),:(cols/2)]
                            quad2= image[:(rows/2),(cols/2):]
                            quad3 = image[(rows/2):,:(cols/2)]
                            quad4=image[(rows/2):,(cols/2):]
                        if rows%2==1 and cols%2==1:
                            quad1= image[:((rows+1)/2),:((cols+1)/2)]
                            quad2= image[:((rows+1)/2),(cols/2)+1:]
                            quad3 = image[((rows+1)/2):,:((cols+1)/2)]
                            quad4 = image[((rows+1)/2):,((cols+1)/2):]
                        if rows%2==1 and cols%2==0:
                            quad1= image[:((rows+1)/2),:((cols+1)/2)]
                            quad2 = image[:((rows+1)/2),(cols/2):]
                            quad3 = image[((rows+1)/2):,:((cols)/2)]
                            quad4 = image[((rows+1)/2):,(cols/2):]
                            #quad4 = image[2:, 2:]
                        if rows%2==0 and cols%2==1:
                            quad1= image[:((rows)/2),:((cols+1)/2)]
                            quad2 = image[:((rows)/2),(cols/2)+1:]
                            quad3 = image[(rows/2):,:((cols+1)/2)]
                            quad4 = image[(rows/2):,((cols+1)/2):]
                        vector1 = __get_image_dense_features(quad1)
                        vector2 = __get_image_dense_features(quad2)
                        vector3 = __get_image_dense_features(quad3)
                        vector4 = __get_image_dense_features(quad4)
                        bow1 = histCalculator.hist(vector1)
                        bow2 = histCalculator.hist(vector2)
                        bow3 = histCalculator.hist(vector3)
                        bow4 = histCalculator.hist(vector4)
                        bow  = np.hstack((bow1,bow2,bow3,bow4))
                        bow = __from_array_to_matrix(bow)
                        totalPredictions += 1
                        correctResponse = classifier.evaluateData(bow, label)
                        if not correctResponse:
                            wrongPredictions += 1
                        
                    except Exception, Argument:
                        print "Exception happened: ", Argument
                        traceback.print_stack()
            
            print ("Label %s results:\n%d were wrongly predicted from %d" % (d, wrongPredictions, totalPredictions))
    
    print ("Final results:\n%d were wrongly predicted from %d" % (classifier.getErrorCount(), classifier.getEvaluationsCount()))
    

def training(path, output_file, vocab_file, dictionary_output_file):
    __check_dir_condition(path)
    __check_file_condition(vocab_file)
    
    __init_histogram_calculator(vocab_file)
    
    label = 0
    labelsVector = None
    bowVector = None
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
                        rows = len(image)
                        cols = len(image[0])
                        if rows%2==0 and cols%2==0:
                            quad1= image[:(rows/2),:(cols/2)]
                            quad2= image[:(rows/2),(cols/2):]
                            quad3 = image[(rows/2):,:(cols/2)]
                            quad4=image[(rows/2):,(cols/2):]
                        if rows%2==1 and cols%2==1:
                            quad1= image[:((rows+1)/2),:((cols+1)/2)]
                            quad2= image[:((rows+1)/2),(cols/2)+1:]
                            quad3 = image[((rows+1)/2):,:((cols+1)/2)]
                            quad4 = image[((rows+1)/2):,((cols+1)/2):]
                        if rows%2==1 and cols%2==0:
                            quad1= image[:((rows+1)/2),:((cols+1)/2)]
                            quad2 = image[:((rows+1)/2),(cols/2):]
                            quad3 = image[((rows+1)/2):,:((cols)/2)]
                            quad4 = image[((rows+1)/2):,(cols/2):]
                            #quad4 = image[2:, 2:]
                        if rows%2==0 and cols%2==1:
                            quad1= image[:((rows)/2),:((cols+1)/2)]
                            quad2 = image[:((rows)/2),(cols/2)+1:]
                            quad3 = image[(rows/2):,:((cols+1)/2)]
                            quad4 = image[(rows/2):,((cols+1)/2):]
                        vector1 = __get_image_dense_features(quad1)
                        vector2 = __get_image_dense_features(quad2)
                        vector3 = __get_image_dense_features(quad3)
                        vector4 = __get_image_dense_features(quad4)
                        bow1 = histCalculator.hist(vector1)
                        bow2 = histCalculator.hist(vector2)
                        bow3 = histCalculator.hist(vector3)
                        bow4 = histCalculator.hist(vector4)
                        bow  = np.hstack((bow1,bow2,bow3,bow4))

                        if bowVector == None:
                            bowVector = bow
                        else:
                            bowVector = np.vstack((bowVector, bow))
                        if labelsVector == None:
                            labelsVector = np.array(correctLabel)
                        else:
                            labelsVector = np.insert(labelsVector, labelsVector.size, correctLabel)
                    except Exception, Argument:
                        print "Exception happened: ", Argument
                        traceback.print_stack()
            
            if label == correctLabel:
                label += 1
    try:
        print "Training Classifier"
        
        global trainingDataMat
        trainingDataMat = __from_array_to_matrix(bowVector) 
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

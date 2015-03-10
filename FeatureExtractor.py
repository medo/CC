import numpy
import cv2 as cv

class FeatureExtractor:
  
  def __init__(self, image,memory):
    self.image = image
    self.memory = memory

  def create_dense_descriptor(self):
    if self.memory==False:
        imgGray=cv.imread(self.image.get_path(),0)
        detector=cv.FeatureDetector_create("Dense")
        points=detector.detect(imgGray) #points
        extractor = cv.DescriptorExtractor_create("SIFT")
        (points, descriptors) = extractor.compute(imgGray,points)
        return descriptors
    else:
        detector=cv.FeatureDetector_create("Dense")
        points=detector.detect(self.image) #points
        extractor = cv.DescriptorExtractor_create("SIFT")
        (points, descriptors) = extractor.compute(self.image,points)
        return descriptors

  def extract_feature_vector(self):
    if self.memory==False:
        img = cv.imread(self.image.get_path())
        img_resized = cv.resize(img, (300, 250)) #resizing the image (those values should be in the configurations)
        gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
        gray_equalized = cv.equalizeHist(gray) #equalizing the histogram (must take a gray scale image)
        sift = cv.SIFT()
        kp, des = sift.detectAndCompute(gray_equalized, None)
    else:
        img_resized = cv.resize(self.image, (300, 250)) #resizing the image (those values should be in the configurations)
        gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
        gray_equalized = cv.equalizeHist(gray) #equalizing the histogram (must take a gray scale image)
        sift = cv.SIFT()
        kp, des = sift.detectAndCompute(gray_equalized, None)
    return des
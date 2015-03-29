import os, csv, sys, traceback
from CategoriesManager import CategoriesManager

def __init_classes_hashtable():
	global classesHashtable
	classesHashtable = CategoriesManager()

def labelToCategory(path,pascalData):
	__init_classes_hashtable()
	global allCategories
	allCategories=[]
	for d in os.listdir(pascalData):
		subdir = ("%s/%s" % (pascalData, d))
		if subdir.endswith(".txt") and "_train" in subdir and "val" not in subdir:
			try:
				catName= d.split("_")
				allCategories.append(catName[0])
			except Exception, Argument:
				print "Exception happened: ", Argument
				traceback.print_stack()

	allCategories2=[]
	for d in os.listdir(pascalData):
		subdir = ("%s/%s" % (pascalData, d))
		if subdir.endswith(".txt") and "_trainval" in subdir:
			try:
				catName= d.split("_")
				allCategories2.append(catName[0])
			except Exception, Argument:
				print "Exception happened: ", Argument
				traceback.print_stack()

	allCategories3=[]
	for d in os.listdir(pascalData):
		subdir = ("%s/%s" % (pascalData, d))
		if subdir.endswith(".txt") and "_trainval" in subdir:
			try:
				catName= d.split("_")
				allCategories3.append(catName[0])
			except Exception, Argument:
				print "Exception happened: ", Argument
				traceback.print_stack()

	if set(allCategories)==set(allCategories2)==set(allCategories3):	
		allCategories= list(set(allCategories))
		with open(("%s/%s" % (path, 'label_to_category.csv')), 'w+') as labelCatDict:
			for i in xrange(0,len(allCategories)): 
				fileWriter = csv.writer(labelCatDict, delimiter=',', quoting=csv.QUOTE_MINIMAL) 
				label = [i+1] 
				catName = [allCategories[i]]
				classesHashtable.addClass((i+1),allCategories[i])
				fileWriter.writerow(label+catName)
		classesHashtable.saveToFile("/Users/ZzimBoo/anaconda/envs/py27opencv/finalPascal/image-categoriy-classifier/dict_file")
		#return allCategories

	else:
		print "Exception Happened: Some .txt files are missing, each category must have three text files: train, trainval and val!"
		#return None

def generateTrainDB(path):
	pascalTrainFolder = "pascal_train"
	pascalTrainFolder = ("%s/%s" % (path, pascalTrainFolder))
	pascalData = ("%s/%s/%s/%s" % (path, "VOC2007","ImageSets","Main"))
	if not os.path.exists(pascalTrainFolder):
		os.makedirs(pascalTrainFolder)
	#allCategories=[]
	#allCategories = labelToCategory(path,pascalData)
	labelToCategory(path,pascalData)
	with open(("%s/%s" % (path, 'label_to_category.csv'))) as labelCatDict: 
		for i in xrange(0,len(allCategories)): 
			catFileRead = allCategories[i] + "_train.txt"
			fileNameWrite = "train_" + allCategories[i] + ".csv"
			with open(("%s/%s" % (pascalData, catFileRead))) as fileReader:
				csvReader = csv.reader(fileReader, delimiter=' ')
				for i in csvReader:
					if '-1' not in i and '1' in i:
						with open(("%s/%s" % (pascalTrainFolder, fileNameWrite)), 'a') as fileWriter2:
							catWriter = csv.writer(fileWriter2, delimiter=',', quoting=csv.QUOTE_MINIMAL)
							imgName = ("%s/%s/%s/%s" % (path,"VOC2007" ,"JPEGImages",i[0]))
							imgName += ".jpg"
							catWriter.writerow([imgName])

def generateTrainValidateDB(path):
	pascTrValFolder = "pascal_trainval"
	pascTrValFolder = ("%s/%s" % (path, pascTrValFolder))
	pascalData = ("%s/%s/%s/%s" % (path, "VOC2007","ImageSets","Main"))
	if not os.path.exists(pascTrValFolder):
		os.makedirs(pascTrValFolder)
	#allCategories = labelToCategory(path,pascalData)
	with open(("%s/%s" % (path, 'label_to_category.csv'))) as labelCatDict: 
		for i in xrange(0,len(allCategories)): 
			catFileRead = allCategories[i] + "_trainval.txt" 
			fileNameWrite = "trainval_" + allCategories[i] + ".csv"
			with open(("%s/%s" % (pascalData, catFileRead))) as fileReader:
				csvReader = csv.reader(fileReader, delimiter=' ')
				for i in csvReader:
					if '-1' not in i and '1' in i:
						with open(("%s/%s" % (pascTrValFolder, fileNameWrite)), 'a') as fileWriter2:
							catWriter = csv.writer(fileWriter2, delimiter=',', quoting=csv.QUOTE_MINIMAL)
							imgName = ("%s/%s/%s/%s" % (path,"VOC2007" ,"JPEGImages",i[0]))
							imgName += ".jpg"
							catWriter.writerow([imgName])

def generateValidateDB(path):
	pascValFolder = "pascal_val"
	pascValFolder = ("%s/%s" % (path, pascValFolder))
	pascalData = ("%s/%s/%s/%s" % (path, "VOC2007","ImageSets","Main"))
	if not os.path.exists(pascValFolder):
		os.makedirs(pascValFolder)
	#allCategories = labelToCategory(path,pascalData)
	with open(("%s/%s" % (path, 'label_to_category.csv'))) as labelCatDict:
		for i in xrange(0,len(allCategories)): 
			catFileRead = allCategories[i] + "_val.txt"
			fileNameWrite = "val_" + allCategories[i] + ".csv"
			with open(("%s/%s" % (pascalData, catFileRead))) as fileReader:
				csvReader = csv.reader(fileReader, delimiter=' ')
				for i in csvReader:
					if '-1' not in i and '1' in i:
						with open(("%s/%s" % (pascValFolder, fileNameWrite)), 'a') as fileWriter2:
							catWriter = csv.writer(fileWriter2, delimiter=',', quoting=csv.QUOTE_MINIMAL)
							imgName = ("%s/%s/%s/%s" % (path,"VOC2007" ,"JPEGImages",i[0]))
							imgName += ".jpg"
							catWriter.writerow([imgName])
	
def main(args):
	print "Running Pascal Script..."
	#print "args = ", args
	generateTrainDB(args)
	generateTrainValidateDB(args)
	generateValidateDB(args)

if __name__ == "__main__":
	main(sys.argv[1])
all: 
	mkdir tmp
	mkdir tmp/images
	mkdir tmp/vocab


vocabulary:
	cp images/test.png tmp/images
	python run.py -v tmp/images -o tmp/vocab/vocab.sift



training:
	mkdir tmp/images/cat
	cp images/test.png tmp/images/cat
	python run.py -t tmp/images -r tmp/vocab/vocab.sift -o tmp/classifier -d tmp/categories

evaluate:
	python run.py -e -r tmp/vocab/vocab.sift -c tmp/classifier -d tmp/categories

clean:
	rm -R tmp
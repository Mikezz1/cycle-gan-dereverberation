
mkdir data
mkdir checkpoints

cd data
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
# wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar –xf LJSpeech-1.1.tar.bz2
#rm train-clean-100.tar.gz

pip install -r requirements.txt

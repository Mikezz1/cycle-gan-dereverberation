
mkdir data
mkdir checkpoints

cd data
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar –xvzf train-clean-100.tar.gz
rm train-clean-100.tar.gz

pip install -r requirements.txt

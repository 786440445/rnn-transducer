CURDIR=$(cd $(dirname $0); pwd)
cd $CURDIR

data=/opt/tiger/speech_data/
data_url=https://openslr.magicdatatech.com/resources/33

# mkdir -p $data
# local/download_and_untar.sh $data $data_url data_aishell || exit 1;
# local/download_and_untar.sh $data $data_url resource_aishell || exit 1;

cd ../../
python3 ./pre_process.py --data_dir $data/data_aishell
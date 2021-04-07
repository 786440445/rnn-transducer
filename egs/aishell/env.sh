
data=/opt/tiger/speech_data
data_url=www.openslr.org/resources/33

local/download_and_untar.sh $data $data_url data_aishell || exit 1;
local/download_and_untar.sh $data $data_url resource_aishell || exit 1;

python3 ./pre_process.py --data_dir $data
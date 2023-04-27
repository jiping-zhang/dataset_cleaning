python ./train.py --train_set_name sst2 --train_set_path ./dataset/sst2/train.txt --valid_set_path ./dataset/sst2/dev.txt --max_len 64 --batch_size 8

python ./train.py --train_set_name="sst2_T(0.67)" --train_set_path "./dataset/sst2/cleaned/sst2_cleaned_cmT(0.67)_split2_e4_b8_ml64_lr2e-05_lfCE_l21e-05_s42.txt" --valid_set_path ./dataset/sst2/dev.txt --max_len 64 --batch_size 8

python ./train.py --train_set_name imdb --train_set_path ./dataset/imdb/train.txt --valid_set_path ./dataset/imdb/dev.txt --max_len 240 --batch_size 4
python ./eval.py --model_path "./model/B_lr2e-05_l21e-05_e4_b8_ml64_sst2_T(0.67).pth" --testset_path ./dataset/sst2/test.txt --n_label 2 --max_len 64


python ./eval.py --model_path "./model/B_lr2e-05_l21e-05_e4_b8_ml64_sst2.pth" --testset_path "./dataset/sst2/cleaned/sst2_test_cleaned_cmT(0.1)_split2_e4_b8_ml64_lr2e-05_lfCE_l21e-05_s42.txt" --n_label 2 --max_len 64
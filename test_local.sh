CUDA_VISIBLE_DEVICES=0 python process.py --many \
  --input_zadc_dir=/usr/mvl2/itdfh/dev/bonbidhie2023_algorithm/input/2Z_ADC \
  --input_adc_ss_dir=/usr/mvl2/itdfh/dev/bonbidhie2023_algorithm/input/1ADC_ss \
  --output_dir=/usr/mvl2/itdfh/dev/BONBID-HIE-MICCAI2023/bonbidhie_eval/test-correct \
  --net_cfg=configs/bonbid_swinunetr.yml \
  --net_ckpt=model/bonbid_swin_unetr_48_no-empty-overfit-48-dice+loghaus-lr=0.0001.ckpt \
  --rf_ckpt=model/RandomForestClassifier.pkl \
  --input_transform=configs/transform.yml
#!/bin/bash

cd /root/ms_recommender_original

if test $5 = "gen_news"; then
  echo "gen_news"
  python inference_nrms.py \
    --data_dir /root/MIND/ \
    --epochs 5 \
    --device_no $1 \
    --infer_model_path /root/inference/large_nrms/large_nrms/ckpt_3_0.6737 \
    --save_dir $2 \
    --news_file '/root/MIND/MINDlarge_test/news.tsv' \
    --behaviors_file $3 \
    --gen_news \
    >$4
else
  python inference_nrms.py \
    --data_dir /root/MIND/ \
    --epochs 5 \
    --device_no $1 \
    --infer_model_path /root/inference/large_nrms/large_nrms/ckpt_3_0.6737 \
    --save_dir $2 \
    --news_file '/root/MIND/MINDlarge_test/news.tsv' \
    --behaviors_file $3 \
    >$4
fi

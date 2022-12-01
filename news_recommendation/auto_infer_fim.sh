#!/bin/bash

#PyTorch / 1.9.1 / 11.1.1 / 3.8

mkdir -p /root/MIND

unzip /root/autodl-nas/MINDlarge_train.zip -d /root/MIND/MINDlarge_train/

unzip /root/autodl-nas/MINDlarge_dev.zip -d /root/MIND/MINDlarge_dev/

unzip /root/autodl-nas/MINDlarge_utils.zip -d /root/MIND/MINDlarge_utils/

unzip /root/autodl-nas/MINDlarge_test.zip -d /root/MIND/MINDlarge_test/

unzip /root/autodl-nas/large_fim.zip -d /root/MIND/large_fim/

unzip /root/autodl-nas/News-Recommendation.zip -d /root

unzip /root/autodl-nas/mind_candidate_news_201911_interest_without_labels.zip -d /root/autodl-tmp/
unzip /root/autodl-nas/mind_candidate_news_201911_biasprop_without_labels.zip -d /root/autodl-tmp/

cd /root/News-Recommendation/src/
pip install transformers
pip install pandas
pip install sklearn

cd /root/News-Recommendation/src/

while read p; do
  behaviors_path="${p}"
  behaviors_temp="${behaviors_path#*labels/}"
  behaviors_suffix="${behaviors_temp%.tsv}"

  rm -rf "/root/News-Recommendation/src/data"
  rm -rf "/root/MIND/MINDlarge_test/behaviors.tsv"
  cp "${behaviors_path}" "/root/MIND/MINDlarge_test/behaviors.tsv"
  wc -l "/root/MIND/MINDlarge_test/behaviors.tsv"

  python -m main.fim \
    --scale 'large' \
    --data-root /root/ \
    --infer-dir "${2}" \
    --suffix "${behaviors_suffix}" \
    --checkpoint "/root/MIND/large_fim/data/ckpts/FIM/large/898790.model" \
    --batch-size 16 \
    --world-size 1 \
    --device 0 \
    --mode "test" \
    --batch-size-eval 16
done <"${1}"

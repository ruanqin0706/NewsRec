#!/bin/bash

#PyTorch / 1.9.1 / 11.1.1 / 3.8

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

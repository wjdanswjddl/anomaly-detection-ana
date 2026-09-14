#!/bin/bash
cd /exp/sbnd/app/users/munjung/anomaly-detection
exec /exp/sbnd/app/users/munjung/env/bin/python -u parallel_handscan_inference.py \
  --mem-budget-gb 15 \
  --worker-rss-estimate-gb 1.35 \
  --poll-sec 8 \
  --stale-sec 900

#!/bin/bash
#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1FrhO9ehMAEcxtjFntcAmj_NvQff4jGo3' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1FrhO9ehMAEcxtjFntcAmj_NvQff4jGo3" -O 'caption_large_best.pt' && rm -rf /tmp/cookies.txt
pip install gdown
gdown --id '1FrhO9ehMAEcxtjFntcAmj_NvQff4jGo3' --output caption_large_best.pt

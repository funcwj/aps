set -eu

dict=data/dict
conf=conf/ctx.yaml
epoches=100
batch_size=96
eval_interval=-1

echo "$0 $@"

[ $# -ne 2 ] && echo "Script format error: $0 <exp-id>" && exit 1

exp_id=$1

./asr/train.py \
  --conf $conf \
  --dict $dict \
  --checkpoint exp/$exp_id \
  --batch-size $batch_size \
  --epoches $epoches \
  --eval-interval $eval_interval
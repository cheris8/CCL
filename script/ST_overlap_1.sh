CUDA_VISIBLE_DEVICES=1 python /home/intern/seungjun/commonsense/CCL/train_with_no_overlap.py --lm 'roberta-large'  --training_size 9000 --pre_task cmqa --cur_task piqa --training_type ST --split_type num --batch_size 4
CUDA_VISIBLE_DEVICES=1 python /home/intern/seungjun/commonsense/CCL/train_with_no_overlap.py --lm 'roberta-large'  --training_size 9000 --pre_task cmqa --cur_task siqa --training_type ST --split_type num --batch_size 4
CUDA_VISIBLE_DEVICES=1 python /home/intern/seungjun/commonsense/CCL/train_with_no_overlap.py --lm 'roberta-large'  --training_size 9000 --pre_task cmqa --cur_task csqa --training_type ST --split_type num --batch_size 4
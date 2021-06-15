# Semi-Autoregressive Image Captioning

## Requirements
- Python 3.6
- Pytorch 1.6

## Prepare data
1. Please use **git clone --recurse-submodules** to clone this repository and remember to follow initialization steps in coco-caption/README.md.
2. Download the preprocessd dataset from this [link](https://drive.google.com/file/d/1nF4lSK51oki46EfAvSJCudRv8un9HSwX/view?usp=sharing) and extract it to data/.
3. Please follow this [instruction](https://github.com/ruotianluo/self-critical.pytorch/blob/master/data/README.md#convert-from-peteanderson80s-original-file) to prepare the **adaptive** bottom-up features and place them under data/mscoco/. Please follow this [instruction](https://github.com/ruotianluo/self-critical.pytorch#evaluate-on-coco-test-set) to prepare the features and place them under data/cocotest/ for online test evaluation.
4. Download part checkpoints from [here](https://drive.google.com/file/d/1RLXRMpIgMQM4OGONUItW9K430atJ53OO/view?usp=sharing) and extract them to save/.

## Offline Evaluation
To reproduce the results, such as SATIC(K=2, bw=1) after self-critical training, just run

```
python3 eval.py  --model  save/nsc-sat-2-from-nsc-seqkd/model-best.pth   --infos_path  save/nsc-sat-2-from-nsc-seqkd/infos_nsc-sat-2-from-nsc-seqkd-best.pkl    --batch_size  1   --beam_size   1   --id  nsc-sat-2-from-nsc-seqkd   
```

## Online Evaluation
Please first run
```
python3 eval_cocotest.py  --input_json  data/cocotest.json  --input_fc_dir data/cocotest/cocotest_bu_fc --input_att_dir  data/cocotest/cocotest_bu_att   --input_label_h5    data/cocotalk_label.h5  --num_images -1    --language_eval 0
--model  save/nsc-sat-4-from-nsc-seqkd/model-best.pth   --infos_path  save/nsc-sat-4-from-nsc-seqkd/infos_nsc-sat-4-from-nsc-seqkd-best.pkl    --batch_size  32   --beam_size   3   --id   captions_test2014_alg_results  
```
and then follow the [instruction](https://cocodataset.org/#captions-eval) to upload results.
## Training
1.  In the first training stage, such as SATIC(K=2) model with sequence-level distillation and weight initialization,  run 
```
python3  train.py   --noamopt --noamopt_warmup 20000 --label_smoothing 0.0  --seq_per_img 5 --batch_size 10 --beam_size 1 --learning_rate 5e-4 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --learning_rate_decay_start 0 --scheduled_sampling_start 0  --save_checkpoint_every 3000 --language_eval 1 --val_images_use 5000 --max_epochs 15    --input_label_h5   data/cocotalk_seq-kd-from-nsc-transformer-baseline-b5_label.h5   --checkpoint_path   save/sat-2-from-nsc-seqkd   --id   sat-2-from-nsc-seqkd   --K  2
```

2. Then in the second training stage, copy the above pretrained model first

```
cd save
./copy_model.sh  sat-2-from-nsc-seqkd    nsc-sat-2-from-nsc-seqkd
cd ..
``` 
and then run
```
python3  train.py    --seq_per_img 5 --batch_size 10 --beam_size 1 --learning_rate 1e-5 --num_layers 6 --input_encoding_size 512 --rnn_size 2048  --save_checkpoint_every 3000 --language_eval 1 --val_images_use 5000 --self_critical_after 10  --max_epochs    40   --input_label_h5    data/cocotalk_label.h5   --start_from   save/nsc-sat-2-from-nsc-seqkd   --checkpoint_path   save/nsc-sat-2-from-nsc-seqkd  --id  nsc-sat-2-from-nsc-seqkd    --K 2
```

## Citation

```
```

## Acknowledgements
This repository is built upon [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch). Thanks for the released  code.

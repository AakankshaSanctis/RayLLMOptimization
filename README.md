# RayLLMOptimization
Experimenting with different optimization techniques for a faster training and inference pipeline using Ray

The data/datasets folder contains the datasets that we finetune the LLMs on.

The finetuning folder contains the self explanatory notebooks that one can run in order to finetune both Bert and Roberta on the IMDB datasets.

The script `imdb_data_finetuning_deepspeed.py` can be run as follows: 

`python imdb_data_finetuning_deepspeed.py --dataset imdb --model distilroberta-base --num_workers 4 --cpus_per_worker 1 --batch_size 8 --wandb_project_name roberta_4_gpu_4_cpu_8_bs`

Here are the results: 

https://wandb.ai/hpml3/roberta_4_gpu_4_cpu_32_bs/reports/Bert-vs-Roberta-4-4-32--Vmlldzo0MzM1NjIw
https://wandb.ai/hpml3/bert_4_gpu_4_cpu_32_bs/reports/Bert-vs-Roberta-4-4-8--Vmlldzo0MzM1NjEz
https://wandb.ai/hpml3/bert_1_gpu_2_cpu_32_bs/reports/Bert-1-2-32-vs-2-4-32-vs-4-4-32--Vmlldzo0MzM0OTkw
https://wandb.ai/hpml3/roberta_2_gpu_4_cpu_32_bs/reports/Roberta-2-4-8-vs-2-4-32--Vmlldzo0MzM0ODUz
https://wandb.ai/hpml3/roberta_1_gpu_2_cpu_32_bs/reports/Roberta-2vs4-CPUs--Vmlldzo0MzM0NjI1
https://wandb.ai/hpml3/roberta_4_gpu_4_cpu_8_bs/reports/Roberta-2-4-8-vs-4-4-8--Vmlldzo0MzM0ODk1

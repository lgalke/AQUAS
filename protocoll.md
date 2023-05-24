# deNBI VM
deNBI VM: scp -P 30327  -i ~/.ssh/id_rsa.pub  data/ready_dataset.csv ubuntu@129.70.51.6:/home/ubuntu
VM parameter: VM: 1; Flavour: de.NBI GPU medium
total core: 14
total RAM: 64GB
total GPUs: 1
Storage Limit 500 GB
Volume Counter 
Bert: https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/bert/modeling_bert.py#L1533

# BERT problems with long texts
- maximum position embedding is 512 tokens in BERT; also when more tokens are set the memory consumption is "unaffordable because all the activations are stored for back-propagation during training" (Ding et al. 2020:2) 
- sliding window approach: 
  - disadvantages: lack of long-distance attention (Ding et al. 2020:2); special relevance of first and last sentence of text paragraph cannot be considerd by sliding window (Ding et al. 2020:6)




ready_dataset.csv: 927 items (3 x 309)
BERT: 
- epochs=3, batch_size=8
  optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0)
  Validation Loss: 9.3224 Accuracy: 0.3297

- epoch 6, batch_size = 8
  Validation Loss: 10.5421 Accuracy: 0.3459



    BERT-dmis-lab/biobert-v1.1

Tokenz 2048
epochs: 5
 Run history:
wandb: accuracy ▆▁▇▃█
wandb:       f1 ▆▁▇▄█
wandb: 
wandb: Run summary:
wandb: accuracy 0.96608
wandb:       f1 0.96618
wandb: 
wandb: 🚀 View run dark-cantina-63 at: https://wandb.ai/zbmed/AQUAS/runs/iq2w99fh

Tokenz 2048
epochs 10
 Run history:
wandb: accuracy ▆▇█▃▄▂▄▂▁▂
wandb:       f1 ▅▇█▃▄▂▄▂▁▂
wandb: 
wandb: Run summary:
wandb: accuracy 0.42741
wandb:       f1 0.27495
wandb: 
wandb: 🚀 View run devoted-silence-69 at: https://wandb.ai/zbmed/AQUAS/runs/64hlj2xu

Tokens 2048
epochs 15
 accuracy ▇█▆▇█▁████
wandb:       f1 ▇█▆▇█▁████
wandb: 
wandb: Run summary:
wandb: accuracy 0.96065
wandb:       f1 0.96063
wandb: 
wandb: 🚀 View run dazzling-sponge-70 at: https://wandb.ai/zbmed/AQUAS/runs/evd8282r
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20230510_081203-evd8282r/logs







Tokenz 10k
Epochs 5
 Run history:
wandb: accuracy ██▁▁█
wandb:       f1 ██▁▁█
wandb: 
wandb: Run summary:
wandb: accuracy 0.59566
wandb:       f1 0.47973
wandb: 
wandb: 🚀 View run stellar-fleet-64 at: https://wandb.ai/zbmed/AQUAS/runs/e3ekztxw




Tokenz 10k:
epochs: 10
   wandb: Run history:
 wandb: accuracy █▃▄▇▄▇▁▁▁▁
   wandb:       f1 █▄▅█▅█▁▁▁▁
   wandb: 
   wandb: Run summary:
   wandb: accuracy 0.31343
   wandb:       f1 0.14959
   wandb: expert-snowflake-58


Tokens 10k
epochs: 15

accuracy ▅▆███▆▁▁▂▂▂▂▂▂▂
wandb:       f1 ▅▇███▅▁▁▂▃▂▃▃▂▂
wandb: 
wandb: Run summary:
wandb: accuracy 0.43691
wandb:       f1 0.26569
  https://wandb.ai/zbmed/AQUAS/runs/hes6ebv2









    BERT base uncased
okens 2048
epochs 5
accuracy ▁▆▇█▇█████
wandb:       f1 ▁▆▇█▇█████
wandb: 
wandb: Run summary:
wandb: accuracy 0.98372
wandb:       f1 0.98379
wandb: 
wandb: 🚀 View run bumbling-snowflake-72 at: https://wandb.ai/zbmed/AQUAS/runs/r1it4j1o



Tokenz: 2048
epochs: 10
 Run history:
wandb: accuracy ▂▇█▅▁▅▇▇▇█
wandb:       f1 ▂▇█▅▁▅▇▇▇█
wandb: 
wandb: Run summary:
wandb: accuracy 0.97422
wandb:       f1 0.97436
wandb: 
wandb: 🚀 View run winter-durian-60 at: https://wandb.ai/zbmed/AQUAS/runs/202v9ovs

Tokens: 2048
epochs: 15
db: accuracy ▇█▃▁▁▁▁▃▃▃
wandb:       f1 ▇█▃▁▁▁▁▃▃▃
wandb: 
wandb: Run summary:
wandb: accuracy 0.6445
wandb:       f1 0.54669
wandb: 
wandb: 🚀 View run fresh-river-71 at: https://wandb.ai/zbmed/AQUAS/runs/c58u1cd8
wandb: 


Tokens 10k
epochs: 5
accuracy █▇██▄▄▄▄▄▁
wandb:       f1 █▇██▄▃▃▄▄▁
wandb: 
wandb: Run summary:
wandb: accuracy 0.43826
wandb:       f1 0.26709
wandb: 
wandb: 🚀 View run different-snowflake-73 at: https://wandb.ai/zbmed/AQUAS/runs/a20sj6fu


Tokens: 10k
epochs: 10
 Waiting for W&B process to finish... (success).
wandb: / 0.002 MB of 0.002 MB uploaded (0.000 MB deduped)
wandb: Run history:
wandb: accuracy ▇███▄▃▁▁▁▁
wandb:       f1 ▇███▅▃▁▁▁▁
wandb: 
wandb: Run summary:
wandb: accuracy 0.44776
wandb:       f1 0.27697
wandb: 
wandb: 🚀 View run holographic-ewok-62 at: https://wandb.ai/zbmed/AQUAS/runs/u14vrskl


tokens: 10k
Epochs 15
accuracy ▁▆▇▇███████████
wandb:       f1 ▁▆█▇███████████
wandb: 
wandb: Run summary:
wandb: accuracy 0.90366
wandb:       f1 0.90324
wandb: 
wandb: 🚀 View run stellar-admiral-65 at: https://wandb.ai/zbmed/AQUAS/runs/n8zorpxt


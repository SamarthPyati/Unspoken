| Phase | Model | Run | n | s/problem | GPU-hours | GPU |
|---|---|---|---|---|---|---|
| T6 dev | 3B | single_cot | 300 | 12.9 | 1.1 | 1 |
| T6 dev | 3B | text_mas | 300 | 40.3 | 3.4 | 1 |
| T6 dev | 3B | latent_mas_m0 | 300 | ~9.7 | 0.8 | 0 |
| T6 dev | 3B | latent_mas_m5 | 300 | 10.8 | 0.9 | 0 |
| T6 dev | 3B | latent_mas_m10 | 300 | 10.0 | 0.8 | 1 |
| T6 dev | 3B | latent_mas_m20 | 300 | ~11.5 | 1.0 | 1 |
| T6 dev | 3B | latent_mas_m40 | 300 | ~14.6 | 1.2 | 0 |
| T6 dev | 3B | latent_mas_m0_drop | 300 | 12.6 | 1.0 | 0 |
| T6 dev | 3B | latent_mas_m10_raw | 300 | ~10.0 | 0.8 | 0 |
| T6 dev | 3B | latent_mas_m10_norm_only | 300 | ~10.0 | 0.8 | 1 |
| T7 final | 3B | single_cot | 1319 | 12.9 | 4.7 | 0 |
| T7 final | 3B | text_mas | 1319 | 40.3 | 14.8 | 0 |
| T7 final | 3B | latent_mas_m10 | 1319 | 10.0 | 3.6 | 0 |
| T7 final | 1.5B | single_cot | 1319 | 10.0 | 3.7 | 1 |
| T7 final | 1.5B | text_mas | 1319 | 24.9 | 9.1 | 1 |
| T7 final | 1.5B | latent_mas_m10 | 1319 | 21.2 | 7.8 | 1 |

Total GPU-hours: 55.6
Wall-clock on 2 GPUs (longest lane): 28.0 h (lanes: 28.0, 27.6)
Kaggle sessions of 11 h: 2.5; weeks of a 30 h quota: 0.9 (assumes quota is charged per session hour, not per GPU: verify on Kaggle)

# Commands, grouped by GPU. Runs are resumable: rerun the same command in the next session to continue.

# --- GPU 0  (~28.0 h)
CUDA_VISIBLE_DEVICES=0 python -m scripts.run --method latent_mas --latent_steps 0 --model Qwen/Qwen2.5-3B-Instruct --split train --n 300 --offset 5000
CUDA_VISIBLE_DEVICES=0 python -m scripts.run --method latent_mas --latent_steps 5 --model Qwen/Qwen2.5-3B-Instruct --split train --n 300 --offset 5000
CUDA_VISIBLE_DEVICES=0 python -m scripts.run --method latent_mas --latent_steps 40 --model Qwen/Qwen2.5-3B-Instruct --split train --n 300 --offset 5000
CUDA_VISIBLE_DEVICES=0 python -m scripts.run --method latent_mas --latent_steps 0 --model Qwen/Qwen2.5-3B-Instruct --m0_cache drop --split train --n 300 --offset 5000
CUDA_VISIBLE_DEVICES=0 python -m scripts.run --method latent_mas --latent_steps 10 --model Qwen/Qwen2.5-3B-Instruct --realign raw --split train --n 300 --offset 5000
CUDA_VISIBLE_DEVICES=0 python -m scripts.run --method single_cot --model Qwen/Qwen2.5-3B-Instruct --split test --n 1319
CUDA_VISIBLE_DEVICES=0 python -m scripts.run --method text_mas --model Qwen/Qwen2.5-3B-Instruct --split test --n 1319
CUDA_VISIBLE_DEVICES=0 python -m scripts.run --method latent_mas --latent_steps 10 --model Qwen/Qwen2.5-3B-Instruct --split test --n 1319

# --- GPU 1  (~27.6 h)
CUDA_VISIBLE_DEVICES=1 python -m scripts.run --method single_cot --model Qwen/Qwen2.5-3B-Instruct --split train --n 300 --offset 5000
CUDA_VISIBLE_DEVICES=1 python -m scripts.run --method text_mas --model Qwen/Qwen2.5-3B-Instruct --split train --n 300 --offset 5000
CUDA_VISIBLE_DEVICES=1 python -m scripts.run --method latent_mas --latent_steps 10 --model Qwen/Qwen2.5-3B-Instruct --split train --n 300 --offset 5000
CUDA_VISIBLE_DEVICES=1 python -m scripts.run --method latent_mas --latent_steps 20 --model Qwen/Qwen2.5-3B-Instruct --split train --n 300 --offset 5000
CUDA_VISIBLE_DEVICES=1 python -m scripts.run --method latent_mas --latent_steps 10 --model Qwen/Qwen2.5-3B-Instruct --realign norm_only --split train --n 300 --offset 5000
CUDA_VISIBLE_DEVICES=1 python -m scripts.run --method single_cot --model Qwen/Qwen2.5-1.5B-Instruct --split test --n 1319
CUDA_VISIBLE_DEVICES=1 python -m scripts.run --method text_mas --model Qwen/Qwen2.5-1.5B-Instruct --split test --n 1319
CUDA_VISIBLE_DEVICES=1 python -m scripts.run --method latent_mas --latent_steps 10 --model Qwen/Qwen2.5-1.5B-Instruct --split test --n 1319

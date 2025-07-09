# gpt-bert-70M

Example Usage 


```
git clone https://github.com/suchirsalhan/gpt-bert-70M
bash setup.sh
pip install -e .  # From the repo root with the pyproject.toml
```


```
python train.py \
  --dataset_name babylm_gpt-bert-70M_single_shuffle
  --config_file configs/base.json \
  --tokenizer_path Talking-Babies/orpo_opt_base_tokenizer \
  --output_dir ./checkpoints/babylm_run
```

`babylm_gpt-bert-70M_single_shuffle` is the default.  `["babylm_gpt-bert-70M_single_shuffle", "kidlm_gpt-bert-70M_single_shuffle", "fineweb_gpt-bert-70M_single_shuffle"]` are the other choices. 



# IDOL


- 📚 Repo for our paper `"IDOL: Indicator-oriented Logic Pre-training for Logical Reasoning"` accepted to the Findings of ACL 2023. [[Link]](https://aclanthology.org/2023.findings-acl.513/#)
- 🏆 Ranked $1^{st}$  system on ReClor Leaderboard from 2022.12 to 2023.10

<div align="center">
<img src=./imgs/ReClorChampionSanpshot.png width=100% />
</div>

<!-- ![system-flowchat](/imgs/system-flowchart.png){:height="50%" width="50%"} -->
<div align="center">
<img src=./imgs/system-flowchart.png width=100% />
</div>

## Pre-training Dataset (LGP)

### Step 1
Download wikipedia data at [WikiDumps](https://dumps.wikimedia.org/backup-index.html). Then, extract texts with the help of [WikiExtractor](https://github.com/attardi/wikiextractor).

### Step 2
Extract logic-related texts and give them LCP labels after tokenization with the help of the functions in `scripts/LGP/utils.py`. Here, we take RoBERTa for example, the IDOL pre-training dataset for RoBERTa is available at [GoogleDrive](https://drive.google.com/file/d/1D_LOSJ1bC4UF1G5gCLe6_NmJaoe1oyjf/view?usp=sharing).

## IDOL Pre-training
- During pre-training with IDOL, models learns via MLM and LCP simultaneously as follows:
<div align="center">
<img src=./imgs/IDOL-pretraining.png width=40% />
</div>

- About the training environment dependencies, please refer to `./idol_environment.yml`. As for the library `transformers`, please use the one provided in `./transformers`.
- Steps
  ```shell
  1. cd /scripts/pretrain
  2. Put LGP in ./data
  3. Change the values of parameters to your prefered ones in logic_pretrain.sh
  4. sh logic_pretrain.sh

  ```

- Examples of checkpoints further pre-trained with IDOL are available at:

  |**Model**|Link|Model|Link|
  |:-------:|:-------:|:-------:|:-------:|
  | **BERT** | [Google Drive](https://drive.google.com/drive/folders/1btwpE6_3z1qefoqSpxZNslPXSuQMQNmH?usp=drive_link) | **RoBERTa** | [Google Drive](https://drive.google.com/drive/folders/1qqAnHY4U-z5_VdxfARLsQd_ClvvYE-iB?usp=drive_link) |
  | **ALBERT** | [Google Drive](https://drive.google.com/drive/folders/1zWwzuPqtjnE01Lo0K1m9-rtHtMHlQNCO?usp=drive_link) | **DeBERTa** | [Google Drive](https://drive.google.com/drive/folders/1jMxL569gCENpLdLWtygxcrenDC3yjGbi?usp=drive_link) |

## Downstream Fine-tuning
Our implementation is based on the official framework provided by the ReClor team and we made some customization. ReClor, LogiQA, RACE are supported in our example in `/scripts/finetune`.

```shell
1. cd /scripts/finetune
2. Put the downstream task datasets in ./data
3. Change the values of parameters to your prefered ones # especially task_name
4. sh run_ft.sh

```

## Citation
```
@inproceedings{xu-etal-2023-idol,
    title = "{IDOL}: Indicator-oriented Logic Pre-training for Logical Reasoning",
    author = "Xu, Zihang  and
      Yang, Ziqing  and
      Cui, Yiming  and
      Wang, Shijin",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.513",
    pages = "8099--8111",
    abstract = "In the field of machine reading comprehension (MRC), existing systems have surpassed the average performance of human beings in many tasks like SQuAD. However, there is still a long way to go when it comes to logical reasoning. Although some methods for it have been put forward, they either are designed in a quite complicated way or rely too much on external structures. In this paper, we proposed IDOL (InDicator-Oriented Logic Pre-training), an easy-to-understand but highly effective further pre-training task which logically strengthens the pre-trained models with the help of 6 types of logical indicators and a logically rich dataset LoGic Pre-training (LGP). IDOL achieves state-of-the-art performance on ReClor and LogiQA, the two most representative benchmarks in logical reasoning MRC, and is proven to be capable of generalizing to different pre-trained models and other types of MRC benchmarks like RACE and SQuAD 2.0 while keeping competitive general language understanding ability through testing on tasks in GLUE. Besides, at the beginning of the era of large language models, we take several of them like ChatGPT into comparison and find that IDOL still shows its advantage.",
}


```


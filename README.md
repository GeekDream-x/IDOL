# IDOL

### * ***Under Construction*** *

This is the repo for our paper "IDOL: Indicator-oriented Logic Pre-training for Logical Reasoning" accepted to the Findings of ACL 2023. Available at ACL anthology at [LINK](https://aclanthology.org/2023.findings-acl.513/#).

Resources like codes, datasets and models will be provided here recently.

<!-- ![system-flowchat](/imgs/system-flowchart.png){:height="50%" width="50%"} -->
<div align="center">
<img src=./imgs/system-flowchart.png width=100% />
</div>

## Pre-training Data (LGP)

### Step 1
Download wikipedia data at [WikiDumps](https://dumps.wikimedia.org/backup-index.html).

### Step 2
Extract logic-related texts and give them LCP labels after tokenization with the help of the functions in ~~utils.py~~. Here, we take RoBERTa for example, the IDOL pre-training dataset for RoBERTa is available at [GoogleDrive](https://drive.google.com/file/d/1D_LOSJ1bC4UF1G5gCLe6_NmJaoe1oyjf/view?usp=sharing).

## IDOL Pre-training
During pre-training with IDOL, models learns via MLM and LCP as follows:
<div align="center">
<img src=./imgs/IDOL-pretraining.png width=40% />
</div>

## Downstream Fine-tuning
Our implementation is based on the official framework provided by the ReClor team and we made some customization.


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


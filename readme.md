# msr22-defects
This repo illustrates source code of experiments in 
*Evaluating the effectiveness of local explanation methods on 
source code-based defect prediction* published in MSR'22 as well 
as the adopted datasets *lucene (2.0.0 and 2.2.0)*, *poi (1.5.0 and 2.0.0)*, 
and *xalan (2.5.0 and 2.6.0)*.

We implement token frequency-based models such as TF [1] and TF-IDF, 
and deep learning-based models such as DBN model [2], CNN model [3] and 
simple BiLSTM model based on [4, 5]. However, in our publication, only TF, DBN
and CNN are evaluated.
## Environment Setup
Python 3.8 with following packages are required.
```
javalang        0.13.0
keras           2.6.0
matplotlib      3.3.2
numpy           1.19.5
pandas          1.1.3
prettytable     3.2.0
sklearn         0.24.2
scipy           1.7.3
tensorflow      2.6.0
tqdm            4.50.2
```
We conduct our experiments on a Windows 10 desktop PC with Intel Core i7 10700 CPU and 64GB Memory. 
Evaluated deep learning-based defect prediction models such as DBN and CNN can be
trained on CPU with acceptable time consumption, and we notice that one CPU core and 
4GB memory is enough for running a single evaluation. Though, a typical GPU may be necessary 
if more complex models will be evaluated.

## Running the Evaluation

First, unzip datasets named `lucene.7z`, `poi.7z` and `xalan.7z` at `./test/data/promise/` in current directory.

Make sure such directories exist:

`./test/data/promise/lucene/2.0.0/`

`./test/data/promise/lucene/2.2.0/`

`./test/data/promise/poi/1.5.0/`

`./test/data/promise/poi/2.0.0/`

`./test/data/promise/xalan/2.5.0/`

`./test/data/promise/xalan/2.6.0/`


Then, use `run.py` to (1) preprocess the dataset, (2) train the models and (3) evaluate the local explanation
methods.

### Preprocess the datasets 
```cmd
$ python run.py --preprocess-dataset --project {project} --output {output}
```
In this step ASTs from the source code files will be extracted, and storage it to a seperate file to speed up 
the process in training and evaluation stage. We ignore java files which do not contain class definition since `javalang` cannot parse
such files. We also ignore missing files which are included in the dataset from PROMISE repositories [6].
#### Parameters
The parameter `{project}` could be one of `lucene-200-220`, `poi-150-200` and `xalan-250-260` which are 
evaluated in our paper. 

The parameter `{output}` specifies the filename for output. 

### Train a source code-based defect prediction model
```cmd
$ python run.py --train-model --project {project} --output {output} --model {model}
```
In this step the specified source code-based defect prediction model will be trained.

#### Parameters

The parameter `{project}` specifies a project of 2 continous versions for model training. It can be 
one of `lucene-200-220`, `poi-150-200` and `xalan-250-260` which are evaluated in our paper. 

The parameter `{output}` specifies the filename for output. 

The parameter `{model}`  specifies a model for training. It can be one of `TF`, `TFIDF`, `DBN`, 
`CNN`, and `LSTM`.

### Evaluate the local explanation methods
Use `run.py` to evaluate the local explanation methods on several source code-based defect prediction models.
```cmd
$ python run.py --project {project} --output {output} --model {model} --k {k}
```
In this step two local explanation methods as well as a baseline will be evaluated.

#### Parameters
The parameter `{project}` specifies a project of 2 continous versions for model training. It can be 
one of `lucene-200-220`, `poi-150-200` and `xalan-250-260` which are evaluated in our paper. 

The parameter `{output}` specifies the filename for output. 

The parameter `{model}`  specifies a model for evaluation. It can be one of `TF`, `TFIDF`, `DBN`, 
`CNN`, and `LSTM`.

The parameter `{explanator}` specifies local explanation methods for evaluation. The local explanation
methods are WO (Word Omission), LIME, and RG (Random Guessing). More than 1 methods can be specified 
which are separated by comma, e.g., `--explanator WO,LIME,RG`.

The parameter `{k}` is the number of features used for explanation. In our study we set `k` from 
`3` to `10`.

## References
[1] Supatsara Wattanakriengkrai, Patanamon Thongtanunam, Chakkrit Tantithamthavorn, Hideaki Hata 
and Kenichi Matsumoto. 2020. Predicting defective lines using a model-agnostic technique. 
CoRR. https://arxiv.org/abs/2009.03612.

[2] Song Wang, Taiyue Liu and Lin Tan. 2016. Automatically learning semantic features for defect 
prediction. In Proceedings of the 38th International Conference on Software Engineering (ICSE'16). 
Association for Computing Machinery, New York, NY, USA, 297–308. 

[3]	Jian Li, Pinjia He, Jieming Zhu and Michael R. Lyu. 2017. Software Defect Prediction via 
Convolutional Neural Network. In Proceedings of 2017 IEEE International Conference on Software 
Quality, Reliability, and Security (QRS'17). IEEE, 318–328.

[4] Guisheng Fan, Xuyang Diao, Huiqun Yu, Kang Yang and Liqiong Chen. 2019. Deep Semantic Feature 
Learning with Embedded Static Metrics for Software Defect Prediction. In Proceedings of the 26th 
Asia-Pacific Software Engineering Conference (APSEC'19). IEEE, 244–251.

[5] Hao Li, Xiaohong Li, Xiang Chen, Xiaofei Xie, Yanzhou Mu and Zhiyong Feng. 2019. Cross-project 
Defect Prediction via ASTToken2Vec and BLSTM-based Neural Network. In Proceedings of 2019 
International Joint Conference on Neural Networks (IJCNN'19). IEEE, 1–8.

[6] Marian Jureczko and Lech Madeyski. 2010. Towards identifying software project clusters with 
regard to defect prediction. In Proceedings of the 6th International Conference on Predictive Models
in Software Engineering (PROMISE'10). Association for Computing Machinery, New York, NY, USA, 1-10.
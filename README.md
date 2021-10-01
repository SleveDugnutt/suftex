# suftex
python code to train a language model to generate texts from end to start using [performer-pytorch](https://github.com/lucidrains/performer-pytorch)
# dependencies
```bash
$ pip install tqdm
$ pip install sentencepiece
$ pip install torch
$ pip install performer-pytorch
```
# usage
```bash
$ git clone https://github.com/SleveDugnutt/suftex
$ cd suftex
```
You need a csv file containing a column of texts.\
First, train [sentencepiece](https://github.com/google/sentencepiece) on the data.
```bash
$ python suftex.py \
-d PATH_TO_CSV_FILE \
-col COLUMN_NAME_TO_USE \
--train_sp
```
which makes a sentencepiece model file named ```suftex.model```.\
Then, train a model.
```bash
$ python suftex.py \
-d PATH_TO_CSV_FILE \
-col COLUMN_NAME_TO_USE \
-sp PATH_TO_suftex.model \
--train
```
to continue training from a checkpoint(.pt file),
```bash
$ python suftex.py \
-d PATH_TO_CSV_FILE \
-col COLUMN_NAME_TO_USE \
-sp PATH_TO_suftex.model \
--train \
-cpt PATH_CHECKPOINT_TO_USE
```
## options
when training sentencepiece, you can set ```vocab_size``` and ```character_coverage```.
```bash
$ python suftex.py \
-d PATH_TO_CSV_FILE \
-col COLUMN_NAME_TO_USE \
--train_sp
--vocab_size INTEGER(default : 10000) \
--character_coverage FLOAT(default : 0.995)
```
when traing a model, you can set the number of training epochs and batch size.
```bash
$ python suftex.py \
-d PATH_TO_CSV_FILE \
-col COLUMN_NAME_TO_USE \
-sp PATH_TO_suftex.model \
--train \
-n EPOCHS(default : 10) \
-b BATCH_SIZE(default : 32)
```

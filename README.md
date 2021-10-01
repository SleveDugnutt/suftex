# suftex
python code to train a model to generate texts ending with specific words using [performer-pytorch](https://github.com/lucidrains/performer-pytorch)
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
First, train sentencepiece on the data.
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

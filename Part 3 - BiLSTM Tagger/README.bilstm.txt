README for bilstmTrain.py and bilstmPredict.py:

In the current working directory should be:
1. bilstmTrain.py
2. bilstmTrain_utils.py
2. bilstmPredict.py
4. bilstmPredict_utils.py
5. modelFile
6. train set (of the relevant task - pos/ner)
7. dev set (of the same task)
8. test set (of the same task)


Five parameters to bilstmTrain.py are:
1. The chosen representation (a/b/c/d)
2. The train file
3. modelFile
4. The dev file
5. The task you want(pos/ner).

For example in the command line you should enter: 
python3 bilstmTrain.py a trainFile modelFile devFile pos
python3 bilstmTrain.py b trainFile modelFile devFile ner

* The dictFile will be outputted to the current working directory.


Five parameters to bilstmPredict.py are:
1. The chosen representation (a/b/c/d)
2. modelFile
3. The test set
4. dictFile (is output of bilstmTrain.py) 
5. The task you want(pos / ner).

For example in the command line you should enter: 
python3 bilstmPredict.py a modelFile testFile dictFile pos
python3 bilstmPredict.py b modelFile testFile dictFile ner



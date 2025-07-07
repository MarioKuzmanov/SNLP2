# Assignment 3: Segmentation (Sequence Labeling)

**Deadline:** Jul 25, 2025

This assignment is on segmentation, a common task in NLP.
In particular, the problem we are solving is
"compound splitting" for German.

As before, you can find the data in your repository as two separate files.
The file [nlp2-a3.train](nlp2-a3.train) contain the training instances,
and the file [nlp2-a3.test](nlp2-a3.test) contains the test instances.
Both files contain a single word per line.
The training data contains parts of the compounds delimited with spaces.
Your task is inserting the spaces between the components of compound
words in the test file using a machine learning method.
The data comes from an earlier study
[(Ma et al.  2016)](https://aclanthology.org/W16-2012).
Although the data set is known to have some (systematic) errors,
we will use it as is,
and you are not recommended to fix the errors in the training set
(the test set has the same errors).

You are allowed to use any additional unlabeled / unsegmented data.
However, you are not allowed to use any data set particularly
prepared for German compound splitting.

A common approach to supervised segmentation is formulating it as
a sequence labeling task where we label each character as
'**B**: beginning of a unit' (in our case units are _morph_s)
and '**I**: inside a unit'.
For example, an example input and output pair for the classifier can be:

```
input:  computerlinguistik
output: BIIIIIIIBIIIIIIIII
```

In some problems it is also useful to introduce another class,
'**O**: outside a unit'.
As a result this approach is often called 'B/I/O' segmentation.
This is a common approach for a wide range of NLP tasks from
tokenization to named-entity recognition (sometimes with additional labels).
Note that even though we are classifying the individual characters,
the context and other decisions in the sequence matters.
Hence, it is a common practice to use methods for sequence prediction
for this problem.

If you follow this approach,
a substantial part of your code will be converting the input into a format
that can be fed to the ML algorithm,
and then convert the predictions back to a readable format.

For our relatively simplified problem,
you can potentially get good results using standard classifiers
attending to a fixed context.
However, you are recommended to try (multiple) methods of sequence learning
for this assignment.

## Submission information

As in previous assignments, you are required to submit your predictions
as a text file. The predictions file for this experiment is similar to
the training file, where each line contains a word from the test file,
and if the word is a compound, a single space character should be
inserted between the parts of the word.
For the example above, the prediction file should contain
a line like the following.

```
computer linguistik
```

The filename for the predictions file should be like `a3-<method>.predictions`,
where, `<method>` is a (short) string that identifies your approach.
The method name you use for the file will be visible on the scoreboard
(see below).
You are recommended to use understandable method names.
For example, if your method is based on a recurrent network,
`rnn` may be a good choice.
You are encouraged to submit multiple sets of predictions.

For each approach, the main Python script should be named `a3-<method>.py`.
You are free to structure your code as multiple files
(and share code across different approaches).
However, the main file for each approach should be named as instructed above.

Unlike previous assignments, we do not provide a baseline.
You are strongly recommended to make use of the lab sessions
where examples solution (for similar tasks) will be provided.

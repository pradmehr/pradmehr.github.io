---
layout: "posts"
title: "Generating Poetry by using LSTM-RNN"
subtitle: "Project Report - First Revision"
date: 2020-05-22
permalink: /Projects/PoetryGeneration/
categories: project
tags: [poems, poetry generation, LSTM, RNN, data science, text analysis]
header:
  image: /images/Project/Shakespeare.jpg
mathjax: "true"
---

![]({{ site.url }}{{ site.baseurl }}/images/Project/LSTM/unnamed-chunk-1-1.png)

Introduction
============

In this present post, I am going to investigate poems generation by using Recurrent Neural Network (RNN). Since a poem is a collection of words in a sequence, I decide to deploy Long Short-Term Memory (LSTM) as it can capture this sequence of words. Each cell of LSTM can process data in a sequence way and use its hidden layers as a new input. After this short introduction, I explain the chosen dataset. Then, I do some preprocessing tasks to make the dataset prepared for neural network analysis. By conducting Explanatory Data Analysis (EDA), we will get more insight from the dataset. We will take a look at word frequency, word cloud, and n-gram. In the model section, we discuss LSTM and the results.

Data
====

The dataset[^fn1] used in this project is a collection gathered from the Poetry Foundation webpage poetryFoundation[^fn2] webpage . It contains 573 poems from 67 poets,
including `WILLIAM SHAKESPEARE` and `SIR PHILIP SIDNEY`. Columns and
their description are as follows.

| featureNames | Description                                                                      |
|:-------------|:---------------------------------------------------------------------------------|
| Author       | Poets’ Name                                                                      |
| content      | content of Poems                                                                 |
| poem.name    | Poems’ Name                                                                      |
| age          | binary value representing whether poems are “Classic” or belong to “Renaissance” |
| type         | The theme of poems: “Love”, “Nature”, and “Mythology & Folklore”                 |

For example, the first row contains the William Shakespeare’s[^fn3] poem called
`The Phoenix and the Turtle`. It starts with:

> “*Let the bird of loudest lay On the sole Arabian tree Herald sad and
> trumpet be, To whose sound chaste wings obey.*”
>
> <footer>
> William Shakespeare
> </footer>


Preprocessing
=============

Before conducting any Machine Learning algorithm on the text, it is
necessary to take some preprocessing steps. The abovementioned dataset
contains both modern and classic poems. This implies that we should
expect to see some archaic words (such as thee, thy) and some accented
characters (e.g. ã, á) in the poems.

It could contain abbreviations form of words, e.g. `won't` instead of
`will not`, `'d` instead of `would`, etc. Besides, we must remove
additional spaces and punctuations. Last but not the least, this dataset
is collected from [Poetry Foundation](https://www.poetryfoundation.org/)
by using web scraping, which may result in some unwanted characters. We
must deal with these type of characters.

EDA
===

*Word frequency table*
----------------------

![Top 6 most frequent words in the poetry dataset]({{ site.url }}{{ site.baseurl }}/images/Project/LSTM/unnamed-chunk-4-1.png)

The first step we can take in Explanatory Data Analysis (EDA) is to
display the word frequency. It can be useful to find out the
high-frequent words used in poetry. It should not be surprising if we
see some romantic and sweet words. After tokenizing the content of the
poem into words, we can remove the words called **stop words** such as
`and`, `that`, etc.

Since there are some classic poems in this dataset, we expect to observe
many archaic pronouns (e.g. thee, thou, and thy). It can affect the
realtive frequency of other words. In order to correct this issue, these
words are filterd from this EDA step. As we expected, the most frequent
words are sweet and romantics such as `Love`, `Heart`, `sweet`, … If we
divide corresponding values by the total number of poems in the dataset
(506), there would be at least one word `Love` in each poem (on
average).

![Word Cloud (for those whose frequency is higher than 10)]({{ site.url }}{{ site.baseurl }}/images/Project/LSTM/unnamed-chunk-5-1.png)

*Word Cloud*
------------

Word cloud is another way of displaying word frequency in a more fancy
and schematic way. Different colors and font sizes represent how their
corresponding words are frequent. The Word cloud of the whole dataset is
depicted in figure 2. In concordance with figure 1, it is obvious that
the word *Love* is the most frequent. Words in blue such as *eyes* and
*heart* are the next frequent ones. Words in orange color are the next
in the list and light green color words are the least frequent.

Figure 2 gives a big picture of the whole dataset, but how if we depict
word cloud separately for renaissance and modern poetry. According to
figures 3 and 4, they are two different clouds both in color and word
variety - as we expected. It seems that in modern poetry, poets use a
diverse list of words, while in renaissance poets used words repeatedly.
And not surprisingly, **Love** has been the most frequent word in all
periods. It seems it is an unsolved problem of humanity! Or at least an
attractive topic for poets of all time.

![Word Cloud of Renaissance poetry (for those whose frequency is higher than 10)]({{ site.url }}{{ site.baseurl }}/images/Project/LSTM/unnamed-chunk-6-1.png)

![Word Cloud of Modern poetry (for those whose frequency is higher than 10)]({{ site.url }}{{ site.baseurl }}/images/Project/LSTM/unnamed-chunk-7-1.png)

*Bigram List*
-------------

The next tool is bigram[^fn4] or in simple language: *the analysis of a pair
of consecutive words*. Not only it can give us a holistic view of highly
consecutive words, but it also allows us to monitor how our
preprocessing works. This dataset is collected by web scrapping and it
is not a clean dataset. So, we expect to see some anomalies even after
preprocessing. For example, it can be possible that a poet dedicated
his/her poem to a person at the onset of the poet. In another scenario,
the publisher’s information can be included in the content. After
conducting the first bigram, I noticed that these anomalies exist. So, I
clean the data at a higher level by removing publishers’ information,
poet’s name, poems’ name.

A bigram is a specific case of n-gram when *n* = 2. In table 2, you can
observe the top 10 most frequently consecutive words. Again, it shows
that not only words such *Love*, *sweet*, and *day* are highly frequent
as a single form, but also their collocation with other words are
prominent.

| word1  | word2  |    n|
|:-------|:-------|----:|
| thou   | art    |   37|
| eccho  | ring   |   21|
| bridal | day    |   20|
| love   | doth   |   20|
| run    | softly |   20|
| softly | till   |   20|
| sweet  | thames |   20|
| thames | run    |   20|
| thou   | hast   |   17|
| wilt   | thou   |   16|

*Bigram Network*
----------------

Another common visualization tool is to convert the bigram into a
network. By doing this, we understand better how words are connected.
Bigram network is depicted in figure 5. This network is a graphical
format of the Markov chain, a simple and common method of text
predicting. It works by calculating the probability and finding the most
probable word. In the Markov chain, predicting the next word depends
only on the previous word. Since we are going to deploy LSTM-RNN, it
does not need to delve more into bigram network visualization.

![Bigram Network]({{ site.url }}{{ site.baseurl }}_post/image/unnamed-chunk-9-1.png)/

Model
=====

*RNN*
-----

Traditional neural networks cannot keep information. They are not good
at predicting events which are dependent on the previous event.
Recurrent neural networks (RNN) can address this problem by deploying
loops in itself. This loop acts like a memory to persist information.

![An unrolled simple RNN]({{ site.url }}{{ site.baseurl }}/images/Project/LSTM/unnamed-chunk-10-1.png)

If we unroll this loop, it turns out that they can be considered as a
consecutive normal neural networks. RNN is nothing just several serie
copies of the same network. Each copy can pass information to its
successor. It can be depicted as following figure:

![Unrolled RNN - a consecutively repeated NN]({{ site.url }}{{ site.baseurl }}/images/Project/LSTM/unnamed-chunk-11-1.png)

*LSTM*
------

RNN has the great ability to address problems in speech recognition,
image captioning, translation, etc. While RNN performs well from the
previous event, it fails to predict when there is a big gap between
events. Long Short Term Memory (LSTM)[^fn5] networks - a specific type in RNN
class- can handle this issue. It is capable of learning long term gaps.
In 1997, Hochreiter and Schmidhuber[^fn6] introduced LSTM [(pdf version can be
found here)](http://www.bioinf.jku.at/publications/older/2604.pdf).

> “*LSTM also solves complex, artificial long time lag tasks that have
> never been solved by previous recurrent network algorithms.*”
>
> <footer>
> — Hochreiter and Schmidhuber, 1997
> </footer>

``` r
#```{marginfigure}
#\break
#\break
#Figure 6-8 are extracted from the brilliant [Christopher Olah\'s #blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
```

Remembering events and maintaining information for long periods is in
LSTM’s blood. While RNN contains a chain of simple neural networks, LSTM
includes a chain of complex neural networks. For example, in Figure 8
they have these four interacting layers in each state. It has three
*sigmoid* and one *tanh* activation functions and
multiple pointwise operations. These elements give LSTM the possibility
to remember, forget and learn information through different states.

![An unrolled LSTM containing 4 internal layers]({{ site.url }}{{ site.baseurl }}/images/Project/LSTM/unnamed-chunk-13-1.png)

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png"
     alt="test"
     style="float: left; margin-right: 10px;" />

*Results*
---------

To fit the LSTM model on our data, we need to convert poems (which are
in character type) into a one-hot vector. This vectorization should
happen after cutting the text into specific chunks (with the length
equal to `maxlen`) and tokenizing these chunks. In the EDA section, we
tokenized the text into words since we wanted to analyze the words
separately or jointly. To apply LSTM, we have two choices for
tokenizing: by word or by character(letter). The former avoids sparsity
problem while the latter is more accurate. We choose the second option
for the sake of accuracy.

Previously, I ran the model by choosing epochs = 10 and batch size = 64.
At the time of first report, results were not satisfactory and even were
ludicrous. I had several problems in installing and building Keras
layers. Besides, building and fitting LSTM layers are very time taking
(at least for CPU resource that I have). Afterward, it returned some
simple but interesting things. It learned that there is a `space`
between words and predicted articles (`a` & `the`) before words. Finding
the best values for the `epoch` and `batch size` is not an easy task. It
is time taking and depends on the problem. After several trial and
error, I found that epochs = 8, batch size = 128 and maxlen = 30 return
more sensible results.

By training LSTM on modern age poems, it returns this results:

> **the moon and in a fire , the sent , is steet , and dead , and the
> rivers of the eye to the floors , and green heaven flowing dream , and
> the larring in the fools , and the shall the startly and the first
> wings , and the startly , where the can the veiets of the moon soul of
> the start of the door for the floor , and the look it a steet , and
> the rivers , the moon , and to trumper of the til**

And, it returns for archaic poems:

> **when the self in the rist , but in the still now with beard , that
> so those fair and beate , so thou have so , bells i should , that i
> can thee in the rest . but now the ground of many with state , i we
> leave , displaine , so for the face , that all , with them belly thee
> , that in the with a mind , or the soul of she was and her bears
> speed;**

As one can see, it is far away from human-written poems. Taking a deeper
look, we can find that it learned to use space character, punctuation,
recognize pronouns, and put them before verbs. There is still much space
to learn since it generates meaningless words. That being said, it is
interesting to me when I look at the “the … of …” phrases that LSTM can
generate. We should not forget this fact that we tokenized at the level
of characters. LSTM can generate these pieces of poems without knowing a
single word. Besides, it can mimic the format of the real world poem.
Sentences and verses are in different lengths!

Just for the sake of curiosity, I imported all Shakespeare’s poems from
the Gutenberg project[^fn7] by using gutenbergr package[^fn8]. Since the results
were not suitable to present, I exclude the results and the
corresponding code.

Conclusion
==========

In this present post, after introducing the dataset and the required
preprocessing steps, I explained the explanatory data analysis that shows
old-world poets had been reluctant to choose various words,
comparing to modern poets. We observed that the word `Love` has been
used in all periods. LSTM-RNN can detect some simple features such as article noun
and spaces between words. By changing maxlen, epoch, and batch sizes,
which control under fitting and over fitting, it can generate more
sophisticated results that are still far away from a well-written poem.


[^fn1]: https://www.kaggle.com/ultrajack/modern-renaissance-poetry

[^fn2]: https://www.poetryfoundation.org/

[^fn3]: The phoenix and the turtle, Escutcheon Press, Shakespeare William, 1991

[^fn4]: wikipedia, Bigram, https://en.wikipedia.org/wiki/Bigram

[^fn5]: Understanding LSTM Networks -- colah's blog, http://colah.github.io/posts/2015-08-Understanding-LSTMs/

[^fn6]: LONG SHORT-TERM MEMORY, journal of Neural Computation, Hochreiter and Schmidhuber, 1997, p.1738–1780

[^fn7]: Project Gutenberg, https://www.gutenberg.org/

[^fn8]: robinson, Download and Process Public Domain Works from Project Gutenberg [R package gutenbergr version 0.1.5]

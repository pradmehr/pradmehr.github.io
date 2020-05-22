---
layout: "posts"
title: "Generating Poetry by using LSTM-RNN"
subtitle: "Project Report - First Revision"
date: 2020-05-20
#output: tint::tintHtml
citation_package: natbib
bibliography: ref
link-citations: no
permalink: /Projects/

tags: [poems, poetry generation, LSTM, RNN, data science, text analysis]
header:
  image: "/images/Project/Shakespeare.jpg"
mathjax: "true"
---



# Introduction
\begin{marginfigure}
\includegraphics{_posts/2020-05-22-Poetry-Generation/figure-latex/unnamed-chunk-1-1} \end{marginfigure}

In this present report, I investigate poems generation by using Recurrent Neural Network (RNN). Since a poem is a collection of words in a sequence, I decide to deploy Long Short-Term Memory (LSTM) as it can capture this sequence of words. Each cell of LSTM can process data in a sequence way and use its hidden layers as a new input. After this short introduction, I explain the chosen dataset. Then, I do some preprocessing tasks to make the dataset prepared for neural network analysis. By conducting Explanatory Data Analysis (EDA), we will get more insight from the dataset. We will take a look at word frequency, word cloud, and n-gram. In the model section, we discuss LSTM and the results.

# Data

The dataset \cite{Kaggle} used in this project is a collection gathered from the Poetry Foundation webpage \cite{poetryFoundation}. It contains 573 poems from 67 poets, including `WILLIAM SHAKESPEARE` and `SIR PHILIP SIDNEY`. Columns and their description are as follows.


Table: Dataset description

featureNames   Description                                                                      
-------------  ---------------------------------------------------------------------------------
Author         Poets' Name                                                                      
content        content of Poems                                                                 
poem.name      Poems' Name                                                                      
age            binary value representing whether poems are "Classic" or belong to "Renaissance"
type           The theme of poems: "Love", "Nature", and "Mythology & Folklore"                 

For example, the first row contains the William Shakespeare's \cite{Shakespear} poem called `The Phoenix and the Turtle`. It starts with:

> "_Let the bird of loudest lay On the sole Arabian tree Herald sad and trumpet be, To whose sound chaste wings obey._"
>
> \hfill --- William Shakespeare

# Preprocessing

Before conducting any Machine Learning algorithm on the text, it is necessary to take some preprocessing steps. The abovementioned dataset contains both modern and classic poems. This implies that we should expect to see some archaic words (such as thee, thy) and some accented characters (e.g. ã, á) in the poems.

It could contain abbreviations form of words, e.g. `won't` instead of `will not`, `'d` instead of `would`, etc. Besides, we must remove additional spaces and punctuations. Last but not the least, this dataset is collected from [Poetry Foundation](https://www.poetryfoundation.org/) by using web scraping, which may result in some unwanted characters. We must deal with these type of characters.






# EDA
## *Word frequency table*
\begin{marginfigure}
\includegraphics{_posts/2020-05-22-Poetry-Generation/figure-latex/unnamed-chunk-4-1} \caption[Top 6 most frequent words in the poetry dataset]{Top 6 most frequent words in the poetry dataset}\label{fig:unnamed-chunk-4}
\end{marginfigure}

The first step we can take in Explanatory Data Analysis (EDA) is to display the word frequency. It can be useful to find out the high-frequent words used in poetry. It should not be surprising if we see some romantic and sweet words. After tokenizing the content of the poem into words, we can remove the words called **stop words** such as `and`, `that`, etc.

Since there are some classic poems in this dataset, we expect to observe many archaic pronouns (e.g. thee, thou, and thy). It can affect the relative frequency of other words. In order to correct this issue, these words are filtered from this EDA step. As we expected, the most frequent words are sweet and romantics such as `Love`, `Heart`, `sweet`, ... If we divide corresponding values by the total number of poems in the dataset (506), there would be at least one word `Love` in each poem (on average).

\begin{marginfigure}
\includegraphics{_posts/2020-05-22-Poetry-Generation/figure-latex/unnamed-chunk-5-1} \caption[Word Cloud (for those whose frequency is higher than 10)]{Word Cloud (for those whose frequency is higher than 10)}\label{fig:unnamed-chunk-5}
\end{marginfigure}

## *Word Cloud*

Word cloud is another way of displaying word frequency in a more fancy and schematic way. Different colors and font sizes represent how their corresponding words are frequent. The Word cloud of the whole dataset is depicted in figure 2. In concordance with figure 1, it is obvious that the word *Love* is the most frequent. Words in blue such as *eyes* and *heart* are the next frequent ones. Words in orange color are the next in the list and light green color words are the least frequent.

Figure 2 gives a big picture of the whole dataset, but how if we depict word cloud separately for renaissance and modern poetry. According to figures 3 and 4, they are two different clouds both in color and word variety - as we expected. It seems that in modern poetry, poets use a diverse list of words, while in renaissance poets used words repeatedly. And not surprisingly, **Love** has been the most frequent word in all periods. It seems it is an unsolved problem of humanity! Or at least an attractive topic for poets of all time.

\begin{marginfigure}
\includegraphics{_posts/2020-05-22-Poetry-Generation/figure-latex/unnamed-chunk-6-1} \caption[Word Cloud of Renaissance poetry (for those whose frequency is higher than 10)]{Word Cloud of Renaissance poetry (for those whose frequency is higher than 10)}\label{fig:unnamed-chunk-6}
\end{marginfigure}

\begin{marginfigure}
\includegraphics{_posts/2020-05-22-Poetry-Generation/figure-latex/unnamed-chunk-7-1} \caption[Word Cloud of Modern poetry (for those whose frequency is higher than 10)]{Word Cloud of Modern poetry (for those whose frequency is higher than 10)}\label{fig:unnamed-chunk-7}
\end{marginfigure}

## *Bigram List*

The next tool is bigram \cite{wikipedia} or in simple language: *the analysis of a pair of consecutive words*. Not only it can give us a holistic view of highly consecutive words, but it also allows us to monitor how our preprocessing works. This dataset is collected by web scrapping and it is not a clean dataset. So, we expect to see some anomalies even after preprocessing. For example, it can be possible that a poet dedicated his/her poem to a person at the onset of the poet. In another scenario, the publisher's information can be included in the content. After conducting the first bigram, I noticed that these anomalies exist. So, I clean the data at a higher level by removing publishers' information, poet's name, poems' name.

A bigram is a specific case of n-gram when $n=2$. In table 2, you can observe the top 10 most frequently consecutive words. Again, it shows that not only words such *Love*, *sweet*, and *day* are highly frequent as a single form, but also their collocation with other words are prominent.



Table: 10 most frequent bigram

word1    word2      n
-------  -------  ---
thou     art       37
eccho    ring      21
bridal   day       20
love     doth      20
run      softly    20
softly   till      20
sweet    thames    20
thames   run       20
thou     hast      17
wilt     thou      16

## *Bigram Network*

Another common visualization tool is to convert the bigram into a network. By doing this, we understand better how words are connected. Bigram network is depicted in figure 5. This network is a graphical format of the Markov chain, a simple and common method of text predicting. It works by calculating the probability and finding the most probable word. In the Markov chain, predicting the next word depends only on the previous word. Since we are going to deploy LSTM-RNN, it does not need to delve more into bigram network visualization.

\begin{marginfigure}
\includegraphics{_posts/2020-05-22-Poetry-Generation/figure-latex/unnamed-chunk-9-1} \caption[Bigram Network]{Bigram Network}\label{fig:unnamed-chunk-9}
\end{marginfigure}

# Model

## *RNN*

Traditional neural networks cannot keep information. They are not good at predicting events which are dependent on the previous event. Recurrent neural networks (RNN) can address this problem by deploying loops in itself. This loop acts like a memory to persist information.

\begin{marginfigure}
\includegraphics{_posts/2020-05-22-Poetry-Generation/figure-latex/unnamed-chunk-10-1} \caption[An unrolled simple RNN]{An unrolled simple RNN}\label{fig:unnamed-chunk-10}
\end{marginfigure}

If we unroll this loop, it turns out that they can be considered as a consecutive normal neural networks. RNN is nothing just several serie copies of the same network. Each copy can pass information to its successor. It can be depicted as following figure:

\begin{figure*}

{\centering \includegraphics{_posts/2020-05-22-Poetry-Generation/figure-latex/unnamed-chunk-11-1}

}

\caption[Unrolled RNN - a consecutively repeated NN]{Unrolled RNN - a consecutively repeated NN}\label{fig:unnamed-chunk-11}
\end{figure*}


## *LSTM*

RNN has the great ability to address problems in speech recognition, image captioning, translation, etc. While RNN performs well from the previous event, it fails to predict when there is a big gap between events. Long Short Term Memory (LSTM) \cite{LSTM} networks - a specific type in RNN class- can handle this issue. It is capable of learning long term gaps. In 1997, Hochreiter and Schmidhuber \cite{article1} introduced LSTM [(pdf version can be found here)](http://www.bioinf.jku.at/publications/older/2604.pdf).

> "_LSTM also solves complex, artificial long time lag tasks that have never been solved by previous recurrent network algorithms._"
>
> \hfill --- Hochreiter and Schmidhuber, 1997

\begin{marginfigure}
\break
\break

Figure 6-8 are extracted from the brilliant
\href{http://colah.github.io/posts/2015-08-Understanding-LSTMs/}{Christopher
Olah's blog}
\end{marginfigure}

Remembering events and maintaining information for long periods is in LSTM's blood. While RNN contains a chain of simple neural networks, LSTM includes a chain of complex neural networks. For example, in Figure 8 they have these four interacting layers in each state. It has three $sigmoid$ and one $tanh$ activation functions and multiple point-wise operations. These elements give LSTM the possibility to remember, forget and learn information through different states. \break

\begin{figure*}

{\centering \includegraphics{_posts/2020-05-22-Poetry-Generation/figure-latex/unnamed-chunk-13-1}

}

\caption[An unrolled LSTM containing 4 internal layers]{An unrolled LSTM containing 4 internal layers}\label{fig:unnamed-chunk-13}
\end{figure*}


## *Results*

To fit the LSTM model on our data, we need to convert poems (which are in character type) into a one-hot vector. This vectorization should happen after cutting the text into specific chunks (with the length equal to `maxlen`) and tokenizing these chunks. In the EDA section, we tokenized the text into words since we wanted to analyze the words separately or jointly. To apply LSTM, we have two choices for tokenizing: by word or by character(letter). The former avoids sparsity problem while the latter is more accurate. We choose the second option for the sake of accuracy.

Previously, I ran the model by choosing $\text{epochs} = 10$ and $\text{batch size} = 64$. At the time of first report, results were not satisfactory and even were ludicrous. I had several problems in installing and building Keras layers. Besides, building and fitting LSTM layers are very time taking (at least for CPU resource that I have). Afterward, it returned some simple but interesting things. It learned that there is a `space` between words and predicted articles (`a` & `the`) before words. Finding the best values for the `epoch` and `batch size` is not an easy task. It is time taking and depends on the problem. After several trial and error, I found that $\text{epochs} = 8$, $\text{batch size} = 128$ and $\text{maxlen} = 30$ return more sensible results.


By training LSTM on modern age poems, it returns this results:

>**the moon and in a fire , the sent , is steet , and dead , \break
and the rivers of the eye to the floors , \break
and green heaven flowing dream , \break
and the larring in the fools , \break
and the shall the startly \break
and the first wings , and the startly , \break
where the can the veiets of the moon soul of the start of the door for the floor ,\break
and the look it a steet , \break
and the rivers , the moon , \break
and to trumper of the til**\break


And, it returns for archaic poems:

>**when the self in the rist , \break
but in the still now with beard , \break
that so those fair and beate , \break
so thou have so , bells i should , \break
that i can thee in the rest . \break
but now the ground of many with state , \break
i we leave , displaine , \break
so for the face , that all , with them belly thee , \break
that in the with a mind , \break
or the soul of she was and her bears speed;** \break

As one can see, it is far away from human-written poems. Taking a deeper look, we can find that it learned to use space character, punctuation, recognize pronouns, and put them before verbs. There is still much space to learn since it generates meaningless words. That being said, it is interesting to me when I look at the "the ... of ..." phrases that LSTM can generate. We should not forget this fact that we tokenized at the level of characters. LSTM can generate these pieces of poems without knowing a single word. Besides, it can mimic the format of the real world poem. Sentences and verses are in different lengths!

Just for the sake of curiosity, I imported all Shakespeare's poems from the Gutenberg project \cite{gproj} by using gutenbergr \cite{robinson} package. Since the results were not suitable to present, I exclude the results and the corresponding code.

# Conclusion

In this post, after introducing the dataset and the required preprocessing steps, I explained the explanatory data analysis that shows old-world poets had been reluctant to choose various words, comparing to modern poets. We observed that the word `Love` has been used in all periods. By running LSTM for the first time, we noticed that the designed neural network did not perform well. By changing maxlen, epoch, and batch sizes, which control under fitting and over fitting, it can generate more sophisticated (close to human-written) results that are still far away from a well-written poem.

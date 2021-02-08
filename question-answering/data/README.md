### QA Leaderboards
* [NLP-progress QA](http://nlpprogress.com/english/question_answering.html)
* [paperswithcode QA](https://paperswithcode.com/task/question-answering)

------
### QA datasets
```
Remark: 
The datasets marked as bold are considered good candidates for open QA.
```

* QASent (Wang et al., 2007)
    - Drawbacks:
        + Question distribution bias: containing questions from human editors
        + Candidate selection bias: 
            1. output from systems participating TREC-QA
            2. sentences need to share non-stop words from the questions
        + Excluding questions with no correct answer
------
* WebQuestions ([Berant et al., 2013](https://www.aclweb.org/anthology/D13-1160.pdf))
    - Description:
        - 5k question-answer pairs from web queries
        - each question is accompanied by a FreeBase (Bollacker et al., 2008) entry
    - Drawbacks:
        - seems to take extra effort to convert freebase entries to text, maybe not worth it
        - quite old dataset, seems to be less popular
    - Websites:
        - [Github repo](https://github.com/brmson/dataset-factoid-webquestions)
    - SOTA:
        + T5-11B (Roberts et al., 2020) + SSM (Guu et al., 2020), EM=44.7
        + dense retrieval (Kaarpukhin et al., 2020), EM=42.4
        
------
* **WikiQA** ([Yang et al., 2015](https://www.aclweb.org/anthology/D15-1237.pdf))
    - Description:
        - 3k questions sampled from Bing query logs (factoid, WH-questions)
        - each question is accompanied with a set of candidate answers from summary paragraphs of Wikipedia pages
        - includes questions with no correct answers
        - adjustment to open domain QA: extract question-answer pairs by selecting the correct answer sentence from the candidates per question
    - Websites:
        + [Microsoft data](https://www.microsoft.com/en-us/download/details.aspx?id=52419)
        + [ACL archive](https://www.aclweb.org/anthology/D15-1237/)
    - Side notes: 
        1. knowledge base approach is still lacking (Freebase can answer 25% of search queries)
        2. free text approach in open domain QA is still important, high quality, reliable text e.g. Wikipedia, news articles

------
* NewsQA ([Trischler et al., 2016](https://arxiv.org/pdf/1611.09830.pdf))
    - Description:
        + 120k question-answer pairs from CNN articles
        + accompanying passages are relatively long (600 words)
    - Websites:
        + [Microsoft data](https://www.microsoft.com/en-us/research/project/newsqa-dataset/)
        
------
* **TriviaQA** ([Joshi et al., 2017](https://arxiv.org/pdf/1705.03551.pdf))
    - Description:
        - 650k (question, answer, evidence) triples
            + remark: each question has a standard answer as well as alternative/alias answers
            + dev: 200k QA pairs, 11k unique questions
            + train: 1500k QA pairs, 87k unique questions
            + test: test set answers are not shared
        - factoid questions from quiz-league website
        - evidence from Wikipedia articles, web domain
    - SOTA:
        + EM=80.86, F1=84.50 [Official leaderboard](https://competitions.codalab.org/competitions/17208#results) 
        + MemoReader (Back et al., 2018), 
        + T5-11B (Roberts et al., 2020) + SSM (Guu et al., 2020), EM=60.5
        + Dense retrieval (Karpukhin et al., 2020), EM=57.9
    - Websites:
        - [Official site](http://nlp.cs.washington.edu/triviaqa/)

------
* **Quasar-T** ([Dhingra et al., 2017](https://arxiv.org/pdf/1707.03904.pdf))
    - Description:
        - 43k factoid/trivia question-answer pairs
            + dev: 3k; test: 3k; train: 37k
        - each question is accompanied with both long and short contexts
    - Websites:
        - [Github repo](https://github.com/bdhingra/quasar)
    - SOTA:
        - EM=42.3, F1=49.6 [Leaderboard](https://paperswithcode.com/sota/open-domain-question-answering-on-quasar?p=gated-attention-readers-for-text)
    - Side notes:
        - The Quasar-S consists of cloze style questions over software entities, irrelevant for QA
        - The Quasar-T consists of trivia questions.
            
------    
* **SearchQA** ([Dunn et al., 2017](https://arxiv.org/pdf/1704.05179.pdf))
    - Description:
        - 140k Jeopardy! question-answer pairs
        - each pair has on average 50 snippets as evidence
        - dev: 21k;test: 43k; train:151k
    - Websites:
        - [Github repo](https://github.com/nyu-dl/dl4ir-searchQA)
    - SOTA:
        - EM=56.8, F1=63.6 [Leaderboard](https://paperswithcode.com/sota/open-domain-question-answering-on-searchqa?p=multi-granular-sequence-encoding-via-dilated)
    - Side notes:
        +  aims to emulate the search and retrieval process in
question answering applications
        + starting from J! archive question-answers pairs
        + as opposed to staring from articles, then generate question-answer pairs (like SQuAD)
        + evidence retrieved via Google

------
* NarativeQA ([Kocisky et al., 2017](https://arxiv.org/pdf/1712.07040.pdf))
    - Description:
        - 46k question-answer pairs with supporting snippets
        - (question, answer, short snippet/summary, long snippet)
        - 1,572 stories based on book, movie scripts, wikipedia articles
        - some human generated summaries
        - dev:3k; test:10k; train:32k
    - Websites:
        - [Github repo](https://github.com/deepmind/narrativeqa)
    - Side notes:
        - The Q,A pairs heavily depend on given snippets, not so well suited for open domain QA

------
* **SQuAD2.0** ([Rajpurkar et al., 2018](https://arxiv.org/pdf/1806.03822.pdf))
    - Description:
        - 100k (q,a) pairs in SQuAD1.1 + 50k unanswerable questions
        - Questions are created by people after given a Wikipedia article.
        - dev:20k; train:86k
    - Websites:
        - [official site](https://rajpurkar.github.io/SQuAD-explorer/)
    - SOTA:
        - Human performance, EM=86.8, F1=89.5
        - Ensemble methods achieves above-human performance, see leader board on the official site      

------
* BoolQ ([Clark et al., 2019](https://arxiv.org/pdf/1905.10044.pdf))
    - Description: Natural Yes/No questions    

------
* **Natural Questions** ([Kwiatkowski et al., 2019](https://www.mitpressjournals.org/doi/full/10.1162/tacl_a_00276))
    - Description:
        - questions from web queries
            + dev: 8k questions
        - each question is accompanied by a Wikipedia article that contains the answer
        - answer forms:
            1. unanswerable
            2. short answer (suitable for open QA)
            3. long answer (suitable for open QA, or passage retrieval)
            4. yes/no answer (suitable for open QA)
            5. multiple answers (suitable for open QA, partial match)
    - Websites:
        - [Google AI blog](https://ai.googleblog.com/2019/01/natural-questions-new-corpus-and.html)
        - [Official website](https://ai.google.com/research/NaturalQuestions)
    - SOTA:
        + dense vectors (Karpukhin et al., 2020), EM=41.5
        + T5.1.1 (pretrained on unlabeled data), EM=37.9
        + GPT3-175B, Accuracy=29.9 ? [reference](https://paperswithcode.com/sota/question-answering-on-natural-questions)


    


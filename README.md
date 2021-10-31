# SMM638 Network Analytics MTP

<img src="https://github.com/simoneSantoni/net-analysis-smm638/blob/master/midTermProject/images/communities.png" height="200">

The MTP revolves around the topic of network modularity, and, especially communities and core-peripehery structure (see [week 3](https://github.com/simoneSantoni/net-analysis-smm638/blob/e75c31ac89c976cd4dbd4d6175315d0272149384/lectureNotes/week3) of the module).

---

## **Description of the project**

Students — working in groups — are supposed:

1. to get access to a network dataset to put at the center of the MTP. There are two options to do that: reusing data available in digital repositories, e.g., [SNAP](http://snap.stanford.edu/data/index.html) or [Kaggle](https://www.kaggle.com/), or gathering data ex-novo (e.g., using a crawler 🕸🕷). Inter-personal, inter-organizational, technological, and physical networks are admitted types of networks for the MTP

2. to identify the relationhsip(s) included in the selected network dataset and to appreciate the economic, organizational, managerial, and/or social relevance of the relationship(s)

3. to use Python:

> 1. to analyze and plot the degree distribution of the network
> 2. to plot the network
> 3. to produce descriptive statistics about relevant node-level properties (e.g., betweeness centrality)
> 4. to asssess how and to what extent the network shows a modular structure (e.g., a community or core-periphery structure)
> 5. to assess the consequences of the modular structure for the individual nodes, the individual communities, and/or the functioning of the network. For example, one may want to correlate the network position of an individual (core Vs. periphery) with her attitudes, behavior, preferences.
> 6. to produce a companion document that — in plain English — describes and comments the outcome of the previous point 3.1 - 3.5.


## **Deliverables**

By Nov 15 (5:00 PM) ⏰ 💣, students are supposed to submit the following package:

- Python 🐍 code behind 3.1 - 3.5
- a companion document in .pdf format. No word limit applies; however, I really appreciate documents with a high added value per page (see the bestseller [How to Write Short](https://www.amazon.co.uk/How-Write-Short-Craft-Times-ebook/dp/B00FOQRPT4/ref=sr_1_1?dchild=1&keywords=how+to+write+short&qid=1634742402&sr=8-1))
- a 10-frame slideshow that summarizes the project (needed if you're selected to present your project to the class).

## **Assessment criteria**

1. appropriate use of notions and frameworks discussed in class.
2. effectiveness of the proposed answer or solution.
3. originality/creativity of the proposed answer or solution.
4. organization and clarity of submitted materials. 

All criteria carry out the equal weight in terms of marking.


## **Structure of the project**
1. Documents:
    - docs: for some detail information about the code and implementation
    - utils: for some usually used functions or methods
    - data: save the dataset in this file
2. Files:
    - analyze: for analyzing the data to the cleaned data
    - clean: for cleaning the data in the data
   
## **How to start**
1. Install the relative packages:
   ``` 
   pip install -r requirements.txt
   ```
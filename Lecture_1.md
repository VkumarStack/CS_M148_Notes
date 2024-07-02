# Lecture 1
## Types of Data
- **Vector / Tabular** data is stored in a tabular manner, with a row-column structure
- **Set** data is not necessarily stored in a tabular manner, but are rather treated as a set (multiple entries, but not necessarily tabularized)
- **Text** data refers to text
- **Sequence** data refers to sequence of characters (similar to text)
    - Related to this is **time series** data
- **Graph / Network** data is also common
- **Image** data is also prevalent (especially with deep learning)
## Functions with Data
- **Supervised** tasks allows for **classification** and **regression** of data to be performed
    - This takes inputs *features* and *label*
    - In regression, a label is predicted but it is *continuous*
    - In classification, a label is predicted but it is *discrete*
- **Unsupervised** tasks allows for intrinsic understandings of data to be learned via **clustering** and **representation learning**
    - This takes inputs of just *features*
    - In clustering, data is grouped to form new categories, in a manner such that similarities between datas *in a cluster* is maximized whereas similarity between datas *between clusters* is minimized
## Life Cycle of Data Science
- Process:
    - Ask an interesting question
    - Collect the data
    - Explore and preprocess the data
    - Model the data
    - Communicate and visualize the results
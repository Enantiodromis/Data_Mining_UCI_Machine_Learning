# Data Mining using UCI Machine Learning Repository
## 1. Classification Task:

The [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult) is used to predict if the income of an individual exceeds 50K per year based on 14 attributes.

__Attributes taken into consideration:__

Attribute | Description
------|----------
age| age group
workclass| type of employment
education| level of education reached
education-num| number of education years
martital-status| type of martial status
occupation| occupation domain
relationship| type of relationship involved
race| social category
sex| male or female
capital-gain| class of capital gains
capital-loss| class of capital losses
hours-per-week| category of working hours
native-country| country of birth
</br>
- A table in the report is created with the following information about the adult data set:
    - Number of instances
    - Number of missing values
    - Fraction of missing values over all attribute values
    - Number of instances with missing values
    - Fraction of instances with missing values over all instances
</br>
- All attributes are converted into nominal using _Sikit-learn LabelEncoder_. Then, the set of all possible discrete values are printed for each attribute.
</br>

- Ignoring any instances with missing value(s), _Sickit-learn_ is used to build decision trees for classifying an individual to either a _<= 50K_ and _> 50K_ category. The error rate of the resulting tree is also computed. 
</br>
- In order to investigate methods to handle missing values. Two datasets are created. (i) All instances with at least one missing value (ii) An equal number of randomly selected instances without missing values. Two decision trees are trained with the two data sets, the error rates are compared and comments on the data outlined in the report. 
</br>
## 2. Clustering Task:
The wholesale customers data from the aforementioned _UCI Machine Learning Repository_ is used to identify similar groups of customers based on 6 attributes.

__Attributes taken into consideration:__

Attribute | Description
------|----------
FRESH| Annual expenses on fresh products
MILK| Annual expenses on milk products
GROCERY| Annual expenses grocery products
FROZEN| Annual expenses frozen products
DETERGENTS| Annual expenses detergent products
DELICATESSEN| Annual expenses delicatessen products

</br>

- A table in the report with the mean and range for each attribute is created.

- K-Means with k=3 is run, a scatterplot for each pair of attributes using Pyplot. For better presentation, different clusters appear with different colors.

- K-Means with all values in set {3,5,10} is run. A table with the between cluster distance, within cluster distance and ratio between and withing cluster distance is obtained for each value of k.

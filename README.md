### Santander Product Recommendation
The goal of this competition was to predict which new banking products customers were most likely to buy. 
#### data description 
The training data consists of nearly 1 million users with monthly historical user and product data between January 2015 and May 2016. User data consists of 24 predictors including the age and income of the users. Product data consists of boolean flags for all 24 products and indicates whether the user owned the product in the respective months. The goal is to predict which new products the 929,615 test users are most likely to buy in June 2016. A product is considered new if it is owned in June 2016 but not in May 2016. The next plot shows that most users in the test data set were already present in the first month of the train data and that a relatively large share of test users contains the first training information in July 2015. Nearly all test users contain monthly data between their first appearance in the train data and the end of the training period (May 2016).
#### data analysis
The popularity of products evolves over time but there are also yearly seasonal causes that impact the new product counts. June 2015 (left dotted line in the plot above) is especially interesting since it contains a quite different new product distribution (Cco_fin and Reca_fin in particular) compared to the other months, probably because June marks the end of the tax year in Spain. It will turn out later on in the analysis that the June 2015 new product information is by far the best indicator of new products in June 2016, especially because of divergent behavior in the tax product (Reca_fin) and the checking account (cco_fin). The most popular forum post suggests to restrict the modeling effort to new product records in June 2015 to predict June 2016. A crucial insight which changed the landscape of the competition after it was made public by one of the top competitors.

The interactive application also reveals that there is an important relation between the new product probability and the products that were owned in the previous month. The Nomina product is an extreme case: it is only bought if Nom_pens was owned in the previous month or if it is bought together with Nom_pens in the same month. Another interesting insight of the interactive application relates to the products that are frequently bought together. Cno_fin is frequently bought together with Nomina and Nom_pens. Most other new product purchases seem fairly independent. A final application of the interactive application shows the distribution of the continuous and categorical user predictors for users who bought new products in a specific month.

#### Feature engineering
The feature engineering files are calculated using different lags. The models trained on June 2015 for example are trained on features based on all 24 user data predictors up till and including June 2015 and product information before June 2015. This approach mimics the test data which also contains user data for June 2016. The test features were generated using the most recent months and were based on lag data in order to have similar feature interpretations. Consequently, the model trained on June 2015 which uses 5 lag months is evaluated on the test features calculated on only the lag data starting in January 2016.

Features were added in several iterations. I added similar features based on those that had a strong predictive value in the base models. Most of the valuable features are present in the lag information of previously owned products. I added lagged features of all products at month lags 1 to 6 and 12 and included features of the number of months since the (second) last positive (new product) and negative (dropped product) flanks. The counts of the positive and negative flanks during the entire lag period were also added as features for all products as well as the number of positive/negative flanks for the combination of all products in lags 1 to 6 and 12. An interesting observation was the fact that the income (renta) was non-unique for about 30% of the user base where most duplicates occurred in pairs and groups of size < 10. I assumed that these represented people from the same household and that this information could result in valuable features since people in the same household might show related patterns. Sadly, all the family related features I tried added little value.

I added a boolean flag for users that had data in May 2015 and June 2015 as users that were added after July 2015 showed different purchasing behavior. These features however added little value since the base models were already able to capture this different behavior using the other features. The raw data was always used in its raw form except for the income feature. Here I used the median for the province if it was missing and I also added a flag to indicate that the value was imputed. Categorical features were mapped to numeric features using an intuitive manual reordering for ordinal data and a dummy ordering for nominal data.

Other features were added to incorporate dynamic information in the lag period of the 24 user data predictors. Many of these predictors are however static and added limited value to the overall performance. It would be great to study the impact of changing income on the product purchasing behavior but that was not possible given the static income values in the given data set. I did not include interactions between the most important features and wish that I had after reading the approaches of several of the other top competitors.

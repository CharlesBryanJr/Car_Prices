# 🚙 Car Prices Prediction 

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

## Introduction

The U.S. car manufacturing industry is thriving, constantly introducing vehicles equipped with advanced features and technology. These innovations, driven by a commitment to enhance the driving experience, ensure that motorists enjoy superior comfort and performance in their journeys.

<img src = "/images/cars.png" width = 350 height = 200/>

## Business Constraints / Key Performance Metrics (KPIs)

A key challenge in automotive sales is setting the optimal price for cars, a factor crucial in driving demand and sales volume. The pricing decision is influenced by multiple variables, including mileage, car size, and the manufacturer, among others. The complexity and multitude of these influencing factors make it a daunting task for human analysis to pinpoint the most effective pricing strategy. An effective solution to this challenge lies in harnessing machine learning and data science. These technologies enable the extraction of deep insights from vast datasets and facilitate accurate predictions. By applying these advanced analytical tools, companies can devise pricing strategies that not only resonate with market trends and consumer preferences but also bolster profitability.

## Machine Learning and Deep Learning

* Machine Learning and Deep Learning Advancements: Significant growth in machine learning and deep learning over the past decade, impacting various industries, including automotive.
* Predictive Pricing Models: Potential to predict car prices using key features like horsepower, make, and others, using machine learning techniques. 
* Optimal Pricing Strategies: Companies can utilize machine learning to determine the ideal price based on factors such as make, horsepower, and mileage, enhancing profitability. 
* Maximizing Profits Through Precise Pricing: Machine learning models enable setting the right price for new cars, ensuring maximum profit for manufacturers. 
* Car Prices Prediction Project: Focused on developing machine learning models to assist in accurate pricing of new cars, leading to cost savings for manufacturers.
* Data Analysis and Visualization: Initial steps include visualizing the data to understand critical information for accurate car price predictions.
* Utilizing Regression Techniques: Employing various regression methods to determine the average price of cars based on the dataset. 
* Comprehensive Pricing Insights: Aiming for a thorough understanding of pricing dynamics to aid in informed decision-making in the automotive sector.

<h2> Data Source</h2>

We will be analyzing a substantial dataset comprising approximately 10,000 data points. This dataset will be strategically split into training and test sets to ensure the efficacy of our predictive models. Our focus extends to various popular car models, integral to our daily lives, offering a practical perspective on their sales trends and average pricing structures. For a more in-depth understanding and to appreciate the scope of our analysis, you are encouraged to explore the dataset we utilized for car price prediction.

__Source:__ https://www.kaggle.com/CooperUnion/cardataset

## Metrics

Predicting car prices is an continuous machine learning problem, particularly in the realm of regression analysis. To effectively address this, we have employed several key metrics specifically tailored for regression problems. These metrics have been crucial in our process of predicting car prices, ensuring accuracy and reliability in our models. Here are the metrics we used:


* [__Mean Squared Error (MSE)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
* [__Mean Absolute Error (MAE)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)

## Exploratory Data Analysis (EDA)

In this section of the project, we conducted a thorough exploration of the data to uncover patterns, trends, and intriguing insights. Our analysis yielded several notable observations:

* Dominance of Certain Manufacturers: A significant portion of the cars in the dataset were from Chevrolet, with Ford being the second most common manufacturer.
* Peak Manufacturing Year: The year 2015 emerged as the year with the highest number of cars manufactured, according to the data.
* Data Incompleteness: The 'Market Category' feature had many missing values, along with some missing data in 'Engine HP' and 'Engine Cylinders'.
* Fluctuating Average Prices: The data revealed that the average car prices peaked in 2014 and were at their lowest in 1990.
* High-End Manufacturer Pricing: Cars from Bugatti were exceptionally expensive compared to other manufacturers.  
* Bugatti’s High Horsepower: Bugatti also stood out for its extremely high engine horsepower, as indicated by the graphical analysis in the notebook.
* Correlation Insights: There was a noticeable negative correlation between 'City Mileage' and features like 'Engine Cylinders' and 'Engine HP'. This suggests that cars with higher city mileage tend to have fewer cylinders and lower horsepower.

<h2> Visualizations</h2>

Upon examining the dataset, we observed several key categories including Vehicle Size, City MPG (miles per gallon), Popularity, and Transmission Types. Additionally, there are other features within the dataset that warrant further exploration through visual analysis. These visualizations will help in gaining a deeper understanding of the interplay between these diverse aspects and their impact on vehicle performance and consumer preferences.

<img src = "/images/data_head.png"/>

We will dive into a series of visualizations to gain insights into the dataset. By examining the distribution of car manufacturers represented, it becomes apparent that Chevrolet has the largest presence, with Ford following closely behind. These visual analyses are crucial in understanding the breadth and diversity of car brands within our data.

<img src = "/images/total_cars_by_manufacturer.png"/>

The visualizations clearly show a rising trend in car demand over the years, reflecting the growing automotive market. Additionally, this increase in recent data points suggests that our machine learning models are likely to be more accurate and effective when predicting prices for newer car models, due to the richer dataset available for these vehicles.

<img src = "/images/cars_manufactured_by_year.png"/>

The plot reveals that a majority of the cars in our dataset are automatic, outnumbering their manual counterparts. Additionally, there's a small segment of vehicles categorized as 'unknown', indicating some variability in transmission types beyond the standard automatic and manual classifications.

<img src = "/images/transmission_type.png"/>

The dataset predominantly features cars with regular unleaded engines, with a smaller yet notable presence of electric vehicles. Given their recent emergence, electric cars are less numerous. Additionally, a variety of other fuel types, including diesel, are represented in smaller quantities.

<img src = "/images/fuel_type.png"/>

A significant number of cars in the dataset are compact-sized, but there is also a fairly even distribution among different vehicle size categories.

<img src = "/images/vehicle_size.png"/>

Missingno plots highlight gaps in our dataset, particularly in 'Market Category' and 'Engine HP'. This helps us choose whether to impute or exclude these values to maintain our machine learning models' accuracy in price prediction.

<img src = "/images/missingno.png"/>

The analysis reveals that car prices peaked in 2014, marking it as the year with the highest average prices in our dataset. This is closely followed by 2012. Conversely, 1990 stands out as the year with the lowest average car prices. These trends highlight the year of manufacture as a significant factor in determining car prices, as evidenced by the plot below.

<img src = "/images/average_car_price_by_year.png"/>

The data shows that cars with both automatic and manual transmissions have the highest average prices, followed by automatic-only models. 'Unknown' transmission types in the data should be clarified or removed for more accurate analysis.

<img src = "/images/average_prices_by_transmission_type.png"/>

Luxury brands like Bugatti and Maybach top the average car prices, highlighting their market prestige and exclusivity. Other notable high-end brands include Rolls-Royce, Lamborghini, Bentley, McLaren, and Ferrari. The plot gives a clear view of the pricing trends among various car manufacturers.

<img src = "/images/average_msrp.png"/>

The bar chart showcases the horsepower across different car manufacturers, with Bugatti leading the pack. Despite its premium pricing, Bugatti stands out for its exceptional horsepower, followed by high-performance brands like McLaren and Maybach. These cars are renowned for their top-tier performance capabilities, as reflected in their impressive horsepower ratings.

<img src = "/images/engine_horsepower.png"/>

Popular car brands like Ford and BMW, along with Audi, Ferrari, Honda, and Nissan, lead in terms of consumer demand. Analyzing the influence of brand popularity on car pricing across these diverse manufacturers offers intriguing insights.

<img src = "/images/driven_wheel_configuration.png"/>

Segmenting the dataset by drivetrain reveals that 'all-wheel drive' (AWD) vehicles have the highest average prices, followed by 'rear-wheel drive' (RWD), with 'front-wheel drive' (FWD) being more affordable. This pricing reflects the performance capabilities and market preferences for each drivetrain type.

<img src = "/images/vehicle_size.png"/>

The left-hand plots display highway MPG for all dataset cars, revealing several outliers. Consequently, these outliers are removed, as seen in the right-hand plots. This refinement helps the ML model better learn and represent these features for accurate car price predictions.

<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/Highway%20mgp.png"/><img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/Highway%20mpg%20outliers%20removed.png"/>

<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/Cith%20mpg.png"/><img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/City%20mpg%20outliers%20removed.png"/>

The plot mirrors real-life trends, showing higher highway MPG compared to city MPG for most cars, evident in the differing spreads and means. While outliers exist, we'll retain some to gauge model performance.

<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/City%20Vs%20Highway%20boxplot.png"/>

The heatmap of the correlation matrix reveals a strong relationship between engine horsepower and engine cylinders, as well as a notable correlation between city mpg and highway mpg.

<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/Heatmap%20correlation.png"/>

### Model Performance

We will now focus our attention on the performance of __various models__ on the test data. Scatterplots can help us determine how much of a spread our predictions are from the actual values. Let us go over the performance of many ML models used in our problem of car price prediction. 

[__Linear Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) - After looking at the linear regression plot, it looks like the model is performing quite well. Scatterplots between the predictions and the actual test outputs closely resemble each other. If there are low latency requirements for a deployment setup, linear regression could be used. 

<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/LR%20Plot.png"/>

[__Support Vector Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) - Support vector regression (SVR) can be computational. In addition, the results below indicate that the predictions are far off from the actual car prices. Therefore, alternate models can be explored. 

<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/SVR%20Plot.png"/>

[__K Nearest Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) - K-Nearest Regressor is doing a good job in predicting the car prices as highlighted in the plot below. There is less spread between the test output labels and the predictions generated by the model. Therefore, there are higher chances that the model gives a low mean absolute error and mean squared error. 

<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/K%20Neighbors%20Regressor.png"/>

[__PLS Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html) - This model does a good job overall when it comes to predicting car prices. However, it fails to compare trends and patterns for higher-priced cars well. This is evident due to the fact that there is a lot of spread among higher car price values as shown in the plot. K-Nearest Regressor, on the other hand, also does predictions accurately on higher priced cars. 
 
<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/PLS%20Regressor%20plot.png"/>

[__Decision Tree Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) - Based on all the models tested, the decision tree regressor was performing the best. As shown below, there is a lot of overlap between the predictions and the actual test values. 

<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/Decision%20Tree%20Plot.png"/>

[__Gradient Boosting Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) - The performance of gradient boosted decision regressor is plotted and it shows that it is quite similar to the decision tree. At prices that are extremely high, the model fails to capture the trend in the data. It does a good job overall. 

<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/GBDT%20Plot.png"/>

[__MLP Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) - It does a good job when it comes to predicting car prices. However, there are better models earlier that we can choose as their performance was better than MLP Regressor in this scenario. 

<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/MLP%20Regressor%20plot.png"/>

[__Final Mean Absolute Error (MAE)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html) - After performing feature engineering and hyperparameter tuning the models, the best model that gave the least mean absolute error (lower is better) was Decision Tree Regressor. Other models such as Support Vector Regressors took a long time to train along with giving less optimum results. Along with good performance, Decision Tree Regressors are highly interpretable and they give a good understanding of how a model gave predictions and which feature was the most important for it to decide car prices. 

<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/Final%20MAE.png"/>

[__Final Mean Squared Error (MSE)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) - The performance of the Decision Tree Regressor was also the highest when using mean squared error as the output metric. While the Gradient Boosted Regressor came close to the performance of a Decision Tree Regressor, the latter is highly interpretable and easier to deploy in real time. Therefore, we can choose this model for deployment as it is performing consistently across a large variety of metrics. 

<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/Final%20MSE.png"/>

## Machine Learning Models 

We have to be using various machine learning models to see which model reduces the __mean absolute error (MAE)__ or __mean squared error (MSE)__ on the cross-validation data respectively. Below are the various machine learning models used. 

| __Machine Learning Models__| __Mean Absolute Error__| __Mean Squared Error__|
| :-:| :-:| :-:|
| [__1. Linear Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)| 6737| 364527989|
| [__2. Support Vector Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)|	22525|	2653742304|
|	[__3. K Nearest Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)|	4668|	198923161|
|	[__4. PLS Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html)|	6732|	364661296|
|	[__5. Decision Tree Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)|	__3327__|	__135789622__|
|	[__6. Gradient Boosting Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)|	4432|	175275369|
|	[__7. MLP Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)|	6467|	250908327|

## Outcomes

* The best model that was performing the best in terms of __mean absolute error (MAE)__ and __mean squared error (MSE)__ was __Decision Tree Regressor__.
* __Exploratory Data Analysis (EDA)__ revealed that __Bugatti manufacturer's prices__ were significantly higher than the other manufacturers. 
* __Scatterplots__ between the __actual prices__ and __predicted prices__ were almost __linear__, especially for the __Decision Tree Regressor__ model.

## Future Scope

* It would be great if the best __machine learning model (Decision Tree Regressor)__ is integrated in the live application where a __seller__ is able to add details such as the __manufacturer__, __year of manufacture__ and __engine cylinders__ to determine the best price for cars that generates __large profit margins__ for the seller. 
* Adding __additional data points__ and __features__ could help in also better determining the best prices for the latest cars as well. 

## 👉 Directions to download the repository and run the notebook 

This is for the Washington Bike Demand Prediction repository. But the same steps could be followed for this repository. 

1. You'll have to download and install Git which could be used for cloning the repositories that are present. The link to download Git is https://git-scm.com/downloads.
 
&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(14).png" width = "600"/>
 
2. Once "Git" is downloaded and installed, you'll have to right-click on the location where you would like to download this repository. I would like to store it in the "Git Folder" location. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(15).png" width = "600" />

3. If you have successfully installed Git, you'll get an option called "Gitbash Here" when you right-click on a particular location. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(16).png" width = "600" />


4. Once the Gitbash terminal opens, you'll need to write "Git clone" and then paste the link to the repository.
 
&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(18).png" width = "600" />

5. The link to the repository can be found when you click on "Code" (Green button) and then, there would be an HTML link just below. Therefore, the command to download a particular repository should be "Git clone HTML" where the HTML is replaced by the link to this repository. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(17).png" width = "600" />

6. After successfully downloading the repository, there should be a folder with the name of the repository as can be seen below.

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(19).png" width = "600" />

7. Once the repository is downloaded, go to the start button and search for "Anaconda Prompt" if you have anaconda installed. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(20).png" width = "600" />

8. Later, open the Jupyter notebook by writing "Jupyter notebook" in the Anaconda prompt. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(21).png" width = "600" />

9. Now the following would open with a list of directories. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(22).png" width = "600" />

10. Search for the location where you have downloaded the repository. Be sure to open that folder. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(12).png" width = "600" />

11. You might now run the .ipynb files present in the repository to open the notebook and the python code present in it. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(13).png" width = "600" />

That's it, you should be able to read the code now. Thanks.
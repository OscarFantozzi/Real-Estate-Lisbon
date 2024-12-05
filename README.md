
# Real Estate Analysis in Lisbon

### 🚀 **About the Project**

This project focuses on extracting and analyzing real estate data in Lisbon using the Idealista API. The data provides insights into rental properties, which are then processed and used to train predictive models for pricing and demand estimation.

The ultimate goal is to help users (e.g., real estate agencies, property owners) understand market trends and make informed decisions about property pricing and investments.

---

### ⚙️ **Features**

The dataset extracted contains the following key features:

- **Property Information:**
  - `propertyCode`: Unique identifier for each property.
  - `price`: Rental price of the property (in euros).
  - `size`: Size of the property in square meters.
  - `rooms`: Number of rooms.
  - `bathrooms`: Number of bathrooms.
  - `propertyType`: Type of property (e.g., apartment, house).

- **Location Information:**
  - `latitude` and `longitude`: Geographic coordinates of the property.
  - `address`, `district`, `municipality`: Location details.

- **Additional Features:**
  - `priceByArea`: Price per square meter.
  - `hasLift`: Indicates if the property has an elevator.
  - `parkingSpace`: Availability of parking spaces.
  - `has360`: Indicates if a 360° virtual tour is available.
  - `datetime_scrapy`: Date and time of data extraction.

---

### 🛠️ **Technologies Used**

- **Programming Language:**
  - Python

- **Key Libraries:**
  - `pandas` for data manipulation.
  - `requests` for API integration.
  - `sqlalchemy` for database management.
  - `scikit-learn` for machine learning models.
  - `matplotlib` and `seaborn` for data visualization.

---

### 📄 **Workflow**

1. **Data Extraction:**
   - The data is extracted from the Idealista API using `requests` and authenticated via OAuth tokens.
   - The API provides property details, which are saved in both SQLite and Excel formats for further analysis.

2. **Data Processing:**
   - Features are cleaned, and missing values are handled.
   - Numerical variables are scaled, and categorical variables are encoded.

3. **Modeling:**
   - Models are trained to predict rental prices (`price`) based on property attributes.
   - Regression models like Linear Regression and Random Forest are evaluated.

4. **Evaluation:**
   - Models are evaluated using metrics such as RMSE (Root Mean Square Error) and R².

---

### 📊 **Conclusions**

- **Key Insights:**
  - Properties with parking spaces and elevators tend to have higher rental prices.
  - Central locations (closer to Lisbon's downtown) command a premium price.
  - Features like `priceByArea` and `rooms` are strong predictors of rental prices.

- **Business Implications:**
  - Property owners can use these insights to determine competitive rental prices.
  - Real estate agencies can target specific property types for high-demand areas.

---

### 🔧 **Next Steps**

1. **Expand Data Collection:**
   - Include more features such as historical price trends and proximity to public transport.

2. **Improve Model Performance:**
   - Use advanced machine learning algorithms like Gradient Boosting and XGBoost.
   - Perform hyperparameter tuning for better model accuracy.

3. **Develop a Dashboard:**
   - Build an interactive dashboard using Power BI or Tableau to visualize market trends and predictions.

4. **Real-Time Predictions:**
   - Deploy the model via an API to provide real-time rental price predictions.

---

### 📬 **Contact**

For questions, suggestions, or collaborations, feel free to reach out:

- **GitHub:** [OscarFantozzi](https://github.com/OscarFantozzi)
- **LinkedIn:** [Oscar Fantozzi](https://linkedin.com/in/oscarfantozzi)

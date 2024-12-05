
# Real Estate Price Prediction in Lisbon

### 🚀 **About the Project**

This project leverages the Idealista API to collect data on rental properties in Lisbon. The data is processed, cleaned, and used to train machine learning models that predict rental prices. The goal is to create a robust framework for understanding rental market trends and providing actionable insights for real estate professionals.

---

### ⚙️ **Features**

- **Data Extraction:**
  - Uses the Idealista API to fetch real estate data (location, size, price, etc.).
  - Saves data in SQLite and Excel formats for flexibility.

- **Data Preprocessing:**
  - Encoding and cleaning data for machine learning compatibility.
  - Handles missing values and outliers effectively.

- **Machine Learning Models:**
  - Trains models on historical data to predict rental prices.
  - Evaluates model performance using metrics like R² and MAE.

- **Practical Applications:**
  - Assists real estate professionals in pricing properties.
  - Provides insights into market trends and demand patterns.

---

### 🛠️ **Technologies Used**

- **Programming Language:**
  - Python (Jupyter Notebooks for exploration and documentation)

- **Key Libraries:**
  - `pandas`, `numpy` for data manipulation.
  - `scikit-learn` for model training and evaluation.
  - `sqlite3` for database management.
  - `matplotlib`, `seaborn` for data visualization.
  - `requests` for API calls.

- **API Integration:**
  - Idealista API for property data.

---

### 🧩 **How to Run**

#### Prerequisites
- Python 3.8 or higher.
- Access to the Idealista API (with valid credentials).
- Required dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/Real-Estate-Lisbon.git
   cd Real-Estate-Lisbon
   ```

2. Update the API credentials in `extracted_data.py`:
   ```python
   API_KEY = "your_api_key"
   API_SECRET = "your_api_secret"
   ```

3. Run the data extraction script:
   ```bash
   python scripts/extracted_data.py
   ```

4. Open the Jupyter Notebook:
   ```bash
   jupyter lab notebooks/prices_rent_idealista-V1.ipynb
   ```

---

### 📄 **Workflow**

1. **Data Extraction:**
   - The `extracted_data.py` script retrieves property data from the Idealista API.
   - Saves raw data in the `data/raw/` folder and a SQLite database.

2. **Data Cleaning:**
   - Encodes categorical variables (e.g., property type, location).
   - Normalizes numerical features like area and price.

3. **Model Training:**
   - Trains regression models (e.g., linear regression, random forest) to predict prices.
   - Splits data into training and testing sets for evaluation.

4. **Results and Evaluation:**
   - Visualizes performance metrics and feature importance.
   - Compares different models to identify the best approach.

---

### 📊 **Example Output**

Example of raw data extracted from the Idealista API:
| Location  | Area (m²) | Bedrooms | Price (€) | Property Type |
|-----------|-----------|----------|-----------|---------------|
| Lisbon    | 80        | 2        | 1,200     | Apartment     |
| Alfama    | 50        | 1        | 800       | Studio        |

Model evaluation results:
- **R² Score:** 0.87
- **Mean Absolute Error (MAE):** €120

---

### 🔧 **Details About Data Extraction**

- The script authenticates with the Idealista API using an API key and secret.
- Retrieves property listings based on parameters such as location, property type, and operation type.
- Stores the results in two formats:
  - Excel files in the `data/raw/` folder.
  - SQLite database (`database/bd_houses_rent_api.sqlite`) for structured storage.

#### **How to Configure API Credentials**
- Create a `.env` file in the root directory with the following content:
  ```env
  IDEALISTA_API_KEY=your_api_key
  IDEALISTA_API_SECRET=your_api_secret
  ```

- Make sure to add `.env` to your `.gitignore` to protect sensitive information:
  ```gitignore
  .env
  ```

---

### 🔧 **Potential Improvements**

- Update the dataset with fresh API calls to ensure accuracy.
- Experiment with additional machine learning algorithms.
- Develop an interactive dashboard for data visualization.
- Optimize hyperparameters for better model performance.

---

### 📬 **Contact**

Feel free to reach out if you have any questions or suggestions:

- **GitHub:** [OscarFantozzi](https://github.com/OscarFantozzi)
- **LinkedIn:** [Oscar Fantozzi](https://linkedin.com/in/oscarfantozzi)

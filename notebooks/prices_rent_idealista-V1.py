{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "629dbadb-ae43-41d2-8a3a-924dba70cdc4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 0.0. IMPORTS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b779377b-97e6-46dd-bd89-398be47a3272",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:13:44.415982Z",
     "start_time": "2024-03-01T22:13:37.911164Z"
    }
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import inflection\n",
    "import math\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import pathlib\n",
    "import seaborn as sns\n",
    "import ast\n",
    "import lightgbm as lgb\n",
    "import xgboost as xgb\n",
    "from scipy.stats       import chi2_contingency\n",
    "from category_encoders import TargetEncoder\n",
    "from sklearn          import preprocessing as pp\n",
    "import pickle\n",
    "import os\n",
    "\n",
    "from xgboost                  import XGBRegressor\n",
    "from lightgbm                 import LGBMRegressor\n",
    "from geopy.geocoders          import Nominatim\n",
    "from sqlalchemy               import create_engine, text\n",
    "from matplotlib               import pyplot as plt\n",
    "from sklearn.preprocessing    import RobustScaler, MinMaxScaler, LabelEncoder\n",
    "from boruta                   import BorutaPy\n",
    "from sklearn.ensemble         import RandomForestRegressor\n",
    "from sklearn.model_selection  import train_test_split\n",
    "from sklearn.linear_model     import LinearRegression,  Lasso\n",
    "from sklearn.metrics          import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,r2_score\n",
    "from sklearn.model_selection  import cross_val_score\n",
    "from sklearn.preprocessing    import LabelEncoder\n",
    "from sklearn.model_selection  import KFold\n",
    "import pandas as pd\n",
    "import inflection\n",
    "import math\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import pathlib\n",
    "import seaborn as sns\n",
    "import ast\n",
    "import lightgbm as lgb\n",
    "import xgboost as xgb\n",
    "from scipy.stats       import chi2_contingency\n",
    "from category_encoders import TargetEncoder\n",
    "from sklearn          import preprocessing as pp\n",
    "import pickle\n",
    "import os\n",
    "\n",
    "from xgboost                  import XGBRegressor\n",
    "from lightgbm                 import LGBMRegressor\n",
    "from geopy.geocoders          import Nominatim\n",
    "from sqlalchemy               import create_engine, text\n",
    "from matplotlib               import pyplot as plt\n",
    "from sklearn.preprocessing    import RobustScaler, MinMaxScaler, LabelEncoder\n",
    "from boruta                   import BorutaPy\n",
    "from sklearn.ensemble         import RandomForestRegressor\n",
    "from sklearn.model_selection  import train_test_split\n",
    "from sklearn.linear_model     import LinearRegression,  Lasso\n",
    "from sklearn.metrics          import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,r2_score\n",
    "from sklearn.model_selection  import cross_val_score\n",
    "from sklearn.preprocessing    import LabelEncoder\n",
    "from sklearn.model_selection  import KFold\n",
    "import pandas as pd\n",
    "import inflection\n",
    "import math\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import pathlib\n",
    "import seaborn as sns\n",
    "import ast\n",
    "import lightgbm as lgb\n",
    "import xgboost as xgb\n",
    "from scipy.stats       import chi2_contingency\n",
    "from category_encoders import TargetEncoder\n",
    "from sklearn          import preprocessing as pp\n",
    "import pickle\n",
    "import os\n",
    "\n",
    "from xgboost                  import XGBRegressor\n",
    "from lightgbm                 import LGBMRegressor\n",
    "from geopy.geocoders          import Nominatim\n",
    "from sqlalchemy               import create_engine, text\n",
    "from matplotlib               import pyplot as plt\n",
    "from sklearn.preprocessing    import RobustScaler, MinMaxScaler, LabelEncoder\n",
    "from boruta                   import BorutaPy\n",
    "from sklearn.ensemble         import RandomForestRegressor\n",
    "from sklearn.model_selection  import train_test_split\n",
    "from sklearn.linear_model     import LinearRegression,  Lasso\n",
    "from sklearn.metrics          import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,r2_score\n",
    "from sklearn.model_selection  import cross_val_score\n",
    "from sklearn.preprocessing    import LabelEncoder\n",
    "from sklearn.model_selection  import KFold\n",
    "import pandas as pd\n",
    "import inflection\n",
    "import math\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import pathlib\n",
    "import seaborn as sns\n",
    "import ast\n",
    "import lightgbm as lgb\n",
    "import xgboost as xgb\n",
    "from scipy.stats       import chi2_contingency\n",
    "from category_encoders import TargetEncoder\n",
    "from sklearn          import preprocessing as pp\n",
    "import pickle\n",
    "import os\n",
    "\n",
    "from xgboost                  import XGBRegressor\n",
    "from lightgbm                 import LGBMRegressor\n",
    "from geopy.geocoders          import Nominatim\n",
    "from sqlalchemy               import create_engine, text\n",
    "from matplotlib               import pyplot as plt\n",
    "from sklearn.preprocessing    import RobustScaler, MinMaxScaler, LabelEncoder\n",
    "from boruta                   import BorutaPy\n",
    "from sklearn.ensemble         import RandomForestRegressor\n",
    "from sklearn.model_selection  import train_test_split\n",
    "from sklearn.linear_model     import LinearRegression,  Lasso\n",
    "from sklearn.metrics          import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,r2_score\n",
    "from sklearn.model_selection  import cross_val_score\n",
    "from sklearn.preprocessing    import LabelEncoder\n",
    "from sklearn.model_selection  import KFold\n",
    "import pandas as pd\n",
    "import inflection\n",
    "import math\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import pathlib\n",
    "import seaborn as sns\n",
    "import ast\n",
    "import lightgbm as lgb\n",
    "import xgboost as xgb\n",
    "from scipy.stats       import chi2_contingency\n",
    "from category_encoders import TargetEncoder\n",
    "from sklearn          import preprocessing as pp\n",
    "import pickle\n",
    "import os\n",
    "\n",
    "from xgboost                  import XGBRegressor\n",
    "from lightgbm                 import LGBMRegressor\n",
    "from geopy.geocoders          import Nominatim\n",
    "from sqlalchemy               import create_engine, text\n",
    "from matplotlib               import pyplot as plt\n",
    "from sklearn.preprocessing    import RobustScaler, MinMaxScaler, LabelEncoder\n",
    "from boruta                   import BorutaPy\n",
    "from sklearn.ensemble         import RandomForestRegressor\n",
    "from sklearn.model_selection  import train_test_split\n",
    "from sklearn.linear_model     import LinearRegression,  Lasso\n",
    "from sklearn.metrics          import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,r2_score\n",
    "from sklearn.model_selection  import cross_val_score\n",
    "from sklearn.preprocessing    import LabelEncoder\n",
    "from sklearn.model_selection  import KFold\n",
    "import pandas as pd\n",
    "import inflection\n",
    "import math\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import pathlib\n",
    "import seaborn as sns\n",
    "import ast\n",
    "import lightgbm as lgb\n",
    "import xgboost as xgb\n",
    "from scipy.stats       import chi2_contingency\n",
    "from category_encoders import TargetEncoder\n",
    "from sklearn          import preprocessing as pp\n",
    "import pickle\n",
    "import os\n",
    "\n",
    "from xgboost                  import XGBRegressor\n",
    "from lightgbm                 import LGBMRegressor\n",
    "from geopy.geocoders          import Nominatim\n",
    "from sqlalchemy               import create_engine, text\n",
    "from matplotlib               import pyplot as plt\n",
    "from sklearn.preprocessing    import RobustScaler, MinMaxScaler, LabelEncoder\n",
    "from boruta                   import BorutaPy\n",
    "from sklearn.ensemble         import RandomForestRegressor\n",
    "from sklearn.model_selection  import train_test_split\n",
    "from sklearn.linear_model     import LinearRegression,  Lasso\n",
    "from sklearn.metrics          import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,r2_score\n",
    "from sklearn.model_selection  import cross_val_score\n",
    "from sklearn.preprocessing    import LabelEncoder\n",
    "from sklearn.model_selection  import KFold\n",
    "import pandas as pd\n",
    "import inflection\n",
    "import math\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import pathlib\n",
    "import seaborn as sns\n",
    "import ast\n",
    "import lightgbm as lgb\n",
    "import xgboost as xgb\n",
    "from scipy.stats       import chi2_contingency\n",
    "from category_encoders import TargetEncoder\n",
    "from sklearn          import preprocessing as pp\n",
    "import pickle\n",
    "import os\n",
    "\n",
    "from xgboost                  import XGBRegressor\n",
    "from lightgbm                 import LGBMRegressor\n",
    "from geopy.geocoders          import Nominatim\n",
    "from sqlalchemy               import create_engine, text\n",
    "from matplotlib               import pyplot as plt\n",
    "from sklearn.preprocessing    import RobustScaler, MinMaxScaler, LabelEncoder\n",
    "from boruta                   import BorutaPy\n",
    "from sklearn.ensemble         import RandomForestRegressor\n",
    "from sklearn.model_selection  import train_test_split\n",
    "from sklearn.linear_model     import LinearRegression,  Lasso\n",
    "from sklearn.metrics          import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,r2_score\n",
    "from sklearn.model_selection  import cross_val_score\n",
    "from sklearn.preprocessing    import LabelEncoder\n",
    "from sklearn.model_selection  import KFold\n",
    "import pandas as pd\n",
    "import inflection\n",
    "import math\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import pathlib\n",
    "import seaborn as sns\n",
    "import ast\n",
    "import lightgbm as lgb\n",
    "import xgboost as xgb\n",
    "from scipy.stats       import chi2_contingency\n",
    "from category_encoders import TargetEncoder\n",
    "from sklearn          import preprocessing as pp\n",
    "import pickle\n",
    "import os\n",
    "\n",
    "from xgboost                  import XGBRegressor\n",
    "from lightgbm                 import LGBMRegressor\n",
    "from geopy.geocoders          import Nominatim\n",
    "from sqlalchemy               import create_engine, text\n",
    "from matplotlib               import pyplot as plt\n",
    "from sklearn.preprocessing    import RobustScaler, MinMaxScaler, LabelEncoder\n",
    "from boruta                   import BorutaPy\n",
    "from sklearn.ensemble         import RandomForestRegressor\n",
    "from sklearn.model_selection  import train_test_split\n",
    "from sklearn.linear_model     import LinearRegression,  Lasso\n",
    "from sklearn.metrics          import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,r2_score\n",
    "from sklearn.model_selection  import cross_val_score\n",
    "from sklearn.preprocessing    import LabelEncoder\n",
    "from sklearn.model_selection  import KFold\n",
    "import pandas as pd\n",
    "import inflection\n",
    "import math\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import pathlib\n",
    "import seaborn as sns\n",
    "import ast\n",
    "import lightgbm as lgb\n",
    "import xgboost as xgb\n",
    "from scipy.stats       import chi2_contingency\n",
    "from category_encoders import TargetEncoder\n",
    "from sklearn          import preprocessing as pp\n",
    "import pickle\n",
    "import os\n",
    "\n",
    "from xgboost                  import XGBRegressor\n",
    "from lightgbm                 import LGBMRegressor\n",
    "from geopy.geocoders          import Nominatim\n",
    "from sqlalchemy               import create_engine, text\n",
    "from matplotlib               import pyplot as plt\n",
    "from sklearn.preprocessing    import RobustScaler, MinMaxScaler, LabelEncoder\n",
    "from boruta                   import BorutaPy\n",
    "from sklearn.ensemble         import RandomForestRegressor\n",
    "from sklearn.model_selection  import train_test_split\n",
    "from sklearn.linear_model     import LinearRegression,  Lasso\n",
    "from sklearn.metrics          import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,r2_score\n",
    "from sklearn.model_selection  import cross_val_score\n",
    "from sklearn.preprocessing    import LabelEncoder\n",
    "from sklearn.model_selection  import KFold\n",
    "import pandas as pd\n",
    "import inflection\n",
    "import math\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import pathlib\n",
    "import seaborn as sns\n",
    "import ast\n",
    "import lightgbm as lgb\n",
    "import xgboost as xgb\n",
    "from scipy.stats       import chi2_contingency\n",
    "from category_encoders import TargetEncoder\n",
    "from sklearn          import preprocessing as pp\n",
    "import pickle\n",
    "import os\n",
    "\n",
    "from xgboost                  import XGBRegressor\n",
    "from lightgbm                 import LGBMRegressor\n",
    "from geopy.geocoders          import Nominatim\n",
    "from sqlalchemy               import create_engine, text\n",
    "from matplotlib               import pyplot as plt\n",
    "from sklearn.preprocessing    import RobustScaler, MinMaxScaler, LabelEncoder\n",
    "from boruta                   import BorutaPy\n",
    "from sklearn.ensemble         import RandomForestRegressor\n",
    "from sklearn.model_selection  import train_test_split\n",
    "from sklearn.linear_model     import LinearRegression,  Lasso\n",
    "from sklearn.metrics          import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,r2_score\n",
    "from sklearn.model_selection  import cross_val_score\n",
    "from sklearn.preprocessing    import LabelEncoder\n",
    "from sklearn.model_selection  import KFold\n",
    "import pandas as pd\n",
    "import inflection\n",
    "import math\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import pathlib\n",
    "import seaborn as sns\n",
    "import ast\n",
    "import lightgbm as lgb\n",
    "import xgboost as xgb\n",
    "from scipy.stats       import chi2_contingency\n",
    "from category_encoders import TargetEncoder\n",
    "from sklearn          import preprocessing as pp\n",
    "import pickle\n",
    "import os\n",
    "\n",
    "from xgboost                  import XGBRegressor\n",
    "from lightgbm                 import LGBMRegressor\n",
    "from geopy.geocoders          import Nominatim\n",
    "from sqlalchemy               import create_engine, text\n",
    "from matplotlib               import pyplot as plt\n",
    "from sklearn.preprocessing    import RobustScaler, MinMaxScaler, LabelEncoder\n",
    "from boruta                   import BorutaPy\n",
    "from sklearn.ensemble         import RandomForestRegressor\n",
    "from sklearn.model_selection  import train_test_split\n",
    "from sklearn.linear_model     import LinearRegression,  Lasso\n",
    "from sklearn.metrics          import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,r2_score\n",
    "from sklearn.model_selection  import cross_val_score\n",
    "from sklearn.preprocessing    import LabelEncoder\n",
    "from sklearn.model_selection  import KFold\n",
    "import pandas as pd\n",
    "import inflection\n",
    "import math\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import pathlib\n",
    "import seaborn as sns\n",
    "import ast\n",
    "import lightgbm as lgb\n",
    "import xgboost as xgb\n",
    "from scipy.stats       import chi2_contingency\n",
    "from category_encoders import TargetEncoder\n",
    "from sklearn          import preprocessing as pp\n",
    "import pickle\n",
    "import os\n",
    "\n",
    "from xgboost                  import XGBRegressor\n",
    "from lightgbm                 import LGBMRegressor\n",
    "from geopy.geocoders          import Nominatim\n",
    "from sqlalchemy               import create_engine, text\n",
    "from matplotlib               import pyplot as plt\n",
    "from sklearn.preprocessing    import RobustScaler, MinMaxScaler, LabelEncoder\n",
    "from boruta                   import BorutaPy\n",
    "from sklearn.ensemble         import RandomForestRegressor\n",
    "from sklearn.model_selection  import train_test_split\n",
    "from sklearn.linear_model     import LinearRegression,  Lasso\n",
    "from sklearn.metrics          import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,r2_score\n",
    "from sklearn.model_selection  import cross_val_score\n",
    "from sklearn.preprocessing    import LabelEncoder\n",
    "from sklearn.model_selection  import KFold\n",
    "import pandas as pd\n",
    "import inflection\n",
    "import math\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import pathlib\n",
    "import seaborn as sns\n",
    "import ast\n",
    "import lightgbm as lgb\n",
    "import xgboost as xgb\n",
    "from scipy.stats       import chi2_contingency\n",
    "from category_encoders import TargetEncoder\n",
    "from sklearn          import preprocessing as pp\n",
    "import pickle\n",
    "import os\n",
    "\n",
    "from xgboost                  import XGBRegressor\n",
    "from lightgbm                 import LGBMRegressor\n",
    "from geopy.geocoders          import Nominatim\n",
    "from sqlalchemy               import create_engine, text\n",
    "from matplotlib               import pyplot as plt\n",
    "from sklearn.preprocessing    import RobustScaler, MinMaxScaler, LabelEncoder\n",
    "from boruta                   import BorutaPy\n",
    "from sklearn.ensemble         import RandomForestRegressor\n",
    "from sklearn.model_selection  import train_test_split\n",
    "from sklearn.linear_model     import LinearRegression,  Lasso\n",
    "from sklearn.metrics          import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,r2_score\n",
    "from sklearn.model_selection  import cross_val_score\n",
    "from sklearn.preprocessing    import LabelEncoder\n",
    "from sklearn.model_selection  import KFold\n",
    "import pandas as pd\n",
    "import inflection\n",
    "import math\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import pathlib\n",
    "import seaborn as sns\n",
    "import ast\n",
    "import lightgbm as lgb\n",
    "import xgboost as xgb\n",
    "from scipy.stats       import chi2_contingency\n",
    "from category_encoders import TargetEncoder\n",
    "from sklearn          import preprocessing as pp\n",
    "import pickle\n",
    "import os\n",
    "\n",
    "from xgboost                  import XGBRegressor\n",
    "from lightgbm                 import LGBMRegressor\n",
    "from geopy.geocoders          import Nominatim\n",
    "from sqlalchemy               import create_engine, text\n",
    "from matplotlib               import pyplot as plt\n",
    "from sklearn.preprocessing    import RobustScaler, MinMaxScaler, LabelEncoder\n",
    "from boruta                   import BorutaPy\n",
    "from sklearn.ensemble         import RandomForestRegressor\n",
    "from sklearn.model_selection  import train_test_split\n",
    "from sklearn.linear_model     import LinearRegression,  Lasso\n",
    "from sklearn.metrics          import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,r2_score\n",
    "from sklearn.model_selection  import cross_val_score\n",
    "from sklearn.preprocessing    import LabelEncoder\n",
    "from sklearn.model_selection  import KFold"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3e10c63f-ec0d-476a-9c17-38c29156929f",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 0.1.0 Helper Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2dbeb768-f3de-41df-bffb-219f265d44d2",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:13:44.435545Z",
     "start_time": "2024-03-01T22:13:44.417987Z"
    }
   },
   "outputs": [],
   "source": [
    "# função para consultar a api passando as coordenadas\n",
    "def find_mun( lat,long ):\n",
    "    \n",
    "    geolocator = Nominatim(user_agent=\"Oscar\")\n",
    "    \n",
    "    lat = str( lat )\n",
    "    \n",
    "    long = str( long )\n",
    "    \n",
    "    coord = lat + ',' + long\n",
    "\n",
    "    location = geolocator.reverse( coord  )\n",
    "    \n",
    "    if 'city_district' in location.raw['address']:\n",
    "        \n",
    "        return location.raw['address']['city_district']\n",
    "        \n",
    "    if 'suburb' in location.raw['address']:\n",
    "            \n",
    "        return location.raw['address']['suburb']\n",
    "            \n",
    "    if 'county' in location.raw['address']:\n",
    "            \n",
    "        return location.raw['address']['county']\n",
    "        \n",
    "    else:\n",
    "        \n",
    "        return 'not found'\n",
    "\n",
    "  \n",
    "def stats( df , excepto = [] ):\n",
    "    # separo as feautures numéricas\n",
    "    if excepto != \"\":\n",
    "    \n",
    "        num_attributes = df.select_dtypes( include = ['int64','float64'] )\n",
    "        \n",
    "        num_attributes = num_attributes.drop( excepto , axis = 1 ) \n",
    "        \n",
    "    else:\n",
    "        \n",
    "        num_attributes = df.select_dtypes( include = ['int64','float64'] )\n",
    "    # separo as features categoricas\n",
    "    num_categorical = df.select_dtypes( exclude = ['int64','float64','datetime64[ns]'] )\n",
    "\n",
    "\n",
    "    # medidas de tendencia central - media e mediana\n",
    "    ct1 = pd.DataFrame( num_attributes.apply( np.mean ) ).T # media\n",
    "\n",
    "    ct2 = pd.DataFrame( num_attributes.apply( np.median ) ).T # mediana\n",
    "\n",
    "    # dispersao - std, min , max , range, skew, kurtosis\n",
    "    d1 = pd.DataFrame( num_attributes.apply( np.std ) ).T # std\n",
    "    d2 = pd.DataFrame( num_attributes.apply( np.min ) ).T # minimo\n",
    "    d3 = pd.DataFrame( num_attributes.apply( np.max ) ).T # maximo\n",
    "    d4 = pd.DataFrame( num_attributes.apply( lambda x: x.max() - x.min() ) ).T # range\n",
    "    d5 = pd.DataFrame( num_attributes.apply( lambda x: x.skew() )).T # skew\n",
    "    d6 = pd.DataFrame( num_attributes.apply( lambda x : x.kurtosis() ) ).T # kurtosis\n",
    "\n",
    "    # concatenate - min , max , range,  media, median, std, skew , kurtosis\n",
    "    m = pd.concat( [ d2 , d3, d4 , ct1 , ct2 , d1,  d5 , d6 ] ).T.reset_index()\n",
    "    m.columns = [ 'features', 'min', 'max', 'range', 'mean', 'median', 'std', 'skew', 'kurtosis' ]\n",
    "    pd.set_option( 'display.float_format', '{:,.2f}'.format )\n",
    "    return m\n",
    "\n",
    "def ml_error( model_name, y , y_hat):\n",
    "    mae =  mean_absolute_error( y , y_hat )\n",
    "    mape = mean_absolute_percentage_error( y , y_hat )\n",
    "    rmse = np.sqrt( mean_squared_error( y , y_hat ) )\n",
    "    r2 =   r2_score(y, y_hat)\n",
    "    df = pd.DataFrame( {'Model Name ' :  model_name,\n",
    "                        'MAE' : mae ,\n",
    "                        'MAPE' : mape,\n",
    "                        'RMSE' : rmse,\n",
    "                        'R2_score' : r2  } ,index = [0] )\n",
    "    \n",
    "    return df\n",
    "\n",
    "def remover_outliers( df, coluna ):\n",
    "    \n",
    "    linhas_iniciais = df.shape[0] \n",
    "    \n",
    "    \n",
    "    d1 = df[coluna].quantile(0.25)\n",
    "    d3 = df[coluna].quantile(0.75)\n",
    "\n",
    "    lim_superior = d3 + 1.5*( d3 - d1  )\n",
    "\n",
    "    lim_inferior = d1 - 1.5 * ( d3 - d1 )\n",
    "    \n",
    "    \n",
    "    df = df.loc[ (df[coluna] >= lim_inferior) & (df[coluna] <= lim_superior) , :  ]\n",
    "    \n",
    "    linhas_finais = df.shape[0]\n",
    "    \n",
    "    \n",
    "    print('Foram removidas {} linhas'.format( linhas_iniciais - linhas_finais ) )\n",
    "          \n",
    "    return df\n",
    "\n",
    "def cramer_v( x, y ):\n",
    "    cm = pd.crosstab( x , y ).values\n",
    "\n",
    "    n = cm.sum()\n",
    "\n",
    "    r,k = cm.shape\n",
    "\n",
    "    chi2 = chi2_contingency( cm )[0]\n",
    "    chi2corr = max( 0 , chi2 - (k-1) * (r-1) / (n-1) ) # chi2 corrigido\n",
    "    kcorr = k - (k-1)**2 / (n-1) # k corrigido\n",
    "    rcorr = r - (k-1)**2 / (n-1) # r corrigido\n",
    "    \n",
    "    v = np.sqrt( ( chi2corr/n ) / ( min ( kcorr-1 , rcorr-1 ) )  )\n",
    "    \n",
    "    return v"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "79e7e0b9-5cde-4e7b-bc03-5e3436b6307f",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:13:44.473617Z",
     "start_time": "2024-03-01T22:13:44.439539Z"
    }
   },
   "outputs": [],
   "source": [
    "class Idealista:\n",
    "    def __init__( self ):\n",
    "        \n",
    "        self.df_region = pd.read_csv( 'datasets/df_region.csv' )\n",
    "        self.home_path                    = r'C:\\Users\\oscar\\Documents\\repos\\api_houses_Lisbon\\encodings'\n",
    "\n",
    "        self.te_status                    = pickle.load(open(os.path.join(self.home_path, 'te_status.pkl'), 'rb'))\n",
    "        self.te_province                  = pickle.load(open(os.path.join(self.home_path,'te_province.pkl') , 'rb'))\n",
    "        self.te_property_type             = pickle.load(open(os.path.join(self.home_path,'te_property_type.pkl'), 'rb'))\n",
    "        self.te_municipality              = pickle.load(open(os.path.join(self.home_path,'te_municipality.pkl'), 'rb' ))\n",
    "        self.te_district                  = pickle.load(open(os.path.join(self.home_path,'te_district.pkl'),'rb'))\n",
    "        self.le_detailed_type             = pickle.load(open(os.path.join(self.home_path,'le_detailed_type.pkl'),'rb' ))\n",
    "        self.freq_encoding_address        = pickle.load(open(os.path.join(self.home_path,'freq_encoding_address.pkl'), 'rb' ))\n",
    "        \n",
    "    \n",
    "    \n",
    "    \n",
    "    def transform_data(self, df, colunas_irrelevantes ):\n",
    "\n",
    "        cols_old = df.columns.to_list()\n",
    "\n",
    "        # def função snake_case\n",
    "        snake_case = lambda x: inflection.underscore( x )\n",
    "\n",
    "        # defino as novas colunas\n",
    "        cols_new = list(map( snake_case, cols_old )) \n",
    "\n",
    "        # atribuo as novas colunas ao df\n",
    "        df.columns = cols_new\n",
    "\n",
    "        # removo as linhas em branco que possam ter vindo na extração\n",
    "        df = df.loc[~ df['property_code'].isna() , : ]\n",
    "\n",
    "        # removo duplicadas\n",
    "        df = df.drop_duplicates( subset= ['property_code'], keep = 'last' )\n",
    "\n",
    "        # preencho os vazios\n",
    "        df = fill_na( df )\n",
    "\n",
    "        # deleto as colunaas irrelevantes\n",
    "        df = delete_columns( df , colunas_irrelevantes )\n",
    "   \n",
    "        return df\n",
    "\n",
    "\n",
    "#=======================================================================================================================#\n",
    "\n",
    "    def feature_engineering(self, df , transform_data = False, merge_datasets = False   ):\n",
    "        \n",
    "        if transform_data == True and merge_datasets == False :\n",
    "            \n",
    "            # mes\n",
    "            df['datetime_scrapy'] = pd.to_datetime( df['datetime_scrapy'] )\n",
    "            df['month'] = df['datetime_scrapy'].dt.month \n",
    "\n",
    "            # ano\n",
    "            df['year'] = df['datetime_scrapy'].dt.year\n",
    "\n",
    "            # excluir a coluna datetime\n",
    "            df = df.drop( 'datetime_scrapy', axis = 1 )\n",
    "\n",
    "            print( 'Dados Transformados' )\n",
    "\n",
    "            return df\n",
    "        \n",
    "    \n",
    "        if merge_datasets == True and transform_data == False:\n",
    "        \n",
    "            # Df com a média de preço por município\n",
    "            df_municipality_mean_price = self.df_region.groupby('municipality')[['price']].mean().reset_index()\n",
    "\n",
    "            # alteração nome coluna\n",
    "            df_municipality_mean_price = df_municipality_mean_price.rename(columns={'price': 'municipality_mean_price'})\n",
    "\n",
    "            # exportar para csv\n",
    "            #d f2_municipality_mean_price.to_csv( 'datasets/municipality.csv' )\n",
    "\n",
    "            # acrescentando a coluna ao df2\n",
    "            df = pd.merge( df, df_municipality_mean_price, on = 'municipality' , how = 'left' )\n",
    "\n",
    "#======================================================================================================================#\n",
    "\n",
    "            # calculando a média por 'province'\n",
    "            df2_province_mean_price = self.df_region.groupby( 'province' )[['price']].mean().reset_index().rename( columns = { 'price': 'province_mean_price' } )\n",
    "\n",
    "            # exportando para csv\n",
    "            # df2_province_mean_price.to_csv( 'datasets/province.csv' )\n",
    "\n",
    "            # acrescentando a coluna ao df2\n",
    "            df = pd.merge( df, df2_province_mean_price , on = 'province' , how = 'left' )\n",
    "\n",
    "    #=======================================================================================================================#\n",
    "\n",
    "            # calculando média por district\n",
    "            df2_district_mean_price = self.df_region.groupby( 'district' )[['price']].median().reset_index().rename( columns = { 'price' : 'district_mean_price' } )\n",
    "\n",
    "            # exportando para csv \n",
    "            #df2_district_mean_price.to_csv( 'datasets/district.csv' )\n",
    "\n",
    "            # acrescentando a coluna ao df2\n",
    "            df = pd.merge( df , df2_district_mean_price , on = 'district', how = 'left' )\n",
    "\n",
    "            print( 'Datasets combinados com preços regioes' )\n",
    "\n",
    "            return df\n",
    "    \n",
    "        if transform_data == True and merge_datasets == True:\n",
    "\n",
    "            # mes\n",
    "            df['datetime_scrapy'] = pd.to_datetime( df['datetime_scrapy'] )\n",
    "            df['month'] = df['datetime_scrapy'].dt.month \n",
    "\n",
    "            # ano\n",
    "            df['year'] = df['datetime_scrapy'].dt.year\n",
    "\n",
    "            # excluir a coluna datetime\n",
    "            df = df.drop( 'datetime_scrapy', axis = 1 )\n",
    "\n",
    "            # Df com a média de preço por município\n",
    "            df_municipality_mean_price = self.df_region.groupby('municipality')[['price']].mean().reset_index()\n",
    "\n",
    "            # alteração nome coluna\n",
    "            df_municipality_mean_price = df_municipality_mean_price.rename(columns={'price': 'municipality_mean_price'})\n",
    "\n",
    "            # exportar para csv\n",
    "            #d f2_municipality_mean_price.to_csv( 'datasets/municipality.csv' )\n",
    "\n",
    "            # acrescentando a coluna ao df2\n",
    "            df = pd.merge( df, df_municipality_mean_price, on = 'municipality' , how = 'left' )\n",
    "\n",
    "    # ======================================================================================================================#\n",
    "\n",
    "            # calculando a média por 'province'\n",
    "            df2_province_mean_price = self.df_region.groupby( 'province' )[['price']].mean().reset_index().rename( columns = { 'price': 'province_mean_price' } )\n",
    "\n",
    "            # exportando para csv\n",
    "            # df2_province_mean_price.to_csv( 'datasets/province.csv' )\n",
    "\n",
    "            # acrescentando a coluna ao df2\n",
    "            df = pd.merge( df, df2_province_mean_price , on = 'province' , how = 'left' )\n",
    "\n",
    "    #=======================================================================================================================#\n",
    "\n",
    "            # calculando média por district\n",
    "            df2_district_mean_price = self.df_region.groupby( 'district' )[['price']].median().reset_index().rename( columns = { 'price' : 'district_mean_price' } )\n",
    "\n",
    "            # exportando para csv \n",
    "            #df2_district_mean_price.to_csv( 'datasets/district.csv' )\n",
    "\n",
    "            # acrescentando a coluna ao df2\n",
    "            df = pd.merge( df , df2_district_mean_price , on = 'district', how = 'left' )\n",
    "\n",
    "            print( 'datasets combinados e dados transformados' )\n",
    "        \n",
    "            return df\n",
    "        \n",
    "        else:\n",
    "\n",
    "            print( 'Nenhuma Transformação' )\n",
    "\n",
    "            return df\n",
    "            \n",
    "#=======================================================================================================================#\n",
    "    \n",
    "    def filter_variables(self, df, filter_variables = False):\n",
    "    \n",
    "        if filter_variables:\n",
    "\n",
    "            # Vou querer saber somente os preços na province de Lisboa\n",
    "            df = df.loc[ df['province'] == 'Lisboa' , :  ]\n",
    "\n",
    "            # vou querer somente as habitações que tenham de 0 a no máximo 4 quartos\n",
    "            df = df.loc[ df['rooms'].isin( [ 0,1,2,3,4 ] ) , : ]\n",
    "\n",
    "            print('Variavies filtradas')\n",
    "\n",
    "            return df\n",
    "\n",
    "        else:\n",
    "\n",
    "            print('Variavies não filtradas')\n",
    "\n",
    "            return df\n",
    "\n",
    "#=======================================================================================================================#\n",
    "\n",
    "    # função para remover outliers\n",
    "    def remove_outliers(self, df, coluna = None ,  keep_outliers = True ):\n",
    "\n",
    "        if keep_outliers:\n",
    "            print( 'Outliers mantidos' )\n",
    "            return df\n",
    "\n",
    "        else:\n",
    "            # variavel resposta sem outliers\n",
    "            df = remover_outliers( df4 , coluna )\n",
    "            print('Outliers removidos')\n",
    "            return df\n",
    "\n",
    "#=======================================================================================================================#\n",
    "    \n",
    "    def rescaling_data(self, df, rescale = False ):\n",
    "\n",
    "        if rescale:\n",
    "            rs = RobustScaler() # robusto com outliers\n",
    "            mms = pp.MinMaxScaler() # desvio padrão pequeno e quando não ha distribuição gaussiana\n",
    "\n",
    "    #         # num_photos\n",
    "    #         rs_num_photos = RobustScaler() \n",
    "    #         # num_photos\n",
    "    #         df['num_photos'] = rs_num_photos.fit_transform( df[['num_photos']].values )\n",
    "\n",
    "            # floor\n",
    "#             rs_floor = RobustScaler() \n",
    "#             # num_photos\n",
    "#             df['floor'] = rs_floor.fit_transform( df[['floor']].values )\n",
    "\n",
    "            # size\n",
    "            rs_size = RobustScaler() \n",
    "            # size\n",
    "            df['size'] = rs_size.fit_transform( df[['size']].values )\n",
    "\n",
    "            # rooms\n",
    "    #         mms_rooms = pp.MinMaxScaler() \n",
    "    #         # rooms\n",
    "    #         df['rooms'] = mms_rooms.fit_transform( df[['rooms']].values )\n",
    "\n",
    "            # bathrooms\n",
    "    #         rs_bathrooms = RobustScaler() \n",
    "    #         # bathrooms\n",
    "    #         df['bathrooms'] = rs_bathrooms.fit_transform( df[['bathrooms']].values )\n",
    "\n",
    "            # distance\n",
    "            rs_distance = RobustScaler()\n",
    "            # distance\n",
    "            df['distance'] = rs_distance.fit_transform( df[['distance']].values )\n",
    "\n",
    "\n",
    "\n",
    "            print('Rescaled')\n",
    "\n",
    "            return  df\n",
    "        else:\n",
    "            print('Not rescaled')\n",
    "\n",
    "            return df\n",
    "\n",
    "#=======================================================================================================================#\n",
    "\n",
    "    def encoding(self, df ):\n",
    "\n",
    "        # floor - substituo os andares com letras\n",
    "        df['floor'] = df['floor'].replace( ['bj','st','ss','en'], 0 )\n",
    "        # transformo em inteiros\n",
    "        df['floor'] = df['floor'].astype( 'int64' )\n",
    "\n",
    "        #property_type - label\n",
    "        df['property_type'] = df['property_type'].map( self.te_property_type  ).astype( 'float64' )\n",
    "\n",
    "        # address - label\n",
    "        df['address'] = df['address'].map( self.freq_encoding_address )\n",
    "\n",
    "        # province \n",
    "        df['province'] = df['province'].map( self.te_province ).astype( 'float64' )\n",
    "\n",
    "        # municipality - label\n",
    "        df['municipality'] = df['municipality'].map( self.te_municipality ).astype( 'float64' )\n",
    "\n",
    "        # municipality\n",
    "    #     freq_encoding_municipality = df['municipality'].value_counts(normalize=True)\n",
    "    #     df['municipality'] = df['municipality'].map( freq_encoding_municipality )\n",
    "\n",
    "\n",
    "        # district - label\n",
    "        df['district'] = df['district'].map( self.te_district ).astype( 'float64' )\n",
    "\n",
    "        # district\n",
    "    #     freq_encoding_district = df['district'].value_counts(normalize=True)\n",
    "    #     df['district'] = df['district'].map( freq_encoding_district )\n",
    "\n",
    "\n",
    "        # Show address\n",
    "        encoding = {True: 1 , False: 0}\n",
    "        df['show_address'] = df['show_address'].apply( lambda x : 1 if x == True else 0 )\n",
    "\n",
    "        # description\n",
    "        df['description'] = df['description'].apply( lambda x : len(x) )\n",
    "\n",
    "        # has_video\n",
    "        encoding = {True: 1 , False: 0}\n",
    "        df['has_video'] = df['has_video'].map( encoding )\n",
    "\n",
    "        # status\n",
    "        df['status'] = df['status'].map( self.te_status ).astype( 'float64' )\n",
    "\n",
    "        # label encoder status\n",
    "        #status_encoder = LabelEncoder()\n",
    "        #df['status'] = status_encoder.fit_transform( df['status'] )\n",
    "\n",
    "        # new_development\n",
    "        encoding = {True: 1 , False: 0}\n",
    "        df['new_development'] = df['new_development'].map( encoding )\n",
    "\n",
    "        # detailed type - label\n",
    "        df['detailed_type'] = df['detailed_type'].apply(lambda x: ast.literal_eval(x) )\n",
    "        df['detailed_type'] = df['detailed_type'].apply(lambda x: x['typology'] )\n",
    "        df['detailed_type'] = self.le_detailed_type.transform( df['detailed_type'] )\n",
    "\n",
    "        # suggested text\n",
    "        df['suggested_texts'] = df['suggested_texts'].apply( lambda x : ast.literal_eval(x) )\n",
    "        df['suggested_texts'] = df['suggested_texts'].apply( lambda x : x['title'] )\n",
    "        df['suggested_texts'] = df['suggested_texts'].apply( lambda x : len(x) )\n",
    "\n",
    "        # hasplan\n",
    "        encoding = {True: 1 , False: 0}\n",
    "        df['has_plan'] = df['has_plan'].map( encoding )\n",
    "\n",
    "        # has3_d_tour\n",
    "        encoding = {True: 1 , False: 0}\n",
    "        df['has3_d_tour'] = df['has3_d_tour'].map( encoding )\n",
    "\n",
    "        # has360\n",
    "        encoding = {True: 1 , False: 0}\n",
    "        df['has360'] = df['has360'].map( encoding )\n",
    "\n",
    "        # has_staging\n",
    "        df['has_staging'] = df['has_staging'].apply( lambda x: 1 if x   else 0   )\n",
    "\n",
    "        # top_new_development\n",
    "        df['top_new_development'] = df['top_new_development'].apply( lambda x: 1 if x  else 0 )\n",
    "\n",
    "        #parking_space\n",
    "        encoding = {True: 1 , False: 0}\n",
    "        df['parking_space'] = df['parking_space'].apply(lambda x: ast.literal_eval(x) if x != 0 else x ) \n",
    "        df['parking_space'] = df['parking_space'].apply(lambda x: x['hasParkingSpace'] if x != 0 else x ) \n",
    "        df['parking_space'] = df['parking_space'].map( encoding )\n",
    "        \n",
    "        df = df.fillna( 0 )\n",
    "\n",
    "        return df\n",
    "    \n",
    "    def predicoes(self, df , model ):\n",
    "        \n",
    "        # drop price\n",
    "        X = df.drop( 'price', axis = 1 )\n",
    "        \n",
    "        y_hat_final = model.predict( X )\n",
    "        \n",
    "        df['pred'] = y_hat_final\n",
    "        \n",
    "        return df\n",
    "        \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9a749f86-db89-4859-a5e9-136a4f1448ad",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 0.2.0. Loading Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6cabd038-ceda-4b76-b667-a46295b0c943",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:13:57.877288Z",
     "start_time": "2024-03-01T22:13:44.477609Z"
    }
   },
   "outputs": [],
   "source": [
    "# caminho\n",
    "caminho = pathlib.Path('data')\n",
    "\n",
    "df = []\n",
    "\n",
    "for arquivo in caminho.iterdir():\n",
    "    print(arquivo.name)\n",
    "    \n",
    "    # ler o arquivo\n",
    "    df_read = pd.read_excel( arquivo )\n",
    "    \n",
    "    # apender o df\n",
    "    df.append(df_read)\n",
    "\n",
    "# concatena os df's\n",
    "df_raw = pd.concat( df )\n",
    "\n",
    "df_raw = df_raw[['propertyCode', 'thumbnail', 'numPhotos', 'floor', 'price',\n",
    "       'propertyType', 'operation', 'size', 'exterior', 'rooms', 'bathrooms',\n",
    "       'address', 'province', 'municipality', 'district', 'country',\n",
    "       'latitude', 'longitude', 'showAddress', 'url', 'distance',\n",
    "       'description', 'hasVideo', 'status', 'newDevelopment', 'hasLift',\n",
    "       'priceByArea', 'detailedType', 'suggestedTexts', 'hasPlan', 'has3DTour',\n",
    "       'has360', 'hasStaging', 'topNewDevelopment', 'superTopHighlight',\n",
    "       'parkingSpace', 'externalReference', 'labels', 'neighborhood', 'pagina',\n",
    "       'datetime_scrapy', 'newDevelopmentFinished', 'highlight']]\n",
    "\n",
    "# resetar index\n",
    "df_raw = df_raw.reset_index( drop = True )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "956547f8-0776-4abd-8dbd-765eabb78b77",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 0.2.1 Separar Dados de Teste"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b8bb2656-b454-4ccc-b78c-1c6911416779",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:13:57.883126Z",
     "start_time": "2024-03-01T22:13:57.879291Z"
    }
   },
   "outputs": [],
   "source": [
    "# Separar Dados de Validação\n",
    "divisor = df_raw.shape[0] - 100\n",
    "df_val = df_raw.iloc[ divisor: ] # pego 1000 linhas para teste"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eb6b2856-122d-4ecf-b25b-a967a322323d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1.0. PASSO 01 - DESCRIÇÃO DOS DADOS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9890c0f3-a187-46e9-b954-37295d52e021",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:13:57.895455Z",
     "start_time": "2024-03-01T22:13:57.886121Z"
    }
   },
   "outputs": [],
   "source": [
    "df1 = df_raw.copy()[:divisor]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cd28340e-0b34-4ffd-998d-98fdabbee179",
   "metadata": {
    "heading_collapsed": true
   },
   "outputs": [],
   "source": [
    "## Explicação das colunas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "725f5565-459e-44f2-9148-e96b8884991c",
   "metadata": {
    "hidden": true
   },
   "outputs": [],
   "source": [
    "* property_code - ID unico para cada imovel\n",
    "\n",
    "\n",
    "* thumbnail - mostra a thumbnail do imovel\n",
    "\n",
    "\n",
    "* external_reference - A feature \"external_reference\" geralmente se refere a um código único atribuído a um imóvel por outra fonte, como outra plataforma imobiliária ou um agente imobiliário. Essa referência externa pode ser usada para rastrear e identificar o imóvel em diferentes sistemas e plataformas.\n",
    "\n",
    "\n",
    "* num_photos - número de fotos do anúncio\n",
    "\n",
    "\n",
    "* price - preço da renda\n",
    "\n",
    "\n",
    "* property_type - tipo de propriedade\n",
    "\n",
    "\n",
    "* operation - renda ou venda ( nesse caso tenho somente renda )\n",
    "\n",
    "\n",
    "* size - tamanho em m2\n",
    "\n",
    "\n",
    "* exterior - deve ser se o imóvel tem área exterior ( array([ 0., nan]) \n",
    "\n",
    "\n",
    "* rooms - quantidade de quartos\n",
    "\n",
    "\n",
    "* bathrooms - quantidade de casas de banho\n",
    "\n",
    "\n",
    "* address - endereço do imóvel\n",
    "\n",
    "\n",
    "* province - são as regiões ( Lisboa, Setúbal, Santarém - vieram na extração )\n",
    "\n",
    "\n",
    "* municipality - municipios\n",
    "\n",
    "\n",
    "* district - são os Distritos\n",
    "\n",
    "\n",
    "* country - PT \n",
    "\n",
    "\n",
    "* latitude - latitude\n",
    "\n",
    "\n",
    "* longitude - longitude\n",
    "\n",
    "\n",
    "* show_address - boolean - se o anunciante exibe o endereço\n",
    "\n",
    "\n",
    "* url - a url do anuncio\n",
    "\n",
    "\n",
    "* distance - distancia do centro em m\n",
    "\n",
    "\n",
    "* description - descricao do apartamento\n",
    "\n",
    "\n",
    "* has_video - diz se o anuncio tem vídeo ou não ( array([ 0.,  1., nan]) )\n",
    "\n",
    "\n",
    "* status - diz sobre o estado do imóvel array(['good', 'renew', 'newdevelopment', None], dtype=object)\n",
    "\n",
    "\n",
    "* new_development - boolean - diz se o empreendimento é novo ou não \n",
    "\n",
    "\n",
    "* price_by_area - preco divido pelo tamanho do imóvel price / size\n",
    "\n",
    "\n",
    "* detailed_type - descrição detalhada do tipo de imóvel\n",
    "\n",
    "\n",
    "* suggested_texts - títulos e subtítulos\n",
    "\n",
    "\n",
    "* has_plan - se o imóvel tem plano ou não\n",
    "\n",
    "\n",
    "* has3_d_tour - indica se o imóvel tem tour 3d\n",
    "\n",
    "\n",
    "* has360 - se tem tour 360º\n",
    "\n",
    "\n",
    "* has_staging - se tem uma decoração na propriedade\n",
    "\n",
    "\n",
    "* top_new_development - indica os empreendimentos mais novos\n",
    "\n",
    "\n",
    "* super_top_highlight - indica os imóveis mais valiosos ou populares em uma determinada área\n",
    "\n",
    "\n",
    "* floor - indica o andar do imóvel\n",
    "\n",
    "\n",
    "* has_lift - indica se tem elevador\n",
    "\n",
    "\n",
    "* parking_space - indica se tem vaga de garagem\n",
    "\n",
    "\n",
    "* neighborhood - Essa feature é útil para compradores e inquilinos que têm preferência por um determinado bairro ou querem morar em uma área específica da cidade por motivos como proximidade do trabalho, escolas, transporte público, entre outros.\n",
    "\n",
    "\n",
    "* labels - etiquetas ou rótulos associados aos imoveis\n",
    "\n",
    "\n",
    "* pagina - a pagina da extração ( definida na extração )\n",
    "\n",
    "\n",
    "* datetime_scrapy - indica a data da extração ( definida na extração )\n",
    "\n",
    "\n",
    "* newDevelopmentFinished - \n",
    "\n",
    "\n",
    "* highlight - "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4f0b24d1-4fb6-4a51-afe6-71a748d71702",
   "metadata": {
    "heading_collapsed": true
   },
   "outputs": [],
   "source": [
    "## 1.1. Rename Columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d49c9438-07d4-4594-8008-79e496eabb2f",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:13:57.921098Z",
     "start_time": "2024-03-01T22:13:57.898452Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "cols_old = [ 'propertyCode', 'thumbnail', 'numPhotos', 'floor', 'price',\n",
    "       'propertyType', 'operation', 'size', 'exterior', 'rooms', 'bathrooms',\n",
    "       'address', 'province', 'municipality', 'district', 'country',\n",
    "       'latitude', 'longitude', 'showAddress', 'url', 'distance',\n",
    "       'description', 'hasVideo', 'status', 'newDevelopment', 'hasLift',\n",
    "       'priceByArea', 'detailedType', 'suggestedTexts', 'hasPlan', 'has3DTour',\n",
    "       'has360', 'hasStaging', 'topNewDevelopment', 'superTopHighlight',\n",
    "       'parkingSpace', 'externalReference', 'labels', 'neighborhood', 'pagina',\n",
    "       'datetime_scrapy', 'newDevelopmentFinished', 'highlight']\n",
    "\n",
    "# def função snake_case\n",
    "snake_case = lambda x: inflection.underscore( x )\n",
    "\n",
    "# defino as novas colunas\n",
    "cols_new = list(map( snake_case, cols_old )) \n",
    "\n",
    "df1.columns = cols_new\n",
    "\n",
    "df1 = df1.loc[~ df1['property_code'].isna() , : ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d5faa165-6a04-49e4-b41d-7bffe42cf022",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-02-03T10:32:07.033474Z",
     "start_time": "2024-02-03T10:32:07.029868Z"
    },
    "heading_collapsed": true
   },
   "outputs": [],
   "source": [
    "## 1.1.2 Drop Duplicates"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9266d55c-1c33-4da2-8287-6b395ee3a4bf",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:13:57.939708Z",
     "start_time": "2024-03-01T22:13:57.922100Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "df1 = df1.drop_duplicates( subset= ['property_code'], keep = 'last' )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8b6bcbd5-3d2b-43e6-b02e-c53badf1f38b",
   "metadata": {
    "heading_collapsed": true
   },
   "outputs": [],
   "source": [
    "## 1.2. Data Dimensions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c389576f-dbea-449a-8b28-0799202c81ce",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:13:57.946188Z",
     "start_time": "2024-03-01T22:13:57.941705Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "print( 'Number of rows {:,}'.format( df1.shape[0] ) )\n",
    "print( 'Number of columns {:,}'.format( df1.shape[1] ) )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c44722dd-8dd1-49d7-90d2-42126dffc770",
   "metadata": {
    "heading_collapsed": true
   },
   "outputs": [],
   "source": [
    "## 1.3. Data Types"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fbe18249-28e8-4a43-9de0-9c088535fff6",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:13:57.957946Z",
     "start_time": "2024-03-01T22:13:57.953215Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "# exibindo os tipos de dados\n",
    "print( df1.dtypes )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a7461625-3847-41a9-a573-24c879f8cd89",
   "metadata": {
    "heading_collapsed": true
   },
   "outputs": [],
   "source": [
    "## 1.4. Check NA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e70249ae-d955-4513-8d2b-de5490f16659",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:13:58.048518Z",
     "start_time": "2024-03-01T22:13:57.959943Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "df1.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "87de54c8-2d4e-4cfa-a33e-bb33d738afba",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:13:58.236261Z",
     "start_time": "2024-03-01T22:13:58.050518Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "# calculando os % de dados vazios\n",
    "dfna = df1.isna().sum().to_frame().reset_index()\n",
    "dfna['% dados Vazios'] = ( df1.isna().sum().to_frame().reset_index()[0] / df1.shape[0] ) *100\n",
    "dfna.loc[ dfna['% dados Vazios'] != 0 , : ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9529bfd9-de09-4c0b-a145-a1b2a6cd3112",
   "metadata": {
    "heading_collapsed": true
   },
   "outputs": [],
   "source": [
    "## 1.5 - Excluindo colunas irrelevantes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bb51765b-2bb5-4ea2-aed5-6095e8a761de",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:13:58.243441Z",
     "start_time": "2024-03-01T22:13:58.238259Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "# função para excluir colunas se for passada uma lista\n",
    "def delete_columns( df , list_columns = None ):\n",
    "    \n",
    "    if list_columns is None:\n",
    "        \n",
    "        df = df\n",
    "        \n",
    "        return df\n",
    "    \n",
    "    else:\n",
    "        \n",
    "        df = df.drop( list_columns , axis = 1)\n",
    "        \n",
    "        return df\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7a13786a-2800-46a8-a8bd-61382a38787f",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:13:58.256800Z",
     "start_time": "2024-03-01T22:13:58.245443Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "# colunas vazias ou irrelevantes\n",
    "colunas_irrelevantes = ['property_code',\n",
    "           'external_reference',\n",
    "           'labels',\n",
    "           'neighborhood',\n",
    "           'new_development_finished',\n",
    "           'highlight',\n",
    "           'exterior',\n",
    "           'super_top_highlight',\n",
    "           'operation',\n",
    "           'country',\n",
    "           'url',\n",
    "           'price_by_area']\n",
    "\n",
    "df1 = delete_columns( df1 , colunas_irrelevantes )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5dadd18b-a7a0-404e-915a-aa6bb919b2ed",
   "metadata": {
    "heading_collapsed": true
   },
   "outputs": [],
   "source": [
    "## 1.5.1 Fillout NA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6cc5cccc-073f-4781-859c-5986b2b2be8e",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:13:58.266665Z",
     "start_time": "2024-03-01T22:13:58.257800Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "def fill_na(df):\n",
    "    # thumbnail - vou substituir anúncios que possuem thumbnail por 1 e aqueles que não por 0\n",
    "    df.loc[:,'thumbnail'] = np.where( df['thumbnail'].isna(), 0 , 1 )\n",
    "\n",
    "    # floor \n",
    "    df.loc[:,'floor'] = df['floor'].apply( lambda x : 0 if pd.isna( x ) else x )\n",
    "\n",
    "    # district - vou fazer uma chamada na api para procurar os districts passando as coordenadas\n",
    "    df.loc[:,'district'] = df.apply(lambda x: find_mun( x['latitude'], x['longitude'] ) if pd.isna( x['district'] ) else x['district'],axis = 1)\n",
    "\n",
    "    # description\n",
    "    df.loc[:,'description'] = df['description'].apply( lambda x : 'no description' if pd.isna( x ) else x )\n",
    "\n",
    "    # has_lift - vou considerar que se não foi informado na base o apto imovel não tem elevador\n",
    "    df.loc[:,'has_lift'] = df['has_lift'].apply( lambda x: 0 if pd.isna( x ) else x  )\n",
    "\n",
    "    # parking_space \n",
    "    df.loc[:,'parking_space'] = df['parking_space'].apply( lambda x: 0 if pd.isna( x ) else x ) \n",
    "    \n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9a8dbdc7-a171-49cb-81ae-9f9cae161250",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:26:51.045632Z",
     "start_time": "2024-03-01T22:13:58.270665Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "df1 = fill_na( df1 )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d3362cb1-04b1-4116-aeab-4fe52771a028",
   "metadata": {
    "heading_collapsed": true
   },
   "outputs": [],
   "source": [
    "## 1.7. Descriptive Statistical"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "538e87f0-6b60-4866-a38d-bbcb397b2d37",
   "metadata": {
    "heading_collapsed": true
   },
   "outputs": [],
   "source": [
    "## 1.7.1 Numerical Attributes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "079c0c72-f9cc-4a84-a309-17492abdcd3b",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:26:51.095649Z",
     "start_time": "2024-03-01T22:26:51.047919Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "df1_stats = stats( df1 , ['latitude','longitude','pagina'] )\n",
    "df1_stats\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cbf28517-4609-48d4-95fd-1299bb5a32ca",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-03-04T23:29:56.937888Z",
     "start_time": "2023-03-04T23:29:56.914814Z"
    },
    "heading_collapsed": true
   },
   "outputs": [],
   "source": [
    "## 1.7.2 Categorical Attributes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "baa89fa7-57f7-4fdd-956f-5071117736b8",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:26:51.349502Z",
     "start_time": "2024-03-01T22:26:51.099046Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "df1['province'] = df1['province'].apply( lambda x : 'Lisboa' if x == 'lisboa' else 'Setúbal' if x == 'setubal' else 'Santarém' if x == 'santarem'  else x)\n",
    "\n",
    "num_categorical = df1.select_dtypes( exclude = ['int64','float64','datetime64[ns]'] )\n",
    "# verifico quantos tipos diferentes tenho nas var categoricas\n",
    "print( num_categorical.apply( lambda x: x.unique().shape[0] ) )\n",
    "\n",
    "print( '='*100 )\n",
    "\n",
    "# verifico quantos tipos diferentes tenho nas var categoricas\n",
    "print( num_categorical.apply( lambda x: x.unique() ) )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "73ada44a-87c9-4e13-a0da-57d2a0cb8265",
   "metadata": {
    "heading_collapsed": true
   },
   "outputs": [],
   "source": [
    "# 2.0 Feature Engineering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c746393b-e1fd-4a2b-9303-e5ac9b87059d",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:26:51.368858Z",
     "start_time": "2024-03-01T22:26:51.351504Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "df2 = df1.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "af8f0abf-a7d8-463f-8e6a-95ebdfbf4b99",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:26:52.236643Z",
     "start_time": "2024-03-01T22:26:51.370860Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "# será usado para médias das regiões\n",
    "df2.to_csv( 'datasets/df_region.csv' )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ebff0440-dc7b-41f0-86f5-53f274ed52a2",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:26:52.254800Z",
     "start_time": "2024-03-01T22:26:52.238154Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "def feature_engineering( df , df_region, transform_data = False, merge_datasets = False ):\n",
    "    \n",
    "    if transform_data and merge_datasets == False :\n",
    "        \n",
    "        # mes\n",
    "        df['datetime_scrapy'] = pd.to_datetime( df['datetime_scrapy'] )\n",
    "        df['month'] = df['datetime_scrapy'].dt.month \n",
    "\n",
    "        # ano\n",
    "        df['year'] = df['datetime_scrapy'].dt.year\n",
    "\n",
    "        # excluir a coluna datetime\n",
    "        df = df.drop( 'datetime_scrapy', axis = 1 )\n",
    "        \n",
    "        print( 'Dados Transformados' )\n",
    "        \n",
    "        return df\n",
    "        \n",
    "    \n",
    "    if merge_datasets == True and transform_data == False:\n",
    "        \n",
    "        # Df com a média de preço por município\n",
    "        df_municipality_mean_price = df_region.groupby('municipality')[['price']].median().reset_index()\n",
    "\n",
    "        # alteração nome coluna\n",
    "        df_municipality_mean_price = df_municipality_mean_price.rename(columns={'price': 'municipality_mean_price'})\n",
    "\n",
    "        # exportar para csv\n",
    "        #d f2_municipality_mean_price.to_csv( 'datasets/municipality.csv' )\n",
    "\n",
    "        # acrescentando a coluna ao df2\n",
    "        df = pd.merge( df, df_municipality_mean_price, on = 'municipality' , how = 'left' )\n",
    "\n",
    "# ======================================================================================================================#\n",
    "\n",
    "        # calculando a média por 'province'\n",
    "        df2_province_mean_price = df_region.groupby( 'province' )[['price']].median().reset_index().rename( columns = { 'price': 'province_mean_price' } )\n",
    "\n",
    "        # exportando para csv\n",
    "        # df2_province_mean_price.to_csv( 'datasets/province.csv' )\n",
    "\n",
    "        # acrescentando a coluna ao df2\n",
    "        df = pd.merge( df, df2_province_mean_price , on = 'province' , how = 'left' )\n",
    "\n",
    "#=======================================================================================================================#\n",
    "\n",
    "        # calculando média por district\n",
    "        df2_district_mean_price = df_region.groupby( 'district' )[['price']].median().reset_index().rename( columns = { 'price' : 'district_mean_price' } )\n",
    "\n",
    "        # exportando para csv \n",
    "        #df2_district_mean_price.to_csv( 'datasets/district.csv' )\n",
    "\n",
    "        # acrescentando a coluna ao df2\n",
    "        df = pd.merge( df , df2_district_mean_price , on = 'district', how = 'left' )\n",
    "\n",
    "        print( 'Datasets combinados com preços regioes' )\n",
    "\n",
    "        return df\n",
    "    \n",
    "    if transform_data == True and merge_datasets == True:\n",
    "        \n",
    "        # mes\n",
    "        df['datetime_scrapy'] = pd.to_datetime( df['datetime_scrapy'] )\n",
    "        df['month'] = df['datetime_scrapy'].dt.month \n",
    "\n",
    "        # ano\n",
    "        df['year'] = df['datetime_scrapy'].dt.year\n",
    "\n",
    "        # excluir a coluna datetime\n",
    "        df = df.drop( 'datetime_scrapy', axis = 1 )\n",
    "        \n",
    "        # Df com a média de preço por município\n",
    "        df_municipality_mean_price = df_region.groupby('municipality')[['price']].mean().reset_index()\n",
    "\n",
    "        # alteração nome coluna\n",
    "        df_municipality_mean_price = df_municipality_mean_price.rename(columns={'price': 'municipality_mean_price'})\n",
    "\n",
    "        # exportar para csv\n",
    "        #d f2_municipality_mean_price.to_csv( 'datasets/municipality.csv' )\n",
    "\n",
    "        # acrescentando a coluna ao df2\n",
    "        df = pd.merge( df, df_municipality_mean_price, on = 'municipality' , how = 'left' )\n",
    "        \n",
    "# ======================================================================================================================#\n",
    "        \n",
    "        # calculando a média por 'province'\n",
    "        df2_province_mean_price = df_region.groupby( 'province' )[['price']].mean().reset_index().rename( columns = { 'price': 'province_mean_price' } )\n",
    "\n",
    "        # exportando para csv\n",
    "        # df2_province_mean_price.to_csv( 'datasets/province.csv' )\n",
    "\n",
    "        # acrescentando a coluna ao df2\n",
    "        df = pd.merge( df, df2_province_mean_price , on = 'province' , how = 'left' )\n",
    "        \n",
    "#=======================================================================================================================#\n",
    "        \n",
    "        # calculando média por district\n",
    "        df2_district_mean_price = df_region.groupby( 'district' )[['price']].median().reset_index().rename( columns = { 'price' : 'district_mean_price' } )\n",
    "\n",
    "        # exportando para csv \n",
    "        #df2_district_mean_price.to_csv( 'datasets/district.csv' )\n",
    "\n",
    "        # acrescentando a coluna ao df2\n",
    "        df = pd.merge( df , df2_district_mean_price , on = 'district', how = 'left' )\n",
    "        \n",
    "        print( 'datasets combinados e dados transformados' )\n",
    "        \n",
    "        return df\n",
    "        \n",
    "    else:\n",
    "        \n",
    "        print( 'Nenhuma Transformação' )\n",
    "        \n",
    "        return df\n",
    "            "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "646dbcba-286e-4b72-b955-b532d16e71d7",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:26:52.907297Z",
     "start_time": "2024-03-01T22:26:52.257350Z"
    },
    "hidden": true,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "df_mean_region = pd.read_csv( 'datasets/df_region.csv' )\n",
    "\n",
    "\n",
    "df2 = feature_engineering( df = df2 ,  \n",
    "                          df_region= df_mean_region,  transform_data= True, merge_datasets=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "db83c257-caa2-4c44-b678-664752921671",
   "metadata": {
    "heading_collapsed": true
   },
   "outputs": [],
   "source": [
    "# 3.0 Filtragem de Variáveis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ef1d5372-066b-454d-ab51-8757a19c2dae",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:26:52.937963Z",
     "start_time": "2024-03-01T22:26:52.914695Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "df3 = df2.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b2e9ee7e-31ac-4d76-8442-b278a15af867",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:26:52.953571Z",
     "start_time": "2024-03-01T22:26:52.940971Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "def filter_variables(df, filter_variables = False):\n",
    "    \n",
    "    if filter_variables:\n",
    "        \n",
    "        # Vou querer saber somente os preços na province de Lisboa\n",
    "        df = df.loc[ df['province'] == 'Lisboa' , :  ]\n",
    "\n",
    "        # vou querer somente as habitações que tenham de 0 a no máximo 4 quartos\n",
    "        df = df.loc[ df['rooms'].isin( [ 0,1,2,3,4 ] ) , : ]\n",
    "        \n",
    "        print('Variavies filtradas')\n",
    "        \n",
    "        return df\n",
    "    \n",
    "    else:\n",
    "        \n",
    "        print('Variavies não filtradas')\n",
    "        \n",
    "        return df\n",
    "\n",
    "        \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c23777fe-efca-4ebf-8947-38755f278a12",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:26:52.970748Z",
     "start_time": "2024-03-01T22:26:52.957454Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "# vou manter filter_variables = False neste ciclo\n",
    "df3 = filter_variables( df3 )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e39e3303-17e7-4739-a653-4827d16253e4",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-08-11T11:58:12.084853Z",
     "start_time": "2023-08-11T11:58:12.079256Z"
    }
   },
   "outputs": [],
   "source": [
    "# 4.0 Análise Exploratória de Dados"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e299579a-11b5-4a87-b374-54d245e33b59",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:26:52.988131Z",
     "start_time": "2024-03-01T22:26:52.974752Z"
    }
   },
   "outputs": [],
   "source": [
    "df4 = df3.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cbf61b6a-f668-46ca-b54f-53136bd7267a",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 4.1. - Analise Univariada"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2f172122-5a73-4ff4-baa5-7cfdf08d21e8",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:47:52.222055Z",
     "start_time": "2024-03-01T22:47:52.034157Z"
    }
   },
   "outputs": [],
   "source": [
    "aux = df4[['year','month','price']].groupby( ['month','year'] ).mean().reset_index().sort_values( 'year', ascending = True )\n",
    "aux['year_month'] = aux['month'].astype( str ) + '-' +  aux['year'].astype( str )\n",
    "sns.lineplot( data = aux , x = 'year_month' , y = 'price' )\n",
    "aux"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "17e70a2a-221d-4e2f-a166-d4651617b158",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:46.272825Z",
     "start_time": "2024-03-01T22:30:46.267727Z"
    }
   },
   "outputs": [],
   "source": [
    "df4.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e20dc7dd-a8e2-43b9-8bc7-4ab37fd4c07d",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:45:23.199066Z",
     "start_time": "2024-03-01T22:45:22.962345Z"
    }
   },
   "outputs": [],
   "source": [
    "aux1 = df4[['year','price']]\n",
    "sns.scatterplot(data = aux1, x = 'year' , y = 'price')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0b490120-aeb7-4987-9f83-d3048e6cbe24",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:26:53.051441Z",
     "start_time": "2024-03-01T22:26:52.993139Z"
    }
   },
   "outputs": [],
   "source": [
    "# verifico as informações estatísticas\n",
    "stats( df4 , ['latitude','longitude'] )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0507eee4-050d-4db4-adce-f47f0363393a",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:26:53.076519Z",
     "start_time": "2024-03-01T22:26:53.062944Z"
    }
   },
   "outputs": [],
   "source": [
    "# removendo o imóvel de tamanho 97500 m2\n",
    "df4 = df4.loc[ df4['size'] != 97500 , : ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8f11ecf7-7f71-48f8-9049-71455ace198f",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:26:53.157029Z",
     "start_time": "2024-03-01T22:26:53.081544Z"
    }
   },
   "outputs": [],
   "source": [
    "# verifico novamente as informações estatísticas\n",
    "stats( df4 , ['latitude','longitude'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5730525e-1ad7-494a-8c54-184352d99107",
   "metadata": {},
   "outputs": [],
   "source": [
    "### 4.1.1 - Response Variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f14e5099-25fa-42e1-af2c-654c7b8c66e1",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:26:55.518704Z",
     "start_time": "2024-03-01T22:26:53.162513Z"
    }
   },
   "outputs": [],
   "source": [
    "# variavael resposta normal\n",
    "sns.displot( df4['price'] )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4cafc568-ec29-484a-b7ca-2be9a60824d0",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:26:55.524943Z",
     "start_time": "2024-03-01T22:26:55.520708Z"
    }
   },
   "outputs": [],
   "source": [
    "# função para remover outliers\n",
    "def remove_outliers(df, coluna = None ,  keep_outliers = True ):\n",
    "    \n",
    "    if keep_outliers:\n",
    "        print( 'Outliers mantidos' )\n",
    "        return df\n",
    "        \n",
    "    else:\n",
    "        # variavel resposta sem outliers\n",
    "        df = remover_outliers( df4 , coluna )\n",
    "        print('Outliers removidos')\n",
    "        return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0c3e4329-fe13-4f6f-b1e3-0f8f0c952650",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:26:55.532985Z",
     "start_time": "2024-03-01T22:26:55.527514Z"
    }
   },
   "outputs": [],
   "source": [
    "# manter os outliers\n",
    "df4 = remove_outliers( df4 )\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "58c61f2e-38a4-4258-8747-2c56cc3ba5b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "### 4.1.2 - Numerical Variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a34a8332-1f1b-433d-a05e-ef2addd954b6",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:26:58.583956Z",
     "start_time": "2024-03-01T22:26:55.534989Z"
    }
   },
   "outputs": [],
   "source": [
    "df4_num_attributes = df4.select_dtypes( include = ['int64','float64'] )\n",
    "df4_num_attributes.hist( bins = 25, figsize=( 15,15 ) );"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "efe41712-f618-48d7-8f90-2f6f2de04d68",
   "metadata": {
    "heading_collapsed": true
   },
   "outputs": [],
   "source": [
    "## 4.2. -  Análise Bivariada"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "30bee0ec-b232-4c62-afeb-2fc6026e0595",
   "metadata": {
    "heading_collapsed": true,
    "hidden": true
   },
   "outputs": [],
   "source": [
    "### 4.2.1 - Correlação entre Atributos Numéricos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "82ecde12-d7a4-4712-81d1-2e8b2f069b08",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:26:59.471260Z",
     "start_time": "2024-03-01T22:26:58.584961Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "plt.figure( figsize = (15,10) )\n",
    "correlation = df4_num_attributes.corr( method = 'pearson' )\n",
    "sns.heatmap( correlation, annot = True );"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7b5cf88c-5f7e-468f-90c2-4e67eceb7f3f",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-08-14T21:04:02.912732Z",
     "start_time": "2023-08-14T21:04:02.909470Z"
    },
    "heading_collapsed": true,
    "hidden": true
   },
   "outputs": [],
   "source": [
    "### 4.2.2 - Correlação entre Atributos Categóricos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f925a525-f8fe-4ba9-80b4-b358ad7df507",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:29:27.498162Z",
     "start_time": "2024-03-01T22:26:59.473309Z"
    },
    "hidden": true,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "a = df4.select_dtypes( include = ['object'] ) # pego o df mais atualizado\n",
    "\n",
    "# aplico a função cramer v nas variáveis categoricas\n",
    "a1 = cramer_v( a['floor'] , a['floor'] )\n",
    "a2 = cramer_v( a['floor'], a['property_type'] )\n",
    "a3 = cramer_v( a['floor'], a['address'] )\n",
    "a4 = cramer_v( a['floor'], a['province'] )\n",
    "a5 = cramer_v( a['floor'], a['municipality'] )\n",
    "a6 = cramer_v( a['floor'], a['district'] )\n",
    "a7 = cramer_v( a['floor'], a['description'] )\n",
    "a8 = cramer_v( a['floor'], a['status'] )\n",
    "a9 = cramer_v( a['floor'], a['detailed_type'] )\n",
    "a10 = cramer_v( a['floor'], a['suggested_texts'] )\n",
    "a11 = cramer_v( a['floor'], a['parking_space'] )\n",
    "\n",
    "# aplico a função cramer v nas variáveis categoricas\n",
    "a12= cramer_v( a['property_type'], a['floor'] )\n",
    "a13= cramer_v( a['property_type'], a['property_type'] )\n",
    "a14= cramer_v( a['property_type'], a['address'] )\n",
    "a15= cramer_v( a['property_type'], a['province'] )\n",
    "a16= cramer_v( a['property_type'], a['municipality'] )\n",
    "a17= cramer_v( a['property_type'], a['district'] )\n",
    "a18= cramer_v( a['property_type'], a['description'] )\n",
    "a19= cramer_v( a['property_type'], a['status'] )\n",
    "a20= cramer_v( a['property_type'], a['detailed_type'] )\n",
    "a21=cramer_v(  a['property_type'], a['suggested_texts'] )\n",
    "a22=cramer_v(  a['property_type'], a['parking_space'] )\n",
    "\n",
    "# aplico a função cramer v nas variáveis categoricas\n",
    "a23= cramer_v( a['address'], a['floor'] )\n",
    "a24= cramer_v( a['address'], a['property_type'] )\n",
    "a25= cramer_v( a['address'], a['address'] )\n",
    "a26= cramer_v( a['address'], a['province'] )\n",
    "a27= cramer_v( a['address'], a['municipality'] )\n",
    "a28= cramer_v( a['address'], a['district'] )\n",
    "a29= cramer_v( a['address'], a['description'] )\n",
    "a30= cramer_v( a['address'], a['status'] )\n",
    "a31= cramer_v( a['address'], a['detailed_type'] )\n",
    "a32=cramer_v(  a['address'], a['suggested_texts'] )\n",
    "a33=cramer_v(  a['address'], a['parking_space'] )\n",
    "\n",
    "# aplico a função cramer v nas variáveis categoricas\n",
    "a34= cramer_v( a['province'], a['floor'] )\n",
    "a35= cramer_v( a['province'], a['property_type'] )\n",
    "a36= cramer_v( a['province'], a['address'] )\n",
    "a37= cramer_v( a['province'], a['province'] )\n",
    "a38= cramer_v( a['province'], a['municipality'] )\n",
    "a39= cramer_v( a['province'], a['district'] )\n",
    "a40= cramer_v( a['province'], a['description'] )\n",
    "a41= cramer_v( a['province'], a['status'] )\n",
    "a42= cramer_v( a['province'], a['detailed_type'] )\n",
    "a43=cramer_v(  a['province'], a['suggested_texts'] )\n",
    "a44=cramer_v(  a['province'], a['parking_space'] )\n",
    "\n",
    "# aplico a função cramer v nas variáveis categoricas\n",
    "a45= cramer_v( a['municipality'], a['floor'] )\n",
    "a46= cramer_v( a['municipality'], a['property_type'] )\n",
    "a47= cramer_v( a['municipality'], a['address'] )\n",
    "a48= cramer_v( a['municipality'], a['province'] )\n",
    "a49= cramer_v( a['municipality'], a['municipality'] )\n",
    "a50= cramer_v( a['municipality'], a['district'] )\n",
    "a51= cramer_v( a['municipality'], a['description'] )\n",
    "a52= cramer_v( a['municipality'], a['status'] )\n",
    "a53= cramer_v( a['municipality'], a['detailed_type'] )\n",
    "a54=cramer_v(  a['municipality'], a['suggested_texts'] )\n",
    "a55=cramer_v(  a['municipality'], a['parking_space'] )\n",
    "\n",
    "# aplico a função cramer v nas variáveis categoricas\n",
    "a56= cramer_v( a['district'], a['floor'] )\n",
    "a57= cramer_v( a['district'], a['property_type'] )\n",
    "a58= cramer_v( a['district'], a['address'] )\n",
    "a59= cramer_v( a['district'], a['province'] )\n",
    "a60= cramer_v( a['district'], a['municipality'] )\n",
    "a61= cramer_v( a['district'], a['district'] )\n",
    "a62= cramer_v( a['district'], a['description'] )\n",
    "a63= cramer_v( a['district'], a['status'] )\n",
    "a64= cramer_v( a['district'], a['detailed_type'] )\n",
    "a65=cramer_v(  a['district'], a['suggested_texts'] )\n",
    "a66=cramer_v(  a['district'], a['parking_space'] )\n",
    "\n",
    "# aplico a função cramer v nas variáveis categoricas\n",
    "a67= cramer_v( a['description'], a['floor'] )\n",
    "a68= cramer_v( a['description'], a['property_type'] )\n",
    "a69= cramer_v( a['description'], a['address'] )\n",
    "a70= cramer_v( a['description'], a['province'] )\n",
    "a71= cramer_v( a['description'], a['municipality'] )\n",
    "a72= cramer_v( a['description'], a['district'] )\n",
    "a73= cramer_v( a['description'], a['description'] )\n",
    "a74= cramer_v( a['description'], a['status'] )\n",
    "a75= cramer_v( a['description'], a['detailed_type'] )\n",
    "a76=cramer_v(  a['description'], a['suggested_texts'] )\n",
    "a77=cramer_v(  a['description'], a['parking_space'] )\n",
    "\n",
    "# aplico a função cramer v nas variáveis categoricas\n",
    "a78= cramer_v( a['status'], a['floor'] )\n",
    "a79= cramer_v( a['status'], a['property_type'] )\n",
    "a80= cramer_v( a['status'], a['address'] )\n",
    "a81= cramer_v( a['status'], a['province'] )\n",
    "a82= cramer_v( a['status'], a['municipality'] )\n",
    "a83= cramer_v( a['status'], a['district'] )\n",
    "a84= cramer_v( a['status'], a['description'] )\n",
    "a85= cramer_v( a['status'], a['status'] )\n",
    "a86= cramer_v( a['status'], a['detailed_type'] )\n",
    "a87=cramer_v(  a['status'], a['suggested_texts'] )\n",
    "a88=cramer_v(  a['status'], a['parking_space'] )\n",
    "\n",
    "# aplico a função cramer v nas variáveis categoricas\n",
    "a89= cramer_v( a['detailed_type'], a['floor'] )\n",
    "a90= cramer_v( a['detailed_type'], a['property_type'] )\n",
    "a91= cramer_v( a['detailed_type'], a['address'] )\n",
    "a92= cramer_v( a['detailed_type'], a['province'] )\n",
    "a93= cramer_v( a['detailed_type'], a['municipality'] )\n",
    "a94= cramer_v( a['detailed_type'], a['district'] )\n",
    "a95= cramer_v( a['detailed_type'], a['description'] )\n",
    "a96= cramer_v( a['detailed_type'], a['status'] )\n",
    "a97= cramer_v( a['detailed_type'], a['detailed_type'] )\n",
    "a98=cramer_v(  a['detailed_type'], a['suggested_texts'] )\n",
    "a99=cramer_v(  a['detailed_type'], a['parking_space'] )\n",
    "\n",
    "# aplico a função cramer v nas variáveis categoricas\n",
    "a100= cramer_v( a['suggested_texts'], a['floor'] )\n",
    "a101= cramer_v( a['suggested_texts'], a['property_type'] )\n",
    "a102= cramer_v( a['suggested_texts'], a['address'] )\n",
    "a103= cramer_v( a['suggested_texts'], a['province'] )\n",
    "a104= cramer_v( a['suggested_texts'], a['municipality'] )\n",
    "a105= cramer_v( a['suggested_texts'], a['district'] )\n",
    "a106= cramer_v( a['suggested_texts'], a['description'] )\n",
    "a107= cramer_v( a['suggested_texts'], a['status'] )\n",
    "a108= cramer_v( a['suggested_texts'], a['detailed_type'] )\n",
    "a109=cramer_v(  a['suggested_texts'], a['suggested_texts'] )\n",
    "a110=cramer_v(  a['suggested_texts'], a['parking_space'] )\n",
    "\n",
    "# aplico a função cramer v nas variáveis categoricas\n",
    "a111= cramer_v( a['parking_space'], a['floor'] )\n",
    "a112= cramer_v( a['parking_space'], a['property_type'] )\n",
    "a113= cramer_v( a['parking_space'], a['address'] )\n",
    "a114= cramer_v( a['parking_space'], a['province'] )\n",
    "a115= cramer_v( a['parking_space'], a['municipality'] )\n",
    "a116= cramer_v( a['parking_space'], a['district'] )\n",
    "a117= cramer_v( a['parking_space'], a['description'] )\n",
    "a118= cramer_v( a['parking_space'], a['status'] )\n",
    "a119= cramer_v( a['parking_space'], a['detailed_type'] )\n",
    "a120=cramer_v(  a['parking_space'], a['suggested_texts'] )\n",
    "a121=cramer_v(  a['parking_space'], a['parking_space'] )\n",
    "\n",
    "\n",
    "\n",
    "d = pd.DataFrame( {'floor' : [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11],\n",
    "                   'property_type' : [a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22],\n",
    "                   'address' : [a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,a33],\n",
    "                   'province' : [a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44],\n",
    "                   'municipality' : [a45,a46,a47,a48,a49,a50,a51,a52,a53,a54,a55],\n",
    "                   'district' : [a56,a57,a58,a59,a60,a61,a62,a63,a64,a65,a66],\n",
    "                   'description' : [a67,a68,a69,a70,a71,a72,a73,a74,a75,a76,a77],\n",
    "                   'status' : [a78,a79,a80,a81,a82,a83,a84,a85,a86,a87,a88],\n",
    "                   'detailed_type' : [a89,a90,a91,a92,a93,a94,a95,a96,a97,a98,a99],\n",
    "                   'suggested_texts' : [a100,a101,a102,a103,a104,a105,a106,a107,a108,a109,a110],\n",
    "                   'parking_space' : [a111,a112,a113,a114,a115,a116,a117,a118,a119,a120,a121]\n",
    "              \n",
    "              } )\n",
    "\n",
    "d = d.set_index( d.columns )\n",
    "d = d.drop( 'province', axis = 1 )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e862b61d-fdd1-4706-972d-74f4122ea444",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:29:28.165496Z",
     "start_time": "2024-03-01T22:29:27.500168Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10,5))\n",
    "correlation = d.corr( method = 'pearson' )\n",
    "sns.heatmap( correlation, annot = True );"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f182ccc3-7057-4e21-80cb-658a695438f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Data Preparation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7916b559-8298-4ded-9785-e8512a1e120d",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:29:28.194607Z",
     "start_time": "2024-03-01T22:29:28.170116Z"
    }
   },
   "outputs": [],
   "source": [
    "# dividir entre treino e teste\n",
    "X = df4.drop( 'price', axis = 1 )\n",
    "y = df4['price']\n",
    "\n",
    "X_train_n, X_test_n , y_train_n , y_test_n = train_test_split(X, y, test_size=0.2, random_state=39)\n",
    "\n",
    "print('Total Shape {}'.format( df4.shape ))\n",
    "\n",
    "print('X_train shape {}'.format(X_train_n.shape) )\n",
    "print('y_train shape {}'.format(y_train_n.shape) )\n",
    "\n",
    "print('=='*50)\n",
    "\n",
    "print('X_test shape {}'.format(X_test_n.shape) )\n",
    "print('y_test shape {}'.format(y_test_n.shape) )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ecb96771-724d-4735-9b4e-a32188e575b9",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:29:28.210682Z",
     "start_time": "2024-03-01T22:29:28.199183Z"
    }
   },
   "outputs": [],
   "source": [
    "# df treino\n",
    "df_treino = pd.concat( [X_train_n,y_train_n ], axis = 1 )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f925237e-673a-4d37-a6bc-34abcc0f5d1d",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:29:28.223369Z",
     "start_time": "2024-03-01T22:29:28.213729Z"
    }
   },
   "outputs": [],
   "source": [
    "# df teste\n",
    "df_teste = pd.concat( [X_test_n,y_test_n] , axis = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "460d4b12-e1e5-4f41-82f9-284d9a95174e",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:29:28.268917Z",
     "start_time": "2024-03-01T22:29:28.225884Z"
    }
   },
   "outputs": [],
   "source": [
    "df5 = df_treino.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eb2be1d8-b6d3-4f68-b826-6ce3c495f86e",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:29:28.359808Z",
     "start_time": "2024-03-01T22:29:28.276415Z"
    }
   },
   "outputs": [],
   "source": [
    "stats( df5 )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a54d92dc-39ca-4d5d-bcf0-f7b6ad15740d",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:29:28.969660Z",
     "start_time": "2024-03-01T22:29:28.369723Z"
    }
   },
   "outputs": [],
   "source": [
    "sns.displot( df5['price'] )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4a028701-966d-4a58-895c-835e57355bfb",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 5.1. - Rescaling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9cb4e8f4-e0ef-42b7-9d94-ea02e0ba7ccd",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 5.1.2 Check Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4c0038e9-8e4c-4324-9a35-9cee4fceaabe",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:29:29.580826Z",
     "start_time": "2024-03-01T22:29:28.971703Z"
    }
   },
   "outputs": [],
   "source": [
    "plt.subplot(1, 2, 1)\n",
    "# num_photos\n",
    "sns.boxplot( df5['num_photos'] )\n",
    "plt.title( 'num_photos' )\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "# num_photos\n",
    "sns.distplot( df5['num_photos']  )\n",
    "plt.title( 'num_photos' )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dc7bcb1d-260b-4793-adad-4304658df44a",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:29:30.023087Z",
     "start_time": "2024-03-01T22:29:29.582829Z"
    }
   },
   "outputs": [],
   "source": [
    "plt.subplot(1, 2, 1)\n",
    "# size\n",
    "sns.boxplot( df5['size'] )\n",
    "plt.title( 'size' )\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "# size\n",
    "sns.distplot( df5['size']  )\n",
    "plt.title( 'size' )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9bdf1c79-9423-468d-97fd-a27e991d7b4e",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:29:30.468187Z",
     "start_time": "2024-03-01T22:29:30.025090Z"
    }
   },
   "outputs": [],
   "source": [
    "plt.subplot(1, 2, 1)\n",
    "# rooms\n",
    "sns.boxplot( df5['rooms'] )\n",
    "plt.title( 'rooms' )\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "# rooms\n",
    "sns.distplot( df5['rooms']  )\n",
    "plt.title( 'rooms' )\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5eac18a2-9918-40b4-989d-cad495519ffb",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:29:31.000728Z",
     "start_time": "2024-03-01T22:29:30.470196Z"
    }
   },
   "outputs": [],
   "source": [
    "plt.subplot(1, 2, 1)\n",
    "# bathrooms\n",
    "sns.boxplot( df5['bathrooms'] )\n",
    "plt.title( 'bathrooms' )\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "# rooms\n",
    "sns.distplot( df5['bathrooms']  )\n",
    "plt.title( 'bathrooms' )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9c4876e4-4be9-4480-b0a3-6cf2df6ea07b",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:29:31.776791Z",
     "start_time": "2024-03-01T22:29:31.009758Z"
    }
   },
   "outputs": [],
   "source": [
    "plt.subplot(1, 2, 1)\n",
    "# distance\n",
    "sns.boxplot( df5['distance'] )\n",
    "plt.title( 'distance' )\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "# rooms\n",
    "sns.distplot( df5['distance']  )\n",
    "plt.title( 'distance' )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "43b44a3d-6b39-4a04-9c9e-694aedbd1ef4",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:29:31.791944Z",
     "start_time": "2024-03-01T22:29:31.778864Z"
    }
   },
   "outputs": [],
   "source": [
    "def rescaling_data(df, rescale = False ):\n",
    "    \n",
    "    if rescale:\n",
    "        rs = RobustScaler() # robusto com outliers\n",
    "        mms = pp.MinMaxScaler() # desvio padrão pequeno e quando não ha distribuição gaussiana\n",
    "        \n",
    "#         # num_photos\n",
    "#         rs_num_photos = RobustScaler() \n",
    "#         # num_photos\n",
    "#         df['num_photos'] = rs_num_photos.fit_transform( df[['num_photos']].values )\n",
    "\n",
    "        # floor\n",
    "#         rs_floor = RobustScaler() \n",
    "#         # num_photos\n",
    "#         df['floor'] = rs_floor.fit_transform( df[['floor']].values )\n",
    "        \n",
    "        # size\n",
    "        rs_size = RobustScaler() \n",
    "        # size\n",
    "        df['size'] = rs_size.fit_transform( df[['size']].values )\n",
    "        \n",
    "        # rooms\n",
    "#         mms_rooms = pp.MinMaxScaler() \n",
    "#         # rooms\n",
    "#         df['rooms'] = mms_rooms.fit_transform( df[['rooms']].values )\n",
    "        \n",
    "        # bathrooms\n",
    "#         rs_bathrooms = RobustScaler() \n",
    "#         # bathrooms\n",
    "#         df['bathrooms'] = rs_bathrooms.fit_transform( df[['bathrooms']].values )\n",
    "        \n",
    "        # distance\n",
    "        rs_distance = RobustScaler()\n",
    "        # distance\n",
    "        df['distance'] = rs_distance.fit_transform( df[['distance']].values )\n",
    "        \n",
    "        \n",
    "        \n",
    "        print('Rescaled')\n",
    "        \n",
    "        return  df\n",
    "    else:\n",
    "        print('Not rescaled')\n",
    "        \n",
    "        return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "01ef38c1-6f65-431f-a402-8be8d8974ec1",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:29:31.832000Z",
     "start_time": "2024-03-01T22:29:31.793950Z"
    }
   },
   "outputs": [],
   "source": [
    "df5 = rescaling_data( df5, rescale = True  )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9782a9e1-e149-4862-bd56-911d7ff80d63",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:29:31.895067Z",
     "start_time": "2024-03-01T22:29:31.841544Z"
    }
   },
   "outputs": [],
   "source": [
    "df5.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "89fdb9a2-8991-4fdb-ae55-c4bb7249fb79",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 5.2 Encoding "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3547f48c-4ca2-447b-96a3-4dd7d2520c14",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:29:31.925635Z",
     "start_time": "2024-03-01T22:29:31.897138Z"
    }
   },
   "outputs": [],
   "source": [
    "def encoding( df ):\n",
    "    \n",
    "    # floor - substituo os andares com letras\n",
    "    df['floor'] = df['floor'].replace( ['bj','st','ss','en'], 0 )\n",
    "    # transformo em inteiros\n",
    "    df['floor'] = df['floor'].astype( 'int64' )\n",
    "\n",
    "    #property_type - label\n",
    "    te_property_type = {'flat' : 0, 'duplex' : 1, 'chalet' : 2, 'studio': 3, 'penthouse': 4, 'countryHouse':5}\n",
    "    #te_property_type = df.groupby( 'property_type' )['price'].mean()\n",
    "    df['property_type'] = df['property_type'].map( te_property_type )\n",
    "    pickle.dump(te_property_type,open(r'C:\\Users\\oscar\\Documents\\repos\\api_houses_Lisbon\\encodings\\te_property_type.pkl', 'wb' ) )\n",
    "    \n",
    "    # address - label\n",
    "    freq_encoding_address = df['address'].value_counts(normalize=True)\n",
    "    df['address'] = df['address'].map( freq_encoding_address )\n",
    "    pickle.dump( freq_encoding_address, open(r'C:\\Users\\oscar\\Documents\\repos\\api_houses_Lisbon\\encodings\\freq_encoding_address.pkl', 'wb') )\n",
    "                \n",
    "    # province \n",
    "    te_province = df['province'].value_counts( )\n",
    "    #te_province = df.groupby( 'province' )['price'].mean()\n",
    "    df['province'] = df['province'].map( te_province )\n",
    "    pickle.dump( te_province, open( r'C:\\Users\\oscar\\Documents\\repos\\api_houses_Lisbon\\encodings\\te_province.pkl', 'wb' ) )\n",
    "                \n",
    "    # municipality - label\n",
    "    te_municipality = df['municipality'].value_counts()\n",
    "    #te_municipality = df.groupby( 'municipality' )['price'].mean()\n",
    "    df['municipality'] = df['municipality'].map( te_municipality ).astype( 'float64' )\n",
    "    pickle.dump( te_municipality, open( r'C:\\Users\\oscar\\Documents\\repos\\api_houses_Lisbon\\encodings\\te_municipality.pkl', 'wb' ) )            \n",
    "    \n",
    "    # municipality\n",
    "#     freq_encoding_municipality = df['municipality'].value_counts(normalize=True)\n",
    "#     df['municipality'] = df['municipality'].map( freq_encoding_municipality )\n",
    "    \n",
    "\n",
    "    # district - label\n",
    "    te_district = df['district'].value_counts()\n",
    "    #te_district = df.groupby( 'district' )['price'].mean()\n",
    "    df['district'] = df['district'].map( te_district ).astype( 'float64' )\n",
    "    pickle.dump( te_district, open( r'C:\\Users\\oscar\\Documents\\repos\\api_houses_Lisbon\\encodings\\te_district.pkl', 'wb' ) )  \n",
    "    # district\n",
    "#     freq_encoding_district = df['district'].value_counts(normalize=True)\n",
    "#     df['district'] = df['district'].map( freq_encoding_district )\n",
    "    \n",
    "    \n",
    "    # Show address\n",
    "    encoding = {True: 1 , False: 0}\n",
    "    df['show_address'] = df['show_address'].apply( lambda x : 1 if x == True else 0 )\n",
    "\n",
    "    # description\n",
    "    df['description'] = df['description'].apply( lambda x : len(x) )\n",
    "\n",
    "    # has_video\n",
    "    encoding = {True: 1 , False: 0}\n",
    "    df['has_video'] = df['has_video'].map( encoding )\n",
    "\n",
    "    # status\n",
    "    te_status = {'good' : 1, 'newdevelopment' : 2 , 'renew' : 3}\n",
    "    #te_status = df.groupby( 'status' )['price'].mean()\n",
    "    df['status'] = df['status'].map( te_status ).astype( 'float64' )\n",
    "    pickle.dump( te_status, open( r'C:\\Users\\oscar\\Documents\\repos\\api_houses_Lisbon\\encodings\\te_status.pkl', 'wb' ) )  \n",
    "    \n",
    "    # label encoder status\n",
    "    #status_encoder = LabelEncoder()\n",
    "    #df['status'] = status_encoder.fit_transform( df['status'] )\n",
    "    \n",
    "    # new_development\n",
    "    encoding = {True: 1 , False: 0}\n",
    "    df['new_development'] = df['new_development'].map( encoding )\n",
    "\n",
    "    # detailed type - label\n",
    "    le_detailed_type = LabelEncoder()\n",
    "    \n",
    "    df['detailed_type'] = df['detailed_type'].apply(lambda x: ast.literal_eval(x) )\n",
    "    df['detailed_type'] = df['detailed_type'].apply(lambda x: x['typology'] )\n",
    "    df['detailed_type'] = le_detailed_type.fit_transform( df['detailed_type'] )\n",
    "    pickle.dump( le_detailed_type, open( r'C:\\Users\\oscar\\Documents\\repos\\api_houses_Lisbon\\encodings\\le_detailed_type.pkl', 'wb' ) ) \n",
    "\n",
    "    # suggested text\n",
    "    df['suggested_texts'] = df['suggested_texts'].apply( lambda x : ast.literal_eval(x) )\n",
    "    df['suggested_texts'] = df['suggested_texts'].apply( lambda x : x['title'] )\n",
    "    df['suggested_texts'] = df['suggested_texts'].apply( lambda x : len(x) )\n",
    "\n",
    "    # hasplan\n",
    "    encoding = {True: 1 , False: 0}\n",
    "    df['has_plan'] = df['has_plan'].map( encoding )\n",
    "\n",
    "    # has3_d_tour\n",
    "    encoding = {True: 1 , False: 0}\n",
    "    df['has3_d_tour'] = df['has3_d_tour'].map( encoding )\n",
    "\n",
    "    # has360\n",
    "    encoding = {True: 1 , False: 0}\n",
    "    df['has360'] = df['has360'].map( encoding )\n",
    "\n",
    "    # has_staging\n",
    "    df['has_staging'] = df['has_staging'].apply( lambda x: 1 if x   else 0   )\n",
    "\n",
    "    # top_new_development\n",
    "    df['top_new_development'] = df['top_new_development'].apply( lambda x: 1 if x  else 0 )\n",
    "    \n",
    "    #parking_space\n",
    "    encoding = {True: 1 , False: 0}\n",
    "    df['parking_space'] = df['parking_space'].apply(lambda x: ast.literal_eval(x) if x != 0 else x ) \n",
    "    df['parking_space'] = df['parking_space'].apply(lambda x: x['hasParkingSpace'] if x != 0 else x ) \n",
    "    df['parking_space'] = df['parking_space'].map( encoding )\n",
    "\n",
    "    \n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8fb6749e-6bf6-44f8-98c5-658a1c93901f",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:29:32.775176Z",
     "start_time": "2024-03-01T22:29:31.927620Z"
    }
   },
   "outputs": [],
   "source": [
    "df5 = encoding( df5 )\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5eed2893-29e6-4ebf-af09-ebf9bf61d2c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 6.0 Feature Selection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "326b0fb9-e99c-46ef-9a28-d33d5702ca01",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:29:32.786602Z",
     "start_time": "2024-03-01T22:29:32.777276Z"
    }
   },
   "outputs": [],
   "source": [
    "df6 = df5.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8c1bf707-7b7a-4b18-8db6-3a7ab4646ed7",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-08-13T00:21:22.012478Z",
     "start_time": "2023-08-13T00:21:22.009576Z"
    }
   },
   "outputs": [],
   "source": [
    "## 6.1 Boruta"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5ddc7af6-a77f-4d4f-b479-8dc629945dae",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:29:32.813048Z",
     "start_time": "2024-03-01T22:29:32.792625Z"
    }
   },
   "outputs": [],
   "source": [
    "def boruta_selection(df , use_boruta = False):\n",
    "    \n",
    "    if use_boruta:\n",
    "    \n",
    "        # training and test dataset for boruta\n",
    "        X_train = df6.drop( ['price'] , axis= 1 ).values\n",
    "        y_train = df6['price'].values.ravel()\n",
    "\n",
    "        # instancia do Rf\n",
    "        rf = RandomForestRegressor( n_jobs = -1 )\n",
    "\n",
    "        # aplico o boruta\n",
    "        boruta = BorutaPy( rf, n_estimators = 'auto', verbose = 2, random_state = 42 ).fit( X_train, y_train )\n",
    "        \n",
    "        # pega as colunas e coloca na variavel\n",
    "        cols_selected = boruta.support_.tolist()\n",
    "\n",
    "        # df boruta\n",
    "        df_boruta = df6.drop('price', axis = 1).loc[ : , cols_selected]\n",
    "\n",
    "        X_train = df6.drop( ['price'] , axis= 1 )\n",
    "        cols_selected_boruta = X_train.loc[ : , cols_selected ]\n",
    "\n",
    "        # cols not selected boruta\n",
    "        cols_not_selected = list( np.setdiff1d( X_train.columns, cols_selected_boruta.columns ) )\n",
    "        \n",
    "        print('Boruta Used')\n",
    "        \n",
    "        return cols_selected_boruta , cols_not_selected\n",
    "    \n",
    "    else:\n",
    "        print('Boruta skipped')\n",
    "        return df\n",
    "\n",
    "        \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0298ff0b-c764-4c9b-a808-6d8f9dcf3a37",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.222817Z",
     "start_time": "2024-03-01T22:29:32.815047Z"
    }
   },
   "outputs": [],
   "source": [
    "df6, cols_not_sel = boruta_selection(df6, use_boruta = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a2f52558-8cb5-461c-a386-9b5d0a3c386c",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.228892Z",
     "start_time": "2024-03-01T22:30:35.228892Z"
    }
   },
   "outputs": [],
   "source": [
    "df5.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5a259278-ff56-4b7e-ba22-2b1255575e8d",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.230915Z",
     "start_time": "2024-03-01T22:30:35.230915Z"
    }
   },
   "outputs": [],
   "source": [
    "df6 =df5[[ 'num_photos', 'floor', 'property_type', 'size', 'rooms',\n",
    "       'bathrooms', 'province', 'municipality', 'district',\n",
    "       'latitude', 'longitude', 'show_address', 'distance', 'description',\n",
    "        'status','month', 'year',\n",
    "       'municipality_mean_price', 'province_mean_price', 'district_mean_price',\n",
    "       'price']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c09e5396-9815-4d41-a0f4-2e1b40c41627",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.232874Z",
     "start_time": "2024-03-01T22:30:35.232874Z"
    }
   },
   "outputs": [],
   "source": [
    "cols = [ 'num_photos', 'floor', 'property_type', 'size', 'rooms',\n",
    "       'bathrooms', 'province', 'municipality', 'district',\n",
    "       'latitude', 'longitude', 'show_address', 'distance', 'description',\n",
    "        'status','month', 'year',\n",
    "       'municipality_mean_price', 'province_mean_price', 'district_mean_price']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3e134f10-9d88-4db6-a1bc-7c6b723a8302",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-08-11T16:55:19.379220Z",
     "start_time": "2023-08-11T16:55:19.373792Z"
    }
   },
   "outputs": [],
   "source": [
    "# 7.0 Machine Learning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f3a07f3e-01c7-4bc4-bd94-5fade31e09c5",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.234868Z",
     "start_time": "2024-03-01T22:30:35.234868Z"
    }
   },
   "outputs": [],
   "source": [
    "# columns used for training\n",
    "colunas_treino_modelo = df6.columns.to_list()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "719c3557-1d9a-4d6d-bd26-d3fe7f239321",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.238170Z",
     "start_time": "2024-03-01T22:30:35.238170Z"
    }
   },
   "outputs": [],
   "source": [
    "df7 = df6.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "af6995f4-03a2-4872-a1da-a7a3aa98f897",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 7.1 - Divisão entre treino e teste"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6182bb12-874b-4803-a28d-347455307aa1",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.239173Z",
     "start_time": "2024-03-01T22:30:35.239173Z"
    }
   },
   "outputs": [],
   "source": [
    "# treino\n",
    "X_train = df7.drop( 'price', axis = 1 )\n",
    "y_train = df7['price']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3162d214-e266-4d9e-aa2a-daa581891442",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.242165Z",
     "start_time": "2024-03-01T22:30:35.242165Z"
    }
   },
   "outputs": [],
   "source": [
    "# df_teste = df_teste[['size', 'bathrooms', 'address', 'municipality', 'latitude', 'longitude',\n",
    "#        'distance', 'description', 'municipality_mean_price',\n",
    "#        'district_mean_price','price']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2b82226c-79a9-42db-ba2a-ec67fd8916ab",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.243161Z",
     "start_time": "2024-03-01T22:30:35.243161Z"
    }
   },
   "outputs": [],
   "source": [
    "# tratando os dados teste\n",
    "idealista = Idealista()\n",
    "\n",
    "df_teste_0 = idealista.rescaling_data( df_teste , rescale = True )\n",
    "df_teste_1 = idealista.encoding( df_teste_0 )\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4de8906d-0374-4587-880e-e628226f284e",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.245668Z",
     "start_time": "2024-03-01T22:30:35.245668Z"
    }
   },
   "outputs": [],
   "source": [
    "# separando os dados de teste\n",
    "X_test = df_teste_1.drop('price', axis = 1 )\n",
    "y_test = df_teste_1['price']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5e50f138-af0b-416b-8a41-b562ac333e12",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 7.2 - Linear Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6b63a878-daec-4a75-b635-49f0165bce96",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.246686Z",
     "start_time": "2024-03-01T22:30:35.246686Z"
    }
   },
   "outputs": [],
   "source": [
    "# Criar um modelo de regressão linear\n",
    "lr = LinearRegression()\n",
    "\n",
    "# Treinar o modelo\n",
    "lr.fit(X_train, y_train)\n",
    "\n",
    "# Fazer previsões no conjunto de teste\n",
    "y_pred = lr.predict( X_test[cols] )\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "057227b8-9087-44bf-af87-e777029e05e9",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.248998Z",
     "start_time": "2024-03-01T22:30:35.248998Z"
    }
   },
   "outputs": [],
   "source": [
    "lr_metrics = ml_error( 'Linear Regression', y_test   , y_pred )\n",
    "lr_metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2489f2fc-245f-460c-a074-c773484012e3",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 7.2.1 - Linear Regression Cross Validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "85d814d2-164f-464a-bcac-9bebd3754739",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.252024Z",
     "start_time": "2024-03-01T22:30:35.252024Z"
    }
   },
   "outputs": [],
   "source": [
    "# cross validation\n",
    "precision_scores = cross_val_score(lr, X_train, y_train, cv=100, scoring= 'r2')\n",
    "\n",
    "# mean score \n",
    "mean_score = np.mean( precision_scores )\n",
    "cv_lr = np.round(mean_score,2)\n",
    "lr_metrics_cv = pd.concat( [lr_metrics, pd.DataFrame( {'Cv Score' : [cv_lr]} ) ], axis = 1)\n",
    "lr_metrics_cv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "063e443d-2584-4726-9f04-6cc5bcdbf14a",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 7.3 Lassso"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "629f6c79-ab07-46d0-aa7b-6dbb6680ead4",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.253987Z",
     "start_time": "2024-03-01T22:30:35.253987Z"
    }
   },
   "outputs": [],
   "source": [
    "# model\n",
    "lrr = Lasso( alpha = 0.01 ).fit( X_train , y_train )\n",
    "\n",
    "# prediction\n",
    "y_hat_lrr = lrr.predict( X_test[cols] )\n",
    "\n",
    "# performance\n",
    "lrr_result = ml_error( 'Lasso',  y_test  ,  y_hat_lrr  )\n",
    "lrr_result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5c5d3a7e-a7e3-44ba-ba48-42e931dddf63",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-02-03T14:44:22.755594Z",
     "start_time": "2024-02-03T14:44:22.752525Z"
    }
   },
   "outputs": [],
   "source": [
    "## 7.3.1 - Lasso Cross Validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "77dd84ce-82ed-49ce-8bd1-46a8c43bef6b",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.255979Z",
     "start_time": "2024-03-01T22:30:35.255979Z"
    },
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# cross validation\n",
    "precision_scores = cross_val_score(lrr, X_train, y_train, cv=20, scoring= 'r2')\n",
    "\n",
    "# mean score \n",
    "mean_score = np.mean( precision_scores )\n",
    "cv_lasso = np.round( mean_score, 2)\n",
    "lrr_result_cv = pd.concat( [lrr_result, pd.DataFrame( {'Cv Score' : [cv_lasso]} ) ], axis = 1) \n",
    "lrr_result_cv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5454b07e-e690-47d9-8a2e-c8ef1ac377e3",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 7.4 Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e88034bf-4f06-4e05-956b-3056a20ec646",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.259062Z",
     "start_time": "2024-03-01T22:30:35.259062Z"
    }
   },
   "outputs": [],
   "source": [
    "# model\n",
    "rf = RandomForestRegressor( n_estimators = 100 , n_jobs = -1, random_state = 42  )\n",
    "\n",
    "rf.fit( X_train , y_train )\n",
    "\n",
    "# prediction\n",
    "y_hat_rf = rf.predict(  X_test[cols] )\n",
    "\n",
    "# \n",
    "rf_result = ml_error( 'Random Forest Regressor',  y_test  ,  y_hat_rf  )\n",
    "rf_result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8464eb08-3d5e-487c-ab5f-f0581579be44",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-08-13T17:51:19.283637Z",
     "start_time": "2023-08-13T17:51:19.280463Z"
    }
   },
   "outputs": [],
   "source": [
    "## 7.4 Random Forest Cross Validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0aa1a681-bfed-430d-95b9-90c7d8a1a952",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.264043Z",
     "start_time": "2024-03-01T22:30:35.264043Z"
    }
   },
   "outputs": [],
   "source": [
    "# cross validation\n",
    "precision_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring= 'neg_mean_absolute_error')\n",
    "\n",
    "# mean score \n",
    "mean_score = np.mean( precision_scores )\n",
    "rf_cv = np.round( mean_score, 2)\n",
    "rf_result_cv = pd.concat( [rf_result, pd.DataFrame( {'Cv Score' : [rf_cv]} ) ], axis = 1) \n",
    "rf_result_cv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "28993d7d-dfdc-46b6-bf45-df723272807f",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 7.5 LGBM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3a4ce14f-5561-495b-b82b-0351f2933bf1",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.266038Z",
     "start_time": "2024-03-01T22:30:35.265041Z"
    }
   },
   "outputs": [],
   "source": [
    "# Criar um conjunto de dados LightGBM a partir dos dados de treinamento\n",
    "train_data = lgb.Dataset(X_train, label=y_train)\n",
    "\n",
    "# Definir parâmetros básicos\n",
    "params = {\n",
    "    'objective': 'regression',  # Regressão\n",
    "    'metric': 'l2',  # Métrica de avaliação (erro quadrático médio)\n",
    "    'boosting_type': 'gbdt',  # Tipo de aumento (gradient boosting)\n",
    "}\n",
    "\n",
    "# Treinar o modelo LightGBM\n",
    "num_round = 100  # Número de iterações (árvores) de treinamento\n",
    "bst = lgb.train(params, train_data, num_boost_round=num_round)\n",
    "\n",
    "# Fazer previsões no conjunto de teste\n",
    "y_pred_lgbm = bst.predict( X_test[cols] )\n",
    "\n",
    "# performance\n",
    "lgbm = ml_error( 'lgbm',  y_test  ,  y_pred_lgbm  )\n",
    "lgbm\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6c6450ed-01a8-4332-8f53-bfc8abbcc6ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 7.5.1 LGBM Cross Validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fb662262-8cd0-4bae-bb98-75a2d8e40aa1",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.268488Z",
     "start_time": "2024-03-01T22:30:35.268488Z"
    }
   },
   "outputs": [],
   "source": [
    "# Definir parâmetros básicos\n",
    "params = {\n",
    "    'objective': 'regression',  # Regressão\n",
    "    'metric': 'r2',  # Usando R² como métrica de avaliação, se suportado pela sua versão do LightGBM\n",
    "    'boosting_type': 'gbdt',  # Tipo de aumento (gradient boosting)\n",
    "}\n",
    "\n",
    "# Preparar o conjunto de dados LightGBM\n",
    "lgb_data = lgb.Dataset(X_train, label=y_train)\n",
    "\n",
    "# Número de iterações de treinamento\n",
    "num_round = 100\n",
    "\n",
    "# Configuração da validação cruzada\n",
    "nfold = 10\n",
    "seed = 42\n",
    "\n",
    "kf = KFold(n_splits=nfold, random_state=seed, shuffle=True)\n",
    "\n",
    "r2_scores = []\n",
    "\n",
    "# Realizar a validação cruzada\n",
    "for train_idx, val_idx in kf.split(X_train):\n",
    "    X_train_fold = X_train.iloc[train_idx]  # Seleciona as linhas de treinamento\n",
    "    y_train_fold = y_train.iloc[train_idx]  # Seleciona os rótulos de treinamento\n",
    "    X_val_fold = X_train.iloc[val_idx]  # Seleciona as linhas de validação\n",
    "\n",
    "    # Treinar o modelo LightGBM\n",
    "    bst = lgb.train(params, lgb.Dataset(X_train_fold, label=y_train_fold), num_boost_round=num_round)\n",
    "\n",
    "    # Fazer previsões no conjunto de validação\n",
    "    y_pred = bst.predict(X_val_fold)\n",
    "\n",
    "    # Calcular o R² score\n",
    "    r2 = r2_score(y_train.iloc[val_idx], y_pred)\n",
    "    r2_scores.append(r2)\n",
    "\n",
    "# Mostrar a média dos R² scores\n",
    "mean_r2 = np.mean(r2_scores)\n",
    "lgbm_cv = pd.concat( [lgbm, pd.DataFrame( {'Cv Score' : [mean_r2]} ) ], axis = 1) \n",
    "lgbm_cv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3d7f0368-6903-408f-ab2e-87d34ed835e6",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 7.6 XGBOOST"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1d8a1d34-f39c-4232-a73a-1fd138a03a35",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.271481Z",
     "start_time": "2024-03-01T22:30:35.271481Z"
    }
   },
   "outputs": [],
   "source": [
    "# Criar um modelo XGBoost Regressor\n",
    "model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.1)\n",
    "\n",
    "# Treinar o modelo\n",
    "model_xgb.fit(X_train, y_train)\n",
    "\n",
    "# Fazer previsões no conjunto de teste\n",
    "y_pred_xgb = model_xgb.predict( X_test[cols])\n",
    "\n",
    "# performance\n",
    "xgb_performance = ml_error( 'xgb',  y_test  ,  y_pred_xgb  )\n",
    "xgb_performance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0ccc217d-4282-481b-9657-f70183130e76",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-02-03T15:02:22.466913Z",
     "start_time": "2024-02-03T15:02:22.462520Z"
    }
   },
   "outputs": [],
   "source": [
    "## 7.6.1 XGBOOST Cross Validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f6fb33b3-75ab-4cce-b16f-a6b2bb039799",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.274479Z",
     "start_time": "2024-03-01T22:30:35.274479Z"
    }
   },
   "outputs": [],
   "source": [
    "# Realizar validação cruzada com 5 folds (ou o número que preferir) usando o R2 como métrica\n",
    "scores = cross_val_score(model_xgb, X_train, y_train, cv=5, scoring='r2')\n",
    "\n",
    "# Calcular a média dos coeficientes R2\n",
    "mean_r2 = scores.mean()\n",
    "\n",
    "# Imprimir o R2 médio\n",
    "result_xgb = pd.concat( [xgb_performance, pd.DataFrame( {'Cv Score' : [mean_r2]} ) ], axis = 1) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eafde3f1-fe8b-4e8c-97d1-d0253bd8cf1e",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 7.7 Resultados Modelos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2ac2cdd2-1489-4cbd-814b-b4cbe85f30ca",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.279007Z",
     "start_time": "2024-03-01T22:30:35.279007Z"
    }
   },
   "outputs": [],
   "source": [
    "pd.concat( [lr_metrics_cv, lrr_result_cv, rf_result_cv , lgbm_cv , result_xgb] , axis = 0 ).sort_values( by = 'MAE'\n",
    " , ascending = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c21e1f53-13c3-435f-9391-30a8bca99cf9",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 7.8 Resultados Cross Validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3c00abb9-fca7-402b-847b-784d4e5ad2e3",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 8.1 Criando a classe Idealista"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1a7890c7-3657-4582-8677-40a691cbaee7",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.282004Z",
     "start_time": "2024-03-01T22:30:35.282004Z"
    }
   },
   "outputs": [],
   "source": [
    "class Idealista:\n",
    "    def __init__( self ):\n",
    "        \n",
    "        self.df_region = pd.read_csv( 'datasets/df_region.csv' )\n",
    "        self.home_path                    = r'C:\\Users\\oscar\\Documents\\repos\\api_houses_Lisbon\\encodings'\n",
    "\n",
    "        self.te_status                    = pickle.load(open(os.path.join(self.home_path, 'te_status.pkl'), 'rb'))\n",
    "        self.te_province                  = pickle.load(open(os.path.join(self.home_path,'te_province.pkl') , 'rb'))\n",
    "        self.te_property_type             = pickle.load(open(os.path.join(self.home_path,'te_property_type.pkl'), 'rb'))\n",
    "        self.te_municipality              = pickle.load(open(os.path.join(self.home_path,'te_municipality.pkl'), 'rb' ))\n",
    "        self.te_district                  = pickle.load(open(os.path.join(self.home_path,'te_district.pkl'),'rb'))\n",
    "        self.le_detailed_type             = pickle.load(open(os.path.join(self.home_path,'le_detailed_type.pkl'),'rb' ))\n",
    "        self.freq_encoding_address        = pickle.load(open(os.path.join(self.home_path,'freq_encoding_address.pkl'), 'rb' ))\n",
    "        \n",
    "    def transform_data(self, df, colunas_irrelevantes ):\n",
    "\n",
    "        cols_old = df.columns.to_list()\n",
    "\n",
    "        # def função snake_case\n",
    "        snake_case = lambda x: inflection.underscore( x )\n",
    "\n",
    "        # defino as novas colunas\n",
    "        cols_new = list(map( snake_case, cols_old )) \n",
    "\n",
    "        # atribuo as novas colunas ao df\n",
    "        df.columns = cols_new\n",
    "\n",
    "        # removo as linhas em branco que possam ter vindo na extração\n",
    "        df = df.loc[~ df['property_code'].isna() , : ]\n",
    "\n",
    "        # removo duplicadas\n",
    "        df = df.drop_duplicates( subset= ['property_code'], keep = 'last' )\n",
    "\n",
    "        # preencho os vazios\n",
    "        df = fill_na( df )\n",
    "\n",
    "        # deleto as colunaas irrelevantes\n",
    "        df = delete_columns( df , colunas_irrelevantes )\n",
    "   \n",
    "        return df\n",
    "\n",
    "\n",
    "#=======================================================================================================================#\n",
    "\n",
    "    def feature_engineering(self, df , transform_data = False, merge_datasets = False   ):\n",
    "        \n",
    "        if transform_data == True and merge_datasets == False :\n",
    "            \n",
    "            # mes\n",
    "            df['datetime_scrapy'] = pd.to_datetime( df['datetime_scrapy'] )\n",
    "            df['month'] = df['datetime_scrapy'].dt.month \n",
    "\n",
    "            # ano\n",
    "            df['year'] = df['datetime_scrapy'].dt.year\n",
    "\n",
    "            # excluir a coluna datetime\n",
    "            df = df.drop( 'datetime_scrapy', axis = 1 )\n",
    "\n",
    "            print( 'Dados Transformados' )\n",
    "\n",
    "            return df\n",
    "        \n",
    "    \n",
    "        if merge_datasets == True and transform_data == False:\n",
    "        \n",
    "            # Df com a média de preço por município\n",
    "            df_municipality_mean_price = self.df_region.groupby('municipality')[['price']].median().reset_index()\n",
    "\n",
    "            # alteração nome coluna\n",
    "            df_municipality_mean_price = df_municipality_mean_price.rename(columns={'price': 'municipality_mean_price'})\n",
    "\n",
    "            # exportar para csv\n",
    "            #d f2_municipality_mean_price.to_csv( 'datasets/municipality.csv' )\n",
    "\n",
    "            # acrescentando a coluna ao df2\n",
    "            df = pd.merge( df, df_municipality_mean_price, on = 'municipality' , how = 'left' )\n",
    "\n",
    "#======================================================================================================================#\n",
    "\n",
    "#             # calculando a média por 'province'\n",
    "#             df2_province_mean_price = self.df_region.groupby( 'province' )[['price']].median().reset_index().rename( columns = { 'price': 'province_mean_price' } )\n",
    "\n",
    "#             # exportando para csv\n",
    "#             # df2_province_mean_price.to_csv( 'datasets/province.csv' )\n",
    "\n",
    "#             # acrescentando a coluna ao df2\n",
    "#             df = pd.merge( df, df2_province_mean_price , on = 'province' , how = 'left' )\n",
    "\n",
    "    #=======================================================================================================================#\n",
    "\n",
    "            # calculando média por district\n",
    "            df2_district_mean_price = self.df_region.groupby( 'district' )[['price']].median().reset_index().rename( columns = { 'price' : 'district_mean_price' } )\n",
    "\n",
    "            # exportando para csv \n",
    "            #df2_district_mean_price.to_csv( 'datasets/district.csv' )\n",
    "\n",
    "            # acrescentando a coluna ao df2\n",
    "            df = pd.merge( df , df2_district_mean_price , on = 'district', how = 'left' )\n",
    "\n",
    "            print( 'Datasets combinados com preços regioes' )\n",
    "\n",
    "            return df\n",
    "    \n",
    "        if transform_data == True and merge_datasets == True:\n",
    "\n",
    "            # mes\n",
    "            df['datetime_scrapy'] = pd.to_datetime( df['datetime_scrapy'] )\n",
    "            df['month'] = df['datetime_scrapy'].dt.month \n",
    "\n",
    "            # ano\n",
    "            df['year'] = df['datetime_scrapy'].dt.year\n",
    "\n",
    "            # excluir a coluna datetime\n",
    "            df = df.drop( 'datetime_scrapy', axis = 1 )\n",
    "\n",
    "            # Df com a média de preço por município\n",
    "            df_municipality_mean_price = self.df_region.groupby('municipality')[['price']].mean().reset_index()\n",
    "\n",
    "            # alteração nome coluna\n",
    "            df_municipality_mean_price = df_municipality_mean_price.rename(columns={'price': 'municipality_mean_price'})\n",
    "\n",
    "            # exportar para csv\n",
    "            #d f2_municipality_mean_price.to_csv( 'datasets/municipality.csv' )\n",
    "\n",
    "            # acrescentando a coluna ao df2\n",
    "            df = pd.merge( df, df_municipality_mean_price, on = 'municipality' , how = 'left' )\n",
    "\n",
    "    # ======================================================================================================================#\n",
    "\n",
    "#             # calculando a média por 'province'\n",
    "#             df2_province_mean_price = self.df_region.groupby( 'province' )[['price']].mean().reset_index().rename( columns = { 'price': 'province_mean_price' } )\n",
    "\n",
    "#             # exportando para csv\n",
    "#             # df2_province_mean_price.to_csv( 'datasets/province.csv' )\n",
    "\n",
    "#             # acrescentando a coluna ao df2\n",
    "#             df = pd.merge( df, df2_province_mean_price , on = 'province' , how = 'left' )\n",
    "\n",
    "    #=======================================================================================================================#\n",
    "\n",
    "            # calculando média por district\n",
    "            df2_district_mean_price = self.df_region.groupby( 'district' )[['price']].mean().reset_index().rename( columns = { 'price' : 'district_mean_price' } )\n",
    "\n",
    "            # exportando para csv \n",
    "            #df2_district_mean_price.to_csv( 'datasets/district.csv' )\n",
    "\n",
    "            # acrescentando a coluna ao df2\n",
    "            df = pd.merge( df , df2_district_mean_price , on = 'district', how = 'left' )\n",
    "\n",
    "            print( 'datasets combinados e dados transformados' )\n",
    "        \n",
    "            return df\n",
    "        \n",
    "        else:\n",
    "\n",
    "            print( 'Nenhuma Transformação' )\n",
    "\n",
    "            return df\n",
    "            \n",
    "#=======================================================================================================================#\n",
    "    \n",
    "    def filter_variables(self, df, filter_variables = False):\n",
    "    \n",
    "        if filter_variables:\n",
    "\n",
    "            # Vou querer saber somente os preços na province de Lisboa\n",
    "            df = df.loc[ df['province'] == 'Lisboa' , :  ]\n",
    "\n",
    "            # vou querer somente as habitações que tenham de 0 a no máximo 4 quartos\n",
    "            df = df.loc[ df['rooms'].isin( [ 0,1,2,3,4 ] ) , : ]\n",
    "\n",
    "            print('Variavies filtradas')\n",
    "\n",
    "            return df\n",
    "\n",
    "        else:\n",
    "\n",
    "            print('Variavies não filtradas')\n",
    "\n",
    "            return df\n",
    "\n",
    "#=======================================================================================================================#\n",
    "\n",
    "    # função para remover outliers\n",
    "    def remove_outliers(self, df, coluna = None ,  keep_outliers = True ):\n",
    "\n",
    "        if keep_outliers:\n",
    "            print( 'Outliers mantidos' )\n",
    "            return df\n",
    "\n",
    "        else:\n",
    "            # variavel resposta sem outliers\n",
    "            df = remover_outliers( df4 , coluna )\n",
    "            print('Outliers removidos')\n",
    "            return df\n",
    "\n",
    "#=======================================================================================================================#\n",
    "    \n",
    "    def rescaling_data(self, df, rescale = False ):\n",
    "\n",
    "        if rescale:\n",
    "            rs = RobustScaler() # robusto com outliers\n",
    "            mms = pp.MinMaxScaler() # desvio padrão pequeno e quando não ha distribuição gaussiana\n",
    "\n",
    "    #         # num_photos\n",
    "    #         rs_num_photos = RobustScaler() \n",
    "    #         # num_photos\n",
    "    #         df['num_photos'] = rs_num_photos.fit_transform( df[['num_photos']].values )\n",
    "\n",
    "            # floor\n",
    "#             rs_floor = RobustScaler() \n",
    "#             # num_photos\n",
    "#             df['floor'] = rs_floor.fit_transform( df[['floor']].values )\n",
    "\n",
    "            # size\n",
    "            rs_size = RobustScaler() \n",
    "            # size\n",
    "            df['size'] = rs_size.fit_transform( df[['size']].values )\n",
    "\n",
    "            # rooms\n",
    "    #         mms_rooms = pp.MinMaxScaler() \n",
    "    #         # rooms\n",
    "    #         df['rooms'] = mms_rooms.fit_transform( df[['rooms']].values )\n",
    "\n",
    "            # bathrooms\n",
    "    #         rs_bathrooms = RobustScaler() \n",
    "    #         # bathrooms\n",
    "    #         df['bathrooms'] = rs_bathrooms.fit_transform( df[['bathrooms']].values )\n",
    "\n",
    "            # distance\n",
    "            rs_distance = RobustScaler()\n",
    "            # distance\n",
    "            df['distance'] = rs_distance.fit_transform( df[['distance']].values )\n",
    "\n",
    "\n",
    "\n",
    "            print('Rescaled')\n",
    "\n",
    "            return  df\n",
    "        else:\n",
    "            print('Not rescaled')\n",
    "\n",
    "            return df\n",
    "\n",
    "#=======================================================================================================================#\n",
    "\n",
    "    def encoding( self, df ):\n",
    "\n",
    "        # floor - substituo os andares com letras\n",
    "        df['floor'] = df['floor'].replace( ['bj','st','ss','en'], 0 )\n",
    "        # transformo em inteiros\n",
    "        df['floor'] = df['floor'].astype( 'int64' )\n",
    "\n",
    "        #property_type - label\n",
    "        te_property_type = {'flat' : 0, 'duplex' : 1, 'chalet' : 2, 'studio': 3, 'penthouse': 4, 'countryHouse':5}\n",
    "        #te_property_type = df.groupby( 'property_type' )['price'].mean()\n",
    "        df['property_type'] = df['property_type'].map( te_property_type )\n",
    "        pickle.dump(te_property_type,open(r'C:\\Users\\oscar\\Documents\\repos\\api_houses_Lisbon\\encodings\\te_property_type.pkl', 'wb' ) )\n",
    "\n",
    "        # address - label\n",
    "        freq_encoding_address = df['address'].value_counts(normalize=True)\n",
    "        df['address'] = df['address'].map( freq_encoding_address )\n",
    "        pickle.dump( freq_encoding_address, open(r'C:\\Users\\oscar\\Documents\\repos\\api_houses_Lisbon\\encodings\\freq_encoding_address.pkl', 'wb') )\n",
    "\n",
    "        # province \n",
    "        te_province = df['province'].value_counts( )\n",
    "        #te_province = df.groupby( 'province' )['price'].mean()\n",
    "        df['province'] = df['province'].map( te_province )\n",
    "        pickle.dump( te_province, open( r'C:\\Users\\oscar\\Documents\\repos\\api_houses_Lisbon\\encodings\\te_province.pkl', 'wb' ) )\n",
    "\n",
    "        # municipality - label\n",
    "        te_municipality = df['municipality'].value_counts()\n",
    "        #te_municipality = df.groupby( 'municipality' )['price'].mean()\n",
    "        df['municipality'] = df['municipality'].map( te_municipality ).astype( 'float64' )\n",
    "        pickle.dump( te_municipality, open( r'C:\\Users\\oscar\\Documents\\repos\\api_houses_Lisbon\\encodings\\te_municipality.pkl', 'wb' ) )            \n",
    "\n",
    "        # municipality\n",
    "    #     freq_encoding_municipality = df['municipality'].value_counts(normalize=True)\n",
    "    #     df['municipality'] = df['municipality'].map( freq_encoding_municipality )\n",
    "\n",
    "\n",
    "        # district - label\n",
    "        te_district = df['district'].value_counts()\n",
    "        #te_district = df.groupby( 'district' )['price'].mean()\n",
    "        df['district'] = df['district'].map( te_district ).astype( 'float64' )\n",
    "        pickle.dump( te_district, open( r'C:\\Users\\oscar\\Documents\\repos\\api_houses_Lisbon\\encodings\\te_district.pkl', 'wb' ) )  \n",
    "        # district\n",
    "    #     freq_encoding_district = df['district'].value_counts(normalize=True)\n",
    "    #     df['district'] = df['district'].map( freq_encoding_district )\n",
    "\n",
    "\n",
    "        # Show address\n",
    "        encoding = {True: 1 , False: 0}\n",
    "        df['show_address'] = df['show_address'].apply( lambda x : 1 if x == True else 0 )\n",
    "\n",
    "        # description\n",
    "        df['description'] = df['description'].apply( lambda x : len(x) )\n",
    "\n",
    "        # has_video\n",
    "        encoding = {True: 1 , False: 0}\n",
    "        df['has_video'] = df['has_video'].map( encoding )\n",
    "\n",
    "        # status\n",
    "        te_status = {'good' : 1, 'newdevelopment' : 2 , 'renew' : 3}\n",
    "        #te_status = df.groupby( 'status' )['price'].mean()\n",
    "        df['status'] = df['status'].map( te_status ).astype( 'float64' )\n",
    "        pickle.dump( te_status, open( r'C:\\Users\\oscar\\Documents\\repos\\api_houses_Lisbon\\encodings\\te_status.pkl', 'wb' ) )  \n",
    "\n",
    "        # label encoder status\n",
    "        #status_encoder = LabelEncoder()\n",
    "        #df['status'] = status_encoder.fit_transform( df['status'] )\n",
    "\n",
    "        # new_development\n",
    "        encoding = {True: 1 , False: 0}\n",
    "        df['new_development'] = df['new_development'].map( encoding )\n",
    "\n",
    "        # detailed type - label\n",
    "        le_detailed_type = LabelEncoder()\n",
    "\n",
    "        df['detailed_type'] = df['detailed_type'].apply(lambda x: ast.literal_eval(x) )\n",
    "        df['detailed_type'] = df['detailed_type'].apply(lambda x: x['typology'] )\n",
    "        df['detailed_type'] = le_detailed_type.fit_transform( df['detailed_type'] )\n",
    "        pickle.dump( le_detailed_type, open( r'C:\\Users\\oscar\\Documents\\repos\\api_houses_Lisbon\\encodings\\le_detailed_type.pkl', 'wb' ) ) \n",
    "\n",
    "        # suggested text\n",
    "        df['suggested_texts'] = df['suggested_texts'].apply( lambda x : ast.literal_eval(x) )\n",
    "        df['suggested_texts'] = df['suggested_texts'].apply( lambda x : x['title'] )\n",
    "        df['suggested_texts'] = df['suggested_texts'].apply( lambda x : len(x) )\n",
    "\n",
    "        # hasplan\n",
    "        encoding = {True: 1 , False: 0}\n",
    "        df['has_plan'] = df['has_plan'].map( encoding )\n",
    "\n",
    "        # has3_d_tour\n",
    "        encoding = {True: 1 , False: 0}\n",
    "        df['has3_d_tour'] = df['has3_d_tour'].map( encoding )\n",
    "\n",
    "        # has360\n",
    "        encoding = {True: 1 , False: 0}\n",
    "        df['has360'] = df['has360'].map( encoding )\n",
    "\n",
    "        # has_staging\n",
    "        df['has_staging'] = df['has_staging'].apply( lambda x: 1 if x   else 0   )\n",
    "\n",
    "        # top_new_development\n",
    "        df['top_new_development'] = df['top_new_development'].apply( lambda x: 1 if x  else 0 )\n",
    "\n",
    "        #parking_space\n",
    "        encoding = {True: 1 , False: 0}\n",
    "        df['parking_space'] = df['parking_space'].apply(lambda x: ast.literal_eval(x) if x != 0 else x ) \n",
    "        df['parking_space'] = df['parking_space'].apply(lambda x: x['hasParkingSpace'] if x != 0 else x ) \n",
    "        df['parking_space'] = df['parking_space'].map( encoding )\n",
    "\n",
    "        df = df.fillna(0)    \n",
    "\n",
    "        return df\n",
    "\n",
    "\n",
    "    def predicoes(self, df , model ):\n",
    "        \n",
    "        # drop price\n",
    "        X = df.drop( 'price', axis = 1 )\n",
    "        \n",
    "        y_hat_final = model.predict( X )\n",
    "        \n",
    "        df['pred'] = y_hat_final\n",
    "        \n",
    "        return df\n",
    "        \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "db949e29-1f87-46a8-8bf9-20b21613e5d7",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.284996Z",
     "start_time": "2024-03-01T22:30:35.284996Z"
    }
   },
   "outputs": [],
   "source": [
    "a = df3.loc[  df3['address'].isin( ['Moscavide e Portela'] ), : ].head()[cols]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "51ed46be-c890-4b9a-925a-542bf8c2ca4e",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.287069Z",
     "start_time": "2024-03-01T22:30:35.287069Z"
    }
   },
   "outputs": [],
   "source": [
    "a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c0b3b28a-6b46-4071-8aad-32f3da86de68",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.290324Z",
     "start_time": "2024-03-01T22:30:35.290324Z"
    }
   },
   "outputs": [],
   "source": [
    "b = test_prediction( a, bst )\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "263c3409-d859-464c-b227-d52abdb82cda",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.292319Z",
     "start_time": "2024-03-01T22:30:35.292319Z"
    }
   },
   "outputs": [],
   "source": [
    "b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "28ed7642-7b0d-475d-8a2d-464cd23f26a4",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.294318Z",
     "start_time": "2024-03-01T22:30:35.294318Z"
    }
   },
   "outputs": [],
   "source": [
    "df_raw.district.unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1521e554-3b25-42f9-8d3d-8c6a8f69be33",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-02-04T16:13:37.736510Z",
     "start_time": "2024-02-04T16:13:37.733647Z"
    }
   },
   "outputs": [],
   "source": [
    "## 8.1.2 Feature Importances"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "11ad7789-e5a5-436e-ba1c-1ce43a843b67",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.296849Z",
     "start_time": "2024-03-01T22:30:35.296849Z"
    },
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Obter importâncias das features\n",
    "importances = rf.feature_importances_\n",
    "\n",
    "# Converter as importâncias em um DataFrame para facilitar a visualização\n",
    "features_df = pd.DataFrame({\n",
    "    'Feature': X_train.columns,\n",
    "    'Importance': importances\n",
    "})\n",
    "\n",
    "# Ordenar as features pela importância\n",
    "features_df = features_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)\n",
    "\n",
    "ax = sns.barplot(data = features_df, x = 'Feature', y = 'Importance')\n",
    "\n",
    "# Rotacionando os rótulos do eixo x\n",
    "ax.set_xticklabels(ax.get_xticklabels(), rotation=90);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f652d7d0-d4e8-4d1d-94f5-1401691ace85",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.298852Z",
     "start_time": "2024-03-01T22:30:35.298852Z"
    }
   },
   "outputs": [],
   "source": [
    "features_df\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "33c7d134-0ddc-4a0c-8a5f-a8da77cea29e",
   "metadata": {},
   "outputs": [],
   "source": [
    "## 8.1.3 Test com Dados Inseridos Manualmente"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4df4233f-9e5c-4b08-8f95-4e00b02b6108",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.301844Z",
     "start_time": "2024-03-01T22:30:35.301844Z"
    }
   },
   "outputs": [],
   "source": [
    "# vejo a estrutura de uma linha\n",
    "# df_teste.loc[ 16640 ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6fd46e24-9fae-4ca3-a5f4-b7d09431b170",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.304833Z",
     "start_time": "2024-03-01T22:30:35.304833Z"
    }
   },
   "outputs": [],
   "source": [
    "# teste com um dados de um imóvel conhecido\n",
    "test = {\n",
    "'property_code': 0,\n",
    "'thumbnail': 0,\n",
    "'num_photos': 5,\n",
    "'floor': 1,\n",
    "'price' : 0,\n",
    "'property_type': 'flat',\n",
    "'operation': 'rent',\n",
    "'size': 30,\n",
    "'exterior': True,\n",
    "'rooms': 1 ,\n",
    "'bathrooms':1,\n",
    "'address' : 'Moscavide e Portela',\n",
    "'province': 'Lisboa',\n",
    "'municipality': 'Moscavide e Portela',\n",
    "'district' : 'Moscavide e Portela',\n",
    "'country': 'PT',\n",
    "'latitude' : 38.775101412019715,\n",
    "'longitude': -9.104655113802375,\n",
    "'show_address': True,\n",
    "'url': 'www.imoveisportugal.com',\n",
    "'distance': 10000,\n",
    "'description' : 'Apartamento localizado na rua 1º de Maio. T1 remodelado, em frente ao metro, próximo a supermercados e ao parque das nações. 3 cauções adiantadas ou 2 fiadores',\n",
    "'has_video' : False,\n",
    "'status' : 'good',\n",
    "'new_development' : False,\n",
    "'has_lift': 1,\n",
    "'price_by_area' : 0,\n",
    "'detailed_type': str({'typology': 'flat'}),\n",
    "'suggested_texts' : str({'subtitle': 'Moscavide e Portela, Loures', 'title': 'Apartamento'}),\n",
    "'has_plan' : True,\n",
    "'has3_d_tour': False,\n",
    "'has360' : False,\n",
    "'has_staging' : False,\n",
    "'top_new_development' : False,\n",
    "'super_top_highlight' : False,\n",
    "'parking_space' : '0',\n",
    "'external_reference':0,\n",
    "'labels' : 0,                                                                 \n",
    "'neighborhood': 0,                                                         \n",
    "'pagina' : 0,                                                                  \n",
    "'datetime_scrapy':  pd.to_datetime('2024-02-05 10:18:37'),                                         \n",
    "'new_development_finished': 0,                                                  \n",
    "'highlight' : 0 \n",
    "\n",
    "}\n",
    "\n",
    "df_pred_test = pd.DataFrame( [test] )\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "29e40b8c-20ec-43dc-a808-308156cdfbfb",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.307341Z",
     "start_time": "2024-03-01T22:30:35.307341Z"
    }
   },
   "outputs": [],
   "source": [
    "df_pred_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a3884e1e-7f5d-44b9-b7ed-8359f1775f04",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.310337Z",
     "start_time": "2024-03-01T22:30:35.310337Z"
    }
   },
   "outputs": [],
   "source": [
    "def test_prediction( df , model ):\n",
    "    # instancia a classe Idealista\n",
    "    idealista_1 = Idealista()\n",
    "\n",
    "    # limpando os dados\n",
    "    df1_val = idealista_1.transform_data( df , colunas_irrelevantes )\n",
    "\n",
    "    # feature engineering\n",
    "    df2_val = idealista_1.feature_engineering( df1_val , transform_data = True, merge_datasets = True  )\n",
    "\n",
    "    # filtrar variáveis\n",
    "    df3_val = idealista_1.filter_variables( df2_val )\n",
    "\n",
    "    # reomver outliers\n",
    "    df4_val = idealista_1.remove_outliers( df3_val )\n",
    "\n",
    "    # reescalar dados\n",
    "    df5_val = idealista_1.rescaling_data( df4_val, rescale = True )\n",
    "\n",
    "    # encodar dados\n",
    "    df6_val = idealista_1.encoding( df5_val )\n",
    "    \n",
    "    df6_val =df6_val[cols,'price']\n",
    "\n",
    "    df6_val = df6_val.fillna( 0 )\n",
    "    \n",
    "    # predicoes\n",
    "    df7_val = idealista_1.predicoes( df6_val ,model )\n",
    "    \n",
    "    return df6_val"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d60ff368-fbad-4cf6-84df-3850d7454068",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.314327Z",
     "start_time": "2024-03-01T22:30:35.314327Z"
    }
   },
   "outputs": [],
   "source": [
    "a = test_prediction( df_pred_test , rf )\n",
    "print(a.isnull().sum())\n",
    "a.T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "517b23d1-2452-4599-a96e-a82ab0654b78",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.317870Z",
     "start_time": "2024-03-01T22:30:35.317870Z"
    },
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Obter importâncias das features\n",
    "importances = rf.feature_importances_\n",
    "\n",
    "# Converter as importâncias em um DataFrame para facilitar a visualização\n",
    "features_df = pd.DataFrame({\n",
    "    'Feature': a.drop(['price','pred'], axis = 1).columns,\n",
    "    'Importance': importances\n",
    "})\n",
    "\n",
    "# Ordenar as features pela importância\n",
    "features_df = features_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)\n",
    "\n",
    "ax = sns.barplot(data = features_df, x = 'Feature', y = 'Importance')\n",
    "\n",
    "# Rotacionando os rótulos do eixo x\n",
    "ax.set_xticklabels(ax.get_xticklabels(), rotation=90);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "82c4165d-c63f-4dfa-87a7-168ab51f3890",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.322402Z",
     "start_time": "2024-03-01T22:30:35.322402Z"
    }
   },
   "outputs": [],
   "source": [
    "len(df_pred_test.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d3370979-091c-480c-bf60-15d9d5bf017a",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.325907Z",
     "start_time": "2024-03-01T22:30:35.325907Z"
    }
   },
   "outputs": [],
   "source": [
    "df_val = clean_data( df_val )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "445a7bec-f9de-4317-b9a6-6824c07124ba",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.326941Z",
     "start_time": "2024-03-01T22:30:35.326941Z"
    }
   },
   "outputs": [],
   "source": [
    "df_val[['price','pred']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "55b6259d-168a-4421-8bd2-d834768bb1b1",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.331378Z",
     "start_time": "2024-03-01T22:30:35.330381Z"
    }
   },
   "outputs": [],
   "source": [
    "def clean_data( df ):    \n",
    "\n",
    "    transform = Idealista()\n",
    "\n",
    "    df1 = transform.transform_data( df , colunas_irrelevantes )\n",
    "    df2 = transform.feature_engineering( df1, transform_data = True )\n",
    "    df3 = transform.filter_variables( df2 )\n",
    "    df4 = transform.remove_outliers( df3 )\n",
    "    df5 = transform.rescaling_data( df4, rescale = True )\n",
    "    df6 = transform.encoding( df5 )\n",
    "   \n",
    "\n",
    "    return df6\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8a92b414-54d1-496f-94fb-85a0d30ab72a",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-03-01T22:30:35.335878Z",
     "start_time": "2024-03-01T22:30:35.335878Z"
    }
   },
   "outputs": [],
   "source": [
    "y_hat = rf.predict( df_val.drop( 'price', axis = 1 ) )\n",
    "\n",
    "ml_error( 'Rf', df_val['price'], y_hat )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "36278712-fcd7-44b9-a384-b1923aad29c9",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

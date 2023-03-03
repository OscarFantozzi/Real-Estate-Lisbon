# =========================================== imports ===================================================================
import pandas as pd
import json
import urllib
import requests as rq
import base64
import datetime as dt
from sqlalchemy import create_engine, text
from datetime import datetime as dt

# ========================================= functions ===================================================================

# def get_oauth_token(): # define parametos api 
    
#     url = "https://api.idealista.com/oauth/token"    
    
#     apikey= 'jrjxxvzz7nsykuhtac3u5rucpzlb89s5' #sent by idealist
    
#     secret= '28H56Shc5soI' #sent by idealista
    
#     apikey_secret = apikey + ':' + secret
    
#     auth = str(base64.b64encode(bytes(apikey_secret, 'utf-8')))[2:][:-1]
    
#     headers = {'Authorization' : 'Basic ' + auth,'Content-Type': 'application/x-www-form- urlencoded;charset=UTF-8'}
    
#     params = urllib.parse.urlencode({'grant_type':'client_credentials'}) #,'scope':'read'
    
#     content = rq.post(url,headers = headers, params=params)
    
#     bearer_token = json.loads(content.text)['access_token']

#     return bearer_token

# def search_api(token, url): # define o resultado da api
#     headers = {'Content-Type': 'Content-Type: multipart/form-data;', 'Authorization' : 'Bearer ' + token}
    
#     content = rq.post(url, headers = headers)
    
#     result = json.loads(content.text)
    
#     return result

# def get_data():
#     # Parametros da extração
#     api_key= 'jrjxxvzz7nsykuhtac3u5rucpzlb89s5'

#     country = 'pt'

#     operation = 'rent'

#     propertyType = 'homes'

#     center = '38.723,-9.139' # Coordenado do centro de Lisboa

#     distance = 50000

#     locale = 'pt'

#     maxItems = 50

#     # Faço a primeira consulta 
#     url = f'https://api.idealista.com/3.5/{country}/search?locale={locale}&operation={operation}&propertyType={propertyType}&apikey={api_key}&center={center}&distance={distance}&numPage=1&maxItems={maxItems}'

#     # Defino o token
#     token = get_oauth_token()

#     # Guardo a consulta
#     consulta = search_api(token,url)

#     # Pego quantas páginas tenho
#     paginas = consulta['totalPages'] + 1 # Adiciono 1 pois quando uso o for a ultima página não é incluída

#     # Crio um df vazio no qual vou anexar os outros df's ao decorrer do " for "
#     df_final = pd.DataFrame(index = [0])

#     for i in range(1, paginas ):
#         # Faço a consulta
#         url = f'https://api.idealista.com/3.5/{country}/search?locale={locale}&operation={operation}&propertyType={propertyType}&apikey={api_key}&center={center}&distance={distance}&numPage={i}&maxItems={maxItems}' # Faço a consulta
#         # crio a variável consulta
#         consulta = search_api(token,url)
    
#         # Crio o df com a consulta
#         df = pd.DataFrame( consulta['elementList'] )
    
#         # Crio a coluna página da extração
#         df['pagina'] = i

#         # Crio a data de extração
#         df['datetime_scrapy'] = dt.now().strftime( '%Y-%m-%d %H:%M:%S' )

#         # Concateno o df consulta com o df vazio
#         df_final = pd.concat( [df_final, df]  )

#         print('iteration : ', i)
#         print('actual page : ', consulta['actualPage'] )

#     # Exportar os dados
#     return df_final    
    

# def load_data( df_raw ):
    
#     # criar a lista n_cols
#     n_cols = ['propertyCode',
#     'thumbnail',
#     'externalReference',
#     'numPhotos',
#     'price',
#     'propertyType',
#     'operation',
#     'size',
#     'exterior',
#     'rooms',
#     'bathrooms',
#     'address','province',
#     'municipality',
#     'district',
#     'country','latitude',
#     'longitude',
#     'showAddress',
#     'url',
#     'distance',
#     'description',
#     'hasVideo',
#     'status',
#     'newDevelopment',
#     'priceByArea',
#     'detailedType',
#     'suggestedTexts',
#     'hasPlan',
#     'has3DTour',
#     'has360',
#     'hasStaging',
#     'topNewDevelopment',
#     'superTopHighlight',
#     'floor',
#     'hasLift',
#     'parkingSpace',
#     'neighborhood',
#     'labels',
#     'pagina',
#     'datetime_scrapy',
#     'newDevelopmentFinished']

#     if len( df_raw.columns ) == len( n_cols ):

#         # crio a conexao com o banco de dados
#         engine = create_engine('sqlite:///bd_houses_rent_api.sqlite', echo = False)

#         # ordena as colunas 
#         df_raw = df_raw[['propertyCode',
#                         'thumbnail',
#                         'externalReference',
#                         'numPhotos',
#                         'price',
#                         'propertyType',
#                         'operation',
#                         'size',
#                         'exterior',
#                         'rooms',
#                         'bathrooms',
#                         'address','province',
#                         'municipality',
#                         'district',
#                         'country','latitude',
#                         'longitude',
#                         'showAddress',
#                         'url',
#                         'distance',
#                         'description',
#                         'hasVideo',
#                         'status',
#                         'newDevelopment',
#                         'priceByArea',
#                         'detailedType',
#                         'suggestedTexts',
#                         'hasPlan',
#                         'has3DTour',
#                         'has360',
#                         'hasStaging',
#                         'topNewDevelopment',
#                         'superTopHighlight',
#                         'floor',
#                         'hasLift',
#                         'parkingSpace',
#                         'neighborhood',
#                         'labels',
#                         'pagina',
#                         'datetime_scrapy',
#                         'newDevelopmentFinished']]

#         # carrego os dados no db
#         df_raw.to_sql( 'houses', con = engine , if_exists = 'append' ,index = False )

#         # exibe mensagem
#         print( 'dados carregados com sucesso' )
        
#         # fecha a conexao com o bd
        
#         engine.dispose()

#     else:

#         print( 'numero de colunas divergentes' )
        
#     return None

# # ======================================== execução =====================================================================

# # df extraido
# df_extracted = get_data()

# # defino a data de extração
# now = dt.now() 
# dia = now.day # dia 
# mes = now.month # mes
# ano = now.year # ano
# # nome do arquivo
# nome_arquivo = 'extracao' + '_' + str(dia) + '_' + str(mes) + '_' + str(ano) 

# # exporto o arquivo
# df_extracted.to_excel( r'data/{}.xlsx'.format( nome_arquivo ), index = False )

# # leio os dados
# df = pd.read_excel( r'data/{}.xlsx'.format( nome_arquivo ) )

# # carrego para o banco de dados
# load_data( df )

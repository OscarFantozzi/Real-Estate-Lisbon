from sqlalchemy import create_engine, text

def create_table():
    # Criando a banco
    engine = create_engine('sqlite:///bd_houses_rent_api.sqlite', echo = False)

    # Cria as querys para criar as tableas
    query_houses = text('''

    CREATE TABLE houses(

        propertyCode                       INTEGER,
        thumbnail                          TEXT,
        externalReference                  TEXT,
        numPhotos                          INTEGER,
        price                              REAL,
        propertyType                       TEXT,
        operation                          TEXT,
        size                               REAL,
        exterior                           INTEGER,
        rooms                              INTEGER,
        bathrooms                          INTEGER,
        address                            TEXT,
        province                           TEXT,
        municipality                       TEXT,
        district                           TEXT,
        country                            TEXT,
        latitude                           REAL,
        longitude                          REAL,
        showAddress                        INTEGER,
        url                                TEXT,
        distance                           INTEGER,
        description                        TEXT,
        hasVideo                           INTEGER,
        status                             TEXT,
        newDevelopment                     INTEGER,
        priceByArea                        INTGER,
        detailedType                       TEXT,
        suggestedTexts                     TEXT,
        hasPlan                            INTEGER,
        has3DTour                          INTEGER,
        has360                             INTEGER,
        hasStaging                         INTEGER,
        topNewDevelopment                  INTEGER,
        superTopHighlight                  INTEGER,
        floor                              TEXT,
        hasLift                            REAL,
        parkingSpace                       TEXT,
        neighborhood                       TEXT,
        labels                             TEXT,
        pagina                             INTEGER,
        datetime_scrapy                    TEXT,
        newDevelopmentFinished             REAL)''')
    conn = engine.connect()

    
    result = conn.execute(  query_houses ) 
    
    conn.close()

    return result

# cria a tabela
create_table()
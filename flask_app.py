import os
import psycopg2
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from fastapi import FastAPI

load_dotenv() # to load all environment variables

api_key = os.getenv("gemini-apikey")
host = os.getenv("host")
dbname = os.getenv("dbname")
user = os.getenv("user")
password = os.getenv("password")
port = os.getenv("port")

# Instantiate db connection
connection = psycopg2.connect(host=host, dbname=dbname, user=user, password=password, port=port)
# Instantiate app instance
app = FastAPI()


# Helper functions
def db_tables():
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public';
        """)

        tables = cursor.fetchall()
        tables = [table[0] for table in tables]
        return tables
    
    
def format_response(response):
    text = ""
    for row in response:
        text += " | ".join([str(item) for item in row]) + "\n\n"
    return text
   
# get db tables list  
tables = db_tables()


# Prompt template for the sql generator
template = """
You are a PostgreSQL expert. Your task is to generate only SQL queries to be run in a Python environment using the `psycopg2` library.
Use double quotes around table names always
The SQL query should be based on this user query: {query}. The database has the following tables: {tables}.
Use only the table names and columns exactly as provided. Do not include or reference any tables or columns that are not listed.
Make sure to reference the table names columns with the right names and spelling case too.
Ensure the query is formatted correctly as a string for Python, following this pattern:
SELECT column1, column2 FROM table_name WHERE condition = 'value';

Remember:
- Always match the exact spelling and case of the tables and columns.
- Be prepared to generate complex queries if the user's request involves comparing products, locations, or metrics.
- If the user asks to compare products across different metrics, use the relevant "resource" tables provided in the list.
- For example, if asked to compare products across states, use the "StateResource" table to rank them by the relevant metric.

The schema of the database is as follows:
```
table Lga
  id                Int                 @id @default(autoincrement())
  name              String
  stateId           Int
  state             State               @relation(fields: [stateId], references: [id])
  lgaResources      LgaResource[]

table State
  id                Int                 @id @default(autoincrement())
  name              String              @unique
  country           String?
  lgas              Lga[]
  stateResources    StateResource[]

table Category
  id                Int        @id @default(autoincrement())
  name              String
  abbr              String?
  unitOfMeasurement String?
  subCategories SubCategory[]
  resources         ResourceCategory[]

table ResourceCategory
  resourceId Int
  categoryId Int
  resource   Resource @relation(fields: [resourceId], references: [id])
  category   Category @relation(fields: [categoryId], references: [id])

  @@id([resourceId, categoryId])  // Composite primary key

table SubCategory
  id  Int @id @default(autoincrement())
  name String
  abbr String?
  categoryId Int
  category Category @relation(fields: [categoryId], references: [id])

table Resource
  id                Int                 @id @default(autoincrement())
  name              String @unique
  stateResources    StateResource[]
  lgaResources      LgaResource[]
  createdAt         DateTime            @default(now())
  lastModifiedOn    DateTime            @updatedAt
  categories        ResourceCategory[]

  @@index([name])

table StateResource
  id         Int      @id @default(autoincrement())
  resourceId Int
  resource   Resource @relation(fields: [resourceId], references: [id])
  stateId    Int
  totalQuantity Float?
  totalValue Float?
  state      State    @relation(fields: [stateId], references: [id])

  @@index([resourceId, stateId])
  @@unique([stateId, resourceId])

table LgaResource
  id         Int      @id @default(autoincrement())
  identifier  String  @unique
  resourceId Int
  resource   Resource @relation(fields: [resourceId], references: [id])
  lgaId      Int
  lga        Lga      @relation(fields: [lgaId], references: [id])
  quantity   String?
  quality    Float?
  value      Float?
  collectionDate    String?
  collectorName String?
  collectorPhone String?
  SampleId String?
  locationName String?
  locationLat Float?
  locationLong Float?
  locationAltitude Float?
  locationPrecision Float?
  harvestDate String?
  quantityRating Float?
  resourceQuantity Float?
  labAnalysis String?
  moistureContent String?
  pestDiseaseIncidence String?
  storageConditions String?
  marketPrice String?
  environmentalSustainabilty String?
  agricCategorization Float?
  solidMineralCategorization Float?
  accessToMarket Float?
  marketValue Float?
  environmentalImpact Float?
  valueChainAnalysis Float?
  investmentOpportunities Float?
  policiesRegulations Float?
  industryChallenges Float?
  locationLgaWard String?
  townVillage String?
  stakeholderEngagement Float?
  submissionDate DateTime?
  status String?
  collectionStart DateTime?
  collectionEnd DateTime?

  @@index([resourceId, lgaId])```
"""

@app.get("/ask_ai")
def get_response(query: str):
    sql_prompt = template.format(query=query, tables=tables)
    sql_llm = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key)

    # Genenerate explanation
    compare_llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
    compare_template = """
    Given a user query, sql code and corresponding result, Actual explanation to users.
    Consider that the users are laymen with no coding knowledge, so don't explain the sql, just relate the result, context from code and the query.
    query: {query}
    sql_code: {sql_code}
    sql_result: {sql_result}
    If the answer to the question is not in the sql result, say that you could not process it, that the user should word the better, respond professionally.
    Do not mention any technical term in your respoonse, like code, sql, result, etc. use data instead, and don't use personal pronouns like we, our, ...
    And be aware of what the values represent. So, when asked to compare any two variables or talk about a variable, by rating and you see numerical values like (2, 3.0, 5.0, and so on), use the schema below:
    0 - 3 being Low
    4 - 6 being Average
    7 - 10 being High
    """

    generated_query = sql_llm(sql_prompt)[6:-3] # filter out "```sql ... ```" in the query generated

    cursor = connection.cursor() # create connection cursor
    
    done = False
    while not done:
        try:
            cursor.execute(generated_query)
            response = cursor.fetchall()
            response = format_response(response)
            connection.commit()
            compare_prompt = compare_template.format(query=query, sql_code=generated_query, sql_result=response)
            explanation = compare_llm(compare_prompt)
            done = True
                
        except Exception as e:
            connection.rollback()  # Rollback the failed transaction to continue
    return {"ai_response": explanation}

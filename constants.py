from envReader import getValue

def MONGO_URL():
 return getValue("MONGO_URL")

def DB_NAME():
   return getValue("DB_NAME")

def COL_NAME():
   return getValue("COL_NAME")

def OPENAI_KEY():
    return getValue("OPENAI_KEY")

def MIN_COS_SIM_VALUE():
   return float(getValue("MIN_COS_SIM_VALUE"))
import sys
print("Python:", sys.version)

try:
    import numpy as np
    print("numpy:", np.__version__)
except Exception as e:
    print("numpy not found:", e)

try:
    import pandas as pd
    print("pandas:", pd.__version__)
except Exception as e:
    print("pandas not found:", e)

try:
    import sklearn
    print("sklearn:", sklearn.__version__)
except Exception as e:
    print("sklearn not found:", e)

try:
    import statsmodels
    print("statsmodels:", statsmodels.__version__)
except Exception as e:
    print("statsmodels not found:", e)

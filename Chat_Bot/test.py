import os

import pandasai as pai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Load your CSV file
file = pai.read_csv("/path/to/your/file.csv")


pai.api_key.set(os.environ["PANDASAI_API_KEY"])
# Save your dataset configuration
df = pai.DataFrame({
    "country": ["US", "UK", "France"],
    "revenue": [5000, 3000, 4000]
})

# Try minimal query
try:
    print(df.chat("How many rows?"))
except Exception as e:
    print(f"Full error: {type(e).__name__}: {str(e)}")
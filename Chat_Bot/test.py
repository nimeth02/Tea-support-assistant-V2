import pandasai as pai
import pandas as pd

# Load your CSV file
file = pai.read_csv("/path/to/your/file.csv")


pai.api_key.set("PAI-6021dae7-863a-46b4-a2e6-b3d917281892")
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
import pandas as pd

col_names = ["mpg", "cylinders", "displacement", "horsepower", "weight",
            "acceleration", "model year", "origin", "car name"]

cars = pd.read_table('auto-mpg.data', delim_whitespace=True, header=None, names=col_names)

print cars.head()

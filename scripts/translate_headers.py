import os
import glob
from pathlib import Path
import pandas as pd

COLUMN_MAPPING = {
    "BETRIEBSTAG": "DATE",
    "FAHRT_BEZEICHNER": "TRIP_ID",
    "BETREIBER_ID": "OPERATOR_ID",
    "BETREIBER_ABK": "OPERATOR_ABB",
    "BETREIBER_NAME": "OPERATOR_NAME",
    "PRODUKT_ID": "PRODUCT_ID",
    "LINIEN_ID": "LINE_ID",
    "LINIEN_TEXT": "LINE_NAME",
    "UMLAUF_ID": "CIRCULATION_ID",
    "VERKEHRSMITTEL_TEXT": "TRANSPORT_TYPE",
    "ZUSATZFAHRT_TF": "ADDITIONAL_TRIP",
    "FAELLT_AUS_TF": "CANCELLED",
    "BPUIC": "BPUIC",
    "HALTESTELLEN_NAME": "STOP_NAME",
    "ANKUNFTSZEIT": "ARRIVAL_TIME",
    "AN_PROGNOSE": "ARRIVAL_FORECAST",
    "AN_PROGNOSE_STATUS": "ARRIVAL_FORECAST_STATUS",
    "ABFAHRTSZEIT": "DEPARTURE_TIME",
    "AB_PROGNOSE": "DEPARTURE_FORECAST",
    "AB_PROGNOSE_STATUS": "DEPARTURE_FORECAST_STATUS",
    "DURCHFAHRT_TF": "PASS_THROUGH",
}

FOLDER = Path(__file__).resolve().parent.parent / "data" / "raw"
i=0

for filepath in glob.glob(os.path.join(FOLDER, "*.csv")):
    with open(filepath, "r+", encoding="utf-8") as f:
        first_line = f.readline()
        rest = f.read()

        new_header = ";".join(
            COLUMN_MAPPING.get(col, col) for col in first_line.strip().split(";")
        )

        f.seek(0)
        f.write(new_header + "\n" + rest)
        f.truncate()
    i+=1
    print(f"{i}/365")
    print(f"Done: {os.path.basename(filepath)}")

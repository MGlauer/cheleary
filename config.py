import os

LOCAL_SIZE_RESTRICTION = int(os.environ.get("CHEBI_SIZE_CON", -1))
EPOCHS = int(os.environ.get("EPOCHS", 300))
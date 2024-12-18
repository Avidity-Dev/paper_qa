import yaml
import redis

# Load YAML configuration
with open("src/config/vector.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract fields from config
text_fields = config.get("text", [])
tag_fields = config.get("tag", [])
vector_fields = config.get("vector", [])

# Connect to Redis
# Adjust host/port/db as needed
r = redis.Redis(host="localhost", port=6379, db=0)

# Index name
index_name = "idx:docs"

# Drop the index if it exists
try:
    r.execute_command("FT.DROPINDEX", index_name)
except redis.exceptions.ResponseError:
    pass  # Index did not exist

# Begin building the FT.CREATE command
# ON JSON means indexing JSON documents
# PREFIX can be used to index keys with a certain prefix if desired. For example:
# "PREFIX 1 myDoc:" if you store documents as myDoc:<id>. Adjust as necessary.
cmd = ["FT.CREATE", index_name, "ON", "JSON", "SCHEMA"]


# Helper function to build field arguments
def build_text_field_args(field):
    args = [field["name"], "AS", field["as_name"], "TEXT"]
    if "weight" in field and field["weight"] != 1.0:
        args += ["WEIGHT", str(field["weight"])]
    if field.get("no_stem", False):
        args.append("NOSTEM")
    if field.get("sortable", False):
        args.append("SORTABLE")
    return args


def build_tag_field_args(field):
    args = [field["name"], "AS", field["as_name"], "TAG"]
    if field.get("no_index", False):
        # If we do not index this field, we might just skip adding it to schema.
        # But generally "no_index" in RediSearch means a different thing.
        # We'll assume "no_index" = False means we do index it, if True, we skip.
        # For demonstration, if True, just don't add any special param:
        pass
    if "separator" in field:
        args += ["SEPARATOR", field["separator"]]
    return args


def build_vector_field_args(field):
    # For vector fields, specify type: VECTOR and the algorithm block
    # Format (RediSearch 2.4+):
    # <json_path> AS <alias> VECTOR <algo> ... params ...
    # Example:
    # $.embedding AS embedding VECTOR FLAT 6 TYPE FLOAT32 DIM 1536 DISTANCE_METRIC COSINE
    args = [field["name"], "AS", field["as_name"], "VECTOR", field["algorithm"], "6"]
    # The algorithm block requires parameters:
    # TYPE, DIM, DISTANCE_METRIC, etc.
    args += [
        "TYPE",
        field["datatype"],
        "DIM",
        str(field["dims"]),
        "DISTANCE_METRIC",
        field["distance_metric"],
    ]
    # Additional parameters for FLAT or HNSW can be added if required:
    # For FLAT:
    # BLOCK_SIZE, INITIAL_CAP, M, EF_CONSTRUCTION for HNSW, etc.
    # Since not provided in the YAML, we won't add them.
    return args


# Build schema arguments
for t in text_fields:
    # Distinguish if it's a text-like field or numeric. If weight or no_stem are set, likely TEXT.
    # If "weight" not present and "sortable" is True but no "no_stem", it might be numeric/datetime.
    # The YAML includes "published_date" and "created_at" as text fields with only sortable = true.
    # Those might be numeric/datetime fields. RediSearch doesn't have a direct datetime field,
    # you can treat them as NUMERIC if they are stored as numeric timestamps or textual dates.
    # If they're textual dates, and you plan to use numeric range searches, you must store them as numeric.
    # Without more info, let's assume they remain TEXT but just sortable. If you want numeric indexing,
    # you'd have to store them as integers or doubles and mark them NUMERIC.

    # Check if field looks like a normal text field or we need to treat as numeric:
    # If it's no_stem or weight set, likely TEXT. If only sortable and no weight/no_stem, maybe numeric.
    if "weight" in t or "no_stem" in t:
        # It's a text field
        cmd += build_text_field_args(t)
    else:
        # If there's no weight, no no_stem, but sortable = True
        # We can consider it as NUMERIC since we want to sort by date.
        # RediSearch does not parse textual date strings automatically.
        # Typically you'd store date as a timestamp and index as NUMERIC.
        # For demonstration, let's index as TAG or TEXT.
        # Sorting by text fields is possible, but not numeric sorting by date.
        # If you are sure your date can be handled as TEXT and still sort lexicographically,
        # you can keep TEXT. Otherwise, treat it as NUMERIC (assuming your data is numeric).
        # Let's assume TEXT sortable for simplicity.
        cmd += build_text_field_args(t)

for tg in tag_fields:
    cmd += build_tag_field_args(tg)

for v in vector_fields:
    cmd += build_vector_field_args(v)


print(cmd)
# Execute the FT.CREATE command
r.execute_command(*cmd)

print("Index created successfully.")

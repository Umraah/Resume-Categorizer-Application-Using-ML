import pickle

data = {"name": "Alice", "age": 25}

# Serialize (save) the object to a file
with open("test.pkl", "wb") as f:
    pickle.dump(data, f)

# Deserialize (load) the object from the file
with open("test.pkl", "rb") as f:
    loaded_data = pickle.load(f)

print("Loaded Data:", loaded_data)  # Expected output: {'name': 'Alice', 'age': 25}

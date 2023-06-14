import subprocess

# Step 1: Train and save the model
print("Training and saving the model...")
subprocess.call(["python", "train_model.py"])

# Step 2: Run the Streamlit app
print("Running the Streamlit app...")
subprocess.call(["streamlit", "run", "app.py"])

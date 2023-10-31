from tqdm import tqdm
import time

# Define the range for your loop
total_iterations = 100

# Create a tqdm progress bar
for i in tqdm(range(total_iterations), desc="Processing"):
    # Simulate some work
    time.sleep(0.1)  # Replace this with your actual work

print("Finished processing")

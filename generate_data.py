import pandas as pd
import random
from datetime import datetime

data = []

# Start time (today, 10 AM)
start_time = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)

person_id = 0

for hour in range(10, 22):  # 10 AM to 10 PM

    # -------------------------
    # Traffic pattern
    # -------------------------
    if 18 <= hour <= 21:
        customers_per_hour = random.randint(40, 60)   # Peak hours
    elif 12 <= hour <= 14:
        customers_per_hour = random.randint(25, 40)   # Medium traffic
    else:
        customers_per_hour = random.randint(10, 20)   # Low traffic

    for _ in range(customers_per_hour):

        minute = random.randint(0, 59)
        second = random.randint(0, 59)

        timestamp = start_time.replace(hour=hour, minute=minute, second=second)

        # -------------------------
        # Gender distribution logic
        # -------------------------
        if 18 <= hour <= 21:
            weights = [0.6, 0.4]  # Female dominant (evening)
        elif 12 <= hour <= 14:
            weights = [0.5, 0.5]  # Balanced (afternoon)
        else:
            weights = [0.45, 0.55]  # Male dominant (morning)

        gender = random.choices(["Female", "Male"], weights=weights)[0]

        data.append([timestamp, person_id, gender])
        person_id += 1


# -------------------------
# Create DataFrame
# -------------------------
df = pd.DataFrame(data, columns=["timestamp", "person_id", "gender"])

# -------------------------
# Shuffle + Sort (realistic flow)
# -------------------------
df = df.sample(frac=1).reset_index(drop=True)
df = df.sort_values("timestamp").reset_index(drop=True)

# -------------------------
# Reassign ID based on time (IMPORTANT FIX)
# -------------------------
df["person_id"] = df.index

# -------------------------
# Add useful analytics columns
# -------------------------
df["hour"] = df["timestamp"].dt.hour
df["day_type"] = "Weekday"

# -------------------------
# Save
# -------------------------
df.to_csv("simulated_customer_data.csv", index=False)

print(f"✅ Generated {len(df)} realistic customer records!")
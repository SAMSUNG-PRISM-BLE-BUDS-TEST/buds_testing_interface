from datetime import datetime

# Get current time
dt_object = datetime.now()
le=[]
ts=14
le.append([ts,"BLE DEVICE CONNECTED"])
print(str("hrl"))
# Format datetime object to string
formatted_time = dt_object.strftime('%d-%m-%Y %H:%M:%S')

print("Formatted DateTime:", formatted_time)

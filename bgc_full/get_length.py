import os
import sys

# 1. Setup SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib

# 2. LOAD YOUR MAP (Replace 'bgc.net.xml' with your actual filename)
NET_FILE = "final_map.net.xml"  # Ensure this file is in the same directory or provide the correct path

try:
    net = sumolib.net.readNet(NET_FILE)
    
    # 3. CALCULATE
    total_length = 0
    total_lane_km = 0
    
    for edge in net.getEdges():
        # Skip internal edges (connections inside intersections)
        if edge.getFunction() == "internal":
            continue
            
        edge_len = edge.getLength()
        lane_count = edge.getLaneNumber()
        
        total_length += edge_len
        total_lane_km += (edge_len * lane_count)

    print(f"--- MAP STATISTICS ---")
    print(f"Total Road Length: {total_length/1000:.2f} km")
    print(f"Total Lane Length: {total_lane_km/1000:.2f} km (Use this for Density)")

except Exception as e:
    print(f"Error: {e}")
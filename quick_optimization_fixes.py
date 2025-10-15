
# IMMEDIATE FIXES TO IMPLEMENT (Copy to fixed_edge_traffic_controller.py)

# 1. DYNAMIC ADAPTATION INTERVALS
def get_dynamic_adaptation_interval(self, traffic_data):
    total_vehicles = sum(data.get('vehicles', 0) for data in traffic_data.values())
    max_waiting = max(data.get('waiting_time', 0) for data in traffic_data.values())
    
    if total_vehicles >= 15 or max_waiting > 40:
        return 20  # Quick response for heavy traffic
    elif total_vehicles >= 8 or max_waiting > 25:
        return 25  # Medium response
    elif total_vehicles >= 4:
        return 35  # Normal response
    else:
        return 50  # Conservative for light traffic

# 2. DYNAMIC CHANGE LIMITS  
def get_dynamic_change_limit(self, traffic_data):
    max_waiting = max(data.get('waiting_time', 0) for data in traffic_data.values())
    total_vehicles = sum(data.get('vehicles', 0) for data in traffic_data.values())
    
    if max_waiting > 45 or total_vehicles >= 20:
        return 0.25  # Allow big changes for critical situations
    elif max_waiting > 25 or total_vehicles >= 10:
        return 0.20  # Good responsiveness
    elif total_vehicles >= 5:
        return 0.15  # Moderate changes
    else:
        return 0.08  # Conservative for light traffic

# 3. ENHANCED TRAFFIC CATEGORIZATION
def enhanced_categorize_traffic(self, vehicles, waiting_time=0, speed=0):
    # Base category
    if vehicles >= 12: base = 'CRITICAL'
    elif vehicles >= 8: base = 'HEAVY'  
    elif vehicles >= 5: base = 'NORMAL'
    elif vehicles >= 3: base = 'MODERATE'
    elif vehicles >= 1: base = 'LIGHT'
    else: base = 'EMPTY'
    
    # Upgrade based on waiting time
    if waiting_time > 40:
        return 'CRITICAL'
    elif waiting_time > 25 and base != 'EMPTY':
        categories = ['EMPTY', 'LIGHT', 'MODERATE', 'NORMAL', 'HEAVY', 'CRITICAL']
        current_idx = categories.index(base)
        return categories[min(current_idx + 1, len(categories) - 1)]
    
    return base

# 4. CRITICAL SITUATION DETECTION
def is_critical_situation(self, traffic_data):
    for direction, data in traffic_data.items():
        if data.get('waiting_time', 0) > 50:  # Very long wait
            return True
        if data.get('vehicles', 0) > 15:  # Queue too long
            return True
        if data.get('speed', 10) < 1.0 and data.get('vehicles', 0) > 5:  # Stopped traffic
            return True
    return False

# 5. REPLACE should_adapt() METHOD
def should_adapt(self, traffic_data):
    current_time = traci.simulation.getTime()
    
    # Critical situations override timing
    if self.is_critical_situation(traffic_data):
        if current_time - self.last_adaptation_time >= 15:  # Minimum 15s
            return True
    
    # Dynamic interval based on traffic
    required_interval = self.get_dynamic_adaptation_interval(traffic_data)
    return (current_time - self.last_adaptation_time) >= required_interval

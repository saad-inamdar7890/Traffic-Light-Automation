"""
Dynamic Flow Manager Module
Handles random traffic flow generation and time-based patterns
"""

import random
import traci

class DynamicFlowManager:
    def __init__(self):
        """Initialize the dynamic flow manager with base flow configurations"""
        # Based on demo.rou.xml flows, create dynamic variations
        self.flow_configs = {
            'f_0': {'from': 'E0', 'to': 'E0.319', 'base_rate': 300, 'current_rate': 300},
            'f_1': {'from': 'E0', 'to': '-E1.238', 'base_rate': 150, 'current_rate': 150},
            'f_2': {'from': 'E0', 'to': 'E1.200', 'base_rate': 150, 'current_rate': 150},
            'f_3': {'from': '-E1', 'to': '-E1.238', 'base_rate': 300, 'current_rate': 300},
            'f_4': {'from': '-E1', 'to': '-E0.254', 'base_rate': 150, 'current_rate': 150},
            'f_5': {'from': '-E1', 'to': 'E0.319', 'base_rate': 150, 'current_rate': 150},
            'f_6': {'from': '-E0', 'to': '-E1.238', 'base_rate': 150, 'current_rate': 150},
            'f_7': {'from': '-E0', 'to': 'E1.200', 'base_rate': 150, 'current_rate': 150},
            'f_8': {'from': 'E1', 'to': '-E0.254', 'base_rate': 150, 'current_rate': 150},
            'f_9': {'from': 'E1', 'to': 'E0.319', 'base_rate': 150, 'current_rate': 150},
            'f_10': {'from': 'E1', 'to': 'E1.200', 'base_rate': 300, 'current_rate': 300},
            'f_11': {'from': '-E0', 'to': '-E0.254', 'base_rate': 300, 'current_rate': 300}
        }
        
        self.flow_update_interval = 120  # Update flows every 2 minutes
        self.last_update = 0
        self.flow_variations = {}
        
        # Time-based traffic patterns
        self.time_patterns = {
            'night': {'factor': 0.3, 'hours': [22, 23, 0, 1, 2, 3, 4, 5, 6]},
            'morning_rush': {'factor': 1.8, 'hours': [7, 8, 9]},
            'day': {'factor': 1.0, 'hours': [10, 11, 12, 13, 14, 15, 16]},
            'evening_rush': {'factor': 1.6, 'hours': [17, 18, 19]},
            'evening': {'factor': 0.7, 'hours': [20, 21]}
        }
        
    def get_time_factor(self, step):
        """Get traffic factor based on time of day"""
        hour = int((step / 3600) % 24)
        
        for pattern_name, pattern_data in self.time_patterns.items():
            if hour in pattern_data['hours']:
                return pattern_data['factor']
        
        return 1.0  # Default factor
    
    def generate_random_variation(self):
        """Generate random variation factor"""
        return random.uniform(0.8, 1.2)  # ±20% variation

    def calculate_new_flow_rate(self, base_rate, time_factor, random_factor):
        """Calculate new flow rate with constraints"""
        new_rate = base_rate * time_factor * random_factor
        return max(50, min(2000, int(new_rate))) 
    
    def update_flow_rates(self, step):
        """Update flow rates dynamically based on time and random factors"""
        if step - self.last_update < self.flow_update_interval:
            return False
            
        hour = (step / 3600) % 24
        time_factor = self.get_time_factor(step)
        
        print(f"\n🔄 UPDATING DYNAMIC FLOWS (Hour: {hour:.1f}, Factor: {time_factor:.1f})")
        
        for flow_id, config in self.flow_configs.items():
            random_factor = self.generate_random_variation()
            new_rate = self.calculate_new_flow_rate(
                config['base_rate'], 
                time_factor, 
                random_factor
            )
            
            config['current_rate'] = new_rate
            self.flow_variations[flow_id] = new_rate
            
            print(f"   {flow_id}: {config['from']} → {config['to']} = {new_rate} veh/h")
        
        self.last_update = step
        return True
    
    def get_current_flow_rate(self, flow_id):
        """Get current flow rate for a specific flow"""
        return self.flow_configs.get(flow_id, {}).get('current_rate', 500)
    
    def get_flow_summary(self):
        """Get summary of current flow rates"""
        summary = {}
        for flow_id, config in self.flow_configs.items():
            summary[flow_id] = {
                'from': config['from'],
                'to': config['to'],
                'base_rate': config['base_rate'],
                'current_rate': config['current_rate'],
                'efficiency': (config['current_rate'] / config['base_rate']) * 100
            }
        return summary
    
    def apply_flow_changes(self):
        """Apply flow rate changes to simulation (placeholder for future implementation)"""
        try:
            # In a real implementation, this would modify SUMO flows
            # For now, we log the changes for analysis
            if self.flow_variations:
                print(f"📊 Flow changes recorded: {len(self.flow_variations)} flows modified")
                return True
        except Exception as e:
            print(f"Note: Flow modifications logged for analysis: {e}")
        return False

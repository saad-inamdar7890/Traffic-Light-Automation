"""
Dynamic Flow Manager Module
Handles random traffic flow generation and time-based patterns
"""

import random
import traci

class DynamicFlowManager:
    def __init__(self):
        """Initialize the dynamic flow manager with base flow configurations"""
        # Based on demo.rou.xml flows, create dynamic variations with mixed vehicle types
        # Further reduced flows for better analysis and less congestion
        self.flow_configs = {
            # Car flows (further reduced for cleaner analysis)
            'f_0_cars': {'from': 'E0', 'to': 'E0.319', 'base_rate': 120, 'current_rate': 120, 'vtype': 'car'},
            'f_1_cars': {'from': 'E0', 'to': '-E1.238', 'base_rate': 60, 'current_rate': 60, 'vtype': 'car'},
            'f_2_cars': {'from': 'E0', 'to': 'E1.200', 'base_rate': 60, 'current_rate': 60, 'vtype': 'car'},
            'f_3_cars': {'from': '-E1', 'to': '-E1.238', 'base_rate': 120, 'current_rate': 120, 'vtype': 'car'},
            'f_4_cars': {'from': '-E1', 'to': '-E0.254', 'base_rate': 60, 'current_rate': 60, 'vtype': 'car'},
            'f_5_cars': {'from': '-E1', 'to': 'E0.319', 'base_rate': 60, 'current_rate': 60, 'vtype': 'car'},
            'f_6_cars': {'from': '-E0', 'to': '-E1.238', 'base_rate': 60, 'current_rate': 60, 'vtype': 'car'},
            'f_7_cars': {'from': '-E0', 'to': 'E1.200', 'base_rate': 60, 'current_rate': 60, 'vtype': 'car'},
            'f_8_cars': {'from': 'E1', 'to': '-E0.254', 'base_rate': 60, 'current_rate': 60, 'vtype': 'car'},
            'f_9_cars': {'from': 'E1', 'to': 'E0.319', 'base_rate': 60, 'current_rate': 60, 'vtype': 'car'},
            'f_10_cars': {'from': 'E1', 'to': 'E1.200', 'base_rate': 120, 'current_rate': 120, 'vtype': 'car'},
            'f_11_cars': {'from': '-E0', 'to': '-E0.254', 'base_rate': 120, 'current_rate': 120, 'vtype': 'car'},
            
            # Motorcycle/Bike flows (also reduced)
            'f_0_bikes': {'from': 'E0', 'to': 'E0.319', 'base_rate': 80, 'current_rate': 80, 'vtype': 'motorcycle'},
            'f_1_bikes': {'from': 'E0', 'to': '-E1.238', 'base_rate': 40, 'current_rate': 40, 'vtype': 'motorcycle'},
            'f_2_bikes': {'from': 'E0', 'to': 'E1.200', 'base_rate': 40, 'current_rate': 40, 'vtype': 'motorcycle'},
            'f_3_bikes': {'from': '-E1', 'to': '-E1.238', 'base_rate': 80, 'current_rate': 80, 'vtype': 'motorcycle'},
            'f_4_bikes': {'from': '-E1', 'to': '-E0.254', 'base_rate': 40, 'current_rate': 40, 'vtype': 'motorcycle'},
            'f_5_bikes': {'from': '-E1', 'to': 'E0.319', 'base_rate': 40, 'current_rate': 40, 'vtype': 'motorcycle'},
            'f_6_bikes': {'from': '-E0', 'to': '-E1.238', 'base_rate': 40, 'current_rate': 40, 'vtype': 'motorcycle'},
            'f_7_bikes': {'from': '-E0', 'to': 'E1.200', 'base_rate': 40, 'current_rate': 40, 'vtype': 'motorcycle'},
            'f_8_bikes': {'from': 'E1', 'to': '-E0.254', 'base_rate': 40, 'current_rate': 40, 'vtype': 'motorcycle'},
            'f_9_bikes': {'from': 'E1', 'to': 'E0.319', 'base_rate': 40, 'current_rate': 40, 'vtype': 'motorcycle'},
            'f_10_bikes': {'from': 'E1', 'to': 'E1.200', 'base_rate': 80, 'current_rate': 80, 'vtype': 'motorcycle'},
            'f_11_bikes': {'from': '-E0', 'to': '-E0.254', 'base_rate': 80, 'current_rate': 80, 'vtype': 'motorcycle'}
        }

        self.flow_update_interval = 300  # Update flows every 5 minutes
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
        print(f"🚗🏍️  Mixed Traffic: Cars + Motorcycles/Bikes")
        
        car_flows = 0
        bike_flows = 0
        
        for flow_id, config in self.flow_configs.items():
            random_factor = self.generate_random_variation()
            new_rate = self.calculate_new_flow_rate(
                config['base_rate'], 
                time_factor, 
                random_factor
            )
            
            config['current_rate'] = new_rate
            self.flow_variations[flow_id] = new_rate
            
            vehicle_type = config.get('vtype', 'car')
            if vehicle_type == 'car':
                car_flows += new_rate
            else:
                bike_flows += new_rate
            
            type_icon = "🚗" if vehicle_type == 'car' else "🏍️"
            print(f"   {type_icon} {flow_id}: {config['from']} → {config['to']} = {new_rate} veh/h")
        
        print(f"📊 Total Flow Summary: Cars: {car_flows} veh/h | Bikes: {bike_flows} veh/h")
        
        self.last_update = step
        return True
    
    def get_current_flow_rate(self, flow_id):
        """Get current flow rate for a specific flow"""
        return self.flow_configs.get(flow_id, {}).get('current_rate', 500)
    
    def get_flow_summary(self):
        """Get summary of current flow rates"""
        summary = {}
        total_cars = 0
        total_bikes = 0
        
        for flow_id, config in self.flow_configs.items():
            vehicle_type = config.get('vtype', 'car')
            summary[flow_id] = {
                'from': config['from'],
                'to': config['to'],
                'base_rate': config['base_rate'],
                'current_rate': config['current_rate'],
                'efficiency': (config['current_rate'] / config['base_rate']) * 100,
                'vehicle_type': vehicle_type
            }
            
            if vehicle_type == 'car':
                total_cars += config['current_rate']
            else:
                total_bikes += config['current_rate']
        
        summary['totals'] = {
            'total_cars': total_cars,
            'total_bikes': total_bikes,
            'total_all': total_cars + total_bikes,
            'car_percentage': (total_cars / (total_cars + total_bikes)) * 100 if (total_cars + total_bikes) > 0 else 0
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

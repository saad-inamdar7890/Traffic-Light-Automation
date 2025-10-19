#!/usr/bin/env python3
"""
Quick Test for Corrected Complex Network
"""

import os
import subprocess

def test_corrected_network():
    """Test the corrected network files."""
    
    print("🧪 Testing Corrected Complex Network...")
    
    # Test configuration
    test_config = """<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="complex_network_fixed.net.xml"/>
        <route-files value="complex_routes_fixed.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="300"/>
        <step-length value="1"/>
    </time>
    <processing>
        <max-depart-delay value="60"/>
    </processing>
    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
        <duration-log.disable value="true"/>
    </report>
</configuration>"""
    
    # Write test config
    with open("test_corrected.sumocfg", "w") as f:
        f.write(test_config)
    
    try:
        # Run test
        cmd = ["sumo", "-c", "test_corrected.sumocfg", "--no-warnings"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   ✅ Corrected network test PASSED!")
            print("   🎉 Network is ready for use!")
            
            # Clean up
            if os.path.exists("test_corrected.sumocfg"):
                os.remove("test_corrected.sumocfg")
            
            return True
        else:
            print(f"   ❌ Test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_corrected_network()
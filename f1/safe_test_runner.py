"""
Safe test runner with proper encoding handling
"""
import sys
import os
import locale

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Import the main test
from test_improved_algorithm import run_comprehensive_test, generate_comprehensive_report, create_comprehensive_visualization
import time

def main():
    """Main test function with safe encoding"""
    print("TRAFFIC LIGHT ALGORITHM TEST")
    print("=" * 50)
    print("Testing improved adaptive algorithm vs baseline")
    print("=" * 50)
    
    try:
        # Run the test
        start_time = time.time()
        results = run_comprehensive_test()
        end_time = time.time()
        
        print(f"\nTest completed in {end_time - start_time:.1f} seconds")
        print("Algorithm testing finished successfully!")
        
        return results
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
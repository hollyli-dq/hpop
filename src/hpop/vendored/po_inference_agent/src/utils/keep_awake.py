#!/usr/bin/env python3
"""
Simple Keep Awake Script
This script keeps your computer awake indefinitely using Windows API calls.
Run this in a separate terminal while your MCMC scripts are running.
"""

import ctypes
import time
from datetime import datetime

# Windows API constants
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

def keep_awake():
    """Keep the computer awake indefinitely."""
    
    kernel32 = ctypes.windll.kernel32
    
    print("=" * 60)
    print("KEEP AWAKE SYSTEM ACTIVATED")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("This script will keep your computer awake indefinitely.")
    print("Your computer will NOT go to sleep while this is running.")
    print("Press Ctrl+C to stop and allow normal sleep behavior.")
    print("=" * 60)
    print()
    
    try:
        while True:
            # Set both system and display required to prevent all types of sleep
            kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            )
            
            # Show status every 5 minutes
            current_time = datetime.now()
            print(f"[{current_time.strftime('%H:%M:%S')}] Computer kept awake - Sleep prevention active")
            
            # Sleep for 5 minutes before next refresh
            time.sleep(300)
            
    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("STOPPING KEEP AWAKE SYSTEM")
        print("=" * 60)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Normal sleep behavior will be restored.")
        print("Your computer can now go to sleep normally.")
        
        # Restore normal sleep behavior
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)

if __name__ == "__main__":
    keep_awake()

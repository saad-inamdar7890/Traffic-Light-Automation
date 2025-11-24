"""
Automatic Checkpoint Management for Kaggle Training
===================================================

This script automates checkpoint saving, zipping, and dataset folder creation
during background training on Kaggle. It monitors the training process and
automatically packages checkpoints as they're created.

Features:
- Automatic checkpoint detection
- Background zip creation (doesn't block training)
- Dataset folder structure creation
- Periodic checkpoint packaging (every N episodes)
- Auto-cleanup of old zips (keep last N)

Usage in Kaggle Notebook:
-------------------------
# Cell 1: Start training in background
!nohup python s1/mappo_k1_implementation.py --num-episodes 200 --max-hours 9 > training.log 2>&1 &

# Cell 2: Start auto checkpoint manager (runs separately)
!python kaggle_auto_checkpoint.py --monitor --interval 3600 --keep-last 3

# Cell 3: Check status anytime
!tail -50 training.log
!ls -lh /kaggle/working/mappo-checkpoint-dataset/
"""

import os
import sys
import time
import shutil
import argparse
import json
from pathlib import Path
from datetime import datetime
from threading import Thread
import glob


class KaggleCheckpointManager:
    """Manages automatic checkpoint packaging for Kaggle"""
    
    def __init__(self, 
                 checkpoint_base_dir="mappo_models",
                 dataset_folder="mappo-checkpoint-dataset",
                 working_dir="/kaggle/working",
                 keep_last=3):
        """
        Args:
            checkpoint_base_dir: Where training saves checkpoints
            dataset_folder: Dataset folder name for Kaggle
            working_dir: Kaggle working directory
            keep_last: Number of recent checkpoint zips to keep
        """
        self.checkpoint_base_dir = checkpoint_base_dir
        self.dataset_folder = dataset_folder
        self.working_dir = working_dir
        self.keep_last = keep_last
        
        # Create dataset folder
        self.dataset_path = os.path.join(working_dir, dataset_folder)
        os.makedirs(self.dataset_path, exist_ok=True)
        
        # Track processed checkpoints
        self.processed_checkpoints = set()
        self._load_processed_list()
        
    def _load_processed_list(self):
        """Load list of already processed checkpoints"""
        processed_file = os.path.join(self.dataset_path, ".processed_checkpoints.json")
        if os.path.exists(processed_file):
            with open(processed_file, 'r') as f:
                self.processed_checkpoints = set(json.load(f))
    
    def _save_processed_list(self):
        """Save list of processed checkpoints"""
        processed_file = os.path.join(self.dataset_path, ".processed_checkpoints.json")
        with open(processed_file, 'w') as f:
            json.dump(list(self.processed_checkpoints), f)
    
    def find_new_checkpoints(self):
        """Find checkpoints that haven't been zipped yet"""
        checkpoint_pattern = os.path.join(self.checkpoint_base_dir, "checkpoint_*")
        all_checkpoints = glob.glob(checkpoint_pattern)
        
        new_checkpoints = []
        for ckpt_dir in all_checkpoints:
            if os.path.isdir(ckpt_dir):
                ckpt_name = os.path.basename(ckpt_dir)
                if ckpt_name not in self.processed_checkpoints:
                    # Verify checkpoint is complete (has critical files)
                    if self._is_checkpoint_complete(ckpt_dir):
                        new_checkpoints.append(ckpt_dir)
        
        return sorted(new_checkpoints)
    
    def _is_checkpoint_complete(self, checkpoint_dir):
        """Check if checkpoint has all necessary files"""
        required_files = [
            "training_state.json",
            "critic.pt",
            "config.json"
        ]
        
        for req_file in required_files:
            if not os.path.exists(os.path.join(checkpoint_dir, req_file)):
                return False
        
        # Check for at least one actor file
        actor_files = glob.glob(os.path.join(checkpoint_dir, "actor_*.pt"))
        return len(actor_files) > 0
    
    def zip_checkpoint(self, checkpoint_dir):
        """
        Zip a checkpoint and move to dataset folder
        
        Returns:
            str: Path to created zip file, or None if failed
        """
        checkpoint_name = os.path.basename(checkpoint_dir)
        
        print(f"📦 Zipping checkpoint: {checkpoint_name}")
        start_time = time.time()
        
        try:
            # Create zip in dataset folder
            output_zip_base = os.path.join(self.dataset_path, checkpoint_name)
            shutil.make_archive(output_zip_base, 'zip', checkpoint_dir)
            
            zip_path = output_zip_base + '.zip'
            zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
            elapsed = time.time() - start_time
            
            print(f"   ✅ Created: {os.path.basename(zip_path)}")
            print(f"   📊 Size: {zip_size_mb:.2f} MB")
            print(f"   ⏱️  Time: {elapsed:.1f}s")
            
            # Mark as processed
            self.processed_checkpoints.add(checkpoint_name)
            self._save_processed_list()
            
            return zip_path
            
        except Exception as e:
            print(f"   ❌ Error zipping {checkpoint_name}: {e}")
            return None
    
    def cleanup_old_zips(self):
        """Keep only the last N checkpoint zips"""
        zip_pattern = os.path.join(self.dataset_path, "checkpoint_*.zip")
        all_zips = sorted(glob.glob(zip_pattern), key=os.path.getmtime)
        
        if len(all_zips) > self.keep_last:
            to_remove = all_zips[:-self.keep_last]
            for old_zip in to_remove:
                try:
                    os.remove(old_zip)
                    print(f"🗑️  Removed old zip: {os.path.basename(old_zip)}")
                except Exception as e:
                    print(f"⚠️  Could not remove {os.path.basename(old_zip)}: {e}")
    
    def process_all_new(self):
        """Process all new checkpoints"""
        new_checkpoints = self.find_new_checkpoints()
        
        if not new_checkpoints:
            return 0
        
        print(f"\n{'='*70}")
        print(f"🔍 Found {len(new_checkpoints)} new checkpoint(s) to process")
        print(f"{'='*70}")
        
        processed_count = 0
        for ckpt_dir in new_checkpoints:
            if self.zip_checkpoint(ckpt_dir):
                processed_count += 1
        
        if processed_count > 0:
            self.cleanup_old_zips()
            self.print_summary()
        
        return processed_count
    
    def print_summary(self):
        """Print summary of dataset folder"""
        print(f"\n{'='*70}")
        print(f"📂 Dataset Folder Summary")
        print(f"{'='*70}")
        print(f"Location: {self.dataset_path}")
        
        zip_pattern = os.path.join(self.dataset_path, "checkpoint_*.zip")
        all_zips = sorted(glob.glob(zip_pattern), key=os.path.getmtime)
        
        if all_zips:
            total_size = sum(os.path.getsize(z) for z in all_zips)
            print(f"Checkpoints: {len(all_zips)} zips")
            print(f"Total size: {total_size / (1024*1024):.2f} MB")
            print(f"\nLatest checkpoints:")
            for zip_path in all_zips[-3:]:
                zip_name = os.path.basename(zip_path)
                zip_size = os.path.getsize(zip_path) / (1024*1024)
                mod_time = datetime.fromtimestamp(os.path.getmtime(zip_path))
                print(f"  • {zip_name} ({zip_size:.1f} MB) - {mod_time.strftime('%Y-%m-%d %H:%M')}")
        else:
            print("No checkpoints zipped yet")
        
        print(f"{'='*70}\n")
    
    def monitor_loop(self, check_interval=3600):
        """
        Continuously monitor for new checkpoints
        
        Args:
            check_interval: Seconds between checks (default: 1 hour)
        """
        print(f"🔄 Starting checkpoint monitor")
        print(f"   Check interval: {check_interval}s ({check_interval/60:.0f} min)")
        print(f"   Dataset folder: {self.dataset_path}")
        print(f"   Keeping last: {self.keep_last} checkpoints")
        print(f"   Press Ctrl+C to stop\n")
        
        check_count = 0
        try:
            while True:
                check_count += 1
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{timestamp}] Check #{check_count}: Scanning for new checkpoints...")
                
                processed = self.process_all_new()
                
                if processed == 0:
                    print(f"   No new checkpoints found")
                
                print(f"   Next check in {check_interval/60:.0f} minutes\n")
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print(f"\n✋ Monitor stopped by user")
            self.print_summary()


def package_latest_checkpoint(checkpoint_dir=None, dataset_folder="mappo-checkpoint-dataset"):
    """
    Quick function to package the latest checkpoint (for manual use)
    
    Args:
        checkpoint_dir: Specific checkpoint to package, or None for latest
        dataset_folder: Dataset folder name
    """
    manager = KaggleCheckpointManager(dataset_folder=dataset_folder)
    
    if checkpoint_dir is None:
        # Find latest checkpoint
        checkpoint_pattern = "mappo_models/checkpoint_*"
        all_checkpoints = sorted(glob.glob(checkpoint_pattern), key=os.path.getmtime)
        if not all_checkpoints:
            print("❌ No checkpoints found in mappo_models/")
            return
        checkpoint_dir = all_checkpoints[-1]
    
    print(f"📦 Packaging checkpoint: {checkpoint_dir}")
    zip_path = manager.zip_checkpoint(checkpoint_dir)
    
    if zip_path:
        manager.print_summary()
        
        # Show download link if in Jupyter/IPython
        try:
            from IPython.display import FileLink
            print(f"\n📥 Download link:")
            return FileLink(zip_path)
        except ImportError:
            print(f"\n📥 Download from: {zip_path}")


def main():
    parser = argparse.ArgumentParser(description='Kaggle Checkpoint Auto-Packaging')
    parser.add_argument('--monitor', action='store_true', 
                       help='Monitor mode: continuously check for new checkpoints')
    parser.add_argument('--interval', type=int, default=3600,
                       help='Check interval in seconds (default: 3600 = 1 hour)')
    parser.add_argument('--keep-last', type=int, default=3,
                       help='Number of recent checkpoint zips to keep (default: 3)')
    parser.add_argument('--checkpoint-dir', type=str,
                       help='Package specific checkpoint directory')
    parser.add_argument('--dataset-folder', type=str, default='mappo-checkpoint-dataset',
                       help='Dataset folder name (default: mappo-checkpoint-dataset)')
    parser.add_argument('--working-dir', type=str, default='/kaggle/working',
                       help='Working directory (default: /kaggle/working)')
    
    args = parser.parse_args()
    
    if args.monitor:
        # Monitor mode: continuous background processing
        manager = KaggleCheckpointManager(
            dataset_folder=args.dataset_folder,
            working_dir=args.working_dir,
            keep_last=args.keep_last
        )
        manager.monitor_loop(check_interval=args.interval)
    
    elif args.checkpoint_dir:
        # Package specific checkpoint
        package_latest_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            dataset_folder=args.dataset_folder
        )
    
    else:
        # Package latest checkpoint
        package_latest_checkpoint(dataset_folder=args.dataset_folder)


if __name__ == '__main__':
    main()

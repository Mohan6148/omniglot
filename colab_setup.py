#!/usr/bin/env python3
"""
Google Colab Setup Script for Gaussian Prototypical Networks
This script automates the setup process for running the project in Google Colab
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """Run a shell command and print status"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"✗ {description} failed")
        if result.stderr:
            print(result.stderr)
        return False
    return True

def main():
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║   Gaussian Prototypical Networks - Google Colab Setup      ║
    ║   Few-Shot Learning on Omniglot Dataset                    ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Create necessary directories
    print("\n📁 Creating necessary directories...")
    directories = ['data', 'tmp_imgs', 'model_checkpoints']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ Created: {directory}/")
    
    # Step 2: Download Omniglot dataset
    print("\n📥 Downloading Omniglot dataset...")
    os.chdir('data')
    
    # Download background images
    if not os.path.exists('images_background'):
        run_command(
            'wget -q https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
            'Downloading background images'
        )
        run_command('unzip -q images_background.zip', 'Extracting background images')
        run_command('rm images_background.zip', 'Cleaning up background zip')
    else:
        print("  ✓ Background images already exist")
    
    # Download evaluation images
    if not os.path.exists('images_evaluation'):
        run_command(
            'wget -q https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip',
            'Downloading evaluation images'
        )
        run_command('unzip -q images_evaluation.zip', 'Extracting evaluation images')
        run_command('rm images_evaluation.zip', 'Cleaning up evaluation zip')
    else:
        print("  ✓ Evaluation images already exist")
    
    os.chdir('..')
    
    # Step 3: Generate file lists
    print("\n📝 Generating file lists...")
    os.chdir('data')
    
    # Generate background list
    if not os.path.exists('images_background_list.txt'):
        run_command(
            'find images_background -name "*.png" | sort > images_background_list.txt',
            'Creating background images list'
        )
    
    # Generate evaluation list
    if not os.path.exists('images_evaluation_list.txt'):
        run_command(
            'find images_evaluation -name "*.png" | sort > images_evaluation_list.txt',
            'Creating evaluation images list'
        )
    
    # Generate small subsets for quick prototyping
    if not os.path.exists('images_background_small1_list.txt'):
        run_command(
            'head -n 400 images_background_list.txt > images_background_small1_list.txt',
            'Creating small background subset 1'
        )
    
    if not os.path.exists('images_background_small2_list.txt'):
        run_command(
            'head -n 800 images_background_list.txt | tail -n 400 > images_background_small2_list.txt',
            'Creating small background subset 2'
        )
    
    os.chdir('..')
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║                    Setup Complete! ✓                       ║
    ╚════════════════════════════════════════════════════════════╝
    
    You can now run the classifier:
        python classifier7.py
    
    The default configuration will:
    - Train on a small subset (400 images) for quick testing
    - Use a small encoder (64-dim embeddings)
    - Run for 30 epochs
    - Save checkpoints to model_checkpoints/
    - Save visualizations to tmp_imgs/
    """)

if __name__ == "__main__":
    main()

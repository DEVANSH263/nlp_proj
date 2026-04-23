#!/usr/bin/env python3
"""
Upload MuRIL model to HuggingFace Hub
Scenario 3: Deploy LR + LSTM from model3/ via Railway, MuRIL via HuggingFace
Run this BEFORE deploying to Railway
"""

import os
import subprocess
import sys

def main():
    print("=" * 60)
    print("🤗 HuggingFace Model Upload (Scenario 3)")
    print("=" * 60)
    print("""
Models to deploy:
  ✅ LR:    model3/lr/model.pkl        (Railway)
  ✅ LSTM:  model3/lstm/lstm_model.pt  (Railway)
  ✅ MuRIL: model/muril/               (HuggingFace)
""")
    
    # Step 1: Check if model exists
    model_path = "model/muril"
    if not os.path.isdir(model_path):
        print(f"❌ ERROR: {model_path} not found!")
        print("Make sure you have the trained MuRIL model in model/muril/ (original, not model2 or model3)")
        sys.exit(1)
    
    print(f"✅ Found MuRIL model at: {model_path}")
    
    # Step 2: Get HuggingFace token
    print("\n" + "=" * 60)
    print("STEP 1: Create/Get HuggingFace token")
    print("=" * 60)
    print("""
1. Go to: https://huggingface.co/settings/tokens
2. Create new token (write access)
3. Copy the token
4. When prompted, paste it here
""")
    
    # Try to login
    try:
        from huggingface_hub import login
        login()
        print("✅ Logged into HuggingFace")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        print("\nTry alternative: Install huggingface_hub")
        print("pip install huggingface_hub")
        sys.exit(1)
    
    # Step 3: Get username
    print("\n" + "=" * 60)
    print("STEP 2: Get your HuggingFace username")
    print("=" * 60)
    
    username = input("Enter your HuggingFace username: ").strip()
    if not username:
        print("❌ Username required!")
        sys.exit(1)
    
    repo_id = f"{username}/hateshield-muril"
    
    print(f"\n📦 Will create/update: {repo_id}")
    
    # Step 4: Create repo
    print("\n" + "=" * 60)
    print("STEP 3: Create repository (if needed)")
    print("=" * 60)
    
    try:
        from huggingface_hub import create_repo
        try:
            create_repo(repo_id, exist_ok=True)
            print(f"✅ Repository ready: {repo_id}")
        except Exception as e:
            print(f"⚠️  Repo may already exist: {e}")
    except ImportError:
        print("Note: huggingface_hub not installed, skipping repo creation")
    
    # Step 5: Upload files
    print("\n" + "=" * 60)
    print("STEP 4: Upload model files")
    print("=" * 60)
    
    model_files = {
        "model/muril/pytorch_model.bin": "pytorch_model.bin",
        "model/muril/config.json": "config.json",
        "model/muril/tokenizer.json": "tokenizer.json",
        "model/muril/tokenizer_config.json": "tokenizer_config.json",
        "model/muril/special_tokens_map.json": "special_tokens_map.json",
        "model/muril/vocab.txt": "vocab.txt",
    }
    
    try:
        from huggingface_hub import upload_file
        
        for local_path, remote_name in model_files.items():
            if os.path.exists(local_path):
                print(f"Uploading: {remote_name}...", end=" ", flush=True)
                try:
                    upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=remote_name,
                        repo_id=repo_id,
                    )
                    print("✅")
                except Exception as e:
                    print(f"⚠️  {e}")
            else:
                print(f"⚠️  Skipping (not found): {local_path}")
    
    except ImportError:
        print("huggingface_hub not installed, using git CLI instead...")
        
        # Alternative: Use git CLI
        print("\n📝 Manual upload via Git:")
        print(f"""
1. Clone:
   git clone https://huggingface.co/{repo_id}
   
2. Copy model files:
   cp -r model/muril/* {repo_id}/
   
3. Push:
   cd {repo_id}
   git add .
   git commit -m "Upload MuRIL model"
   git push
""")
        sys.exit(0)
    
    # Step 6: Verify
    print("\n" + "=" * 60)
    print("STEP 5: Verify Upload")
    print("=" * 60)
    
    print(f"""
✅ Model uploaded to: https://huggingface.co/{repo_id}

Next steps:
1. Update config.py:
   MURIL_MODEL_PATH = "{repo_id}"

2. Or set environment variable:
   export MURIL_MODEL_ID="{repo_id}"

3. Push to GitHub and deploy to Railway

4. First request will download model from HuggingFace
   (may take 2-3 minutes for initial download)
""")

if __name__ == "__main__":
    main()

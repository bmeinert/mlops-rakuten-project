#!/usr/bin/env python3
"""
dvc_push_manager.py
DVC Push Logic for ML Pipeline
Handles DVC tracking, Git operations, and error recovery more robustly.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DVCPushManager:
    # Manages DVC and Git operations for ML pipeline artifacts
    
    def __init__(self):
        self.cwd = os.getcwd()
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_owner = os.getenv("GITHUB_REPO_OWNER") 
        self.repo_name = os.getenv("GITHUB_REPO_NAME")
        self.credentials_configured = False
        
    def _run_command(self, cmd: List[str], cwd: Optional[str] = None, 
                    capture_output: bool = False, check: bool = True) -> subprocess.CompletedProcess:
        # Execute a command with proper error handling
        cwd = cwd or self.cwd
        try:
            logger.info(f"Running: {' '.join(cmd)} in {cwd}")
            result = subprocess.run(
                cmd, 
                cwd=cwd, 
                capture_output=capture_output, 
                text=True, 
                check=check
            )
            if capture_output and result.stdout:
                logger.debug(f"Output: {result.stdout}")
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {' '.join(cmd)}")
            logger.error(f"Error: {e}")
            if capture_output and e.stdout:
                logger.error(f"Stdout: {e.stdout}")
            if capture_output and e.stderr:
                logger.error(f"Stderr: {e.stderr}")
            raise
    
    def setup_git_credentials(self) -> bool:
        if self.credentials_configured:
            return True

        if not all([self.github_token, self.repo_owner, self.repo_name]):
            logger.error("GitHub credentials not properly set in environment")
            return False

        try:
            # Remote-URL 
            remote_url = (
                f"https://github.com/{self.repo_owner}/{self.repo_name}.git"
            )
            self._run_command(["git", "remote", "set-url", "origin", remote_url])

            # .git-credentials 
            cred_file = os.path.expanduser("~/.git-credentials")
            os.makedirs(os.path.dirname(cred_file), exist_ok=True)
            with open(cred_file, "w") as fh:
                fh.write(
                    f"https://x-access-token:{self.github_token}@github.com\n"
                )

            self._run_command(
                ["git", "config", "--global", "credential.helper", "store"]
            )
            self._run_command(
                ["git", "config", "--global", "user.email", "pipeline@example.com"]
            )
            self._run_command(
                ["git", "config", "--global", "user.name", "ML Pipeline"]
            )

            self.credentials_configured = True
            logger.info("Git credentials configured successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to setup git credentials: {e}")
            return False

    def find_dvc_files(self) -> List[Path]:
        # Find all DVC files in the repository
        dvc_files = list(Path(self.cwd).rglob("*.dvc"))
        logger.info(f"Found {len(dvc_files)} DVC files: {[str(f.relative_to(self.cwd)) for f in dvc_files]}")
        return dvc_files
    
    def check_dvc_status(self) -> Tuple[List[str], List[str]]:
        # Check DVC status to find modified files
        try:
            result = self._run_command(["dvc", "status"], capture_output=True, check=False)
            if result.returncode == 0 and not result.stdout.strip():
                logger.info("DVC status: No changes detected")
                return [], []
            
            # Parse DVC status output
            modified_files = []
            new_files = []
            
            if result.stdout:
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('DVC'):
                        if 'modified:' in line:
                            modified_files.append(line.split('modified:')[1].strip())
                        elif 'new file:' in line:
                            new_files.append(line.split('new file:')[1].strip())
            
            return modified_files, new_files
            
        except Exception as e:
            logger.warning(f"Could not check DVC status: {e}")
            return [], []
    
    def update_dvc_files(self, force_all: bool = False) -> List[str]:
        # Update DVC tracking for modified files
        updated_files = []
        
        if force_all:
            # Force update all DVC files
            dvc_files = self.find_dvc_files()
            for dvc_file in dvc_files:
                tracked_path = str(dvc_file).replace('.dvc', '')
                if os.path.exists(tracked_path):
                    try:
                        rel_path = os.path.relpath(tracked_path, self.cwd)
                        self._run_command(["dvc", "add", "--force", rel_path])
                        updated_files.append(str(dvc_file.relative_to(self.cwd)))
                        logger.info(f"Force updated DVC tracking for {rel_path}")
                    except Exception as e:
                        logger.warning(f"Could not force update {tracked_path}: {e}")
        else:
            # Only update modified files
            modified_files, new_files = self.check_dvc_status()
            all_files = modified_files + new_files
            
            for file_path in all_files:
                try:
                    self._run_command(["dvc", "add", file_path])
                    dvc_file = file_path + '.dvc'
                    if os.path.exists(dvc_file):
                        updated_files.append(dvc_file)
                    logger.info(f"Updated DVC tracking for {file_path}")
                except Exception as e:
                    logger.warning(f"Could not update DVC tracking for {file_path}: {e}")
        
        return updated_files
    
    def commit_and_push_git(self, updated_files: List[str], description: str) -> bool:
        # Commit and push changes to Git
        if not updated_files:
            logger.info("No DVC files to commit")
            return True
        
        try:
            # Add DVC files to git
            for dvc_file in updated_files:
                self._run_command(["git", "add", dvc_file])
                logger.info(f"Added {dvc_file} to git")
            
            # Check if there are changes to commit
            result = self._run_command(["git", "status", "--porcelain"], capture_output=True)
            if not result.stdout.strip():
                logger.info("No changes to commit")
                return True
                
            # Commit changes
            commit_msg = f"dvc: {description}"
            self._run_command(["git", "commit", "-m", commit_msg])
            logger.info(f"Committed changes: {description}")
            
            # Push to git
            self._run_command(["git", "push"])
            logger.info("Successfully pushed to Git")
            
            return True
            
        except Exception as e:
            logger.error(f"Git operations failed: {e}")
            return False
    
    def push_dvc(self) -> bool:
        # Push DVC data to remote storage
        try:
            self._run_command(["dvc", "push"])
            logger.info("Successfully pushed to DVC remote")
            return True
        except Exception as e:
            logger.error(f"DVC push failed: {e}")
            return False
    
    def track_and_push(self, description: str, force_all: bool = False) -> bool:
        # Main method to track and push DVC changes
        logger.info(f"Starting DVC track and push: {description}")
        logger.info(f"Current working directory: {self.cwd}")
        logger.info(f"Directory contents: {os.listdir(self.cwd)}")
        
        # Setup git credentials
        if not self.setup_git_credentials():
            logger.error("Failed to setup git credentials, continuing anyway...")
            return False
        
        try:
            # Update DVC files
            updated_files = self.update_dvc_files(force_all=force_all)
            if not updated_files:
                logger.info("No DVC files were updated")
                return True
            
            # Commit and push to git
            git_success = self.commit_and_push_git(updated_files, description)
            if not git_success:
                logger.error("Git operations failed, skipping DVC push")
                return False
            
            # Push to DVC remote
            dvc_success = self.push_dvc()
            
            if git_success and dvc_success:
                logger.info(f"Successfully processed DVC files: {', '.join(updated_files)}")
                return True
            else:
                logger.warning("Some operations failed, but continuing...")
                return False
                
        except Exception as e:
            logger.error(f"Error during track and push: {e}")
            return False



# Track and Push with retry logic
def track_and_push_with_retry(description: str, max_retries: int = 3, force_all: bool = False) -> bool:
    manager = DVCPushManager()
    
    for attempt in range(max_retries):
        logger.info(f"Attempt {attempt + 1}/{max_retries} for: {description}")
        success = manager.track_and_push(description, force_all=force_all)
        
        if success:
            return True
        
        if attempt < max_retries - 1:
            logger.info(f"Retrying in 5 seconds...")
            import time
            time.sleep(5)
    
    logger.error(f"All {max_retries} attempts failed for: {description}")
    return False

if __name__ == "__main__":
    description = sys.argv[1] if len(sys.argv) > 1 else "test push"
    force_all = "--force" in sys.argv
    
    success = track_and_push_with_retry(description, force_all=force_all)
    sys.exit(0 if success else 1)
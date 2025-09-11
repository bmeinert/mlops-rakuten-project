# How To: DVC Setup

1. Create Virtual Enviroment - Run in Console (Optional): 
- python -m venv env
- env\Scripts\activate

2. Install DVC - Run in Console: 
- pip install dvc
- pip install "dvc[s3]"

3. Setup Credentials (has to be done by every colaborator individually) 
- Access Key can be found on: https://dagshub.com/BeritM/mlops-rakuten-project/src/develop
- Follow: Remote --> Data --> DVC --> S3 --> Scroll down to see: Setup credentials

4. Run in Console:
- dvc remote modify origin --local access_key_id <your access key id>
- dvc remote modify origin --local secret_access_key <your access key>

5. Now execute: dvc pull and it should be running with Dagshub as the remote storage. 
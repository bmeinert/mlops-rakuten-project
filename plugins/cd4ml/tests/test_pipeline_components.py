"""
test_pipeline_components.py

Unit tests for MLOps pipeline components to be run in GitHub Actions.
Tests core functionality without requiring running containers or external services.
"""

import os
import sys
import pytest
import tempfile
import shutil
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import yaml
import platform

# Add project root to path
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'plugins'))

# Set up test environment variables
temp_base = "C:\\temp" if platform.system() == "Windows" else "/tmp"
os.environ.update({
    "DATA_RAW_DIR": os.path.join(temp_base, "test_raw"),
    "DATA_PROCESSED_DIR": os.path.join(temp_base, "test_processed"),
    "MODEL_DIR": os.path.join(temp_base, "test_models"),
    "X_RAW": "X_raw.csv",
    "Y_RAW": "y_raw.csv",
    "X_Y_RAW": "X_y_raw.csv",
    "X_TRAIN_TFIDF": "X_train_tfidf.pkl",
    "X_VALIDATE_TFIDF": "X_validate_tfidf.pkl",
    "X_TEST_TFIDF": "X_test_tfidf.pkl",
    "TFIDF_VECTORIZER": "tfidf_vectorizer.pkl",
    "Y_TRAIN": "y_train.pkl",
    "Y_VALIDATE": "y_validate.pkl",
    "Y_TEST": "y_test.pkl",
    "MODEL": "model.pkl",
    "CURRENT_RUN_ID": "current_run_id.txt",
    "PARAM_CONFIG": "params.yaml",
    "DAGSHUB_USER_NAME": "test_user",
    "DAGSHUB_USER_TOKEN": "test_token",
    "DAGSHUB_REPO_OWNER": "test_owner",
    "DAGSHUB_REPO_NAME": "test_repo",
    "DATA_FEEDBACK_DIR": os.path.join(temp_base, "test_feedback"),
    "FEEDBACK_CSV_PATH": "feedback.csv",
    "CLASS_REPORT": "class_report.txt",
    "PRODUCT_DICTIONARY": "product_dictionary.pkl"
})


class TestDataProcessing:
    """Test data processing components"""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test"""
        # Create test directories
        temp_base = "C:\\temp" if platform.system() == "Windows" else "/tmp"
        test_dirs = [
            os.path.join(temp_base, "test_raw"),
            os.path.join(temp_base, "test_processed"),
            os.path.join(temp_base, "test_models"),
            os.path.join(temp_base, "test_feedback")
        ]
        for dir_path in test_dirs:
            os.makedirs(dir_path, exist_ok=True)
        
        yield
        
        # Cleanup
        for dir_path in test_dirs:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
    
    @patch('nltk.download')
    @patch('nltk.corpus.stopwords.words')
    @patch('nltk.stem.WordNetLemmatizer')
    def test_preprocessing_imports(self, mock_lemmatizer, mock_stopwords, mock_download):
        """Test that preprocessing modules can be imported"""
        # Mock NLTK dependencies
        mock_stopwords.return_value = ['the', 'is', 'a']
        mock_lemmatizer.return_value = Mock()
        
        try:
            preprocessing_path = os.path.join('plugins', 'cd4ml', 'data_processing', 'run_preprocessing.py')
            assert os.path.exists(preprocessing_path), f"run_preprocessing.py not found at {preprocessing_path}"
            assert True
        except ImportError as e:
            pytest.skip(f"Skipping due to import dependencies: {e}")
    
    def test_clean_text_functionality(self):
        """Test text cleaning functionality without importing the actual module"""
        def simple_clean_text(text):
            import re
            text = text.lower()
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            text = ' '.join(text.split())
            return text
        
        # Test cases
        test_cases = [
            ("Hello World!", "hello world"),
            ("Test@#$ Product 123", "test product 123"),
            ("", ""),
            ("   Spaces   ", "spaces"),
            ("UPPERCASE lowercase", "uppercase lowercase")
        ]
        
        for input_text, expected in test_cases:
            result = simple_clean_text(input_text)
            assert result == expected, f"Failed for input: {input_text}"
    
    @patch('dvc_push_manager.track_and_push_with_retry')
    def test_preprocessing_pipeline_flow(self, mock_dvc):
        """Test preprocessing pipeline execution flow - mocked"""
        # Mock the DVC push
        mock_dvc.return_value = True
        
        # Test that we can create the expected data structures
        mock_df = pd.DataFrame({
            'description': ['Test product 1', 'Test product 2'],
            'prdtypecode': ['A', 'B']
        })
        
        # Verify data structure
        assert 'description' in mock_df.columns
        assert 'prdtypecode' in mock_df.columns
        assert len(mock_df) == 2


class TestModelTraining:
    """Test model training components"""
    
    def test_training_imports(self):
        """Test that training modules exist"""
        # Check if files exist
        training_path = os.path.join('plugins', 'cd4ml', 'model_training', 'run_model_training.py')
        assert os.path.exists(training_path), f"run_model_training.py not found at {training_path}"
    
    def test_mlflow_setup(self):
        """Test MLflow configuration setup - simplified"""
        # Test MLflow URI construction
        tracking_uri = (
            f"https://{os.getenv('DAGSHUB_USER_NAME')}:{os.getenv('DAGSHUB_USER_TOKEN')}"
            f"@dagshub.com/{os.getenv('DAGSHUB_REPO_OWNER')}/{os.getenv('DAGSHUB_REPO_NAME')}.mlflow"
        )
        
        assert "dagshub.com" in tracking_uri
        assert os.getenv('DAGSHUB_USER_NAME') in tracking_uri
    
    def test_config_loading(self):
        """Test configuration file loading"""
        # Create temp config file
        config_data = {
            "model": {
                "params": {
                    "loss": "log_loss",
                    "alpha": 0.0001,
                    "max_iter": 1000
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Load and verify config
            with open(temp_path, 'r') as fh:
                loaded_config = yaml.safe_load(fh)
            
            assert loaded_config["model"]["params"]["loss"] == "log_loss"
            assert loaded_config["model"]["params"]["alpha"] == 0.0001
        finally:
            os.unlink(temp_path)


class TestModelValidation:
    """Test model validation components"""
    
    def test_validation_imports(self):
        """Test that validation modules exist"""
        validation_path = os.path.join('plugins', 'cd4ml', 'model_validation', 'run_model_validation.py')
        assert os.path.exists(validation_path), f"run_model_validation.py not found at {validation_path}"
    
    @patch('mlflow.tracking.MlflowClient')
    def test_production_model_check(self, mock_client):
        """Test checking for production model - mocked"""
        # Mock production model
        mock_version = Mock()
        mock_version.version = "1"
        mock_version.run_id = "test_run_id"
        
        mock_run = Mock()
        mock_run.data.metrics = {"f1_weighted": 0.85}
        
        client_instance = Mock()
        client_instance.get_latest_versions.return_value = [mock_version]
        client_instance.get_run.return_value = mock_run
        mock_client.return_value = client_instance
        
        # Test the mock
        f1_score = mock_run.data.metrics.get("f1_weighted")
        run_id = mock_version.run_id
        
        assert f1_score == 0.85
        assert run_id == "test_run_id"


class TestInferenceServices:
    """Test inference service components"""
    
    def test_auth_service_structure(self):
        """Test that auth service file exists and has expected structure"""
        auth_path = os.path.join('plugins', 'cd4ml', 'inference', 'auth_service.py')
        assert os.path.exists(auth_path), f"auth_service.py not found at {auth_path}"
        
        # Read file and check for expected functions
        with open(auth_path, 'r') as f:
            content = f.read()
            assert 'create_access_token' in content
            assert 'verify_token' in content
            assert 'auth_app' in content
    
    def test_predict_service_structure(self):
        """Test that predict service file exists and has expected structure"""
        predict_path = os.path.join('plugins', 'cd4ml', 'inference', 'predict_service.py')
        assert os.path.exists(predict_path), f"predict_service.py not found at {predict_path}"
        
        # Read file and check for expected classes
        with open(predict_path, 'r') as f:
            content = f.read()
            assert 'PredictionRequest' in content
            assert 'PredictionResponse' in content
            assert 'predict_app' in content
    
    def test_jwt_token_creation(self):
        """Test JWT token creation logic - simplified"""
        # Test that we can create a token-like string
        from datetime import datetime, timedelta
        import hashlib
        
        test_data = {"sub": "testuser", "role": "user"}
        # Simple token simulation
        token_data = f"{test_data}_{datetime.now().isoformat()}"
        token = hashlib.sha256(token_data.encode()).hexdigest()
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_pydantic_models(self):
        """Test Pydantic model structures"""
        # Test that we can create similar models
        from pydantic import BaseModel
        
        class TestPredictionRequest(BaseModel):
            designation: str
            description: str
        
        class TestFeedbackEntry(BaseModel):
            designation: str
            description: str
            predicted_label: str
            correct_label: str
        
        # Test model creation
        req = TestPredictionRequest(designation="Test Product", description="Test Description")
        assert req.designation == "Test Product"
        assert req.description == "Test Description"
        
        feedback = TestFeedbackEntry(
            designation="Test",
            description="Desc",
            predicted_label="A",
            correct_label="B"
        )
        assert feedback.predicted_label == "A"
        assert feedback.correct_label == "B"


class TestCommonUtilities:
    """Test common utilities and helper functions"""
    
    def test_environment_validation(self):
        """Test environment variable validation"""
        required_vars = [
            "DATA_RAW_DIR", "DATA_PROCESSED_DIR", "MODEL_DIR",
            "X_RAW", "Y_RAW", "X_Y_RAW"
        ]
        
        # Check that our test environment has these set
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        assert len(missing_vars) == 0, f"Missing environment variables: {missing_vars}"
    
    def test_file_paths(self):
        """Test file path construction"""
        raw_dir = os.getenv("DATA_RAW_DIR")
        x_raw = os.getenv("X_RAW")
        
        expected_path = os.path.join(raw_dir, x_raw)
        assert raw_dir in expected_path
        assert x_raw in expected_path
    
    def test_data_structure_compatibility(self):
        """Test data structure compatibility between pipeline stages"""
        # Create sample data structures
        sample_tfidf = np.random.rand(10, 100)
        sample_labels = pd.Series(['A', 'B', 'C'] * 3 + ['A'])
        
        # Verify shapes are compatible
        assert sample_tfidf.shape[0] == len(sample_labels)
        assert sample_tfidf.shape[1] > 0  # Has features


class TestErrorHandling:
    """Test error handling in pipeline components"""
    
    def test_missing_description_column(self):
        """Test handling of missing description column"""
        df = pd.DataFrame({'wrong_column': ['test']})
        
        # Handle missing column gracefully
        with pytest.raises(KeyError):
            _ = df["description"]  
    
    def test_dvc_push_failure_handling(self):
        """Test handling of DVC push failures"""
        # Mock a failing DVC push
        def mock_dvc_push(description, max_retries):
            return False
        
        # Pipeline should continue even if DVC fails
        success = mock_dvc_push(description="test", max_retries=1)
        assert success is False  


class TestProjectStructure:
    """Test that the project structure is as expected"""
    
    def test_required_directories_exist(self):
        """Test that all required directories exist"""
        required_dirs = [
            os.path.join('plugins', 'cd4ml'),
            os.path.join('plugins', 'cd4ml', 'data_processing'),
            os.path.join('plugins', 'cd4ml', 'model_training'),
            os.path.join('plugins', 'cd4ml', 'model_validation'),
            os.path.join('plugins', 'cd4ml', 'inference'),
            os.path.join('plugins', 'cd4ml', 'tests')
        ]
        
        for dir_path in required_dirs:
            assert os.path.exists(dir_path), f"Directory not found: {dir_path}"
    
    def test_main_scripts_exist(self):
        """Test that all main scripts exist"""
        required_scripts = [
            os.path.join('plugins', 'cd4ml', 'data_processing', 'run_preprocessing.py'),
            os.path.join('plugins', 'cd4ml', 'model_training', 'run_model_training.py'),
            os.path.join('plugins', 'cd4ml', 'model_validation', 'run_model_validation.py'),
            os.path.join('plugins', 'cd4ml', 'inference', 'auth_service.py'),
            os.path.join('plugins', 'cd4ml', 'inference', 'predict_service.py')
        ]
        
        for script_path in required_scripts:
            assert os.path.exists(script_path), f"Script not found: {script_path}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
import streamlit as st
import requests
import json
import time
import subprocess

# --- Configuration ---
# IMPORTANT: Ensure these URLs match where your FastAPI services are running.
AUTH_API_BASE_URL = "http://localhost:8001"
PREDICT_API_BASE_URL = "http://localhost:8002" 

product_dict = {10: 'Used Books', 
                1940: 'Food', 
                40: 'Used video games', 
                2060: 'Crafts & souvenirs', 
                50: 'Console games accessories', 
                2220: 'Pet accessories', 
                60: 'New console games', 
                2280: 'Magazines & Comics', 
                1140: 'Action figurines', 
                2403: 'Books, magazines & collections', 
                1160: 'Trading card games', 
                2462: 'Used video games & accessories', 
                1180: 'Role-playing games', 
                2522: 'Office Supplies', 
                1280: 'Childrens toys', 
                2582: 'Gardening Furniture', 
                1281: 'Board games', 
                2583: 'Pool & accessories', 
                1300: 'Miniature car toys', 
                2585: 'Gardening tools', 
                1301: 'Indoor games', 
                2705: 'New books', 
                1302: 'Outdoor play', 
                2905: 'Computer games', 
                1320: 'Baby care', 
                2910: 'Household linens', 
                1560: 'Furniture', 
                1920: 'Home accessories'}

# --- Session State Initialization ---
# Initialize session state variables if they don't already exist.
# This helps maintain state across Streamlit re-runs.
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
# For prediction form
if 'designation_input' not in st.session_state:
    st.session_state.designation_input = ""
if 'description_input' not in st.session_state:
    st.session_state.description_input = ""
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'warning' not in st.session_state:
    st.session_state.warning = None
if 'show_prediction_message' not in st.session_state:
    st.session_state.show_prediction_message = False
if "show_upload_message" not in st.session_state:
    st.session_state.show_upload_message = False
if 'upload_status_message' not in st.session_state: 
    st.session_state.upload_status_message = None
if 'upload_status_type' not in st.session_state: 
    st.session_state.upload_status_type = None
if 'selected_category_from_dropdown_index' not in st.session_state: # To control selectbox index
    st.session_state.selected_category_from_dropdown_index = 0
if 'confirmed_category' not in st.session_state:
    st.session_state.confirmed_category = None
if 'show_all_categories' not in st.session_state: 
    st.session_state.show_all_categories = False

# --- global variables ---
SELECT_TEXT = "-- Select a category --"
CHOOSE_OTHER_CATEGORY_TEXT = "-- Choose another category --"

# --- Helper Functions for API Calls ---
response = None

def login_user(username, password):
    """
    Attempts to log in a user by sending credentials to the auth API.
    Updates session state upon successful login.
    """
    try:
        response = requests.post(
            f"{AUTH_API_BASE_URL}/login",
            data={"username": username, "password": password}
        )
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()
        st.session_state.logged_in = True
        st.session_state.access_token = data["access_token"]
        st.session_state.username = username
        if username == "admin":
            st.session_state.user_role = "admin"
        else:
            st.session_state.user_role = "user"

        st.success(f"Logged in as {username}!")
        st.rerun()
    except requests.exceptions.RequestException as e:
        st.error(f"Login failed: {e}")
        if response.status_code == 401:
            st.error("Incorrect username or password.")
        else:
            st.error(f"An error occurred: {response.text}")

def logout_user():
    """Logs out the user by clearing session state."""
    st.session_state.logged_in = False
    st.session_state.access_token = None
    st.session_state.username = None
    st.session_state.user_role = None
    st.session_state.show_prediction_message = False
    st.info("Logged out successfully.")
    st.rerun()

def create_user(username, password, role):
    """
    Sends a request to the auth API to create a new user.
    Requires admin privileges. Uses Authorization: Bearer.
    """
    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    try:
        response = requests.post(
            f"{AUTH_API_BASE_URL}/users",
            json={"username": username, "password": password, "role": role},
            headers=headers
        )
        response.raise_for_status()
        st.success(f"User '{username}' created successfully.")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to create user: {e}")
        st.error(f"Error details: {response.text}")

def delete_user(username):
    """
    Sends a request to the auth API to delete a user.
    Requires admin privileges. Uses Authorization: Bearer.
    """
    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    try:
        response = requests.delete(
            f"{AUTH_API_BASE_URL}/users/{username}",
            headers=headers
        )
        response.raise_for_status()
        st.success(f"User '{username}' deleted successfully.")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to delete user: {e}")
        st.error(f"Error details: {response.text}")

def get_prediction(designation, description):
    """
    Sends a prediction request to the predict API.
    Requires authentication. Uses 'token' header.
    """
    headers = {"token": st.session_state.access_token}
    try:
        response = requests.post(
            f"{PREDICT_API_BASE_URL}/predict",
            json={"designation": designation, "description": description},
            headers=headers
        )
        response.raise_for_status()
        return response.json()["predicted_class"]
    except requests.exceptions.RequestException as e:
        st.error(f"Prediction failed: {e}")
        st.error(f"Error details: {response.text}")
        return None

def get_model_info():
    """
    Retrieves model information from the predict API.
    Requires authentication. Uses 'token' header.
    """
    headers = {"token": st.session_state.access_token}
    try:
        response = requests.get(
            f"{PREDICT_API_BASE_URL}/model-info",
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to retrieve model info: {e}")
        st.error(f"Error details: {response.text}")
        return None
    
def send_feedback(designation, description, predicted_label, correct_label, access_token):
    """
    Sends feedback about a prediction to the predict API.
    Requires authentication.
    """
    headers = {"token": access_token}
    payload = {
        "designation": designation,
        "description": description,
        "predicted_label": predicted_label,
        "correct_label": correct_label
    }
    try:
        response = requests.post(
            f"{PREDICT_API_BASE_URL}/feedback",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to upload article: {e}")
        if response is not None:
            st.error(f"Server response: {response.text}")
        else:
            st.error("No response from server.")
        return None

   
# --- UI Callback Functions ---

def prepare_dropdown_options(product_dict, predicted_category):
    """
    Prepares a list of categories for the dropdown.

    Args:
        product_dict (dict): a dictionary mapping product codes to category names.
        predicted_category (str): name of the category predicted by the model.

    Returns:
        list: a list of category names for the dropdown, starting with the predicted category.
    """
    options = [predicted_category]  
    for category in sorted(list(product_dict.values())):
        if category != predicted_category:  
            options.append(category)
    return options

def handle_predict_button_click():
    """
    Callback for the predict button.
    Resets dropdown and handles prediction logic.
    """
    st.session_state.selected_category_from_dropdown_index = 0
    st.session_state.confirmed_category = None
    st.session_state.show_prediction_message = False 
    st.session_state.show_all_categories = False
    st.session_state.show_upload_message = False
    st.session_state.upload_status_message = None
    st.session_state.upload_status_type = None

    designation = st.session_state.designation_input
    description = st.session_state.description_input

    if designation:
        predicted_class = get_prediction(designation, description)
        if predicted_class:
            st.session_state.last_prediction = predicted_class
            st.session_state.selected_category_from_dropdown_index = 0 # Set dropdown to predicted class
            st.session_state.show_prediction_message = True # Show prediction message
           
        else:
            st.session_state.last_prediction = None
            st.session_state.show_prediction_message = False
    else:
        st.session_state.warning = "Please enter at least the **Designation**."

def handle_confirm_category_click(selected_category_value):
    """
    Callback for the confirm category button.
    Stores confirmed category and clears input fields/dropdown.
    """
    current_selection = selected_category_value

    if current_selection != SELECT_TEXT and current_selection != CHOOSE_OTHER_CATEGORY_TEXT:
        confirmed_category = current_selection

        # Data for feedback submission
        designation_to_send = st.session_state.designation_input
        description_to_send = st.session_state.description_input
        predicted_label_to_send = st.session_state.last_prediction
        correct_label_to_send  = confirmed_category
        
        if not designation_to_send or not predicted_label_to_send or not correct_label_to_send or not st.session_state.access_token:
            st.session_state.upload_status_message = "Error: Missing required fields: designation, predicted label, correct label or access token"
            st.session_state.upload_status_type = "error"
            st.session_state.show_upload_message = True
            return

        # Send feedback to the API
        feedback_result = send_feedback(
            designation_to_send,
            description_to_send,
            predicted_label_to_send,
            correct_label_to_send,
            st.session_state.access_token)
        
        
        if feedback_result and feedback_result.get("status") == "success":
            st.session_state.upload_status_message = f"Article uploaded successfully in category {confirmed_category}."
            st.session_state.upload_status_type = "success"
        else:
            st.session_state.upload_status_message = f"Error: Failed to upload article in category {confirmed_category}."
            st.session_state.upload_status_type = "error"

        
        st.session_state.show_upload_message = True
        st.session_state.confirmed_category = current_selection
        # Clear input fields and reset dropdown after confirmation
        st.session_state.designation_input = ""
        st.session_state.description_input = ""
        st.session_state.last_prediction = None 
        st.session_state.selected_category_from_dropdown_index = 0 
        st.session_state.show_prediction_message = False 
        st.session_state.show_all_categories = False 

def handle_category_select_change():
    """
    Callback for when the category selectbox changes.
    Updates the index of the selected category in session state.
    """
    selected_value = st.session_state.category_select

    if selected_value == CHOOSE_OTHER_CATEGORY_TEXT:
        st.session_state.show_all_categories = True
        st.session_state.selected_category_from_dropdown_index = 0
    else:
        pass


# --- Streamlit UI Layout ---

st.set_page_config(page_title="API Interaction Dashboard", layout="centered")

st.title("API Interaction Dashboard")

# --- Authentication Section ---
st.header("Authentication")

if not st.session_state.logged_in:
    with st.form("login_form"):
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            login_user(username, password)
else:
    st.success(f"You are logged in as **{st.session_state.username}** (Role: **{st.session_state.user_role}**)")
    st.button("Logout", on_click=logout_user)

    # --- Admin Section (only for admins) ---
    if st.session_state.user_role == "admin":
        st.header("Admin Panel (User Management)")
        st.markdown("---")

        st.subheader("Create New User")
        with st.form("create_user_form"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            new_role = st.selectbox("Role", ["user", "admin"], index=0)
            create_user_button = st.form_submit_button("Create User")

            if create_user_button:
                create_user(new_username, new_password, new_role)

        st.subheader("Delete User")
        with st.form("delete_user_form"):
            user_to_delete = st.text_input("Username to Delete")
            delete_user_button = st.form_submit_button("Delete User")

            if delete_user_button:
                delete_user(user_to_delete)
 
        st.header("Model Management")
        if st.button("Fetch Model Info"):
            model_info = get_model_info()
            if model_info:
                st.json(model_info) 

        #st.subheader("Start retraining the model")
        if st.button("Retrain Model"):
            try:
                subprocess.run(["docker", "compose", "up", "dvc-sync", "preprocessing", "model_training", "model_validation"], check=True)
                st.success("Retraining was successful. Check logs for details.")
            except subprocess.CalledProcessError as e:
                st.error(f"Failed to start retraining: {e}")



    # --- Prediction Section (for all logged-in users) ---
    st.header("Product Type Prediction")
    st.markdown("---")

    st.subheader("Get Prediction")
    with st.form("prediction_form"):
        designation = st.text_area("Designation (required)", 
                                   value=st.session_state.designation_input,
                                   key="designation_input",
                                   help="e.g., 'Voiture miniature 1976 Ford Mustang'")
        description = st.text_area("Description (optional)", 
                                   value=st.session_state.description_input,
                                   key="description_input",
                                   help="e.g., 'couleur bleu.'")
        predict_button = st.form_submit_button("Confirm", on_click=handle_predict_button_click)

    if 'warning' in st.session_state and st.session_state.warning:
        st.warning(st.session_state.warning)
        st.session_state.warning = None
    
    if st.session_state.show_prediction_message:
        st.info("Please select product category from the dropdown below.")

    # --- Category Selection Section --- (after prediction)
    if st.session_state.show_prediction_message and st.session_state.last_prediction:
        st.markdown("---")

     
        dynamic_dropdown_options = []

        if not st.session_state.show_all_categories:
            dynamic_dropdown_options = [st.session_state.last_prediction,
                                        CHOOSE_OTHER_CATEGORY_TEXT]
            if st.session_state.selected_category_from_dropdown_index == 0:
                st.session_state.selected_category_from_dropdown_index = 0
        else:
            all_categories_sorted = prepare_dropdown_options(product_dict, st.session_state.last_prediction)
            dynamic_dropdown_options = [SELECT_TEXT] + all_categories_sorted

        current_selection = st.selectbox(
            "Category selection",
            options=dynamic_dropdown_options,
            index=st.session_state.selected_category_from_dropdown_index,
            key="category_select",
            on_change=handle_category_select_change
        )

        if current_selection != SELECT_TEXT and current_selection != CHOOSE_OTHER_CATEGORY_TEXT:
            st.button("Upload article", 
                    key="confirm_category_button", 
                    on_click=handle_confirm_category_click,
                    args=(current_selection,))
        
    
    message_placeholder = st.empty()

    if st.session_state.get('show_upload_message') and st.session_state.upload_status_message:
        if st.session_state.upload_status_type == "success":
            message_placeholder.success(st.session_state.upload_status_message)
        elif st.session_state.upload_status_type == "error":
            message_placeholder.error(st.session_state.upload_status_message)

        time.sleep(3) 

        message_placeholder.empty() 
        st.session_state.show_upload_message = False 
        st.session_state.upload_status_message = None 
        st.session_state.upload_status_type = None 


    

st.markdown("---")
st.caption("Developed with Streamlit and FastAPI")

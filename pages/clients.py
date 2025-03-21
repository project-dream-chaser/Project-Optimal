import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os

from utils.client_management import (
    add_client, update_client, delete_client, 
    add_spouse, get_client_by_id
)
from utils.questionnaire import (
    generate_risk_questionnaire, send_questionnaire_email
)

def show_clients_page():
    """Display the Clients page with client management functionality."""
    st.title("Client Management")
    
    # Create tabs for client list and client details
    tab1, tab2 = st.tabs(["Client List", "Client Details"])
    
    with tab1:
        show_client_list()
    
    with tab2:
        show_client_details()

def show_client_list():
    """Display the list of clients with filtering and actions."""
    st.subheader("Clients")
    
    # Add a new client button
    if st.button("Add New Client"):
        st.session_state.client_action = "add"
        st.session_state.current_client = None
        st.rerun()
    
    # Check if there are clients to display
    if not st.session_state.clients:
        st.info("No clients found. Add a new client to get started.")
        return
    
    # Convert clients to a DataFrame for display
    clients_data = []
    for client in st.session_state.clients:
        has_spouse = "Yes" if client.spouse else "No"
        clients_data.append({
            "ID": client.id,
            "Name": f"{client.first_name} {client.last_name}",
            "Email": client.email,
            "Date of Birth": client.date_of_birth,
            "Risk Score": client.risk_score if client.risk_score else "Not assessed",
            "Has Spouse": has_spouse
        })
    
    df = pd.DataFrame(clients_data)
    
    # Add a search filter
    search = st.text_input("Search clients", "")
    if search:
        df = df[df["Name"].str.contains(search, case=False) | 
                df["Email"].str.contains(search, case=False)]
    
    # Display the client list with action buttons
    st.dataframe(df, use_container_width=True)
    
    # Select client for actions
    if not df.empty:
        selected_client_id = st.selectbox(
            "Select a client for actions",
            df["ID"].tolist(),
            format_func=lambda x: df[df["ID"] == x]["Name"].iloc[0]
        )
        
        # Actions for the selected client
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("View/Edit Client"):
                st.session_state.client_action = "edit"
                st.session_state.current_client = get_client_by_id(selected_client_id)
                st.rerun()
        
        with col2:
            if st.button("Add Spouse"):
                st.session_state.client_action = "add_spouse"
                st.session_state.current_client = get_client_by_id(selected_client_id)
                st.rerun()
        
        with col3:
            if st.button("Send Questionnaire"):
                client = get_client_by_id(selected_client_id)
                questionnaire_html = generate_risk_questionnaire(client)
                success, message = send_questionnaire_email(client.email, questionnaire_html)
                
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        with col4:
            if st.button("Delete Client"):
                st.session_state.client_action = "delete"
                st.session_state.current_client = get_client_by_id(selected_client_id)
                st.rerun()

def show_client_details():
    """Display the client details form based on the current action."""
    # Check if there's a client action to perform
    if not hasattr(st.session_state, 'client_action'):
        st.session_state.client_action = None
    
    if not hasattr(st.session_state, 'current_client'):
        st.session_state.current_client = None
    
    if st.session_state.client_action == "add":
        add_client_form()
    elif st.session_state.client_action == "edit":
        edit_client_form()
    elif st.session_state.client_action == "delete":
        delete_client_form()
    elif st.session_state.client_action == "add_spouse":
        add_spouse_form()
    else:
        st.info("Select a client action from the Client List tab.")

def add_client_form():
    """Display form for adding a new client."""
    st.subheader("Add New Client")
    
    with st.form("add_client_form"):
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        email = st.text_input("Email")
        date_of_birth = st.date_input("Date of Birth", min_value=datetime(1900, 1, 1))
        
        submitted = st.form_submit_button("Add Client")
        
        if submitted:
            if not first_name or not last_name or not email:
                st.error("Please fill in all required fields.")
            else:
                success, client = add_client(
                    first_name=first_name,
                    last_name=last_name,
                    email=email,
                    date_of_birth=date_of_birth.strftime("%Y-%m-%d")
                )
                
                if success:
                    st.success(f"Client {first_name} {last_name} added successfully!")
                    st.session_state.client_action = None
                    st.rerun()
                else:
                    st.error(f"Error adding client: {client}")

def edit_client_form():
    """Display form for editing an existing client."""
    client = st.session_state.current_client
    
    if not client:
        st.error("No client selected for editing.")
        return
    
    st.subheader(f"Edit Client: {client.first_name} {client.last_name}")
    
    with st.form("edit_client_form"):
        first_name = st.text_input("First Name", value=client.first_name)
        last_name = st.text_input("Last Name", value=client.last_name)
        email = st.text_input("Email", value=client.email)
        
        date_of_birth_obj = datetime.strptime(client.date_of_birth, "%Y-%m-%d") if client.date_of_birth else datetime.now()
        date_of_birth = st.date_input("Date of Birth", value=date_of_birth_obj, min_value=datetime(1900, 1, 1))
        
        risk_score = st.slider("Risk Score (1-10)", 1, 10, value=client.risk_score if client.risk_score else 5)
        
        submitted = st.form_submit_button("Update Client")
        
        if submitted:
            if not first_name or not last_name or not email:
                st.error("Please fill in all required fields.")
            else:
                success, updated_client = update_client(
                    client_id=client.id,
                    first_name=first_name,
                    last_name=last_name,
                    email=email,
                    date_of_birth=date_of_birth.strftime("%Y-%m-%d"),
                    risk_score=risk_score
                )
                
                if success:
                    st.success(f"Client {first_name} {last_name} updated successfully!")
                    st.session_state.client_action = None
                    st.rerun()
                else:
                    st.error(f"Error updating client: {updated_client}")
    
    # Display spouse information if available
    if client.spouse:
        st.subheader("Spouse Information")
        spouse = client.spouse
        st.write(f"Name: {spouse.get('first_name', '')} {spouse.get('last_name', '')}")
        st.write(f"Email: {spouse.get('email', '')}")
        st.write(f"Date of Birth: {spouse.get('date_of_birth', '')}")

def delete_client_form():
    """Display form for confirming client deletion."""
    client = st.session_state.current_client
    
    if not client:
        st.error("No client selected for deletion.")
        return
    
    st.subheader(f"Delete Client: {client.first_name} {client.last_name}")
    st.warning("This action cannot be undone. All client data will be permanently deleted.")
    
    confirm = st.text_input("Type the client's full name to confirm deletion:")
    expected = f"{client.first_name} {client.last_name}"
    
    if st.button("Delete Client"):
        if confirm == expected:
            success, message = delete_client(client.id)
            
            if success:
                st.success(message)
                st.session_state.client_action = None
                st.rerun()
            else:
                st.error(message)
        else:
            st.error("Confirmation does not match client name. Please try again.")

def add_spouse_form():
    """Display form for adding a spouse to an existing client."""
    client = st.session_state.current_client
    
    if not client:
        st.error("No client selected for adding spouse.")
        return
    
    st.subheader(f"Add Spouse for: {client.first_name} {client.last_name}")
    
    # If spouse already exists, show current information
    if client.spouse:
        st.info("This client already has a spouse. The existing spouse information will be updated.")
    
    with st.form("add_spouse_form"):
        # Pre-fill values if spouse exists
        spouse = client.spouse or {}
        
        first_name = st.text_input("First Name", value=spouse.get("first_name", ""))
        last_name = st.text_input("Last Name", value=spouse.get("last_name", ""))
        email = st.text_input("Email", value=spouse.get("email", ""))
        
        date_of_birth_str = spouse.get("date_of_birth", None)
        date_of_birth_obj = datetime.strptime(date_of_birth_str, "%Y-%m-%d") if date_of_birth_str else datetime.now()
        date_of_birth = st.date_input("Date of Birth", value=date_of_birth_obj, min_value=datetime(1900, 1, 1))
        
        submitted = st.form_submit_button("Save Spouse Information")
        
        if submitted:
            if not first_name or not last_name or not email:
                st.error("Please fill in all required fields.")
            else:
                success, message = add_spouse(
                    client_id=client.id,
                    spouse_first_name=first_name,
                    spouse_last_name=last_name,
                    spouse_email=email,
                    spouse_date_of_birth=date_of_birth.strftime("%Y-%m-%d")
                )
                
                if success:
                    st.success("Spouse information saved successfully!")
                    st.session_state.client_action = None
                    st.rerun()
                else:
                    st.error(f"Error saving spouse information: {message}")

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from models.client import Client

def load_clients():
    """Load clients from session state or create empty DataFrame if not exists."""
    try:
        # In a real app, this would load from a database
        if os.path.exists('data/clients.json'):
            with open('data/clients.json', 'r') as f:
                clients_data = json.load(f)
                clients = []
                for client_data in clients_data:
                    client = Client(
                        client_data.get('id'),
                        client_data.get('first_name'),
                        client_data.get('last_name'),
                        client_data.get('email'),
                        client_data.get('date_of_birth'),
                        client_data.get('risk_score'),
                        client_data.get('spouse')
                    )
                    clients.append(client)
                return clients
        return []
    except Exception as e:
        st.error(f"Error loading clients: {e}")
        return []

def save_clients(clients):
    """Save clients to a JSON file."""
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Convert clients to JSON serializable format
        clients_data = []
        for client in clients:
            client_dict = client.__dict__.copy()
            clients_data.append(client_dict)
            
        with open('data/clients.json', 'w') as f:
            json.dump(clients_data, f)
        return True
    except Exception as e:
        st.error(f"Error saving clients: {e}")
        return False

def add_client(first_name, last_name, email, date_of_birth, risk_score=None):
    """Add a new client to the session state."""
    try:
        # Generate a new client ID
        client_id = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Create a new client object
        new_client = Client(
            id=client_id,
            first_name=first_name,
            last_name=last_name,
            email=email,
            date_of_birth=date_of_birth,
            risk_score=risk_score
        )
        
        # Add to session state
        st.session_state.clients.append(new_client)
        
        # Save clients
        save_clients(st.session_state.clients)
        
        return True, new_client
    except Exception as e:
        return False, str(e)

def update_client(client_id, **kwargs):
    """Update an existing client."""
    try:
        for i, client in enumerate(st.session_state.clients):
            if client.id == client_id:
                for key, value in kwargs.items():
                    if hasattr(client, key):
                        setattr(client, key, value)
                
                # Save clients
                save_clients(st.session_state.clients)
                return True, client
        
        return False, "Client not found"
    except Exception as e:
        return False, str(e)

def delete_client(client_id):
    """Delete a client from the session state."""
    try:
        client_to_remove = None
        for client in st.session_state.clients:
            if client.id == client_id:
                client_to_remove = client
                break
                
        if client_to_remove:
            st.session_state.clients.remove(client_to_remove)
            save_clients(st.session_state.clients)
            return True, "Client deleted successfully"
        else:
            return False, "Client not found"
    except Exception as e:
        return False, str(e)

def add_spouse(client_id, spouse_first_name, spouse_last_name, spouse_email, spouse_date_of_birth):
    """Add a spouse to an existing client."""
    try:
        for client in st.session_state.clients:
            if client.id == client_id:
                spouse = {
                    "first_name": spouse_first_name,
                    "last_name": spouse_last_name,
                    "email": spouse_email,
                    "date_of_birth": spouse_date_of_birth
                }
                client.spouse = spouse
                save_clients(st.session_state.clients)
                return True, "Spouse added successfully"
        
        return False, "Client not found"
    except Exception as e:
        return False, str(e)

def get_client_by_id(client_id):
    """Find a client by ID."""
    for client in st.session_state.clients:
        if client.id == client_id:
            return client
    return None

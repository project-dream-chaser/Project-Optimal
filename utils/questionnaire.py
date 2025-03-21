import streamlit as st
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os

def generate_risk_questionnaire(client):
    """
    Generate a risk tolerance questionnaire for a client.
    
    Parameters:
    -----------
    client : Client object
        The client for whom to generate the questionnaire
        
    Returns:
    --------
    str
        HTML content of the questionnaire
    """
    # Create questionnaire HTML
    questionnaire_html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
            .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #1565c0; }}
            .question {{ margin-bottom: 20px; }}
            .options {{ margin-left: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Risk Tolerance Questionnaire</h1>
            <p>Dear {client.first_name} {client.last_name},</p>
            <p>Please complete this questionnaire to help us understand your risk tolerance and financial goals.</p>
            
            <form>
                <div class="question">
                    <h3>1. Investment Experience</h3>
                    <p>How would you describe your investment experience?</p>
                    <div class="options">
                        <input type="radio" name="q1" value="1"> None<br>
                        <input type="radio" name="q1" value="2"> Limited<br>
                        <input type="radio" name="q1" value="3"> Good<br>
                        <input type="radio" name="q1" value="4"> Extensive<br>
                    </div>
                </div>
                
                <div class="question">
                    <h3>2. Investment Horizon</h3>
                    <p>How long do you expect to be investing before you need to access a significant portion of your money?</p>
                    <div class="options">
                        <input type="radio" name="q2" value="1"> Less than 3 years<br>
                        <input type="radio" name="q2" value="2"> 3-5 years<br>
                        <input type="radio" name="q2" value="3"> 6-10 years<br>
                        <input type="radio" name="q2" value="4"> More than 10 years<br>
                    </div>
                </div>
                
                <div class="question">
                    <h3>3. Risk and Return</h3>
                    <p>Which statement best describes your attitude toward investment risk and return?</p>
                    <div class="options">
                        <input type="radio" name="q3" value="1"> I prefer investments with little or no fluctuation in value and am willing to accept lower returns.<br>
                        <input type="radio" name="q3" value="2"> I can tolerate modest fluctuations in value and seek modest returns.<br>
                        <input type="radio" name="q3" value="3"> I can tolerate moderate fluctuations in value and seek higher returns.<br>
                        <input type="radio" name="q3" value="4"> I can tolerate significant fluctuations in value for potentially higher returns.<br>
                    </div>
                </div>
                
                <div class="question">
                    <h3>4. Market Decline</h3>
                    <p>If your investment portfolio lost 20% of its value over a short period, what would you do?</p>
                    <div class="options">
                        <input type="radio" name="q4" value="1"> Sell all remaining investments and move to cash<br>
                        <input type="radio" name="q4" value="2"> Sell some investments to reduce exposure<br>
                        <input type="radio" name="q4" value="3"> Hold steady and make no changes<br>
                        <input type="radio" name="q4" value="4"> Buy more investments to take advantage of lower prices<br>
                    </div>
                </div>
                
                <div class="question">
                    <h3>5. Financial Goals</h3>
                    <p>What is your primary financial goal?</p>
                    <div class="options">
                        <input type="radio" name="q5" value="1"> Preserving capital<br>
                        <input type="radio" name="q5" value="2"> Generating income<br>
                        <input type="radio" name="q5" value="3"> Balanced growth and income<br>
                        <input type="radio" name="q5" value="4"> Long-term growth<br>
                    </div>
                </div>
                
                <div class="question">
                    <h3>6. Income Needs</h3>
                    <p>How important is current income from your investments?</p>
                    <div class="options">
                        <input type="radio" name="q6" value="1"> Very important - I need income now<br>
                        <input type="radio" name="q6" value="2"> Somewhat important - I need some income<br>
                        <input type="radio" name="q6" value="3"> Not very important - I'm focused on growth<br>
                        <input type="radio" name="q6" value="4"> Not important at all - I'm focused on maximum growth<br>
                    </div>
                </div>
                
                <div class="question">
                    <h3>7. Emergency Funds</h3>
                    <p>Do you have adequate emergency funds (typically 3-6 months of expenses)?</p>
                    <div class="options">
                        <input type="radio" name="q7" value="1"> No<br>
                        <input type="radio" name="q7" value="2"> Somewhat, but not fully funded<br>
                        <input type="radio" name="q7" value="3"> Yes, but just the minimum<br>
                        <input type="radio" name="q7" value="4"> Yes, well beyond the minimum<br>
                    </div>
                </div>
                
                <div class="question">
                    <h3>8. Retirement Planning</h3>
                    <p>How confident are you in your current retirement savings plan?</p>
                    <div class="options">
                        <input type="radio" name="q8" value="1"> Not confident at all<br>
                        <input type="radio" name="q8" value="2"> Somewhat concerned<br>
                        <input type="radio" name="q8" value="3"> Moderately confident<br>
                        <input type="radio" name="q8" value="4"> Very confident<br>
                    </div>
                </div>
                
                <div class="question">
                    <h3>9. Investment Decision Making</h3>
                    <p>How do you prefer to make investment decisions?</p>
                    <div class="options">
                        <input type="radio" name="q9" value="1"> I prefer my advisor to make all decisions<br>
                        <input type="radio" name="q9" value="2"> I want to understand but rely on advisor recommendations<br>
                        <input type="radio" name="q9" value="3"> I want to be actively involved in the decision process<br>
                        <input type="radio" name="q9" value="4"> I want to make my own decisions with minimal guidance<br>
                    </div>
                </div>
                
                <div class="question">
                    <h3>10. Additional Information</h3>
                    <p>Please provide any additional information that may help us understand your financial situation and goals:</p>
                    <textarea name="additional" rows="4" cols="50"></textarea>
                </div>
                
                <input type="submit" value="Submit Questionnaire">
            </form>
        </div>
    </body>
    </html>
    """
    
    return questionnaire_html

def send_questionnaire_email(client_email, questionnaire_html):
    """
    Send the risk tolerance questionnaire to a client via email.
    
    Parameters:
    -----------
    client_email : str
        Email address of the client
    questionnaire_html : str
        HTML content of the questionnaire
        
    Returns:
    --------
    bool
        True if email was sent successfully, False otherwise
    str
        Message indicating success or error
    """
    try:
        # Email configuration
        sender_email = os.getenv("EMAIL_SENDER", "advisor@financialplanning.com")
        password = os.getenv("EMAIL_PASSWORD", "")
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = "Risk Tolerance Questionnaire"
        msg['From'] = sender_email
        msg['To'] = client_email
        
        # Create HTML message
        html_part = MIMEText(questionnaire_html, 'html')
        msg.attach(html_part)
        
        # For development/testing purposes, don't actually send the email
        # Just print that it would be sent
        st.info(f"In a production environment, the questionnaire would be sent to {client_email}")
        
        # In a real application, you would uncomment this code to send the email
        """
        # Send email
        server = smtplib.SMTP_SSL('smtp.yourprovider.com', 465)
        server.login(sender_email, password)
        server.sendmail(sender_email, client_email, msg.as_string())
        server.quit()
        """
        
        return True, "Questionnaire sent successfully!"
    except Exception as e:
        return False, f"Error sending questionnaire: {str(e)}"

def calculate_risk_score(responses):
    """
    Calculate a risk score based on questionnaire responses.
    
    Parameters:
    -----------
    responses : dict
        Dictionary of question IDs and response values
        
    Returns:
    --------
    int
        Risk score (1-10)
    str
        Risk profile description
    """
    # Sum the response values
    total = sum(responses.values())
    
    # Calculate score on a scale of 1-10
    max_possible = len(responses) * 4  # Maximum possible score
    normalized_score = int(round((total / max_possible) * 10))
    
    # Ensure score is between 1 and 10
    risk_score = max(1, min(normalized_score, 10))
    
    # Determine risk profile
    if risk_score <= 2:
        profile = "Very Conservative"
    elif risk_score <= 4:
        profile = "Conservative"
    elif risk_score <= 6:
        profile = "Moderate"
    elif risk_score <= 8:
        profile = "Aggressive"
    else:
        profile = "Very Aggressive"
    
    return risk_score, profile

import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
from datetime import datetime

def generate_investment_policy_statement(client, plan, glidepath_result, risk_profile):
    """
    Generate an Investment Policy Statement (IPS) for a client.
    
    Parameters:
    -----------
    client : Client object
        Client information
    plan : Plan object
        Client's financial plan
    glidepath_result : dict
        Results from glidepath optimization
    risk_profile : str
        Client's risk profile
        
    Returns:
    --------
    bytes
        PDF document as bytes
    """
    # Create a temporary file for the PDF
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
        pdf_path = temp_file.name
    
    # Create the PDF document
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Create styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='Heading1',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name='Heading2',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10
    ))
    styles.add(ParagraphStyle(
        name='Normal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8
    ))
    
    # Create story (content)
    story = []
    
    # Title
    story.append(Paragraph(f"Investment Policy Statement", styles['Title']))
    story.append(Paragraph(f"For: {client.first_name} {client.last_name}", styles['Heading1']))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 0.25*inch))
    
    # 1. Introduction
    story.append(Paragraph("1. Introduction", styles['Heading1']))
    story.append(Paragraph(
        "This Investment Policy Statement (IPS) is designed to establish a clear understanding of "
        "the investment objectives and policies applicable to the investor's portfolio. This IPS "
        "will outline the framework for the investment management of assets.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.15*inch))
    
    # 2. Risk Profile
    story.append(Paragraph("2. Risk Profile", styles['Heading1']))
    story.append(Paragraph(
        f"Based on the completed risk assessment questionnaire, the investor's risk profile "
        f"is classified as <b>{risk_profile}</b>. This profile reflects the investor's "
        f"tolerance for investment risk and volatility.",
        styles['Normal']
    ))
    
    # Add risk factors
    story.append(Paragraph("Key Risk Factors:", styles['Heading2']))
    
    risk_items = [
        "Market Risk: Possibility of investment losses due to market movements",
        "Interest Rate Risk: Impact of interest rate changes on fixed income investments",
        "Inflation Risk: Risk that investment returns won't keep pace with inflation",
        "Longevity Risk: Risk of outliving retirement assets"
    ]
    
    for item in risk_items:
        story.append(Paragraph(f"• {item}", styles['Normal']))
    
    story.append(Spacer(1, 0.15*inch))
    
    # 3. Return Expectations
    story.append(Paragraph("3. Return Expectations", styles['Heading1']))
    
    # Calculate expected return from glidepath
    initial_allocation = glidepath_result['glidepath'][0]
    asset_classes = glidepath_result['asset_classes']
    expected_returns = [0.067, 0.023, 0.018, 0.039, 0.052, 0.042]  # Sample expected returns
    
    expected_return = sum(a * r for a, r in zip(initial_allocation, expected_returns))
    expected_return_pct = expected_return * 100
    
    story.append(Paragraph(
        f"Based on the recommended asset allocation, the expected long-term annual return "
        f"is approximately <b>{expected_return_pct:.1f}%</b> (before taxes and fees). This "
        f"expected return is based on long-term capital market assumptions and is not guaranteed.",
        styles['Normal']
    ))
    
    story.append(Paragraph(
        f"The probability of meeting all financial goals based on Monte Carlo analysis "
        f"is approximately <b>{glidepath_result['success_probability']*100:.1f}%</b>.",
        styles['Normal']
    ))
    
    story.append(Spacer(1, 0.15*inch))
    
    # 4. Time Horizon
    story.append(Paragraph("4. Time Horizon", styles['Heading1']))
    
    # Calculate client's age and retirement age
    current_year = datetime.now().year
    birth_year = datetime.strptime(client.date_of_birth, '%Y-%m-%d').year
    current_age = current_year - birth_year
    retirement_age = 65  # Assumption
    
    story.append(Paragraph(
        f"The investor is currently {current_age} years old. The primary investment time horizon "
        f"extends to retirement at age {retirement_age} and beyond. The investment strategy accounts "
        f"for both accumulation phase (pre-retirement) and distribution phase (post-retirement) needs.",
        styles['Normal']
    ))
    
    # Time horizon breakdown
    years_to_retirement = max(0, retirement_age - current_age)
    
    story.append(Paragraph("Investment Time Horizons:", styles['Heading2']))
    
    time_data = [
        ["Time Horizon", "Years", "Purpose"],
        ["Short-term", "0-3 years", "Emergency funds, near-term goals"],
        ["Intermediate", "3-10 years", "Medium-term goals (home purchase, education)"],
        ["Long-term", f"{years_to_retirement}+ years", "Retirement funding"]
    ]
    
    time_table = Table(time_data, colWidths=[1.5*inch, 1*inch, 2.5*inch])
    time_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(time_table)
    story.append(Spacer(1, 0.15*inch))
    
    # 5. Tax Considerations
    story.append(Paragraph("5. Tax Considerations", styles['Heading1']))
    story.append(Paragraph(
        "The investment strategy will consider tax implications and aim to maximize after-tax "
        "returns. This will be achieved through:",
        styles['Normal']
    ))
    
    tax_items = [
        "Tax-efficient asset location (placing less tax-efficient investments in tax-advantaged accounts)",
        "Consideration of tax-exempt or tax-deferred investment vehicles where appropriate",
        "Tax-loss harvesting opportunities",
        "Managing capital gains distributions and realizations"
    ]
    
    for item in tax_items:
        story.append(Paragraph(f"• {item}", styles['Normal']))
    
    story.append(Spacer(1, 0.15*inch))
    
    # 6. Liquidity Requirements
    story.append(Paragraph("6. Liquidity Requirements", styles['Heading1']))
    story.append(Paragraph(
        "The portfolio will maintain sufficient liquidity to meet expected and unexpected "
        "cash flow needs without disrupting the long-term investment strategy:",
        styles['Normal']
    ))
    
    liquidity_items = [
        "Emergency Fund: 3-6 months of living expenses in highly liquid assets",
        "Known Near-term Expenses: Funds for known expenses within 1-2 years will be kept in cash or cash equivalents",
        "Income Needs: For clients taking distributions, 1-2 years of expected withdrawals will be maintained in lower volatility assets"
    ]
    
    for item in liquidity_items:
        story.append(Paragraph(f"• {item}", styles['Normal']))
    
    story.append(Spacer(1, 0.15*inch))
    
    # 7. Unique Circumstances
    story.append(Paragraph("7. Unique Circumstances and Constraints", styles['Heading1']))
    story.append(Paragraph(
        "The following unique circumstances, preferences, and constraints have been considered in developing this investment policy:",
        styles['Normal']
    ))
    
    unique_items = [
        "Legacy/Estate Planning: Consideration for intergenerational wealth transfer",
        "Charitable Giving: Potential for charitable contribution strategies",
        "Special Needs: Any special family circumstances requiring financial planning",
        "Investment Restrictions: Any specific investment types to avoid or include based on client preferences"
    ]
    
    for item in unique_items:
        story.append(Paragraph(f"• {item}", styles['Normal']))
    
    story.append(Spacer(1, 0.15*inch))
    
    # 8. Asset Allocation
    story.append(Paragraph("8. Strategic Asset Allocation", styles['Heading1']))
    story.append(Paragraph(
        "Based on the investor's risk profile, time horizon, and financial goals, the following "
        "strategic asset allocation is recommended:",
        styles['Normal']
    ))
    
    # Create a pyplot figure for the asset allocation
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Initial allocation pie chart
    initial_alloc = glidepath_result['glidepath'][0]
    ax.pie(initial_alloc, labels=asset_classes, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title('Recommended Asset Allocation')
    
    # Save the chart to a BytesIO object
    chart_buffer = io.BytesIO()
    fig.savefig(chart_buffer, format='png', bbox_inches='tight')
    chart_buffer.seek(0)
    
    # Create a ReportLab Image from the BytesIO object
    chart_image = Image(chart_buffer, width=5*inch, height=3.5*inch)
    story.append(chart_image)
    
    # Close the pyplot figure to free memory
    plt.close(fig)
    
    story.append(Spacer(1, 0.15*inch))
    
    # Asset allocation table
    alloc_data = [["Asset Class", "Target %"]]
    for i, asset_class in enumerate(asset_classes):
        alloc_data.append([asset_class, f"{initial_alloc[i]*100:.1f}%"])
    
    alloc_table = Table(alloc_data, colWidths=[3*inch, 1*inch])
    alloc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(alloc_table)
    story.append(Spacer(1, 0.15*inch))
    
    # 9. Glidepath Strategy
    story.append(Paragraph("9. Glidepath Strategy", styles['Heading1']))
    story.append(Paragraph(
        "The investment strategy includes a glidepath approach that will gradually adjust the "
        "asset allocation over time to reduce risk as the investor approaches and enters retirement. "
        "This dynamic approach balances growth potential in early years with capital preservation "
        "in later years.",
        styles['Normal']
    ))
    
    # Create a pyplot figure for the glidepath
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Extract glidepath data
    glidepath = glidepath_result['glidepath']
    ages = glidepath_result['ages']
    
    # Plot as a stacked area chart
    bottom = np.zeros(len(glidepath))
    for i, asset in enumerate(asset_classes):
        values = [allocation[i] for allocation in glidepath]
        ax.fill_between(ages, bottom, bottom + values, label=asset)
        bottom += values
    
    ax.set_xlabel('Age')
    ax.set_ylabel('Allocation (%)')
    ax.set_title('Investment Glidepath Over Time')
    ax.legend(loc='upper right')
    ax.set_xlim(ages[0], ages[-1])
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    
    # Save the chart to a BytesIO object
    chart_buffer = io.BytesIO()
    fig.savefig(chart_buffer, format='png', bbox_inches='tight')
    chart_buffer.seek(0)
    
    # Create a ReportLab Image from the BytesIO object
    chart_image = Image(chart_buffer, width=6*inch, height=3.5*inch)
    story.append(chart_image)
    
    # Close the pyplot figure to free memory
    plt.close(fig)
    
    story.append(Spacer(1, 0.15*inch))
    
    # 10. Monitoring and Review
    story.append(Paragraph("10. Monitoring and Review", styles['Heading1']))
    story.append(Paragraph(
        "This Investment Policy Statement will be reviewed at least annually and updated as needed "
        "based on changes in the investor's financial situation, goals, or market conditions. "
        "Portfolio performance will be evaluated regularly against appropriate benchmarks.",
        styles['Normal']
    ))
    
    monitoring_items = [
        "Quarterly performance reviews",
        "Annual comprehensive portfolio analysis",
        "Rebalancing as needed to maintain target allocation (±5% threshold)",
        "Life event triggered reviews (retirement, inheritance, etc.)"
    ]
    
    for item in monitoring_items:
        story.append(Paragraph(f"• {item}", styles['Normal']))
    
    # Build the PDF
    doc.build(story)
    
    # Read the PDF file into memory
    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()
    
    # Clean up the temporary file
    os.unlink(pdf_path)
    
    return pdf_data

def display_ips_download_button(pdf_data):
    """
    Display a download button for the IPS PDF.
    
    Parameters:
    -----------
    pdf_data : bytes
        PDF document as bytes
    """
    b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
    
    # Display download button
    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="Investment_Policy_Statement.pdf">Download Investment Policy Statement (PDF)</a>'
    st.markdown(href, unsafe_allow_html=True)

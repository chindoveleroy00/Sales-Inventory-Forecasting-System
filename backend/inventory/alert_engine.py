import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AlertEngine:
    """
    Manages the generation and dispatch of inventory alerts via email and in-system notifications.
    Excludes SMS and WhatsApp as per user requirement.
    """

    def __init__(self, email_config: Optional[Dict] = None):
        """
        Initializes the AlertEngine with email configuration.

        Args:
            email_config (Optional[Dict]): Dictionary containing email server details
                                           (e.g., 'smtp_server', 'smtp_port', 'sender_email', 'sender_password').
                                           If None, email alerts will be disabled.
        """
        self.email_config = email_config
        self.in_system_alerts = []  # Stores alerts for in-system display/retrieval

    def check_and_dispatch_alerts(
            self,
            current_inventory: pd.DataFrame,
            forecast_df: pd.DataFrame,
            alert_threshold_days: int = 3,
            recipient_emails: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Checks for potential stockouts based on current inventory and forecasts,
        and dispatches alerts.

        Args:
            current_inventory (pd.DataFrame): DataFrame with 'sku_id' and 'current_stock'.
            forecast_df (pd.DataFrame): DataFrame with 'date', 'sku_id', 'yhat' (predicted demand).
                                        Assumes forecast_df contains daily predictions.
            alert_threshold_days (int): Number of future days to look ahead for stockout risk.
            recipient_emails (Optional[List[str]]): List of email addresses to send alerts to.
                                                    Required if email_config is provided.

        Returns:
            List[Dict]: A list of dispatched alerts (for in-system display).
        """
        if current_inventory.empty or forecast_df.empty:
            logger.info("No data provided for alert checking. Skipping alert generation.")
            return []

        # Ensure date column is datetime and sorted for correct cumulative sum
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        forecast_df = forecast_df.sort_values(['sku_id', 'date'])

        # Calculate cumulative forecasted demand for the alert threshold period
        # Group by SKU and then apply a rolling sum for the next 'alert_threshold_days'
        # This is a simplified approach. A more robust approach would simulate daily stock
        # depletion and reorders.

        # Get the latest forecast date for each SKU
        latest_forecast_dates = forecast_df.groupby('sku_id')['date'].max().reset_index()
        latest_forecast_dates.columns = ['sku_id', 'latest_forecast_date']

        # Merge current inventory with latest forecast date
        inventory_with_dates = pd.merge(current_inventory, latest_forecast_dates, on='sku_id', how='left')
        inventory_with_dates = inventory_with_dates.dropna(
            subset=['latest_forecast_date'])  # Drop SKUs without forecast

        generated_alerts = []

        for index, row in inventory_with_dates.iterrows():
            sku_id = row['sku_id']
            current_stock = row['current_stock']
            latest_forecast_date = row['latest_forecast_date']

            # Filter forecast for the specific SKU and the alert period
            # We need to consider demand *from* the current date *up to* the threshold

            # For simplicity in this example, let's assume the forecast_df already contains
            # predictions starting from the day after the last historical data point.
            # We'll take the first 'alert_threshold_days' predictions for each SKU.

            sku_forecast = forecast_df[
                (forecast_df['sku_id'] == sku_id)
            ].head(alert_threshold_days)  # Take the first N days of forecast

            if sku_forecast.empty:
                logger.warning(f"No sufficient future forecast for SKU: {sku_id} to check alerts.")
                continue

            # Calculate total forecasted demand for the alert period
            # Using yhat_lower for a more conservative (risk-averse) stockout check
            total_forecasted_demand = sku_forecast['yhat_lower'].sum()

            # Check for potential stockout
            if current_stock < total_forecasted_demand:
                alert_message = (
                    f"LOW STOCK ALERT: SKU '{sku_id}' is projected to run out within "
                    f"the next {alert_threshold_days} days. "
                    f"Current stock: {current_stock:.2f}, "
                    f"Projected demand: {total_forecasted_demand:.2f}."
                )
                alert_details = {
                    'sku_id': sku_id,
                    'alert_type': 'Low Stock',
                    'message': alert_message,
                    'current_stock': current_stock,
                    'projected_demand_threshold': total_forecasted_demand,
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                generated_alerts.append(alert_details)

                logger.warning(f"Alert generated for {sku_id}: {alert_message}")

                # Dispatch email if configured
                if self.email_config and recipient_emails:
                    subject = f"SIFS Low Stock Alert: {sku_id}"
                    self._send_email_alert(subject, alert_message, recipient_emails)
                else:
                    logger.info(f"Email alerts disabled or no recipients for {sku_id}.")
            else:
                logger.info(f"SKU '{sku_id}' has sufficient stock for the next {alert_threshold_days} days.")

        self.in_system_alerts.extend(generated_alerts)  # Add to in-system alerts list
        return generated_alerts

    def test_email_connection(self) -> bool:
        """
        Tests the email connection without sending an email.
        Returns True if connection is successful, False otherwise.
        """
        if not self.email_config:
            logger.error("Email configuration not provided. Cannot test connection.")
            return False

        try:
            sender_email = self.email_config['sender_email']
            sender_password = self.email_config['sender_password']
            smtp_server = self.email_config['smtp_server']
            smtp_port = self.email_config['smtp_port']

            logger.info(f"Testing SMTP connection to {smtp_server}:{smtp_port}")

            # Create connection with timeout
            server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
            server.set_debuglevel(1)  # Enable debug output

            # Start TLS encryption
            server.starttls()

            # Login
            server.login(sender_email, sender_password)

            # Close connection
            server.quit()

            logger.info("SMTP connection test successful!")
            return True

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP Authentication failed: {e}")
            logger.error(
                "Check your email and app password. For Gmail, ensure 2FA is enabled and you're using an App Password.")
            return False
        except smtplib.SMTPConnectError as e:
            logger.error(f"SMTP Connection failed: {e}")
            logger.error("Check your SMTP server and port settings.")
            return False
        except smtplib.SMTPServerDisconnected as e:
            logger.error(f"SMTP Server disconnected: {e}")
            logger.error("The server closed the connection unexpectedly. Try again or check server settings.")
            return False
        except Exception as e:
            logger.error(f"SMTP connection test failed: {e}")
            return False

    def _send_email_alert(self, subject: str, body: str, recipients: List[str], max_retries: int = 3) -> None:
        """
        Sends an email alert using the configured SMTP server with retry logic.
        """
        if not self.email_config:
            logger.error("Email configuration not provided. Cannot send email.")
            return

        for attempt in range(max_retries):
            try:
                sender_email = self.email_config['sender_email']
                sender_password = self.email_config['sender_password']
                smtp_server = self.email_config['smtp_server']
                smtp_port = self.email_config['smtp_port']

                msg = MIMEMultipart()
                msg['From'] = sender_email
                msg['To'] = ", ".join(recipients)
                msg['Subject'] = subject
                msg.attach(MIMEText(body, 'plain'))

                # Create connection with timeout and retry logic
                server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
                server.starttls()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipients, msg.as_string())
                server.quit()

                logger.info(f"Email alert sent to {', '.join(recipients)} for subject: '{subject}'")
                return  # Success, exit retry loop

            except smtplib.SMTPAuthenticationError as e:
                logger.error(f"SMTP Authentication failed: {e}")
                logger.error("Check your email credentials. For Gmail, use an App Password.")
                break  # Don't retry authentication errors
            except (smtplib.SMTPConnectError, smtplib.SMTPServerDisconnected) as e:
                logger.warning(f"SMTP connection issue on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to send email alert after {max_retries} attempts")
            except Exception as e:
                logger.error(f"Failed to send email alert on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} email sending attempts failed")

    def get_in_system_alerts(self) -> List[Dict]:
        """
        Retrieves all generated in-system alerts.
        """
        return self.in_system_alerts


# --- Example Usage ---
if __name__ == "__main__":
    logger.info("Running AlertEngine Example Usage...")

    # Mock historical data (similar to what forecasting models would receive)
    dates_sku1 = pd.date_range('2024-01-01', periods=100, freq='D')
    quantity_sku1 = (10 + np.sin(np.arange(100) * 2 * np.pi / 7) * 3 + np.random.normal(0, 1, 100)).clip(min=0)

    dates_sku2 = pd.date_range('2024-01-01', periods=100, freq='D')
    quantity_sku2 = (5 + np.cos(np.arange(100) * 2 * np.pi / 14) * 2 + np.random.normal(0, 0.5, 100)).clip(min=0)

    # Create mock forecast data (this would typically come from your forecasting pipeline)
    # Ensure forecast dates are in the future relative to current inventory check
    forecast_start_date = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)

    forecast_df_sku1 = pd.DataFrame({
        'date': pd.date_range(forecast_start_date, periods=10, freq='D'),
        'sku_id': 'MEALIE_2KG',
        'yhat': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],  # Decreasing demand for testing stockout
        'yhat_lower': [8, 7, 6, 5, 4, 3, 2, 1, 0, 0],  # Lower bound of demand
        'yhat_upper': [12, 11, 10, 9, 8, 7, 6, 5, 4, 3]
    })

    forecast_df_sku2 = pd.DataFrame({
        'date': pd.date_range(forecast_start_date, periods=10, freq='D'),
        'sku_id': 'COOKOIL_2LT',
        'yhat': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],  # Stable demand
        'yhat_lower': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        'yhat_upper': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    })

    mock_forecast_data = pd.concat([forecast_df_sku1, forecast_df_sku2], ignore_index=True)

    # Mock current inventory data
    mock_current_inventory = pd.DataFrame({
        'sku_id': ['MEALIE_2KG', 'COOKOIL_2LT', 'SUGAR_1KG'],
        'current_stock': [15, 50, 100]  # MEALIE_2KG will be low, COOKOIL_2LT will be fine
    })

    # --- Email Configuration Setup ---
    print("\n=== SETTING UP EMAIL CONFIGURATION ===")
    print("IMPORTANT: You need to set up environment variables for email to work:")
    print("1. Set EMAIL_SENDER to your Gmail address")
    print("2. Set EMAIL_PASSWORD to your Gmail App Password (not regular password)")
    print("3. Make sure 2-Factor Authentication is enabled on your Gmail account")
    print("4. Generate an App Password at: https://myaccount.google.com/apppasswords")

    email_config_example = {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender_email': os.getenv('EMAIL_SENDER', 'your_email@example.com'),
        'sender_password': os.getenv('EMAIL_PASSWORD', 'your_app_password')
    }

    # Test email configuration first
    alert_engine_test = AlertEngine(email_config=email_config_example)

    print("\n--- Testing SMTP Connection ---")
    connection_success = alert_engine_test.test_email_connection()

    if connection_success:
        print("✅ Email connection test passed! Proceeding with email alerts...")
        recipient_emails_example = ['chindoveleroy@gmail.com', 'lchindovve@gmail.com']

        # --- Scenario 1: Email alerts enabled ---
        print("\n--- Scenario 1: Testing with Email Alerts Enabled ---")
        alert_engine_email = AlertEngine(email_config=email_config_example)

        try:
            dispatched_alerts_email = alert_engine_email.check_and_dispatch_alerts(
                mock_current_inventory,
                mock_forecast_data,
                alert_threshold_days=5,
                recipient_emails=recipient_emails_example
            )
            print(f"\nDispatched Alerts (Email Scenario): {dispatched_alerts_email}")
            print(f"In-system alerts collected: {alert_engine_email.get_in_system_alerts()}")
        except Exception as e:
            print(f"Error during email alert scenario: {e}")
            logger.error(f"Error during email alert scenario: {e}", exc_info=True)
    else:
        print("❌ Email connection test failed! Skipping email alerts...")
        print("Please check your email configuration and try again.")

    # --- Scenario 2: Email alerts disabled (always run this) ---
    print("\n--- Scenario 2: Testing with Email Alerts Disabled ---")
    alert_engine_no_email = AlertEngine()  # No email config provided
    try:
        dispatched_alerts_no_email = alert_engine_no_email.check_and_dispatch_alerts(
            mock_current_inventory,
            mock_forecast_data,
            alert_threshold_days=5,
            recipient_emails=[]
        )
        print(f"\nDispatched Alerts (No Email Scenario): {dispatched_alerts_no_email}")
        print(f"In-system alerts collected: {alert_engine_no_email.get_in_system_alerts()}")
    except Exception as e:
        print(f"Error during no email alert scenario: {e}")
        logger.error(f"Error during no email alert scenario: {e}", exc_info=True)

    print("\nAlertEngine Example Usage Complete.")
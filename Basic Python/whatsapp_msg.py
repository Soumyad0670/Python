import pywhatkit as kit # Importing the pywhatkit library
import datetime # Importing datetime for handling date and time
from datetime import timedelta # Importing timedelta for time manipulation
def send_whatsapp_message(phone_number, message): # Function to send WhatsApp message
    try:
        now = datetime.datetime.now() # Get current time
        send_time = now + timedelta(minutes=1) # Schedule for 1 minute later
        # Extract hour and minute from the scheduled time
        hour = send_time.hour   
        minute = send_time.minute
        kit.sendwhatmsg(phone_number, message, hour, minute) # Send WhatsApp message
        # Wait for the message to be sent
        print("Message sent successfully!")
    except Exception as e: # Handle exceptions
        print(f"An error occurred: {str(e)}") # Print error message
def main():
    phone = input("Enter phone number with country code (e.g. +911234567890): ") 
    message = input("Enter your message: ")
    send_whatsapp_message(phone, message) # Call the function to send message
if __name__ == "__main__":  # Check if the script is running directly
    main() # Run the main function
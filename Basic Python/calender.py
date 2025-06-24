# Import required modules
import calendar                                              # For calendar-related functionality
import tkinter as tk                                         # Main GUI library
from tkinter import ttk                                      # Themed widgets for better look
import datetime                                              # For handling dates and times
class CalendarApp:                                           # Define the CalendarApp class
    def __init__(self, root):                                # Constructor method
        self.root = root                                     # Store the main window reference
        self.root.title("Calendar Application")              # Set window title
        self.current_date = datetime.datetime.now()          # Get current date
        self.year = self.current_date.year                   # Extract current year
        self.month = self.current_date.month                 # Extract current month
        self.create_widgets()                                # Call method to create GUI elements
    def create_widgets(self):                                # Create a frame to hold all control elements
        controls_frame = ttk.Frame(self.root)                # Create a frame for controls
        controls_frame.pack(padx=10, pady=5)                 # Add padding around frame
        ttk.Label(controls_frame, text="Year:").pack(side=tk.LEFT)  # Create and pack year label and spinbox
        self.year_spin = ttk.Spinbox(controls_frame,         # Create year selector
                                    from_=1900,              # Minimum year
                                    to=2100,                 # Maximum year
                                    width=8)                 # Width of spinbox
        self.year_spin.set(self.year)                        # Set current year
        self.year_spin.pack(side=tk.LEFT, padx=5)            # Pack with left alignment
        ttk.Label(controls_frame, text="Month:").pack(side=tk.LEFT) # Create and pack month label and spinbox
        self.month_spin = ttk.Spinbox(controls_frame,        # Create month selector
                                     from_=1,                # January
                                     to=12,                  # December
                                     width=5)                # Width of spinbox
        self.month_spin.set(self.month)                      # Set current month
        self.month_spin.pack(side=tk.LEFT, padx=5)           # Pack with left alignment
        ttk.Button(controls_frame,                           # Create button to show calendar
                  text="Show Calendar",                      # Button label
                  command=self.update_calendar,              # Button click handler
                  ).pack(side=tk.LEFT, padx=5)               # Pack with left alignment
        # Create text widget to display calendar
        self.cal_display = tk.Text(self.root,                # Create text widget
                                 height=12,                  # Height in lines
                                 width=30)                   # Width in characters
        self.cal_display.pack(padx=10, pady=5)               # Pack with padding
        self.update_calendar()                               # Show initial calendar
    def update_calendar(self):
        try:
            # Get values from spinboxes and convert to integers
            year = int(self.year_spin.get())                # Get year from spinbox
            month = int(self.month_spin.get())              # Get month from spinbox
            # Clear previous calendar display
            self.cal_display.delete(1.0, tk.END)            # Clear text widget
            # Create calendar object starting with Sunday
            cal = calendar.TextCalendar(calendar.SUNDAY)    # Create a TextCalendar object
            # Generate formatted calendar text
            cal_text = cal.formatmonth(year, month)         # Format month for the given year
            # Display calendar in text widget
            self.cal_display.insert(tk.END, cal_text)       # Insert formatted calendar text
        except ValueError:
            # Handle invalid input
            self.cal_display.delete(1.0, tk.END)            # Clear text widget
            self.cal_display.insert(tk.END, "Please enter valid year and month")  # Show error message
def show_text_calendar():
    # Get current year and month
    year = datetime.datetime.now().year                     # Get current year
    month = datetime.datetime.now().month                   # Get current month
    # Print calendar header
    print(f"\nCalendar for {calendar.month_name[month]} {year}") # Print month name and year
    
    # Print calendar in text format
    print(calendar.month(year, month))                      # Print formatted month calendar

def main():
    show_text_calendar()                                    # Show calendar in console first
    root = tk.Tk()                                          # Create main window
    app = CalendarApp(root)                                 # Create calendar application
    root.mainloop()                                         # Start the GUI event loop
if __name__ == "__main__":                                  # Check if this script is run directly
    main()                                                  # Call main function
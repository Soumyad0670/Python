import tkinter as tk
from tkinter import scrolledtext
import openai
import threading

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Python ChatGPT Interface")
        self.root.geometry("800x600")
        
        # Replace with your OpenAI API key
        self.api_key = "your-api-key-here"
        openai.api_key = self.api_key
        
        self.create_widgets()
        
    def create_widgets(self):
        # Chat history display
        self.chat_history = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=30)
        self.chat_history.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Input frame
        input_frame = tk.Frame(self.root)
        input_frame.pack(padx=10, pady=5, fill=tk.X)
        
        # Message input
        self.message_input = tk.Entry(input_frame)
        self.message_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.message_input.bind("<Return>", self.send_message)
        
        # Send button
        send_button = tk.Button(input_frame, text="Send", command=self.send_message)
        send_button.pack(side=tk.RIGHT, padx=5)
        
        # Initial greeting
        self.append_message("Assistant: Hello! How can I help you today?")
        
    def send_message(self, event=None):
        message = self.message_input.get().strip()
        if message:
            # Clear input field
            self.message_input.delete(0, tk.END)
            
            # Display user message
            self.append_message(f"You: {message}")
            
            # Get AI response in a separate thread
            threading.Thread(target=self.get_ai_response, args=(message,)).start()
    
    def get_ai_response(self, message):
        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": message}
                ]
            )
            
            # Get the assistant's response
            ai_response = response.choices[0].message.content
            
            # Display AI response
            self.root.after(0, self.append_message, f"Assistant: {ai_response}")
            
        except Exception as e:
            self.root.after(0, self.append_message, f"Error: {str(e)}")
    
    def append_message(self, message):
        self.chat_history.insert(tk.END, message + "\n\n")
        self.chat_history.see(tk.END)

def main():
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
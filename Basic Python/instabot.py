from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
import time
import random

class InstagramBot:
    def __init__(self, username, password):
        """Initialize bot with login credentials"""
        self.username = username
        self.password = password
        self.driver = webdriver.Chrome()
        self.wait = WebDriverWait(self.driver, 10)

    def login(self):
        """Login to Instagram"""
        try:
            self.driver.get('https://www.instagram.com/accounts/login/')
            time.sleep(2)

            # Enter username
            username_input = self.wait.until(
                EC.presence_of_element_located((By.NAME, "username"))
            )
            username_input.send_keys(self.username)

            # Enter password
            password_input = self.wait.until(
                EC.presence_of_element_located((By.NAME, "password"))
            )
            password_input.send_keys(self.password)
            password_input.send_keys(Keys.RETURN)
            time.sleep(5)

            # Handle save login info popup
            try:
                not_now_button = self.wait.until(
                    EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Not Now')]"))
                )
                not_now_button.click()
            except TimeoutException:
                pass

            print("Successfully logged in!")
            return True

        except Exception as e:
            print(f"Error during login: {str(e)}")
            return False

    def like_posts_by_hashtag(self, hashtag, number_of_posts=10):
        """Like posts for a given hashtag"""
        try:
            self.driver.get(f'https://www.instagram.com/explore/tags/{hashtag}/')
            time.sleep(4)

            # Click first post
            first_post = self.wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "article img")
                )
            )
            first_post.click()

            posts_liked = 0
            while posts_liked < number_of_posts:
                time.sleep(random.uniform(2, 4))  # Random delay

                try:
                    # Click like button
                    like_button = self.wait.until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "section span button")
                        )
                    )
                    like_button.click()
                    posts_liked += 1
                    print(f"Liked post {posts_liked}/{number_of_posts}")

                    # Move to next post
                    next_button = self.wait.until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "button._65Bje")
                        )
                    )
                    next_button.click()

                except Exception as e:
                    print(f"Error liking post: {str(e)}")
                    # Try to move to next post
                    try:
                        next_button = self.wait.until(
                            EC.presence_of_element_located(
                                (By.CSS_SELECTOR, "button._65Bje")
                            )
                        )
                        next_button.click()
                    except:
                        break

            print(f"Liked {posts_liked} posts with hashtag #{hashtag}")

        except Exception as e:
            print(f"Error in like_posts_by_hashtag: {str(e)}")

    def follow_user(self, username):
        """Follow a specific user"""
        try:
            self.driver.get(f'https://www.instagram.com/{username}/')
            time.sleep(3)

            follow_button = self.wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "button._acan._acap._acas")
                )
            )
            follow_button.click()
            print(f"Successfully followed {username}")
            time.sleep(random.uniform(2, 4))

        except Exception as e:
            print(f"Error following user {username}: {str(e)}")

    def close(self):
        """Close the browser"""
        self.driver.quit()

def main():
    # Replace with your Instagram credentials
    USERNAME = "your_username"
    PASSWORD = "your_password"

    # Initialize and login
    bot = InstagramBot(USERNAME, PASSWORD)
    if bot.login():
        try:
            # Like some posts with hashtag
            bot.like_posts_by_hashtag("python", 5)

            # Follow specific users
            users_to_follow = ["user1", "user2"]
            for user in users_to_follow:
                bot.follow_user(user)
                time.sleep(random.uniform(30, 60))  # Random delay between follows

        finally:
            # Always close the browser
            bot.close()

if __name__ == "__main__":
    main()
    
import os
from dotenv import load_dotenv

load_dotenv()
USERNAME = os.getenv('INSTAGRAM_USERNAME')
PASSWORD = os.getenv('INSTAGRAM_PASSWORD')
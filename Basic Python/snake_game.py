import pygame
import random
import time

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = WINDOW_WIDTH // GRID_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // GRID_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class Snake:
    def __init__(self):
        self.length = 1
        self.positions = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.color = GREEN
        self.score = 0

    def update_position(self):
        current = self.positions[0]
        x, y = self.direction
        new = (((current[0] + x) % GRID_WIDTH), (current[1] + y) % GRID_HEIGHT)
        
        if new in self.positions[2:]:
            return False
        else:
            self.positions.insert(0, new)
            if len(self.positions) > self.length:
                self.positions.pop()
        return True

    def reset(self):
        self.length = 1
        self.positions = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.score = 0

class Apple:
    def __init__(self):
        self.position = (0, 0)
        self.color = RED
        self.randomize_position()

    def randomize_position(self):
        self.position = (random.randint(0, GRID_WIDTH-1), 
                        random.randint(0, GRID_HEIGHT-1))

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        self.snake = Snake()
        self.apple = Apple()
        self.game_speed = 10

    def handle_keys(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and self.snake.direction != DOWN:
                    self.snake.direction = UP
                elif event.key == pygame.K_DOWN and self.snake.direction != UP:
                    self.snake.direction = DOWN
                elif event.key == pygame.K_LEFT and self.snake.direction != RIGHT:
                    self.snake.direction = LEFT
                elif event.key == pygame.K_RIGHT and self.snake.direction != LEFT:
                    self.snake.direction = RIGHT

    def draw(self):
        self.screen.fill(BLACK)
        
        # Draw snake
        for position in self.snake.positions:
            rect = (position[0] * GRID_SIZE, position[1] * GRID_SIZE, 
                   GRID_SIZE-2, GRID_SIZE-2)
            pygame.draw.rect(self.screen, self.snake.color, rect)
            
        # Draw apple
        rect = (self.apple.position[0] * GRID_SIZE, 
                self.apple.position[1] * GRID_SIZE, 
                GRID_SIZE-2, GRID_SIZE-2)
        pygame.draw.rect(self.screen, self.apple.color, rect)
        
        # Draw score
        score_text = self.font.render(f'Score: {self.snake.score}', True, WHITE)
        self.screen.blit(score_text, (5, 5))
        
        pygame.display.update()

    def run(self):
        while True:
            self.handle_keys()
            
            if not self.snake.update_position():
                self.snake.reset()
                self.apple.randomize_position()
                
            if self.snake.positions[0] == self.apple.position:
                self.snake.length += 1
                self.snake.score += 1
                self.apple.randomize_position()
                
            self.draw()
            self.clock.tick(self.game_speed)

if __name__ == '__main__':
    game = Game()
    game.run()
import pygame
import random
import numpy as np
import pickle


# === Neural Network ===

class NeuralNetwork:
    def __init__(self, n=3, layer_size=[4, 12, 1]):
        self.n = n
        self.layer_size = layer_size
        self.w = []
        self.b = []
        for i in range(n - 1):
            self.w.append(np.random.randn(layer_size[i + 1], layer_size[i]))
            self.b.append(np.random.randn(layer_size[i + 1], 1))

    def forward(self, input):
        a = input
        for i in range(self.n - 1):
            z = np.dot(self.w[i], a) + self.b[i]
            if i == self.n - 2:
                a = 1 / (1 + np.exp(-z))  # Sigmoid
            else:
                a = np.tanh(z)
        return a

    def decide_action(self, input):
        return self.forward(input) > 0.5

    def copy(self):
        new_net = NeuralNetwork(self.n, self.layer_size)
        new_net.w = [np.copy(w) for w in self.w]
        new_net.b = [np.copy(b) for b in self.b]
        return new_net

    def mutate(self, rate=0.5):
        for i in range(len(self.w)):
            self.w[i] += np.random.randn(*self.w[i].shape) * rate
            self.b[i] += np.random.randn(*self.b[i].shape) * rate

    def save(self, filename="best_network.pkl", score=0):
        with open(filename, "wb") as f:
            pickle.dump((self.n, self.layer_size, self.w, self.b, score), f)

    @staticmethod
    def load(filename="best_network.pkl"):
        with open(filename, "rb") as f:
            n, layer_size, w, b, score = pickle.load(f)
            net = NeuralNetwork(n, layer_size)
            net.w = w
            net.b = b
            return net, score
# === Bird ===

class bird:
    def __init__(self, net=None):
        self.net = net if net else NeuralNetwork()
        self.alive = True
        self.x = 250
        self.y = 300
        self.radius = 20
        self.x_vel = 0
        self.y_vel = 0
        self.x_aks = 0
        self.y_aks = 600.81
        self.score = 0

    def update_vel(self, dt):
        self.x_vel = min(600, self.x_vel + dt * self.x_aks)
        self.y_vel = min(600, self.y_vel + dt * self.y_aks)

    def update_pos(self, dt):
        self.x = self.x + dt * self.x_vel
        self.y = self.y + dt * self.y_vel

    def jump(self):
        self.y_vel = -200

    def validate_hight(self, height):
        return (self.y - self.radius > 0) and (self.y + self.radius < height)

    def draw(self, screen, color):
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.radius)

    def update_score_time(self, dt, next_pipe=None):
        self.score += dt / 10
        if next_pipe is not None:
            center_gap = next_pipe.y_top_gap + next_pipe.gap_height / 2
            self.score += 1 - abs(self.y - center_gap) / 600  # maks 1, lavere jo lenger unna

    def dead(self):
        self.alive = False


# === Pipe ===

class pipe:
    def __init__(self, width, height):
        self.x = width
        self.x_vel = -100
        self.w = 30
        self.gap_height = random.randint(130, 300)
        self.y_top_gap = random.randint(0, height - self.gap_height)

    def update_pos(self, dt):
        self.x = self.x + dt * self.x_vel

    def draw(self, screen, color, height):
        pygame.draw.rect(screen, color, (self.x, 0, self.w, self.y_top_gap))
        pygame.draw.rect(screen, color, (self.x, self.y_top_gap + self.gap_height, self.w, height))


# === Simulation ===

def simulate_generation(birds):
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    WHITE = (255, 255, 255)
    BLUE = (0, 100, 255)
    RED = (255, 0, 0)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Flappy Bird AI")
    clock = pygame.time.Clock()
    previous_ticks = pygame.time.get_ticks()

    obstacles = [pipe(WIDTH, HEIGHT)]
    running = True
    while running:
        dt_ms = clock.tick(120)
        dt = dt_ms / 1000
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Update pipes
        for obstacle in obstacles:
            obstacle.update_pos(dt)

        # Update birds
        alive_count = 0
        for player in birds:
            if player.alive:
                alive_count += 1
                
                player.update_vel(dt)
                player.update_pos(dt)

                # Think
                if len(obstacles) > 0:
                    next_pipe = next((o for o in obstacles if o.x + o.w > player.x), obstacles[0])
                    player.update_score_time((pygame.time.get_ticks() - previous_ticks), next_pipe)
                    inputs = np.array([
                        player.y / HEIGHT,
                        next_pipe.x / WIDTH,
                        next_pipe.gap_height / HEIGHT,
                        next_pipe.y_top_gap / HEIGHT
                    ]).reshape(-1, 1)
                    if player.net.decide_action(inputs):
                        player.jump()
                else:
                    player.update_score_time((pygame.time.get_ticks() - previous_ticks))
                # Collision
                for obstacle in obstacles:
                    if (obstacle.x < player.x + player.radius) and (obstacle.x + obstacle.w > player.x - player.radius):
                        if (player.y - player.radius < obstacle.y_top_gap) or (player.y + player.radius > obstacle.y_top_gap + obstacle.gap_height):
                            player.dead()

                if not player.validate_hight(HEIGHT):
                    player.dead()

                player.draw(screen, BLUE)

        if alive_count == 0:
            running = False

        # Remove offscreen pipes and spawn new
        obstacles = [o for o in obstacles if o.x > -o.w]
        if obstacles[-1].x < 540:
            obstacles.append(pipe(WIDTH, HEIGHT))

        for obstacle in obstacles:
            obstacle.draw(screen, RED, HEIGHT)

        pygame.display.flip()
        previous_ticks = pygame.time.get_ticks()

    pygame.quit()


# === Training Loop ===

def train(num_generations=1000, population_size=100):
    best_network = NeuralNetwork()
    for generation in range(num_generations):
        birds = [bird(best_network.copy()) for _ in range(population_size)]
        for b in birds:
            b.net.mutate(rate=1/(generation+1))

        simulate_generation(birds)
        best_bird = max(birds, key=lambda b: b.score)
        best_network = best_bird.net.copy()
        print(f"Generation {generation + 1} | Best Score: {int(best_bird.score)}")
        try:
            _, best_score_all_time = NeuralNetwork.load("best_network.pkl")
        except FileNotFoundError:
            best_score_all_time = -1

        if best_bird.score > best_score_all_time:
            best_bird.net.save("best_network.pkl", score=best_bird.score)
            print(f"Ny all-time high score: {best_bird.score:.1f} – nettverk lagret!")

def run_best():
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Flappy Bird - Beste Fuglen")
    clock = pygame.time.Clock()

    try:
        best_net, best_score = NeuralNetwork.load("best_network.pkl")
    except FileNotFoundError:
        print("Ingen lagret fugl funnet!")
        return

    bird = bird(network=best_net)
    pipes = [pipe(WIDTH, HEIGHT)]

    running = True
    while running and bird.alive:
        dt = clock.tick(60) / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))

        # Pipe logic
        for pipe in pipes:
            pipe.update_pos(dt)
            pipe.draw(screen, (255, 0, 0), HEIGHT)
        if pipes[-1].x < 540:
            pipes.append(pipe(WIDTH, HEIGHT))
        pipes = [p for p in pipes if p.x > -p.w]

        # Bird logic
        bird.update_score_time(dt)
        bird.update_vel(dt)
        bird.update_pos(dt)
        bird.draw(screen, (0, 100, 255))
        if not bird.validate_hight(HEIGHT):
            bird.dead()

        for pipe in pipes:
            if pipe.x < bird.x + bird.radius and pipe.x + pipe.w > bird.x - bird.radius:
                if bird.y - bird.radius < pipe.y_top_gap or bird.y + bird.radius > pipe.y_top_gap + pipe.gap_height:
                    bird.dead()

        # Decide action
        pipe = next((p for p in pipes if p.x + p.w > bird.x), None)
        if pipe:
            input_vec = np.array([[bird.y], [pipe.x], [pipe.gap_height], [pipe.y_top_gap]])
            if bird.net.decide_action(input_vec):
                bird.jump()

        pygame.display.flip()

    pygame.quit()


train()

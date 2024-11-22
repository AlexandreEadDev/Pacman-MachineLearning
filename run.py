from PacMan_Game.main import GameController

if __name__ == "__main__":
    # Create an instance of GameController and run the game
    game = GameController()
    game.startGame()
    while True:
        game.update()

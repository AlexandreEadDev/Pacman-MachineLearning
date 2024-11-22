from PacMan.main import GameController


def main():
    game = GameController()
    game.startGame()
    while True:
        game.update()


if __name__ == "__main__":
    main()
